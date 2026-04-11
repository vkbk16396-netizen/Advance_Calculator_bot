import os
import re
import math
import asyncio
import sqlite3
from io import BytesIO

import numpy as np
import sympy as sp
import matplotlib
import matplotlib.pyplot as plt
import requests
import telebot
from fastapi import FastAPI, Request
from sympy.stats import Normal, density
from google import genai

# Prevent Matplotlib from attempting to open GUI windows on a server
matplotlib.use('Agg')

app = FastAPI()

# ================= TOKEN =================
TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("BOT_TOKEN is not set")

bot = telebot.TeleBot(TOKEN, parse_mode=None)

# ================= GEMINI =================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# ================= DATABASE =================
conn = sqlite3.connect("history.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute(
    "CREATE TABLE IF NOT EXISTS history (chat_id INTEGER, expr TEXT, result TEXT)"
)
conn.commit()

chat_variables = {}
db_lock = asyncio.Lock()

# ================= ASYNC HELPERS =================
async def async_send(chat_id: int, text: str):
    """Helper to send messages asynchronously without blocking the event loop."""
    await asyncio.to_thread(bot.send_message, chat_id, text)

# ================= PREPROCESS =================
def preprocess(expr: str) -> str:
    expr = expr.strip()
    if not expr:
        return ""
    expr = expr.lower()
    
    # common math symbols
    expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
    
    # allow sin 30 / cos 60 / tan 45
    expr = re.sub(r"\b(sin|cos|tan)\s+(-?\d+(\.\d+)?)\b", r"\1(\2)", expr)
    
    superscripts = {
        "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
        "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    }
    
    new_expr = ""
    power_mode = False
    for ch in expr:
        if ch in superscripts:
            if not power_mode:
                new_expr += "**"
                power_mode = True
            new_expr += superscripts[ch]
        else:
            power_mode = False
            new_expr += ch
            
    expr = new_expr
    
    fractions = {
        "½": "1/2", "¼": "1/4", "¾": "3/4", "⅓": "1/3", "⅔": "2/3",
        "⅕": "1/5", "⅖": "2/5", "⅗": "3/5", "⅘": "4/5", "⅙": "1/6",
        "⅚": "5/6", "⅛": "1/8", "⅜": "3/8", "⅝": "5/8", "⅞": "7/8",
    }
    for k, v in fractions.items():
        expr = expr.replace(k, v)
        
    # mixed fraction like 1 1/2 -> (1+1/2)
    expr = re.sub(r"(\d+)\s+(\d+/\d+)", r"(\1+\2)", expr)
    
    # percent
    expr = re.sub(r"(\d+(\.\d+)?)\s*%", r"(\1/100)", expr)
    
    return expr

# ================= SAFE =================
def safe_locals(chat_id: int) -> dict:
    x = sp.symbols("x")
    
    # degree-based trig for user-friendly calculator input
    def sin_deg(v): return sp.sin(sp.pi * v / 180)
    def cos_deg(v): return sp.cos(sp.pi * v / 180)
    def tan_deg(v): return sp.tan(sp.pi * v / 180)
    
    return {
        "x": x,
        "pi": sp.pi,
        "e": sp.E,
        "sin": sin_deg,
        "cos": cos_deg,
        "tan": tan_deg,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "sqrt": sp.sqrt,
        "log": sp.log,
        "ln": sp.log,
        "abs": sp.Abs,
        "factorial": sp.factorial,
        "diff": sp.diff,
        "integrate": sp.integrate,
        "Matrix": sp.Matrix,
        "det": lambda m: sp.Matrix(m).det(),
        "inv": lambda m: sp.Matrix(m).inv(),
        "transpose": lambda m: sp.Matrix(m).T,
        "mean": lambda *a: sum(a) / len(a),
        "variance": lambda *a: float(np.var(a)),
        "std": lambda *a: float(np.std(a)),
        "Normal": Normal,
        "pdf": density,
        **chat_variables.get(chat_id, {}),
    }

# ================= EVALUATE =================
def evaluate(expr: str, chat_id: int):
    expr = preprocess(expr)
    if not expr:
        return None
    
    safe = safe_locals(chat_id)
    
    # Matrix(...)
    if expr.startswith("matrix"):
        try:
            inside = expr[len("matrix"):].strip()
            if not inside.startswith("(") or not inside.endswith(")"):
                return None
            matrix_data = sp.sympify(inside[1:-1], locals=safe)
            return str(sp.Matrix(matrix_data))
        except Exception:
            return None
            
    try:
        res = sp.sympify(expr, locals=safe)
        
        # simplify exact trig etc.
        try:
            res = sp.simplify(res)
        except Exception:
            pass
            
        if getattr(res, "free_symbols", None):
            return str(res)
        return float(res.evalf())
    except Exception:
        return None

# ================= GEMINI =================
def gemini_reply(text: str) -> str:
    try:
        if not client:
            return "⚠️ Gemini API key not configured"
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=text
        )
        if not response or not getattr(response, "text", None):
            return "⚠️ Empty Gemini response"
        return response.text.strip()
    except Exception as e:
        print("GEMINI ERROR:", e)
        return f"⚠️ Gemini error\n{str(e)[:100]}"

# ================= SAVE =================
async def save_history(chat_id: int, expr: str, result):
    async with db_lock:
        await asyncio.to_thread(
            cursor.execute,
            "INSERT INTO history VALUES (?, ?, ?)",
            (chat_id, expr, str(result))
        )
        await asyncio.to_thread(conn.commit)

# ================= PLOT =================
def plot(expr: str):
    x = sp.symbols("x")
    funcs = [f.strip() for f in expr.split(",") if f.strip()]
    if not funcs:
        return None
        
    xs = np.linspace(-10, 10, 400)
    plt.figure()
    plotted = False
    
    for f in funcs:
        try:
            parsed = sp.sympify(preprocess(f), locals={"x": x, "pi": sp.pi, "e": sp.E})
            f_np = sp.lambdify(x, parsed, "numpy")
            ys = f_np(xs)
            ys = np.array(ys, dtype=float)
            ys[np.abs(ys) > 1e6] = np.nan
            plt.plot(xs, ys, label=f)
            plotted = True
        except Exception:
            pass
            
    if not plotted:
        plt.close()
        return None
        
    plt.axhline(0)
    plt.axvline(0)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid(True)
    plt.legend()
    
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return buf

# ================= UTIL =================
def convert_units(text: str):
    lower = text.lower().strip()
    parts = lower.split()
    if len(parts) != 4 or parts[2] != "to":
        return None
        
    value_str, unit1, _, unit2 = parts
    try:
        value = float(value_str)
    except Exception:
        return None
        
    conv = {
        ("km", "m"): 1000,
        ("m", "km"): 0.001,
        ("kg", "g"): 1000,
        ("g", "kg"): 0.001,
        ("cm", "m"): 0.01,
        ("m", "cm"): 100,
        ("mm", "m"): 0.001,
        ("m", "mm"): 1000,
    }
    
    factor = conv.get((unit1, unit2))
    if factor is None:
        return None
    return value * factor

def solve_equation(raw_expr: str, chat_id: int):
    eq = preprocess(raw_expr)
    safe = safe_locals(chat_id)
    x = safe["x"]
    
    if not eq:
        return None
        
    if "=" in eq:
        left, right = eq.split("=", 1)
        expr = sp.sympify(left, locals=safe) - sp.sympify(right, locals=safe)
    else:
        expr = sp.sympify(eq, locals=safe)
        
    return sp.solve(expr, x)

def shorten_url(url: str) -> str:
    url = url.strip()
    if not url:
        return "❌ Please provide a URL"
    try:
        response = requests.get(
            "http://tinyurl.com/api-create.php",
            params={"url": url},
            timeout=10
        )
        return response.text.strip()
    except Exception as e:
        return f"❌ URL shortener error: {e}"

def get_history(chat_id: int):
    cursor.execute("SELECT expr, result FROM history WHERE chat_id=?", (chat_id,))
    return cursor.fetchall()

# ================= WEBHOOK =================
@app.post("/webhook")
async def webhook(request: Request):
    try:
        data = await request.json()
    except Exception:
        return {"ok": False, "error": "Invalid JSON"}
        
    if "message" not in data:
        return {"ok": True}
        
    msg = data["message"]
    chat_id = msg["chat"]["id"]
    text = msg.get("text", "").strip()
    lower = text.lower().strip()
    
    if not text:
        return {"ok": True}

    # ===== START (UNCHANGED) =====
    if lower == "/start":
        await async_send(
            chat_id,
            "✨ *Welcome to Most Advanced Calculator* 🤖\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "🚀 *Fast • Powerful • Intelligent*\n\n"
            "🧮 Solve any calculation instantly\n"
            "📊 Plot graphs & analyze functions\n"
            "📐 Perform calculus & algebra\n"
            "📦 Work with matrices & statistics\n\n"
            "👨‍💻 *Developed by:* @Sudhakaran12\n\n"
            "👉 Use /help to explore all features\n"
            "💡 *Try:* `2²`, `cos 60`, `sin(30)`"
        )

    # ===== HELP (REDESIGNED) =====
    elif lower == "/help":
        await async_send(
            chat_id,
            "📘 MOST ADVANCED CALCULATOR - HELP MENU\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "This bot can do calculations, algebra, calculus, matrices, graphs, unit conversion, URL shortening, history export, and AI replies.\n\n"
            "🧮 1) BASIC CALCULATIONS\n"
            "Use normal math expressions.\n"
            "Examples:\n"
            "2+2\n"
            "15-7\n"
            "8*9\n"
            "10/4\n"
            "2^5\n"
            "2²\n"
            "3³\n"
            "50%\n"
            "1 1/2\n"
            "¾ + ¼\n\n"
            "📐 2) TRIGONOMETRY\n"
            "Use sin, cos, tan.\n"
            "Examples:\n"
            "sin(30)\n"
            "cos(60)\n"
            "tan(45)\n"
            "sin 30\n"
            "cos 90\n\n"
            "📘 3) ALGEBRA WITH x\n"
            "Use x in expressions.\n"
            "Examples:\n"
            "x+2\n"
            "x^2+5*x+6\n"
            "sqrt(x)\n"
            "log(x)\n\n"
            "📊 4) CALCULUS\n"
            "Differentiate and integrate expressions.\n"
            "Examples:\n"
            "diff(x^2,x)\n"
            "diff(sin(x),x)\n"
            "integrate(x^2,x)\n"
            "integrate(cos(x),x)\n\n"
            "🧠 5) EQUATION SOLVER\n"
            "Use /solve to solve equations.\n"
            "Examples:\n"
            "/solve x^2-4=0\n"
            "/solve x^2+5*x+6=0\n"
            "/solve x^2-9\n\n"
            "📦 6) MATRICES\n"
            "Create and work with matrices.\n"
            "Examples:\n"
            "Matrix([[1,2],[3,4]])\n"
            "det([[1,2],[3,4]])\n"
            "inv([[1,2],[3,4]])\n"
            "transpose([[1,2],[3,4]])\n\n"
            "📈 7) GRAPH PLOTTING\n"
            "Use /plot to draw one or more functions.\n"
            "Examples:\n"
            "/plot x^2\n"
            "/plot sin(x)\n"
            "/plot sin(x),cos(x)\n"
            "/plot x^2,x^3\n\n"
            "📉 8) STATISTICS\n"
            "Use mean, variance, std.\n"
            "Examples:\n"
            "mean(2,4,6,8)\n"
            "variance(1,2,3,4,5)\n"
            "std(1,2,3,4,5)\n\n"
            "🔔 9) NORMAL DISTRIBUTION\n"
            "You can use Normal and pdf.\n"
            "Examples:\n"
            "pdf(Normal('X',0,1))(0)\n\n"
            "🔄 10) UNIT CONVERSION\n"
            "Use: value unit to unit\n"
            "Supported now:\n"
            "km to m\n"
            "m to km\n"
            "kg to g\n"
            "g to kg\n"
            "cm to m\n"
            "m to cm\n"
            "mm to m\n"
            "m to mm\n"
            "Examples:\n"
            "10 km to m\n"
            "5000 m to km\n"
            "2 kg to g\n\n"
            "🔗 11) SHORT URL\n"
            "Use /short followed by a URL.\n"
            "Example:\n"
            "/short https://example.com\n\n"
            "📤 12) EXPORT HISTORY\n"
            "Use /export to download your saved calculations.\n"
            "Example:\n"
            "/export\n\n"
            "🤖 13) AI REPLIES\n"
            "If expression is not solved by calculator, bot sends it to Gemini AI.\n"
            "Example:\n"
            "Explain Newton's law\n"
            "Write a Python program\n\n"
            "💾 14) HISTORY\n"
            "Every solved calculation is saved automatically.\n"
            "Use /export to get all saved history.\n\n"
            "✨ 15) SPECIAL SYMBOLS SUPPORTED\n"
            "The bot supports many special math inputs.\n"
            "Examples:\n"
            "² ³ ⁴ ⁵ ⁶ ⁷ ⁸ ⁹\n"
            "½ ¼ ¾ ⅓ ⅔ ⅕ ⅖ ⅗ ⅘ ⅙ ⅚ ⅛ ⅜ ⅝ ⅞\n"
            "× ÷ %\n\n"
            "⚡ QUICK EXAMPLES TO TRY\n"
            "2² + 5²\n"
            "sin 30\n"
            "diff(x^3,x)\n"
            "/solve x^2-9=0\n"
            "/plot sin(x),cos(x)\n"
            "10 km to m\n"
            "mean(10,20,30)\n\n"
            "👨‍💻 Developer: @Sudhakaran12"
        )

    elif lower.startswith("/short"):
        url = text[6:].strip()
        result = await asyncio.to_thread(shorten_url, url)
        await async_send(chat_id, result)

    elif lower.startswith("/plot"):
        img = await asyncio.to_thread(plot, text[5:].strip())
        if img is None:
            await async_send(chat_id, "❌ Invalid function(s) for plotting")
        else:
            await asyncio.to_thread(bot.send_photo, chat_id, img)

    elif lower.startswith("/solve"):
        try:
            expr = text[6:].strip()
            if not expr:
                await async_send(chat_id, "❌ Please provide an equation")
            else:
                res = solve_equation(expr, chat_id)
                await async_send(chat_id, f"🧠 {res}")
        except Exception as e:
            await async_send(chat_id, f"❌ Solve error: {str(e)[:120]}")

    elif lower == "/export":
        async with db_lock:
            rows = await asyncio.to_thread(get_history, chat_id)

        if not rows:
            await async_send(chat_id, "❌ No history")
        else:
            file_name = f"history_{chat_id}.txt"
            
            # Write to file
            with open(file_name, "w", encoding="utf-8") as f:
                for e, r in rows:
                    f.write(f"{e} = {r}\n")
                    
            # Send file to telegram
            with open(file_name, "rb") as f:
                await asyncio.to_thread(bot.send_document, chat_id, f)
                
            # Clean up the file so server storage doesn't fill up
            if os.path.exists(file_name):
                os.remove(file_name)

    else:
        # unit conversion
        conversion = convert_units(text)
        if conversion is not None:
            await async_send(chat_id, f"🔄 {conversion}")
            return {"ok": True}

        # calculator first
        result = evaluate(text, chat_id)
        if result is not None:
            await save_history(chat_id, text, result)
            await async_send(chat_id, f"✅ {result}")
        else:
            reply = await asyncio.to_thread(gemini_reply, text)
            await async_send(chat_id, reply)

    return {"ok": True}

@app.get("/")
async def root():
    return {"status": "Bot is running"}
    
