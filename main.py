import os
import re
import math
import asyncio
import sqlite3
import concurrent.futures
from io import BytesIO

import numpy as np
import sympy as sp
import matplotlib
import matplotlib.pyplot as plt
import requests
import telebot
from fastapi import FastAPI, Request, BackgroundTasks
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
    
    expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
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
        
    expr = re.sub(r"(\d+)\s+(\d+/\d+)", r"(\1+\2)", expr)
    expr = re.sub(r"(\d+(\.\d+)?)\s*%", r"(\1/100)", expr)
    
    return expr

# ================= SAFE LOCALS =================
def safe_locals(chat_id: int) -> dict:
    x = sp.symbols("x")
    
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
        "limit": sp.limit,
        "series": sp.series,
        "bin": lambda val: bin(int(val)),
        "hex": lambda val: hex(int(val)),
        "oct": lambda val: oct(int(val)),
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

# ================= TIMEOUT EXECUTOR =================
def run_with_timeout(func, *args, timeout=3.0):
    """Prevents malicious/complex math from hanging the server."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return "TIMEOUT"
        except Exception:
            return None

# ================= EVALUATE =================
def _eval_math(expr: str, chat_id: int):
    safe = safe_locals(chat_id)
    if expr.startswith("matrix"):
        inside = expr[len("matrix"):].strip()
        if not inside.startswith("(") or not inside.endswith(")"):
            return None
        matrix_data = sp.sympify(inside[1:-1], locals=safe)
        return str(sp.Matrix(matrix_data))
        
    res = sp.sympify(expr, locals=safe)
    try:
        res = sp.simplify(res)
    except Exception:
        pass
        
    if getattr(res, "free_symbols", None) or isinstance(res, str):
        return str(res)
    return float(res.evalf())

def evaluate(expr: str, chat_id: int):
    expr = preprocess(expr)
    if not expr:
        return None
    return run_with_timeout(_eval_math, expr, chat_id)

# ================= GEMINI =================
def gemini_reply(text: str) -> str:
    try:
        if not client:
            return "⚠️ Gemini API key not configured"
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=text
        )
        if not response or not getattr(response, "text", None):
            return "⚠️ Empty Gemini response"
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini error\n{str(e)[:100]}"

# ================= DATABASE OPS =================
async def save_history(chat_id: int, expr: str, result):
    async with db_lock:
        await asyncio.to_thread(
            cursor.execute,
            "INSERT INTO history VALUES (?, ?, ?)",
            (chat_id, expr, str(result))
        )
        await asyncio.to_thread(conn.commit)

async def clear_history(chat_id: int):
    async with db_lock:
        await asyncio.to_thread(cursor.execute, "DELETE FROM history WHERE chat_id=?", (chat_id,))
        await asyncio.to_thread(conn.commit)

def get_history(chat_id: int):
    cursor.execute("SELECT expr, result FROM history WHERE chat_id=?", (chat_id,))
    return cursor.fetchall()

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
        
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
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
        
    conv_multipliers = {
        ("km", "m"): 1000, ("m", "km"): 0.001,
        ("kg", "g"): 1000, ("g", "kg"): 0.001,
        ("cm", "m"): 0.01, ("m", "cm"): 100,
        ("mm", "m"): 0.001, ("m", "mm"): 1000,
        ("km/h", "mph"): 0.621371, ("mph", "km/h"): 1.60934
    }
    
    # Check standard multipliers
    if (unit1, unit2) in conv_multipliers:
        return value * conv_multipliers[(unit1, unit2)]
        
    # Check temperature
    if unit1 == "c" and unit2 == "f":
        return (value * 9/5) + 32
    if unit1 == "f" and unit2 == "c":
        return (value - 32) * 5/9
        
    return None

def _solve_eq(raw_expr: str, chat_id: int):
    eq = preprocess(raw_expr)
    safe = safe_locals(chat_id)
    x = safe["x"]
    if "=" in eq:
        left, right = eq.split("=", 1)
        expr = sp.sympify(left, locals=safe) - sp.sympify(right, locals=safe)
    else:
        expr = sp.sympify(eq, locals=safe)
    return sp.solve(expr, x)

def solve_equation(raw_expr: str, chat_id: int):
    return run_with_timeout(_solve_eq, raw_expr, chat_id)

def shorten_url(url: str) -> str:
    url = url.strip()
    if not url:
        return "❌ Please provide a URL"
    try:
        response = requests.get("http://tinyurl.com/api-create.php", params={"url": url}, timeout=5)
        return response.text.strip()
    except Exception as e:
        return f"❌ URL shortener error: {e}"

# ================= BACKGROUND TASK WORKER =================
async def process_message(msg: dict):
    chat_id = msg["chat"]["id"]
    text = msg.get("text", "").strip()
    lower = text.lower().strip()
    
    if not text:
        return

    if lower == "/start":
        await async_send(
            chat_id,
            "✨ *Welcome to Most Advanced Calculator* 🤖\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "🚀 *Fast • Powerful • Intelligent*\n\n"
            "🧮 Solve calculations & base conversions (hex, bin)\n"
            "📊 Plot graphs & analyze functions\n"
            "📐 Perform calculus (diff, integrate, limit, series)\n"
            "📦 Work with matrices & statistics\n\n"
            "👨‍💻 *Developed by:* @Sudhakaran12\n\n"
            "👉 Use /help to explore all features\n"
            "💡 *Try:* `2²`, `cos 60`, `limit(sin(x)/x, x, 0)`"
        )

    elif lower == "/help":
        await async_send(chat_id, "📘 Use standard math operations, or try special functions like `limit()`, `series()`, `hex()`, `bin()`. Use `/plot x^2`, `/solve x^2-4=0`, `/export` for history, or `/clear` to wipe history. Developer: @Sudhakaran12")

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
        expr = text[6:].strip()
        if not expr:
            await async_send(chat_id, "❌ Please provide an equation")
        else:
            res = await asyncio.to_thread(solve_equation, expr, chat_id)
            if res == "TIMEOUT":
                await async_send(chat_id, "❌ Equation is too complex (Timed out).")
            elif res is not None:
                await async_send(chat_id, f"🧠 {res}")
            else:
                await async_send(chat_id, "❌ Could not solve equation.")

    elif lower == "/export":
        async with db_lock:
            rows = await asyncio.to_thread(get_history, chat_id)

        if not rows:
            await async_send(chat_id, "❌ No history found.")
        else:
            # Using memory buffer to save disk I/O
            buf = BytesIO()
            for e, r in rows:
                buf.write(f"{e} = {r}\n".encode("utf-8"))
            buf.seek(0)
            buf.name = f"history_{chat_id}.txt"
            await asyncio.to_thread(bot.send_document, chat_id, buf)
            
    elif lower == "/clear":
        await clear_history(chat_id)
        await async_send(chat_id, "🗑️ Your calculation history has been cleared!")

    else:
        # 1. Check unit conversion
        conversion = convert_units(text)
        if conversion is not None:
            await async_send(chat_id, f"🔄 {conversion}")
            return

        # 2. Check Calculator
        result = await asyncio.to_thread(evaluate, text, chat_id)
        if result == "TIMEOUT":
            await async_send(chat_id, "❌ Calculation took too long and was aborted.")
        elif result is not None:
            await save_history(chat_id, text, result)
            await async_send(chat_id, f"✅ {result}")
        else:
            # 3. Fallback to Gemini
            reply = await asyncio.to_thread(gemini_reply, text)
            await async_send(chat_id, reply)

# ================= WEBHOOK =================
@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        data = await request.json()
    except Exception:
        return {"ok": False, "error": "Invalid JSON"}
        
    if "message" in data:
        # Immediately acknowledge Telegram and pass work to background task
        background_tasks.add_task(process_message, data["message"])
        
    return {"ok": True}

@app.get("/")
async def root():
    return {"status": "Bot is running"}
    
