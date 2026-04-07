import os
import re
import asyncio
import sqlite3
from io import BytesIO

import numpy as np
import sympy as sp
import matplotlib
matplotlib.use("Agg")  # ← MUST be before pyplot import on headless servers
import matplotlib.pyplot as plt
import requests
import telebot
from fastapi import FastAPI, Request
from sympy.stats import Normal, density
from google import genai

app = FastAPI()

# ================= TOKENS =================
TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("BOT_TOKEN is not set")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

bot = telebot.TeleBot(TOKEN, parse_mode=None)
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# ================= DATABASE =================
conn = sqlite3.connect("history.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute(
    "CREATE TABLE IF NOT EXISTS history (chat_id INTEGER, expr TEXT, result TEXT)"
)
conn.commit()

chat_variables: dict[int, dict] = {}
db_lock = asyncio.Lock()


# ================= PREPROCESS =================
SUPERSCRIPTS = {"⁰":"0","¹":"1","²":"2","³":"3","⁴":"4","⁵":"5","⁶":"6","⁷":"7","⁸":"8","⁹":"9"}
FRACTIONS = {"½":"1/2","¼":"1/4","¾":"3/4","⅓":"1/3","⅔":"2/3","⅕":"1/5","⅖":"2/5",
             "⅗":"3/5","⅘":"4/5","⅙":"1/6","⅚":"5/6","⅛":"1/8","⅜":"3/8","⅝":"5/8","⅞":"7/8"}

def preprocess(expr: str) -> str:
    expr = expr.strip().lower()
    if not expr:
        return ""

    expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
    expr = re.sub(r"\b(sin|cos|tan)\s+(-?\d+(\.\d+)?)\b", r"\1(\2)", expr)

    # superscripts
    new_expr, power_mode = "", False
    for ch in expr:
        if ch in SUPERSCRIPTS:
            if not power_mode:
                new_expr += "**"
                power_mode = True
            new_expr += SUPERSCRIPTS[ch]
        else:
            power_mode = False
            new_expr += ch
    expr = new_expr

    for k, v in FRACTIONS.items():
        expr = expr.replace(k, v)

    expr = re.sub(r"(\d+)\s+(\d+/\d+)", r"(\1+\2)", expr)   # mixed fractions
    expr = re.sub(r"(\d+(\.\d+)?)\s*%", r"(\1/100)", expr)   # percent
    return expr


# ================= SAFE LOCALS =================
def safe_locals(chat_id: int) -> dict:
    x = sp.symbols("x")

    def sin_deg(v): return sp.sin(sp.pi * v / 180)
    def cos_deg(v): return sp.cos(sp.pi * v / 180)
    def tan_deg(v): return sp.tan(sp.pi * v / 180)

    return {
        "x": x, "pi": sp.pi, "e": sp.E,
        "sin": sin_deg, "cos": cos_deg, "tan": tan_deg,
        "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
        "sqrt": sp.sqrt, "log": sp.log, "ln": sp.log,
        "abs": sp.Abs, "factorial": sp.factorial,
        "diff": sp.diff, "integrate": sp.integrate,
        "Matrix": sp.Matrix,
        "det": lambda m: sp.Matrix(m).det(),
        "inv": lambda m: sp.Matrix(m).inv(),
        "transpose": lambda m: sp.Matrix(m).T,
        "mean": lambda *a: sum(a) / len(a),
        "variance": lambda *a: float(np.var(a)),
        "std": lambda *a: float(np.std(a)),
        "Normal": Normal, "pdf": density,
        **chat_variables.get(chat_id, {}),
    }


# ================= EVALUATE =================
def evaluate(expr: str, chat_id: int):
    expr = preprocess(expr)
    if not expr:
        return None

    safe = safe_locals(chat_id)

    if expr.startswith("matrix"):
        try:
            inside = expr[len("matrix"):].strip()
            if inside.startswith("(") and inside.endswith(")"):
                return str(sp.Matrix(sp.sympify(inside[1:-1], locals=safe)))
        except Exception:
            pass
        return None

    try:
        res = sp.sympify(expr, locals=safe)
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
    if not client:
        return "⚠️ Gemini API key not configured."
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=text
        )
        return response.text.strip() if getattr(response, "text", None) else "⚠️ Empty response from AI."
    except Exception as e:
        err = str(e)
        if "429" in err or "quota" in err.lower():
            return "⚠️ AI quota reached. Please try again later."
        return f"❌ AI error: {err[:120]}"


# ================= HISTORY =================
async def save_history(chat_id: int, expr: str, result):
    async with db_lock:
        cursor.execute(
            "INSERT INTO history VALUES (?, ?, ?)",
            (chat_id, expr, str(result))
        )
        conn.commit()


# ================= PLOT =================
def plot(expr: str) -> BytesIO | None:
    x = sp.symbols("x")
    funcs = [f.strip() for f in expr.split(",") if f.strip()]
    if not funcs:
        return None

    xs = np.linspace(-10, 10, 400)
    fig, ax = plt.subplots()
    plotted = False

    for f in funcs:
        try:
            parsed = sp.sympify(preprocess(f), locals={"x": x, "pi": sp.pi, "e": sp.E})
            f_np = sp.lambdify(x, parsed, "numpy")
            ys = np.array(f_np(xs), dtype=float)
            ys[np.abs(ys) > 1e6] = np.nan
            ax.plot(xs, ys, label=f)
            plotted = True
        except Exception:
            pass

    if not plotted:
        plt.close(fig)
        return None

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.grid(True)
    ax.legend()

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


# ================= UNIT CONVERSION =================
UNIT_CONVERSIONS: dict[tuple, float] = {
    ("km","m"):1000, ("m","km"):0.001,
    ("kg","g"):1000, ("g","kg"):0.001,
    ("cm","m"):0.01, ("m","cm"):100,
    ("mm","m"):0.001, ("m","mm"):1000,
    ("mi","km"):1.60934, ("km","mi"):0.621371,
    ("lb","kg"):0.453592, ("kg","lb"):2.20462,
    ("ft","m"):0.3048, ("m","ft"):3.28084,
    ("in","cm"):2.54, ("cm","in"):0.393701,
}

def convert_units(text: str) -> str | None:
    parts = text.lower().strip().split()
    if len(parts) != 4 or parts[2] != "to":
        return None
    try:
        value = float(parts[0])
    except ValueError:
        return None
    factor = UNIT_CONVERSIONS.get((parts[1], parts[3]))
    if factor is None:
        return None
    return f"{value} {parts[1]} = {value * factor} {parts[3]}"


# ================= EQUATION SOLVER =================
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


# ================= URL SHORTENER =================
def shorten_url(url: str) -> str:
    url = url.strip()
    if not url:
        return "❌ Please provide a URL."
    try:
        r = requests.get("http://tinyurl.com/api-create.php", params={"url": url}, timeout=10)
        return r.text.strip()
    except Exception as e:
        return f"❌ URL shortener error: {e}"


# ================= MESSAGES =================
START_MSG = (
    "✨ Welcome to Most Advanced Calculator 🤖\n"
    "━━━━━━━━━━━━━━━━━━━━━━\n\n"
    "🚀 Fast • Powerful • Intelligent\n\n"
    "🧮 Solve any calculation instantly\n"
    "📊 Plot graphs & analyze functions\n"
    "📐 Perform calculus & algebra\n"
    "📦 Work with matrices & statistics\n\n"
    "👨‍💻 Developed by: @Sudhakaran12\n\n"
    "👉 Use /help to explore all features\n"
    "💡 Try: 2², cos 60, sin(30)"
)

HELP_MSG = (
    "📘 MOST ADVANCED CALCULATOR - HELP MENU\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    "🧮 BASIC: 2+2, 5*6, 2^5, 2², 50%, 1 1/2\n"
    "📐 TRIG: sin(30), cos(60), tan 45\n"
    "📘 ALGEBRA: x^2+5*x+6, sqrt(x), log(x)\n"
    "📊 CALCULUS: diff(x^2,x), integrate(x^2,x)\n"
    "🧠 SOLVE: /solve x^2-4=0\n"
    "📦 MATRIX: Matrix([[1,2],[3,4]]), det([[1,2],[3,4]])\n"
    "📈 PLOT: /plot sin(x),cos(x)\n"
    "📉 STATS: mean(2,4,6), std(1,2,3,4,5)\n"
    "🔄 CONVERT: 10 km to m, 5 kg to g\n"
    "🔗 SHORT URL: /short https://example.com\n"
    "📤 EXPORT: /export\n"
    "🤖 AI: Explain Newton's law\n\n"
    "✨ SYMBOLS: ² ³ ½ ¼ ¾ × ÷ %\n\n"
    "👨‍💻 Developer: @Sudhakaran12"
)


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
    if not text:
        return {"ok": True}

    lower = text.lower()

    async def send(txt: str):
        await asyncio.to_thread(bot.send_message, chat_id, txt)

    # ── commands ──────────────────────────────────────────
    if lower == "/start":
        await send(START_MSG)

    elif lower == "/help":
        await send(HELP_MSG)

    elif lower.startswith("/short"):
        await send(await asyncio.to_thread(shorten_url, text[6:].strip()))

    elif lower.startswith("/plot"):
        img = await asyncio.to_thread(plot, text[5:].strip())
        if img:
            await asyncio.to_thread(bot.send_photo, chat_id, img)
        else:
            await send("❌ Invalid function(s) for plotting.")

    elif lower.startswith("/solve"):
        expr = text[6:].strip()
        if not expr:
            await send("❌ Please provide an equation, e.g. /solve x^2-4=0")
        else:
            try:
                res = await asyncio.to_thread(solve_equation, expr, chat_id)
                await send(f"🧠 Solution: {res}")
            except Exception as e:
                await send(f"❌ Solve error: {str(e)[:120]}")

    elif lower == "/export":
        async with db_lock:
            cursor.execute(
                "SELECT expr, result FROM history WHERE chat_id=?", (chat_id,)
            )
            rows = cursor.fetchall()
        if not rows:
            await send("❌ No history saved yet.")
        else:
            file_name = f"history_{chat_id}.txt"
            with open(file_name, "w", encoding="utf-8") as f:
                for e, r in rows:
                    f.write(f"{e} = {r}\n")
            with open(file_name, "rb") as f:
                await asyncio.to_thread(bot.send_document, chat_id, f)

    # ── free text ─────────────────────────────────────────
    else:
        # 1. unit conversion
        conversion = convert_units(text)
        if conversion is not None:
            await send(f"🔄 {conversion}")
            return {"ok": True}

        # 2. calculator
        result = await asyncio.to_thread(evaluate, text, chat_id)
        if result is not None:
            await save_history(chat_id, text, result)
            await send(f"✅ {result}")
        else:
            # 3. AI fallback
            reply = await asyncio.to_thread(gemini_reply, text)
            await send(reply)

    return {"ok": True}

@app.get("/")
async def root():
    return {"status": "Bot is running:"} 
