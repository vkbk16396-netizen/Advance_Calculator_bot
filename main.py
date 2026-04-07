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
async def async_send(chat_id: int, text: str, parse_mode="Markdown"):
    """Helper to send messages asynchronously."""
    await asyncio.to_thread(bot.send_message, chat_id, text, parse_mode=parse_mode)

# ================= PREPROCESS =================
def preprocess(expr: str) -> str:
    expr = expr.strip().lower()
    if not expr:
        return ""
    
    # Standardize operators
    expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
    
    # Auto-wrap trig functions if space is used: "sin 30" -> "sin(30)"
    expr = re.sub(r"\b(sin|cos|tan)\s+(-?\d+(\.\d+)?)\b", r"\1(\2)", expr)
    
    # Handle superscripts
    superscripts = {"⁰":"0","¹":"1","²":"2","³":"3","⁴":"4","⁵":"5","⁶":"6","⁷":"7","⁸":"8","⁹":"9"}
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
    
    # Handle fractions and percents
    fractions = {"½":"1/2","¼":"1/4","¾":"3/4","⅓":"1/3","⅔":"2/3"}
    for k, v in fractions.items():
        expr = expr.replace(k, v)
        
    expr = re.sub(r"(\d+)\s+(\d+/\d+)", r"(\1+\2)", expr) # Mixed fractions
    expr = re.sub(r"(\d+(\.\d+)?)\s*%", r"(\1/100)", expr) # Percents
    
    return expr

# ================= SAFE MATH ENVIRONMENT =================
def safe_locals(chat_id: int) -> dict:
    x = sp.symbols("x")
    def sin_deg(v): return sp.sin(sp.pi * v / 180)
    def cos_deg(v): return sp.cos(sp.pi * v / 180)
    def tan_deg(v): return sp.tan(sp.pi * v / 180)
    
    return {
        "x": x, "pi": sp.pi, "e": sp.E,
        "sin": sin_deg, "cos": cos_deg, "tan": tan_deg,
        "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
        "sqrt": sp.sqrt, "log": sp.log, "ln": sp.log, "abs": sp.Abs,
        "factorial": sp.factorial, "diff": sp.diff, "integrate": sp.integrate,
        "limit": sp.limit, "series": sp.series,
        "Matrix": sp.Matrix, "det": lambda m: sp.Matrix(m).det(),
        "inv": lambda m: sp.Matrix(m).inv(),
        "mean": lambda *a: sum(a) / len(a),
        "bin": lambda v: bin(int(v)), "hex": lambda v: hex(int(v)),
        **chat_variables.get(chat_id, {}),
    }

# ================= CORE LOGIC =================
def _eval_math(expr: str, chat_id: int):
    safe = safe_locals(chat_id)
    # Handle Matrix shorthand
    if expr.startswith("matrix"):
        data = sp.sympify(expr[6:].strip()[1:-1], locals=safe)
        return str(sp.Matrix(data))
    
    res = sp.sympify(expr, locals=safe)
    try: res = sp.simplify(res)
    except: pass
    
    if getattr(res, "free_symbols", None): return str(res)
    return float(res.evalf())

async def process_message(msg: dict):
    chat_id = msg["chat"]["id"]
    text = msg.get("text", "").strip()
    if not text: return
    lower = text.lower()

    # 1. START & HELP
    if lower == "/start":
        await async_send(chat_id, "🚀 *Advanced Calculator Online!*\nUse /help to see all features.")
    
    elif lower == "/help":
        help_text = (
            "📖 *Advanced Calculator Guide*\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "🔢 *Basic Math*\n"
            "• Just type: `(5 + 5) * 2` or `10% of 50`\n"
            "• Supports: `² ³ ½ ¾ × ÷` symbols.\n\n"
            "📐 *Trigonometry (Degree based)*\n"
            "• `sin 30`, `cos 60`, `tan 45`\n\n"
            "📊 *Calculus & Algebra*\n"
            "• Derivative: `diff(x^2, x)`\n"
            "• Integral: `integrate(sin(x), x)`\n"
            "• Limits: `limit(sin(x)/x, x, 0)`\n"
            "• Solver: `/solve x^2 - 9 = 0`\n\n"
            "📈 *Graphing*\n"
            "• `/plot x^2` or `/plot sin(x), cos(x)`\n\n"
            "📦 *Advanced Tools*\n"
            "• Matrices: `matrix([[1,2],[3,4]])`\n"
            "• Base: `hex(255)`, `bin(10)`\n"
            "• Units: `10 km to m`, `100 c to f`\n\n"
            "🔗 *Other Commands*\n"
            "• `/short [URL]` - Shorten links\n"
            "• `/export` - Download history\n"
            "• `/clear` - Wipe history\n"
            "• *AI Mode:* Just ask any question!"
        )
        await async_send(chat_id, help_text)

    # 2. PLOT
    elif lower.startswith("/plot"):
        img = await asyncio.to_thread(plot, text[5:].strip())
        if img: await asyncio.to_thread(bot.send_photo, chat_id, img)
        else: await async_send(chat_id, "❌ Invalid function for plotting.")

    # 3. SOLVE
    elif lower.startswith("/solve"):
        eq = text[6:].strip()
        res = await asyncio.to_thread(solve_equation, eq, chat_id)
        await async_send(chat_id, f"🧠 *Solution:* `{res}`")

    # 4. EXPORT / CLEAR
    elif lower == "/export":
        rows = get_history(chat_id)
        if not rows: await async_send(chat_id, "📭 No history found.")
        else:
            buf = BytesIO()
            for e, r in rows: buf.write(f"{e} = {r}\n".encode())
            buf.seek(0); buf.name = "history.txt"
            await asyncio.to_thread(bot.send_document, chat_id, buf)

    elif lower == "/clear":
        await clear_history(chat_id)
        await async_send(chat_id, "🗑 History cleared.")

    # 5. EVAL / CONVERT / AI
    else:
        # Unit Conversion
        conv = convert_units(text)
        if conv:
            await async_send(chat_id, f"🔄 *Result:* `{conv}`")
            return

        # Math Eval
        processed = preprocess(text)
        try:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = await asyncio.get_event_loop().run_in_executor(pool, _eval_math, processed, chat_id)
            if result is not None:
                await save_history(chat_id, text, result)
                await async_send(chat_id, f"✅ `{result}`")
                return
        except: pass

        # AI Fallback
        if client:
            reply = await asyncio.to_thread(gemini_reply, text)
            await async_send(chat_id, reply, parse_mode=None)

# ================= (UTILITIES REMAIN SAME AS PREVIOUS STEP) =================
# [Insert plot, solve_equation, convert_units, save_history, gemini_reply from previous code]

@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    if "message" in data:
        background_tasks.add_task(process_message, data["message"])
    return {"ok": True}
            
