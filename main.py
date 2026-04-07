import os
import re
import asyncio
import sqlite3
from io import BytesIO

import numpy as np
import sympy as sp
import matplotlib
import matplotlib.pyplot as plt
import telebot
from fastapi import FastAPI, Request, BackgroundTasks
from google import genai

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

db_lock = asyncio.Lock()

# ================= ASYNC SEND =================
async def async_send(chat_id, text, parse_mode="Markdown"):
    await asyncio.to_thread(bot.send_message, chat_id, text, parse_mode=parse_mode)

# ================= PREPROCESS =================
def preprocess(expr):
    expr = expr.strip().lower()
    expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
    expr = re.sub(r"\b(sin|cos|tan)\s+(\d+)", r"\1(\2)", expr)
    return expr

# ================= SAFE LOCALS =================
def safe_locals():
    x = sp.symbols("x")

    def sin_deg(v): return sp.sin(sp.pi * v / 180)
    def cos_deg(v): return sp.cos(sp.pi * v / 180)
    def tan_deg(v): return sp.tan(sp.pi * v / 180)

    return {
        "x": x,
        "sin": sin_deg,
        "cos": cos_deg,
        "tan": tan_deg,
        "sqrt": sp.sqrt,
        "log": sp.log,
        "diff": sp.diff,
        "integrate": sp.integrate,
    }

# ================= MATH =================
def eval_math(expr):
    res = sp.sympify(expr, locals=safe_locals())
    if getattr(res, "free_symbols", None):
        return str(res)
    return float(res.evalf())

# ================= CONVERT =================
def convert_units(text):
    text = text.lower()

    if "km to m" in text:
        return float(re.findall(r'\d+', text)[0]) * 1000
    if "m to km" in text:
        return float(re.findall(r'\d+', text)[0]) / 1000
    if "c to f" in text:
        v = float(re.findall(r'\d+', text)[0])
        return (v * 9/5) + 32
    if "f to c" in text:
        v = float(re.findall(r'\d+', text)[0])
        return (v - 32) * 5/9

    return None

# ================= SOLVE =================
def solve_equation(eq):
    x = sp.symbols('x')
    if "=" in eq:
        l, r = eq.split("=")
        eq = sp.Eq(sp.sympify(l), sp.sympify(r))
    else:
        eq = sp.sympify(eq)
    return sp.solve(eq, x)

# ================= PLOT =================
def plot(expr):
    try:
        x = sp.symbols('x')
        funcs = expr.split(",")

        x_vals = np.linspace(-10, 10, 400)
        plt.figure()

        for f in funcs:
            f_expr = sp.sympify(f.strip())
            f_l = sp.lambdify(x, f_expr, "numpy")
            plt.plot(x_vals, f_l(x_vals))

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf
    except:
        return None

# ================= HISTORY =================
async def save_history(chat_id, expr, result):
    async with db_lock:
        cursor.execute("INSERT INTO history VALUES (?, ?, ?)", (chat_id, expr, str(result)))
        conn.commit()

def get_history(chat_id):
    cursor.execute("SELECT expr, result FROM history WHERE chat_id=?", (chat_id,))
    return cursor.fetchall()

async def clear_history(chat_id):
    async with db_lock:
        cursor.execute("DELETE FROM history WHERE chat_id=?", (chat_id,))
        conn.commit()

# ================= GEMINI =================
def gemini_reply(text):
    try:
        res = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=text
        )
        return res.text
    except Exception as e:
        return "AI Error: " + str(e)

# ================= MAIN =================
async def process_message(msg):
    chat_id = msg["chat"]["id"]
    text = msg.get("text", "").strip()
    lower = text.lower()

    # ===== START =====
    if lower == "/start":
        await async_send(chat_id, """✨ *Welcome to Most Advanced Calculator* 🤖
━━━━━━━━━━━━━━━━━━━━━━

🚀 *Fast • Powerful • Intelligent*

🧮 Solve any calculation instantly
📊 Plot graphs & analyze functions
📐 Perform calculus & algebra
📦 Work with matrices & statistics

👨‍💻 *Developed by:* @Sudhakaran12

👉 Use /help to explore all features
💡 *Try:* `2²`, `cos 60`, `sin(30)`""")

    # ===== HELP =====
    elif lower == "/help":
        await async_send(chat_id, """📘 MOST ADVANCED CALCULATOR - HELP MENU
━━━━━━━━━━━━━━━━━━━━━━━━━━

🧮 BASIC CALCULATIONS → 2+2, 2², 50%
📐 TRIG → sin(30), cos(60)
📊 CALCULUS → diff(x^2,x)
📈 GRAPH → /plot x^2
🔄 UNIT → 10 km to m
🤖 AI → Ask anything""")

    elif lower.startswith("/plot"):
        img = await asyncio.to_thread(plot, text[5:].strip())
        if img:
            await asyncio.to_thread(bot.send_photo, chat_id, img)

    elif lower.startswith("/solve"):
        res = await asyncio.to_thread(solve_equation, text[6:].strip())
        await async_send(chat_id, f"`{res}`")

    elif lower == "/export":
        rows = get_history(chat_id)
        buf = BytesIO()
        for e, r in rows:
            buf.write(f"{e} = {r}\n".encode())
        buf.seek(0)
        buf.name = "history.txt"
        await asyncio.to_thread(bot.send_document, chat_id, buf)

    elif lower == "/clear":
        await clear_history(chat_id)
        await async_send(chat_id, "Cleared!")

    else:
        conv = convert_units(text)
        if conv:
            await async_send(chat_id, f"`{conv}`")
            return

        try:
            result = await asyncio.to_thread(eval_math, preprocess(text))
            await save_history(chat_id, text, result)
            await async_send(chat_id, f"`{result}`")
        except:
            if client:
                reply = await asyncio.to_thread(gemini_reply, text)
                await async_send(chat_id, reply, parse_mode=None)

# ================= WEBHOOK =================
@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    if "message" in data:
        background_tasks.add_task(process_message, data["message"])
    return {"ok": True}
