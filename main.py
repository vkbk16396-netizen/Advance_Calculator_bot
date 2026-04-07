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
import telebot
from fastapi import FastAPI, Request, BackgroundTasks
from google import genai

# Prevent GUI
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

# ================= ASYNC SEND =================
async def async_send(chat_id: int, text: str, parse_mode="Markdown"):
    await asyncio.to_thread(bot.send_message, chat_id, text, parse_mode=parse_mode)

# ================= PREPROCESS =================
def preprocess(expr: str) -> str:
    expr = expr.strip().lower()
    expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")

    expr = re.sub(r"\b(sin|cos|tan)\s+(-?\d+(\.\d+)?)\b", r"\1(\2)", expr)

    return expr

# ================= SAFE LOCALS =================
def safe_locals(chat_id: int):
    x = sp.symbols("x")

    def sin_deg(v): return sp.sin(sp.pi * v / 180)
    def cos_deg(v): return sp.cos(sp.pi * v / 180)
    def tan_deg(v): return sp.tan(sp.pi * v / 180)

    return {
        "x": x, "pi": sp.pi, "e": sp.E,
        "sin": sin_deg, "cos": cos_deg, "tan": tan_deg,
        "sqrt": sp.sqrt, "log": sp.log,
        "diff": sp.diff, "integrate": sp.integrate,
    }

# ================= MATH =================
def _eval_math(expr, chat_id):
    safe = safe_locals(chat_id)
    res = sp.sympify(expr, locals=safe)
    try:
        res = sp.simplify(res)
    except:
        pass

    if getattr(res, "free_symbols", None):
        return str(res)

    return float(res.evalf())

# ================= CONVERT =================
def convert_units(text: str):
    text = text.lower()

    if "km to m" in text:
        v = float(re.findall(r'\d+', text)[0])
        return v * 1000

    if "m to km" in text:
        v = float(re.findall(r'\d+', text)[0])
        return v / 1000

    if "c to f" in text:
        v = float(re.findall(r'\d+', text)[0])
        return (v * 9/5) + 32

    if "f to c" in text:
        v = float(re.findall(r'\d+', text)[0])
        return (v - 32) * 5/9

    return None

# ================= SOLVE =================
def solve_equation(eq, chat_id):
    x = sp.symbols('x')
    safe = safe_locals(chat_id)

    if "=" in eq:
        l, r = eq.split("=")
        eq = sp.Eq(sp.sympify(l, locals=safe), sp.sympify(r, locals=safe))
    else:
        eq = sp.sympify(eq, locals=safe)

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
            y = f_l(x_vals)
            plt.plot(x_vals, y)

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
        await async_send(chat_id, "✨ *Welcome to Most Advanced Calculator* 🤖
━━━━━━━━━━━━━━━━━━━━━━

🚀 *Fast • Powerful • Intelligent*

🧮 Solve any calculation instantly
📊 Plot graphs & analyze functions
📐 Perform calculus & algebra
📦 Work with matrices & statistics

👨‍💻 *Developed by:* @Sudhakaran12

👉 Use /help to explore all features
💡 *Try:* `2²`, `cos 60`, `sin(30)`")

    # ===== HELP =====
    elif lower == "/help":
        await async_send(
            chat_id,
            "📘 MOST ADVANCED CALCULATOR - HELP MENU
━━━━━━━━━━━━━━━━━━━━━━━━━━

This bot can do calculations, algebra, calculus, matrices, graphs, unit conversion, URL shortening, history export, and AI replies.

🧮 1) BASIC CALCULATIONS
Use normal math expressions.
Examples:
2+2
15-7
8*9
10/4
2^5
2²
3³
50%
1 1/2
¾ + ¼

📐 2) TRIGONOMETRY
Use sin, cos, tan.
Examples:
sin(30)
cos(60)
tan(45)
sin 30
cos 90

📘 3) ALGEBRA WITH x
Use x in expressions.
Examples:
x+2
x^2+5*x+6
sqrt(x)
log(x)

📊 4) CALCULUS
Differentiate and integrate expressions.
Examples:
diff(x^2,x)
diff(sin(x),x)
integrate(x^2,x)
integrate(cos(x),x)

🧠 5) EQUATION SOLVER
Use /solve to solve equations.
Examples:
/solve x^2-4=0
/solve x^2+5*x+6=0
/solve x^2-9

📦 6) MATRICES
Create and work with matrices.
Examples:
Matrix([[1,2],[3,4]])
det([[1,2],[3,4]])
inv([[1,2],[3,4]])
transpose([[1,2],[3,4]])

📈 7) GRAPH PLOTTING
Use /plot to draw one or more functions.
Examples:
/plot x^2
/plot sin(x)
/plot sin(x),cos(x)
/plot x^2,x^3

📉 8) STATISTICS
Use mean, variance, std.
Examples:
mean(2,4,6,8)
variance(1,2,3,4,5)
std(1,2,3,4,5)

🔔 9) NORMAL DISTRIBUTION
You can use Normal and pdf.
Examples:
pdf(Normal('X',0,1))(0)

🔄 10) UNIT CONVERSION
Use: value unit to unit
Supported now:
km to m
m to km
kg to g
g to kg
cm to m
m to cm
mm to m
m to mm
Examples:
10 km to m
5000 m to km
2 kg to g

🔗 11) SHORT URL
Use /short followed by a URL.
Example:
/short https://example.com

📤 12) EXPORT HISTORY
Use /export to download your saved calculations.
Example:
/export

🤖 13) AI REPLIES
If expression is not solved by calculator, bot sends it to Gemini AI.
Example:
Explain Newton's law
Write a Python program

💾 14) HISTORY
Every solved calculation is saved automatically.
Use /export to get all saved history.

✨ 15) SPECIAL SYMBOLS SUPPORTED
The bot supports many special math inputs.
Examples:
² ³ ⁴ ⁵ ⁶ ⁷ ⁸ ⁹
½ ¼ ¾ ⅓ ⅔ ⅕ ⅖ ⅗ ⅘ ⅙ ⅚ ⅛ ⅜ ⅝ ⅞
× ÷ %

⚡ QUICK EXAMPLES TO TRY
2² + 5²
sin 30
diff(x^3,x)
/solve x^2-9=0
/plot sin(x),cos(x)
10 km to m
mean(10,20,30)

👨‍💻 Developer: @Sudhakaran12"
        )

    elif lower.startswith("/plot"):
        img = await asyncio.to_thread(plot, text[5:].strip())
        if img:
            await asyncio.to_thread(bot.send_photo, chat_id, img)

    elif lower.startswith("/solve"):
        res = await asyncio.to_thread(solve_equation, text[6:].strip(), chat_id)
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
            result = await asyncio.to_thread(_eval_math, preprocess(text), chat_id)
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
