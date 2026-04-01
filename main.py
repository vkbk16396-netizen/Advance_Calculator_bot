import os
import re
import asyncio
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import telebot
import sqlite3
import pytesseract
from PIL import Image
from fastapi import FastAPI, Request
from sympy.stats import Normal, density
from openai import OpenAI
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

app = FastAPI()

# ================= TOKENS =================
TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

bot = telebot.TeleBot(TOKEN, parse_mode="Markdown")
client = OpenAI(api_key=OPENAI_API_KEY)

# ================= DATABASE =================
conn = sqlite3.connect("history.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS history (
    chat_id INTEGER,
    expr TEXT,
    result TEXT
)
""")
conn.commit()

chat_variables = {}

# ================= PREPROCESS =================
def preprocess(expr):
    expr = expr.lower().strip()
    if not expr:
        return ""

    expr = re.sub(r'(sin|cos|tan)\s+(\d+)', r'\1(\2)', expr)
    expr = expr.replace("×", "*").replace("÷", "/")
    expr = re.sub(r'(\d+(\.\d+)?)\s*%', r'(\1/100)', expr)

    return expr.replace("^", "**")

# ================= SAFE =================
def safe_locals(chat_id):
    x = sp.symbols('x')
    return {
        "x": x,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "sqrt": sp.sqrt, "log": sp.log,
        "diff": sp.diff, "integrate": sp.integrate,
        "Matrix": sp.Matrix,
        "det": lambda m: sp.Matrix(m).det(),
        "inv": lambda m: sp.Matrix(m).inv(),
        "mean": lambda *a: sum(a)/len(a),
        "variance": lambda *a: float(np.var(a)),
        "Normal": Normal, "pdf": density,
        **chat_variables.get(chat_id, {})
    }

# ================= SAVE HISTORY =================
def save_history(chat_id, expr, result):
    cursor.execute("INSERT INTO history VALUES (?, ?, ?)", (chat_id, expr, str(result)))
    conn.commit()

# ================= AI =================
def ai_chat(prompt):
    try:
        res = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a helpful math assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return res.choices[0].message.content
    except:
        return "❌ AI Error"

# ================= OCR =================
def solve_image(path):
    img = Image.open(path)
    return pytesseract.image_to_string(img)

# ================= EVALUATE =================
def evaluate(expr, chat_id):
    expr = preprocess(expr)
    if not expr:
        return None

    if expr.startswith("matrix"):
        try:
            data = eval(expr[6:], {"__builtins__": {}})
            return sp.Matrix(data)
        except:
            return None

    safe = safe_locals(chat_id)

    if "=" in expr and expr.count("=") == 1:
        left, right = expr.split("=")
        if left.strip().isidentifier():
            val = sp.sympify(right, locals=safe)
            chat_variables.setdefault(chat_id, {})[left.strip()] = val
            return f"{left.strip()} = {val}"

    try:
        res = sp.sympify(expr, locals=safe)
        return str(res) if res.free_symbols else float(res.evalf())
    except:
        return None

# ================= SOLVE =================
def solve_eq(expr):
    try:
        x = sp.symbols('x')
        eq = sp.sympify(preprocess(expr).replace("=", "-(") + ")")
        return sp.solve(eq, x)
    except:
        return None

# ================= PLOT =================
def plot(expr):
    x = sp.symbols('x')
    funcs = expr.split(",")

    xs = np.linspace(-10, 10, 400)
    plt.figure()

    valid = False
    for f in funcs:
        f = f.strip()
        if not f:
            continue
        try:
            f_sym = sp.sympify(preprocess(f))
            f_np = sp.lambdify(x, f_sym, 'numpy')
            plt.plot(xs, f_np(xs), label=f)
            valid = True
        except:
            continue

    if not valid:
        return None

    plt.legend()
    plt.grid()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

# ================= HELP =================
def get_help():
    return """
📘 ULTIMATE CALCULATOR GUIDE

🧮 Basic → 2+2, 5%
📐 Trig → sin(30)
📊 Calc → diff(x^2,x)
📦 Matrix → Matrix([[1,2],[3,4]])
🧠 Solve → /solve x^2-4=0
📈 Plot → /plot sin(x),cos(x)
🤖 AI → /ai explain
📸 Image → send photo
💾 DB → /dbhistory
"""

# ================= BUTTONS =================
def main_buttons():
    markup = InlineKeyboardMarkup()
    markup.row(
        InlineKeyboardButton("📘 Help", callback_data="help"),
        InlineKeyboardButton("📊 Features", callback_data="features")
    )
    markup.row(
        InlineKeyboardButton("📈 Plot", callback_data="plot"),
        InlineKeyboardButton("🧠 Solve", callback_data="solve")
    )
    markup.row(
        InlineKeyboardButton("🤖 AI", callback_data="ai"),
        InlineKeyboardButton("💾 History", callback_data="history")
    )
    return markup

# ================= WEBHOOK =================
@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()

    # BUTTON HANDLER
    if "callback_query" in data:
        call = data["callback_query"]
        chat_id = call["message"]["chat"]["id"]
        btn = call["data"]

        if btn == "help":
            await asyncio.to_thread(bot.send_message, chat_id, get_help())

        elif btn == "features":
            await asyncio.to_thread(bot.send_message, chat_id, "🔥 All features enabled!")

        elif btn == "plot":
            await asyncio.to_thread(bot.send_message, chat_id, "`/plot sin(x),cos(x)`")

        elif btn == "solve":
            await asyncio.to_thread(bot.send_message, chat_id, "`/solve x^2-4=0`")

        elif btn == "ai":
            await asyncio.to_thread(bot.send_message, chat_id, "`/ai explain integration`")

        elif btn == "history":
            await asyncio.to_thread(bot.send_message, chat_id, "`/dbhistory`")

        return {"ok": True}

    if "message" not in data:
        return {"ok": True}

    msg = data["message"]
    chat_id = msg["chat"]["id"]
    text = msg.get("text", "").strip()
    lower = text.lower()

    # IMAGE OCR
    if "photo" in msg:
        file_id = msg["photo"][-1]["file_id"]
        file = bot.get_file(file_id)
        data_file = bot.download_file(file.file_path)

        with open("img.png", "wb") as f:
            f.write(data_file)

        extracted = solve_image("img.png")
        result = evaluate(extracted, chat_id)

        await asyncio.to_thread(bot.send_message, chat_id, f"📸 `{extracted}`")

        if result:
            await asyncio.to_thread(bot.send_message, chat_id, f"✅ `{result}`")

        return {"ok": True}

    # START WITH BUTTONS
    if lower == "/start":
        await asyncio.to_thread(
            bot.send_message,
            chat_id,
            "🤖 *Most Advanced Calculator*\n\n"
            "👉 Use buttons below or /help",
            reply_markup=main_buttons()
        )

    elif lower == "/help":
        await asyncio.to_thread(bot.send_message, chat_id, get_help())

    elif lower.startswith("/solve"):
        res = solve_eq(text[6:].strip())
        await asyncio.to_thread(bot.send_message, chat_id, f"🧠 `{res}`")

    elif lower.startswith("/plot"):
        buf = plot(text[5:].strip())
        if buf:
            await asyncio.to_thread(bot.send_photo, chat_id, buf)

    elif lower.startswith("/ai"):
        reply = ai_chat(text[3:].strip())
        await asyncio.to_thread(bot.send_message, chat_id, reply)

    elif lower == "/dbhistory":
        cursor.execute("SELECT expr, result FROM history WHERE chat_id=?", (chat_id,))
        rows = cursor.fetchall()
        txt = "\n".join([f"{e} = {r}" for e, r in rows[-10:]]) if rows else "No history"
        await asyncio.to_thread(bot.send_message, chat_id, txt)

    else:
        result = evaluate(text, chat_id)
        if result is not None:
            save_history(chat_id, text, result)
            await asyncio.to_thread(bot.send_message, chat_id, f"✅ `{result}`")
        else:
            await asyncio.to_thread(bot.send_message, chat_id, "❌ Invalid input")

    return {"ok": True}
