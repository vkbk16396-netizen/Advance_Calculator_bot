import os
import re
import asyncio
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import telebot
from fastapi import FastAPI, Request
from sympy.stats import Normal, density
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

app = FastAPI()

# ================= TOKEN =================
TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("❌ BOT_TOKEN not set")

bot = telebot.TeleBot(TOKEN, parse_mode="Markdown")

# ================= STORAGE =================
chat_angle_mode = {}
chat_history = {}
chat_variables = {}

# ================= PREPROCESS =================
def preprocess_expression(expr: str) -> str:
    expr = expr.lower().strip()

    expr = re.sub(r'(sin|cos|tan)\s+(\d+)', r'\1(\2)', expr)

    superscripts = {
        "⁰":"^0","¹":"^1","²":"^2","³":"^3","⁴":"^4",
        "⁵":"^5","⁶":"^6","⁷":"^7","⁸":"^8","⁹":"^9"
    }
    for k, v in superscripts.items():
        expr = expr.replace(k, v)

    expr = expr.replace("×", "*").replace("÷", "/")

    expr = re.sub(r'(\d+(\.\d+)?)\s*%', r'(\1/100)', expr)

    return expr

# ================= HISTORY =================
def add_history(chat_id, expr, result):
    chat_history.setdefault(chat_id, []).append((expr, result))
    if len(chat_history[chat_id]) > 30:
        chat_history[chat_id].pop(0)

# ================= SAFE LOCALS =================
def get_safe_locals(chat_id):
    x, y, z = sp.symbols('x y z')

    return {
        "x": x, "y": y, "z": z,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "sqrt": sp.sqrt, "log": sp.log,
        "diff": sp.diff, "integrate": sp.integrate,
        "Matrix": sp.Matrix,
        "mean": lambda *a: sum(a)/len(a),
        "variance": lambda *a: float(np.var(a)),
        "Normal": Normal, "pdf": density,
        **chat_variables.get(chat_id, {})
    }

# ================= EVALUATE =================
def evaluate(expr, chat_id):
    expr = preprocess_expression(expr)
    expr = expr.replace("^", "**")

    # 🔥 MATRIX FIX
    if expr.startswith("matrix"):
        try:
            inside = expr[6:]
            data = eval(inside, {"__builtins__": {}}, {})
            return sp.Matrix(data)
        except:
            return None

    safe = get_safe_locals(chat_id)

    # VARIABLE
    if "=" in expr and expr.count("=") == 1:
        left, right = expr.split("=")
        if left.strip().isidentifier():
            val = sp.sympify(right, locals=safe)
            chat_variables.setdefault(chat_id, {})[left.strip()] = val
            return f"📌 `{left.strip()}` = `{val}`"

    try:
        res = sp.sympify(expr, locals=safe)

        if res.free_symbols:
            return str(res)

        return float(res.evalf())

    except:
        return None

# ================= HELP =================
def get_help():
    return """
╔══════════════════════╗
📘 *ULTIMATE CALCULATOR*
╚══════════════════════╝

🧮 `2+2` `5%`
📐 `sin(30)`
📊 `diff(x^2,x)`
📦 `Matrix([[1,2],[3,4]])`
📊 `mean(1,2,3)`
📈 `/plot sin(x)`
"""

# ================= ROOT =================
@app.get("/")
async def root():
    return {"status": "LIVE 🚀"}

# ================= WEBHOOK =================
@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()

    # ===== BUTTONS =====
    if "callback_query" in data:
        call = data["callback_query"]
        chat_id = call["message"]["chat"]["id"]
        btn = call["data"]

        if btn == "help":
            await asyncio.to_thread(bot.send_message, chat_id, get_help())

        elif btn == "features":
            await asyncio.to_thread(bot.send_message, chat_id, "🔥 All features enabled!")

        return {"ok": True}

    if "message" not in data:
        return {"ok": True}

    msg = data["message"]
    chat_id = msg["chat"]["id"]
    text = msg.get("text", "").strip()
    lower = text.lower()

    # ===== START =====
    if lower == "/start":
        markup = InlineKeyboardMarkup()
        markup.row(
            InlineKeyboardButton("📘 Help", callback_data="help"),
            InlineKeyboardButton("📊 Features", callback_data="features")
        )

        await asyncio.to_thread(
            bot.send_message,
            chat_id,
            "╔══════════════════════╗\n"
            "✨ *Most Advanced Calculator* 🤖\n"
            "╚══════════════════════╝\n\n"
            "👋 Welcome to *Most Advanced Calculator*\n\n"
            "👨‍💻 *Made by:* @Sudhakaran12\n\n"
            "👉 Use /help to see all features",
            reply_markup=markup
        )

    # ===== HELP =====
    elif lower == "/help":
        await asyncio.to_thread(bot.send_message, chat_id, get_help())

    # ===== PLOT =====
    elif lower.startswith("/plot"):
        exprs = text[5:].strip().split(",")
        try:
            x = sp.symbols('x')
            xs = np.linspace(-10, 10, 400)

            plt.figure()

            for ex in exprs:
                f = sp.lambdify(x, sp.sympify(preprocess_expression(ex)), 'numpy')
                plt.plot(xs, f(xs), label=ex)

            plt.legend()
            plt.grid()

            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()

            await asyncio.to_thread(bot.send_photo, chat_id, buf)
        except:
            await asyncio.to_thread(bot.send_message, chat_id, "❌ Plot error")

    # ===== CALCULATE =====
    else:
        result = evaluate(text, chat_id)

        if result is not None:
            add_history(chat_id, text, result)
            await asyncio.to_thread(bot.send_message, chat_id, f"✅\n`{result}`")
        else:
            await asyncio.to_thread(bot.send_message, chat_id, "❌ Invalid input")

    return {"ok": True}
