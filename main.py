import os
import re
import asyncio
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import telebot
from fastapi import FastAPI, Request
from telebot.types import (
    InlineKeyboardMarkup, InlineKeyboardButton,
    ReplyKeyboardMarkup
)

app = FastAPI()

TOKEN = os.getenv("BOT_TOKEN")
bot = telebot.TeleBot(TOKEN, parse_mode="Markdown")

chat_variables = {}

# ================= KEYBOARD =================
def main_menu():
    kb = ReplyKeyboardMarkup(resize_keyboard=True)
    kb.row("🧮 Calculate", "📊 Stats")
    kb.row("📈 Plot", "📘 Help")
    return kb

# ================= PREPROCESS =================
def preprocess(expr):
    expr = expr.lower().strip()

    expr = re.sub(r'(sin|cos|tan)\s+(\d+)', r'\1(\2)', expr)
    expr = expr.replace("×","*").replace("÷","/")
    expr = re.sub(r'(\d+)%', r'(\1/100)', expr)

    return expr.replace("^","**")

# ================= SAFE =================
def safe_locals():
    x = sp.symbols('x')
    return {
        "x":x,
        "sin":sp.sin,"cos":sp.cos,"tan":sp.tan,
        "sqrt":sp.sqrt,"log":sp.log,
        "diff":sp.diff,"integrate":sp.integrate,
        "Matrix":sp.Matrix
    }

# ================= EVALUATE =================
def evaluate(expr):
    expr = preprocess(expr)

    if expr.startswith("matrix"):
        return sp.Matrix(eval(expr, {"__builtins__":{}}))

    return sp.sympify(expr, locals=safe_locals())

# ================= SOLVE =================
def solve_equation(expr):
    x = sp.symbols('x')
    expr = preprocess(expr)
    eq = sp.sympify(expr.replace("=", "-(")+")")
    return sp.solve(eq, x)

# ================= PLOT MULTI =================
def plot_multi(expr):
    x = sp.symbols('x')
    funcs = expr.split(",")

    xs = np.linspace(-10,10,400)

    plt.figure()

    for f in funcs:
        f = sp.lambdify(x, sp.sympify(preprocess(f)), 'numpy')
        plt.plot(xs, f(xs))

    plt.grid()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return buf

# ================= WEBHOOK =================
@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()

    if "message" not in data:
        return {"ok":True}

    msg = data["message"]
    chat_id = msg["chat"]["id"]
    text = msg.get("text","")

    # ===== START =====
    if text.lower()=="/start":
        await asyncio.to_thread(bot.send_message, chat_id,
        "🚀 *GOD MODE CALCULATOR ACTIVATED*\n\n"
        "🔥 Next Level Math Engine\n\n"
        "👉 Try /help",
        reply_markup=main_menu())

    # ===== HELP =====
    elif text.lower()=="/help":
        await asyncio.to_thread(bot.send_message, chat_id,
        "*COMMANDS*\n\n"
        "/solve x^2-4=0\n"
        "/plot sin(x),cos(x)\n"
        "Matrix([[1,2],[3,4]])\n"
        "diff(x^2,x)")

    # ===== SOLVE =====
    elif text.startswith("/solve"):
        expr = text[6:].strip()
        res = solve_equation(expr)
        await asyncio.to_thread(bot.send_message, chat_id, f"🧠 Solution: `{res}`")

    # ===== PLOT =====
    elif text.startswith("/plot"):
        expr = text[5:].strip()
        buf = plot_multi(expr)
        await asyncio.to_thread(bot.send_photo, chat_id, buf)

    # ===== BUTTONS =====
    elif text == "🧮 Calculate":
        await asyncio.to_thread(bot.send_message, chat_id, "Enter expression")

    elif text == "📊 Stats":
        await asyncio.to_thread(bot.send_message, chat_id, "Try: mean(1,2,3)")

    elif text == "📈 Plot":
        await asyncio.to_thread(bot.send_message, chat_id, "/plot sin(x),cos(x)")

    # ===== DEFAULT =====
    else:
        try:
            res = evaluate(text)
            await asyncio.to_thread(bot.send_message, chat_id, f"✅ `{res}`")
        except:
            await asyncio.to_thread(bot.send_message, chat_id, "❌ Error")

    return {"ok":True}
