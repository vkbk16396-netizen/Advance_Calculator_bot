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

TOKEN = os.getenv("BOT_TOKEN")
bot = telebot.TeleBot(TOKEN, parse_mode="Markdown")

chat_history = {}
chat_variables = {}

# ================= PREPROCESS =================
def preprocess(expr):
    expr = expr.lower().strip()

    if not expr:
        return ""

    expr = re.sub(r'(sin|cos|tan)\s+(\d+)', r'\1(\2)', expr)
    expr = expr.replace("×", "*").replace("÷", "/")
    expr = re.sub(r'(\d+(\.\d+)?)\s*%', r'(\1/100)', expr)

    superscripts = {"²": "^2", "³": "^3"}
    for k, v in superscripts.items():
        expr = expr.replace(k, v)

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

# ================= EVALUATE =================
def evaluate(expr, chat_id):
    expr = preprocess(expr)

    if not expr:
        return None

    # MATRIX FIX
    if expr.startswith("matrix"):
        try:
            data = eval(expr[6:], {"__builtins__": {}})
            return sp.Matrix(data)
        except:
            return None

    safe = safe_locals(chat_id)

    # VARIABLE
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
        expr = preprocess(expr)
        eq = sp.sympify(expr.replace("=", "-(") + ")")
        return sp.solve(eq, x)
    except:
        return None

# ================= SAFE PLOT =================
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
        except Exception as e:
            print("Plot Error:", e)
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
📘 FINAL BOSS CALCULATOR 💀

🧮 Basic → 2+2, 5%
📐 Trig → sin(30)
📊 Calc → diff(x^2,x)
📦 Matrix → Matrix([[1,2],[3,4]])
🧠 Solve → /solve x^2-4=0
📊 Stats → mean(1,2,3)
📈 Plot → /plot sin(x),cos(x)
"""

# ================= WEBHOOK =================
@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()

    # BUTTONS
    if "callback_query" in data:
        call = data["callback_query"]
        chat_id = call["message"]["chat"]["id"]
        btn = call["data"]

        if btn == "help":
            await asyncio.to_thread(bot.send_message, chat_id, get_help())

        elif btn == "features":
            await asyncio.to_thread(bot.send_message, chat_id, "💀 FINAL BOSS FEATURES ENABLED")

        return {"ok": True}

    if "message" not in data:
        return {"ok": True}

    msg = data["message"]
    chat_id = msg["chat"]["id"]
    text = msg.get("text", "").strip()
    lower = text.lower()

    # ===== KEEP YOUR ORIGINAL /start =====
    if lower == "/start":
        markup = InlineKeyboardMarkup()
        markup.row(
            InlineKeyboardButton("📘 Help", callback_data="help"),
            InlineKeyboardButton("📊 Features", callback_data="features")
        )

        await asyncio.to_thread(
            bot.send_message,
            chat_id,
            "✨ *Most Advanced Calculator* 🤖\n\n"
            "🚀 PRO MAX MODE ENABLED",
            reply_markup=markup
        )

    elif lower == "/help":
        await asyncio.to_thread(bot.send_message, chat_id, get_help())

    elif lower.startswith("/solve"):
        res = solve_eq(text[6:].strip())
        if res is not None:
            await asyncio.to_thread(bot.send_message, chat_id, f"🧠 `{res}`")
        else:
            await asyncio.to_thread(bot.send_message, chat_id, "❌ Solve error")

    elif lower.startswith("/plot"):
        buf = plot(text[5:].strip())
        if buf:
            await asyncio.to_thread(bot.send_photo, chat_id, buf)
        else:
            await asyncio.to_thread(bot.send_message, chat_id, "❌ Invalid plot input")

    else:
        result = evaluate(text, chat_id)

        if result is not None:
            await asyncio.to_thread(bot.send_message, chat_id, f"✅\n`{result}`")
        else:
            await asyncio.to_thread(bot.send_message, chat_id, "❌ Invalid input")

    return {"ok": True}
