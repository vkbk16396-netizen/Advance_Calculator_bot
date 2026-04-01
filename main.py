import os
import re
import math
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from fastapi import FastAPI, Request
import telebot

# ================== CONFIG ==================
TOKEN = os.getenv("BOT_TOKEN")
bot = telebot.TeleBot(TOKEN)
app = FastAPI()

user_mode = {}     # deg/rad
user_history = {}  # history storage

# ================== NORMALIZE INPUT ==================
def normalize_input(expr: str) -> str:
    superscript_map = {
        '⁰': '0','¹': '1','²': '2','³': '3',
        '⁴': '4','⁵': '5','⁶': '6',
        '⁷': '7','⁸': '8','⁹': '9'
    }

    fraction_map = {
        '½': '1/2',
        '⅓': '1/3','⅔': '2/3',
        '¼': '1/4','¾': '3/4',
        '⅕': '1/5','⅖': '2/5','⅗': '3/5','⅘': '4/5',
        '⅙': '1/6','⅚': '5/6',
        '⅛': '1/8','⅜': '3/8','⅝': '5/8','⅞': '7/8'
    }

    result = ""
    power_mode = False

    for ch in expr:
        if ch in superscript_map:
            if not power_mode:
                result += "^"
                power_mode = True
            result += superscript_map[ch]

        elif ch in fraction_map:
            if result and result[-1].isdigit():
                result += f"+({fraction_map[ch]})"
            else:
                result += f"({fraction_map[ch]})"
            power_mode = False

        else:
            result += ch
            power_mode = False

    return result


# ================== SAFE EVAL ==================
def evaluate(expr, user_id):
    expr = normalize_input(expr)
    expr = expr.replace("^", "**")

    x = sp.symbols('x')

    try:
        result = sp.sympify(expr)

        # Handle trig mode
        if user_mode.get(user_id, "rad") == "deg":
            result = result.subs({
                sp.sin(x): sp.sin(sp.rad(x)),
                sp.cos(x): sp.cos(sp.rad(x)),
                sp.tan(x): sp.tan(sp.rad(x))
            })

        return float(result.evalf())

    except Exception as e:
        return f"❌ Invalid input"


# ================== COMMANDS ==================
@bot.message_handler(commands=['start'])
def start(msg):
    bot.reply_to(msg,
        "👋 Welcome to Most Advanced Calculator 🤖\n\n"
        "Made by @Sudhakaran12\n\n"
        "👉 Use /help to see all features"
    )

@bot.message_handler(commands=['help'])
def help_cmd(msg):
    bot.reply_to(msg,
        "🧮 Calculator Commands:\n\n"
        "/deg → Degree mode\n"
        "/rad → Radian mode\n"
        "/plot → Graph function\n"
        "/latex → Show LaTeX\n"
        "/py → Python eval\n"
        "/history → Show history\n"
        "/vars → Variables\n"
        "/clear → Clear history\n\n"
        "✅ Supports:\n"
        "5^2, 5⁵, 6⅞, sin(30), log(10)"
    )

@bot.message_handler(commands=['deg'])
def deg(msg):
    user_mode[msg.chat.id] = "deg"
    bot.reply_to(msg, "📐 Degree mode ON")

@bot.message_handler(commands=['rad'])
def rad(msg):
    user_mode[msg.chat.id] = "rad"
    bot.reply_to(msg, "📏 Radian mode ON")

@bot.message_handler(commands=['history'])
def history(msg):
    hist = user_history.get(msg.chat.id, [])
    if not hist:
        bot.reply_to(msg, "No history")
    else:
        bot.reply_to(msg, "\n".join(hist[-10:]))

@bot.message_handler(commands=['clear'])
def clear(msg):
    user_history[msg.chat.id] = []
    bot.reply_to(msg, "🧹 Cleared")

# ================== PLOT ==================
@bot.message_handler(commands=['plot'])
def plot_cmd(msg):
    try:
        expr = msg.text.replace("/plot", "").strip()
        expr = normalize_input(expr).replace("^", "**")

        x = sp.symbols('x')
        f = sp.lambdify(x, sp.sympify(expr), "numpy")

        xs = np.linspace(-10, 10, 400)
        ys = f(xs)

        plt.figure()
        plt.plot(xs, ys)

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        bot.send_photo(msg.chat.id, buf)

    except:
        bot.reply_to(msg, "❌ Plot error")

# ================== MAIN HANDLER ==================
@bot.message_handler(func=lambda m: True)
def calc(msg):
    user_id = msg.chat.id
    expr = msg.text.strip()

    result = evaluate(expr, user_id)

    if isinstance(result, float):
        response = f"✅ {result}"
        user_history.setdefault(user_id, []).append(f"{expr} = {result}")
    else:
        response = result

    bot.reply_to(msg, response)


# ================== WEBHOOK ==================
@app.post("/webhook")
async def webhook(req: Request):
    data = await req.json()
    update = telebot.types.Update.de_json(data)
    bot.process_new_updates([update])
    return {"ok": True}
