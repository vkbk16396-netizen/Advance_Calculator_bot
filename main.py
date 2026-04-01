import os
import re
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from fastapi import FastAPI, Request
import telebot

# ================= CONFIG =================
TOKEN = os.getenv("BOT_TOKEN")
bot = telebot.TeleBot(TOKEN)
app = FastAPI()

user_mode = {}
user_history = {}

# ================= NORMALIZE =================
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

# ================= FIX FUNCTIONS =================
def fix_functions(expr: str) -> str:
    expr = expr.lower()

    # sin 30 → sin(30)
    expr = re.sub(r'\b(sin|cos|tan|log|sqrt)\s+([0-9\.]+)', r'\1(\2)', expr)

    # tan45 → tan(45)
    expr = re.sub(r'\b(sin|cos|tan|log|sqrt)([0-9\.]+)', r'\1(\2)', expr)

    return expr

# ================= EVALUATE =================
def evaluate(expr, user_id):
    try:
        expr = normalize_input(expr)
        expr = fix_functions(expr)
        expr = expr.replace("^", "**")

        # Degree mode conversion
        if user_mode.get(user_id, "rad") == "deg":
            expr = re.sub(r'sin\((.*?)\)', r'sin(\1*pi/180)', expr)
            expr = re.sub(r'cos\((.*?)\)', r'cos(\1*pi/180)', expr)
            expr = re.sub(r'tan\((.*?)\)', r'tan(\1*pi/180)', expr)

        result = sp.sympify(expr).evalf()
        return float(result)

    except:
        return "❌ Invalid input"

# ================= COMMANDS =================
@bot.message_handler(commands=['start'])
def start(msg):
    bot.reply_to(msg,
        "👋 Welcome to Most Advanced Calculator 🤖\n\n"
        "Made by @Sudhakaran12\n\n"
        "👉 Use /help to see all features"
    )

@bot.message_handler(commands=['help'])
def help_cmd(msg):
    bot.reply_to(msg, """📘 ULTIMATE CALCULATOR HELP

━━━━━━━━━━━━━━━━━━━━━━
🧮 BASIC
2+2, 5^2, 10/3

📊 ADVANCED
sqrt(16)
log(10)
factorial(5)

━━━━━━━━━━━━━━━━━━━━━━
📐 TRIGONOMETRY
sin(30)
cos 60
tan(45)

Use:
/deg or /rad

━━━━━━━━━━━━━━━━━━━━━━
📈 CALCULUS
diff(x^2,x)
integrate(x^2,x)

━━━━━━━━━━━━━━━━━━━━━━
📦 MATRIX
Matrix([[1,2],[3,4]])

━━━━━━━━━━━━━━━━━━━━━━
📊 STATISTICS
mean(1,2,3)
variance(1,2,3)

━━━━━━━━━━━━━━━━━━━━━━
🎲 PROBABILITY
Normal(0,1)
pdf(Normal(0,1),0)

━━━━━━━━━━━━━━━━━━━━━━
📏 UNIT
10 km to m

━━━━━━━━━━━━━━━━━━━━━━
📊 GRAPH
/plot sin(x)

📐 LATEX
/latex diff(x^2,x)

🐍 PYTHON
/py 2**10

━━━━━━━━━━━━━━━━━━━━━━
📂 MEMORY
x=10
x+5

━━━━━━━━━━━━━━━━━━━━━━
🗂 COMMANDS
/deg /rad
/plot /latex /py
/history /vars
/clear /clearvars /clearfuncs

━━━━━━━━━━━━━━━━━━━━━━

🚀 Enjoy!
""")

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
    bot.reply_to(msg, "\n".join(hist[-10:]) if hist else "No history")

@bot.message_handler(commands=['clear'])
def clear(msg):
    user_history[msg.chat.id] = []
    bot.reply_to(msg, "🧹 Cleared")

# ================= PLOT =================
@bot.message_handler(commands=['plot'])
def plot_cmd(msg):
    try:
        expr = msg.text.replace("/plot", "").strip()
        expr = normalize_input(expr)
        expr = fix_functions(expr)
        expr = expr.replace("^", "**")

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

# ================= MAIN =================
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

# ================= WEBHOOK =================
@app.post("/webhook")
async def webhook(req: Request):
    data = await req.json()
    update = telebot.types.Update.de_json(data)
    bot.process_new_updates([update])
    return {"ok": True}
