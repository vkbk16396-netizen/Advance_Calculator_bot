import os
import re
import asyncio
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import telebot
import sqlite3
import requests
from fastapi import FastAPI, Request
from sympy.stats import Normal, density
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

app = FastAPI()

# ================= TOKEN =================
TOKEN = os.getenv("BOT_TOKEN")
bot = telebot.TeleBot(TOKEN, parse_mode="Markdown")

# ================= DATABASE =================
conn = sqlite3.connect("history.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS history (chat_id INTEGER, expr TEXT, result TEXT)")
conn.commit()

chat_variables = {}

# ================= PREPROCESS =================
def preprocess(expr):
    expr = expr.lower().strip()
    expr = re.sub(r'(sin|cos|tan)\s+(\d+)', r'\1(\2)', expr)
    expr = expr.replace("×","*").replace("÷","/")
    expr = re.sub(r'(\d+(\.\d+)?)\s*%', r'(\1/100)', expr)
    return expr.replace("^","**")

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

    if expr.startswith("matrix"):
        try:
            return sp.Matrix(eval(expr[6:], {"__builtins__":{}}))
        except:
            return None

    safe = safe_locals(chat_id)

    try:
        res = sp.sympify(expr, locals=safe)
        return str(res) if res.free_symbols else float(res.evalf())
    except:
        return None

# ================= SAVE =================
def save_history(chat_id, expr, result):
    cursor.execute("INSERT INTO history VALUES (?, ?, ?)", (chat_id, expr, str(result)))
    conn.commit()

# ================= HELP =================
def get_help():
    return """
✨ *ULTIMATE CALCULATOR PRO* 🤖
━━━━━━━━━━━━━━━━━━━

🧮 *BASIC*
`2+2`, `5^2`, `10/3`, `5%`

📐 *TRIGONOMETRY*
`sin(30)`, `cos 60`

📊 *CALCULUS*
`diff(x^2,x)`
`integrate(x^2,x)`

📦 *MATRIX*
`Matrix([[1,2],[3,4]])`

🎲 *PROBABILITY*
`pdf(Normal(0,1),0)`

📈 *GRAPH*
`/plot sin(x),cos(x)`

🧠 *SOLVE*
`/solve x^2-4=0`

🔗 *URL SHORTENER*
`/short https://example.com`

📂 *EXPORT HISTORY*
`/export`

🔄 *UNIT CONVERTER*
`10 km to m`

📜 *DATABASE*
`/dbhistory`

━━━━━━━━━━━━━━━━━━━
💡 Try: `2²`, `cos 60`, `sin(30)`
"""

# ================= HELP BUTTONS =================
def help_buttons():
    m = InlineKeyboardMarkup()
    m.row(
        InlineKeyboardButton("📈 Plot", callback_data="plot"),
        InlineKeyboardButton("🧠 Solve", callback_data="solve")
    )
    m.row(
        InlineKeyboardButton("🔗 Short URL", callback_data="short"),
        InlineKeyboardButton("📂 Export", callback_data="export")
    )
    return m

# ================= PLOT =================
def plot(expr):
    x = sp.symbols('x')
    funcs = expr.split(",")
    xs = np.linspace(-10,10,400)

    plt.figure()
    for f in funcs:
        try:
            f_np = sp.lambdify(x, sp.sympify(preprocess(f)), 'numpy')
            plt.plot(xs, f_np(xs))
        except:
            pass

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

# ================= WEBHOOK =================
@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()

    # BUTTONS
    if "callback_query" in data:
        call = data["callback_query"]
        chat_id = call["message"]["chat"]["id"]
        btn = call["data"]

        if btn == "plot":
            await asyncio.to_thread(bot.send_message, chat_id, "`/plot sin(x)`")
        elif btn == "solve":
            await asyncio.to_thread(bot.send_message, chat_id, "`/solve x^2-4=0`")
        elif btn == "short":
            await asyncio.to_thread(bot.send_message, chat_id, "`/short https://example.com`")
        elif btn == "export":
            await asyncio.to_thread(bot.send_message, chat_id, "`/export`")

        return {"ok": True}

    if "message" not in data:
        return {"ok": True}

    msg = data["message"]
    chat_id = msg["chat"]["id"]
    text = msg.get("text","").strip()
    lower = text.lower()

    # START (IMPROVED DESIGN)
    if lower == "/start":
        await asyncio.to_thread(
            bot.send_message,
            chat_id,
            "✨ *Welcome to the Most Advanced Calculator* 🤖\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "🚀 *Fast • Powerful • Intelligent*\n\n"
            "🧮 Solve complex calculations instantly\n"
            "📊 Plot graphs & analyze functions\n"
            "🧠 Perform calculus, algebra & more\n"
            "📦 Work with matrices & probability\n\n"
            "👨‍💻 *Developed by:* @Sudhakaran12\n\n"
            "👉 Use /help to explore all features\n"
            "💡 Try: `2²`, `cos 60`, `sin(30)`"
        )

    # HELP (PRO)
    elif lower == "/help":
        await asyncio.to_thread(
            bot.send_message,
            chat_id,
            get_help(),
            reply_markup=help_buttons()
        )

    elif lower.startswith("/short"):
        url = text[6:]
        res = requests.get(f"http://tinyurl.com/api-create.php?url={url}").text
        await asyncio.to_thread(bot.send_message, chat_id, res)

    elif lower.startswith("/plot"):
        await asyncio.to_thread(bot.send_photo, chat_id, plot(text[5:]))

    elif lower.startswith("/solve"):
        res = sp.solve(sp.sympify(text[6:].replace("=","-(")+")"))
        await asyncio.to_thread(bot.send_message, chat_id, f"🧠 `{res}`")

    elif lower == "/export":
        cursor.execute("SELECT expr,result FROM history WHERE chat_id=?", (chat_id,))
        rows = cursor.fetchall()

        if not rows:
            await asyncio.to_thread(bot.send_message, chat_id, "❌ No history")
        else:
            file = f"history_{chat_id}.txt"
            with open(file,"w") as f:
                for e,r in rows:
                    f.write(f"{e}={r}\n")
            with open(file,"rb") as f:
                await asyncio.to_thread(bot.send_document, chat_id, f)

    elif " to " in lower:
        try:
            v,u1,_,u2 = lower.split()
            v=float(v)
            conv = {("km","m"):1000,("m","km"):0.001}
            res = v*conv[(u1,u2)]
            await asyncio.to_thread(bot.send_message, chat_id, f"🔄 {res}")
        except:
            pass

    else:
        result = evaluate(text, chat_id)
        if result is not None:
            save_history(chat_id, text, result)
            await asyncio.to_thread(bot.send_message, chat_id, f"✅ `{result}`")
        else:
            await asyncio.to_thread(bot.send_message, chat_id, "❌ Invalid input")

    return {"ok": True}
