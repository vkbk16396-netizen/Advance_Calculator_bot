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

    # sin 30 → sin(30)
    expr = re.sub(r'(sin|cos|tan)\s+(\d+)', r'\1(\2)', expr)

    # superscript powers
    superscripts = {
        "⁰":"^0","¹":"^1","²":"^2","³":"^3","⁴":"^4",
        "⁵":"^5","⁶":"^6","⁷":"^7","⁸":"^8","⁹":"^9"
    }
    for k, v in superscripts.items():
        expr = expr.replace(k, v)

    # symbols
    expr = expr.replace("×", "*").replace("÷", "/")

    # % → /100
    expr = re.sub(r'(\d+(\.\d+)?)\s*%', r'(\1/100)', expr)

    return expr

# ================= HISTORY =================
def add_history(chat_id, expr, result):
    chat_history.setdefault(chat_id, []).append((expr, result))
    if len(chat_history[chat_id]) > 30:
        chat_history[chat_id].pop(0)

# ================= SAFE LOCALS =================
def get_safe_locals(chat_id):
    mode = chat_angle_mode.get(chat_id, "rad")

    x, y, z = sp.symbols('x y z')

    if mode == "deg":
        trig = {
            "sin": lambda v: sp.sin(sp.rad(v)),
            "cos": lambda v: sp.cos(sp.rad(v)),
            "tan": lambda v: sp.tan(sp.rad(v)),
        }
    else:
        trig = {
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan
        }

    return {
        **trig,
        "x": x,
        "y": y,
        "z": z,
        "sqrt": sp.sqrt,
        "log": sp.log,
        "exp": sp.exp,
        "pi": sp.pi,
        "e": sp.E,
        "factorial": sp.factorial,
        "diff": sp.diff,
        "integrate": sp.integrate,
        "Matrix": sp.Matrix,
        "abs": sp.Abs,
        "Normal": Normal,
        "pdf": density,
        **chat_variables.get(chat_id, {})
    }

# ================= EVALUATE =================
def evaluate(expr, chat_id):
    expr = preprocess_expression(expr)
    expr = expr.replace("^", "**")

    safe = get_safe_locals(chat_id)

    # Variable assignment
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

    except Exception as e:
        print("ERROR:", e)
        return None

# ================= HELP =================
def get_help():
    return """
📘 *ULTIMATE CALCULATOR*

🧮 BASIC
`2+2`, `5^2`, `10/3`, `5%`

📐 TRIG
`sin(30)` `cos 60`
Use `/deg` `/rad`

📊 CALCULUS
`diff(x^2,x)`
`integrate(x^2,x)`

📦 MATRIX
`Matrix([[1,2],[3,4]])`

🎲 PROBABILITY
`pdf(Normal(0,1),0)`

📈 GRAPH
`/plot sin(x)`

📐 LATEX
`/latex diff(x^2,x)`

🐍 PYTHON
`/py 2**10`

📂 MEMORY
`x=10`, `x+5`

COMMANDS:
/history /vars /clear /clearvars
"""

# ================= ROOT =================
@app.get("/")
async def root():
    return {"status": "LIVE 🚀"}

# ================= WEBHOOK =================
@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()

    if "message" not in data:
        return {"ok": True}

    msg = data["message"]
    chat_id = msg["chat"]["id"]
    text = msg.get("text", "").strip()
    lower = text.lower()

    # ===== START =====
    if lower == "/start":
        await asyncio.to_thread(
            bot.send_message,
            chat_id,
            "👋 *Advanced Calculator Bot*\n\nType /help"
        )

    # ===== HELP =====
    elif lower == "/help":
        await asyncio.to_thread(bot.send_message, chat_id, get_help())

    # ===== MODES =====
    elif lower == "/deg":
        chat_angle_mode[chat_id] = "deg"
        await asyncio.to_thread(bot.send_message, chat_id, "📐 Degree mode ON")

    elif lower == "/rad":
        chat_angle_mode[chat_id] = "rad"
        await asyncio.to_thread(bot.send_message, chat_id, "📐 Radian mode ON")

    # ===== HISTORY =====
    elif lower == "/history":
        hist = chat_history.get(chat_id, [])
        txt = "Empty" if not hist else "\n".join(f"`{e}` = {r}" for e, r in hist)
        await asyncio.to_thread(bot.send_message, chat_id, txt)

    elif lower == "/clear":
        chat_history[chat_id] = []
        await asyncio.to_thread(bot.send_message, chat_id, "Cleared")

    # ===== VARIABLES =====
    elif lower == "/vars":
        vars_ = chat_variables.get(chat_id, {})
        txt = "None" if not vars_ else "\n".join(f"{k} = {v}" for k, v in vars_.items())
        await asyncio.to_thread(bot.send_message, chat_id, txt)

    elif lower == "/clearvars":
        chat_variables[chat_id] = {}
        await asyncio.to_thread(bot.send_message, chat_id, "Cleared")

    # ===== PLOT =====
    elif lower.startswith("/plot"):
        expr = preprocess_expression(text[5:].strip())
        try:
            x = sp.symbols('x')
            f = sp.lambdify(x, sp.sympify(expr), 'numpy')
            xs = np.linspace(-10, 10, 400)
            ys = f(xs)

            plt.figure()
            plt.plot(xs, ys)
            plt.grid()

            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()

            await asyncio.to_thread(bot.send_photo, chat_id, buf)
        except:
            await asyncio.to_thread(bot.send_message, chat_id, "❌ Plot error")

    # ===== LATEX =====
    elif lower.startswith("/latex"):
        expr = preprocess_expression(text[6:].strip())
        try:
            latex = sp.latex(sp.sympify(expr))
            await asyncio.to_thread(bot.send_message, chat_id, f"`{latex}`")
        except:
            await asyncio.to_thread(bot.send_message, chat_id, "❌ Error")

    # ===== PY =====
    elif lower.startswith("/py"):
        code = text[3:].strip()
        if "import" in code or "__" in code:
            await asyncio.to_thread(bot.send_message, chat_id, "❌ Unsafe")
        else:
            try:
                res = eval(code, {"__builtins__": {}})
                await asyncio.to_thread(bot.send_message, chat_id, str(res))
            except:
                await asyncio.to_thread(bot.send_message, chat_id, "❌ Error")

    # ===== CALCULATE =====
    else:
        result = ervaluate(text, chat_id)
        if result is not None:
            add_history(chat_id, text, result)
            await asyncio.to_thread(bot.send_message, chat_id, f"✅ `{result}`")
        else:
            await asyncio.to_thread(bot.send_message, chat_id, "❌ Invalid input")

    return {"ok": True}
