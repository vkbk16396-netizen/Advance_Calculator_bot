import os
import re
import asyncio
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import telebot
from fastapi import FastAPI, Request

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
chat_custom_funcs = {}

# ================= SMART INPUT =================
def preprocess_expression(expr: str) -> str:
    expr = expr.lower().strip()
    expr = re.sub(r'(sin|cos|tan)\s+(\d+)', r'\1(\2)', expr)
    return expr

# ================= HISTORY =================
def add_history(chat_id, expr, result):
    chat_history.setdefault(chat_id, []).append((expr, result))
    if len(chat_history[chat_id]) > 30:
        chat_history[chat_id].pop(0)

# ================= SAFE LOCALS =================
def get_safe_locals(chat_id):
    mode = chat_angle_mode.get(chat_id, "rad")

    if mode == "deg":
        trig = {
            "sin": lambda x: sp.sin(sp.rad(x)),
            "cos": lambda x: sp.cos(sp.rad(x)),
            "tan": lambda x: sp.tan(sp.rad(x)),
        }
    else:
        trig = {
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan
        }

    return {
        **trig,
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
        return float(res.evalf())
    except:
        return None

# ================= HELP MENU =================
def get_help():
    return """
📘 *ULTIMATE CALCULATOR HELP*

━━━━━━━━━━━━━━━━━━━━━━
🧮 *BASIC*
`2+2`, `5^2`, `10/3`

📊 *ADVANCED*
`sqrt(16)`
`log(10)`
`factorial(5)`

━━━━━━━━━━━━━━━━━━━━━━
📐 *TRIGONOMETRY*
`sin(30)`
`cos 60`
`tan(45)`

Use:
`/deg` or `/rad`

━━━━━━━━━━━━━━━━━━━━━━
📈 *CALCULUS*
`diff(x^2,x)`
`integrate(x^2,x)`

━━━━━━━━━━━━━━━━━━━━━━
📦 *MATRIX*
`Matrix([[1,2],[3,4]])`

━━━━━━━━━━━━━━━━━━━━━━
📊 *STATISTICS*
`mean(1,2,3)`
`variance(1,2,3)`

━━━━━━━━━━━━━━━━━━━━━━
🎲 *PROBABILITY*
`Normal(0,1)`
`pdf(Normal(0,1),0)`

━━━━━━━━━━━━━━━━━━━━━━
📏 *UNIT*
`10 km to m`

━━━━━━━━━━━━━━━━━━━━━━
📊 *GRAPH*
`/plot sin(x)`

📐 *LATEX*
`/latex diff(x^2,x)`

🐍 *PYTHON*
`/py 2**10`

━━━━━━━━━━━━━━━━━━━━━━
📂 *MEMORY*
`x=10`
`x+5`

━━━━━━━━━━━━━━━━━━━━━━
🗂 *COMMANDS*
`/deg` `/rad`
`/plot` `/latex` `/py`
`/history` `/vars`
`/clear` `/clearvars` `/clearfuncs`

━━━━━━━━━━━━━━━━━━━━━━
🔥 Try:
`cos 60`
`sin(30)`
`x=5`
`x^2`

🚀 Enjoy!
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
            "👋 *Welcome to the Most Advanced Calculator* 🤖\n\n"
            "🚀 With all powerful features built-in\n\n"
            "👨‍💻 Made by @Sudhakaran12\n\n"
            "👉 Type /help to see all features and how to use."
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
        txt = "📜 History empty" if not hist else "\n".join(f"`{e}` = {r}" for e, r in hist)
        await asyncio.to_thread(bot.send_message, chat_id, txt)

    elif lower == "/clear":
        chat_history[chat_id] = []
        await asyncio.to_thread(bot.send_message, chat_id, "🗑 History cleared")

    # ===== VARIABLES =====
    elif lower == "/vars":
        vars_ = chat_variables.get(chat_id, {})
        txt = "📌 No variables" if not vars_ else "\n".join(f"{k} = {v}" for k, v in vars_.items())
        await asyncio.to_thread(bot.send_message, chat_id, txt)

    elif lower == "/clearvars":
        chat_variables[chat_id] = {}
        await asyncio.to_thread(bot.send_message, chat_id, "🗑 Variables cleared")

    elif lower == "/clearfuncs":
        chat_custom_funcs[chat_id] = {}
        await asyncio.to_thread(bot.send_message, chat_id, "🗑 Functions cleared")

    # ===== PLOT =====
    elif lower.startswith("/plot"):
        expr = preprocess_expression(text[5:].strip())
        try:
            x = sp.symbols('x')
            f = sp.lambdify(x, sp.sympify(expr), 'numpy')
            xs = np.linspace(-10, 10, 500)
            ys = np.where(np.isfinite(f(xs)), f(xs), np.nan)

            plt.plot(xs, ys)
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
            sym = sp.sympify(expr)
            latex = sp.latex(sym)
            await asyncio.to_thread(bot.send_message, chat_id, f"📐 `{latex}`")
        except:
            await asyncio.to_thread(bot.send_message, chat_id, "❌ LaTeX error")

    # ===== PY =====
    elif lower.startswith("/py"):
        code = text[3:].strip()
        if "import" in code or "__" in code:
            await asyncio.to_thread(bot.send_message, chat_id, "❌ Unsafe")
        else:
            try:
                res = eval(code, {"__builtins__": {}}, {})
                await asyncio.to_thread(bot.send_message, chat_id, str(res))
            except:
                await asyncio.to_thread(bot.send_message, chat_id, "❌ Error")

    # ===== CALCULATE =====
    else:
        result = evaluate(text, chat_id)
        if result is not None:
            add_history(chat_id, text, result)
            await asyncio.to_thread(bot.send_message, chat_id, f"✅ `{result}`")
        else:
            await asyncio.to_thread(bot.send_message, chat_id, "❌ Invalid input")

    return {"ok": True}
