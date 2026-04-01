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

# ================= STORAGE =================
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

# ================= SAVE HISTORY =================
def save_history(chat_id, expr, result):
    cursor.execute(
        "INSERT INTO history VALUES (?, ?, ?)",
        (chat_id, expr, str(result))
    )
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
    except Exception as e:
        return f"❌ AI Error: {e}"

# ================= OCR =================
def solve_image(path):
    img = Image.open(path)
    text = pytesseract.image_to_string(img)
    return text

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
📘 ULTIMATE CALCULATOR HELP

🧮 BASIC
2+2, 5^2, 10/3

📐 TRIG
sin(30), cos 60

📊 CALCULUS
diff(x^2,x)
integrate(x^2,x)

📦 MATRIX
Matrix([[1,2],[3,4]])

📈 GRAPH
/plot sin(x)

🐍 PYTHON
/py 2**10

📂 MEMORY
x=10, x+5
"""

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

    # ===== IMAGE OCR =====
    if "photo" in msg:
        file_id = msg["photo"][-1]["file_id"]
        file = bot.get_file(file_id)
        data_file = bot.download_file(file.file_path)

        with open("img.png", "wb") as f:
            f.write(data_file)

        extracted = solve_image("img.png")
        result = evaluate(extracted, chat_id)

        await asyncio.to_thread(bot.send_message, chat_id, f"📸 Detected:\n`{extracted}`")

        if result:
            await asyncio.to_thread(bot.send_message, chat_id, f"✅ `{result}`")
        else:
            await asyncio.to_thread(bot.send_message, chat_id, "❌ Could not solve")

        return {"ok": True}

    # ===== START (UNCHANGED) =====
    if lower == "/start":
        await asyncio.to_thread(
            bot.send_message,
            chat_id,
            "👋 Welcome to Most Advanced Calculator 🤖\n\n"
            "Made by @Sudhakaran12\n\n"
            "👉 Use /help to see all features"
        )

    # ===== HELP (UNCHANGED) =====
    elif lower == "/help":
        await asyncio.to_thread(bot.send_message, chat_id, get_help())

    elif lower.startswith("/solve"):
        res = solve_eq(text[6:].strip())
        await asyncio.to_thread(bot.send_message, chat_id, f"🧠 `{res}`")

    elif lower.startswith("/plot"):
        buf = plot(text[5:].strip())
        if buf:
            await asyncio.to_thread(bot.send_photo, chat_id, buf)
        else:
            await asyncio.to_thread(bot.send_message, chat_id, "❌ Invalid plot input")

    elif lower.startswith("/ai"):
        prompt = text[3:].strip()
        reply = ai_chat(prompt)
        await asyncio.to_thread(bot.send_message, chat_id, reply)

    elif lower == "/dbhistory":
        cursor.execute("SELECT expr, result FROM history WHERE chat_id=?", (chat_id,))
        rows = cursor.fetchall()

        if not rows:
            await asyncio.to_thread(bot.send_message, chat_id, "No history")
        else:
            txt = "\n".join([f"{e} = {r}" for e, r in rows[-10:]])
            await asyncio.to_thread(bot.send_message, chat_id, txt)

    else:
        result = evaluate(text, chat_id)

        if result is not None:
            save_history(chat_id, text, result)
            await asyncio.to_thread(bot.send_message, chat_id, f"✅\n`{result}`")
        else:
            await asyncio.to_thread(bot.send_message, chat_id, "❌ Invalid input")

    return {"ok": True}
