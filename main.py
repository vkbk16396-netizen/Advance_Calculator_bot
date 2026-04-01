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
import requests
from PIL import Image
from fastapi import FastAPI, Request
from sympy.stats import Normal, density
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from openai import OpenAI

app = FastAPI()

# ================= TOKENS =================
TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

bot = telebot.TeleBot(TOKEN, parse_mode="Markdown")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ================= DATABASE =================
conn = sqlite3.connect("history.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS history (chat_id INTEGER, expr TEXT, result TEXT)")
conn.commit()

chat_variables = {}

# ================= PREPROCESS =================
def preprocess(expr):
    expr = expr.lower().strip()
    if not expr:
        return ""
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

# ================= SAVE =================
def save_history(chat_id, expr, result):
    cursor.execute("INSERT INTO history VALUES (?, ?, ?)", (chat_id, expr, str(result)))
    conn.commit()

# ================= AI =================
def ai_chat(prompt):
    if not client:
        return "❌ AI not configured"
    try:
        res = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role":"user","content":prompt}]
        )
        return res.choices[0].message.content
    except:
        return "❌ AI error"

# ================= OCR =================
def solve_image(path):
    return pytesseract.image_to_string(Image.open(path))

# ================= URL SHORTENER =================
def shorten_url(url):
    try:
        return requests.get(f"http://tinyurl.com/api-create.php?url={url}").text
    except:
        return "❌ Failed to shorten URL"

# ================= EXPORT =================
def export_history(chat_id):
    cursor.execute("SELECT expr, result FROM history WHERE chat_id=?", (chat_id,))
    rows = cursor.fetchall()

    if not rows:
        return None

    filename = f"history_{chat_id}.txt"
    with open(filename, "w") as f:
        for e, r in rows:
            f.write(f"{e} = {r}\n")

    return filename

# ================= UNIT =================
def unit_convert(text):
    try:
        parts = text.split()
        value = float(parts[0])
        from_unit = parts[1]
        to_unit = parts[3]

        conv = {
            ("km","m"): value*1000,
            ("m","km"): value/1000,
            ("kg","g"): value*1000,
            ("g","kg"): value/1000,
            ("cm","m"): value/100,
            ("m","cm"): value*100,
        }

        return conv.get((from_unit, to_unit), "❌ Not supported")
    except:
        return None

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

# ================= PLOT =================
def plot(expr):
    x = sp.symbols('x')
    funcs = expr.split(",")
    xs = np.linspace(-10,10,400)

    plt.figure()
    valid = False

    for f in funcs:
        f = f.strip()
        if not f:
            continue
        try:
            f_np = sp.lambdify(x, sp.sympify(preprocess(f)), 'numpy')
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

# ================= BUTTONS =================
def buttons():
    m = InlineKeyboardMarkup()
    m.row(
        InlineKeyboardButton("📘 Help", callback_data="help"),
        InlineKeyboardButton("📈 Plot", callback_data="plot")
    )
    m.row(
        InlineKeyboardButton("🧠 Solve", callback_data="solve"),
        InlineKeyboardButton("🤖 AI", callback_data="ai")
    )
    m.row(
        InlineKeyboardButton("💾 History", callback_data="history")
    )
    return m

# ================= HELP (UNCHANGED) =================
def get_help():
    return "Use /help from your existing version"

# ================= WEBHOOK =================
@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()

    if "callback_query" in data:
        call = data["callback_query"]
        chat_id = call["message"]["chat"]["id"]
        btn = call["data"]

        if btn == "help":
            await asyncio.to_thread(bot.send_message, chat_id, get_help())
        elif btn == "plot":
            await asyncio.to_thread(bot.send_message, chat_id, "`/plot sin(x)`")
        elif btn == "solve":
            await asyncio.to_thread(bot.send_message, chat_id, "`/solve x^2-4=0`")
        elif btn == "ai":
            await asyncio.to_thread(bot.send_message, chat_id, "`/ai explain math`")
        elif btn == "history":
            await asyncio.to_thread(bot.send_message, chat_id, "`/dbhistory`")

        return {"ok": True}

    if "message" not in data:
        return {"ok": True}

    msg = data["message"]
    chat_id = msg["chat"]["id"]
    text = msg.get("text","").strip()
    lower = text.lower()

    # IMAGE
    if "photo" in msg:
        file = bot.get_file(msg["photo"][-1]["file_id"])
        data_file = bot.download_file(file.file_path)

        with open("img.png","wb") as f:
            f.write(data_file)

        extracted = solve_image("img.png")
        result = evaluate(extracted, chat_id)

        await asyncio.to_thread(bot.send_message, chat_id, f"📸 `{extracted}`")

        if result:
            await asyncio.to_thread(bot.send_message, chat_id, f"✅ `{result}`")

        return {"ok": True}

    # START (UNCHANGED)
    if lower == "/start":
        await asyncio.to_thread(
            bot.send_message,
            chat_id,
            "👋 Welcome to Most Advanced Calculator 🤖\n\nMade by @Sudhakaran12\n\n👉 Use /help to see all features",
            reply_markup=buttons()
        )

    elif lower == "/help":
        await asyncio.to_thread(bot.send_message, chat_id, get_help())

    elif lower.startswith("/short"):
        url = text[6:].strip()
        await asyncio.to_thread(bot.send_message, chat_id, f"🔗 {shorten_url(url)}")

    elif lower == "/export":
        file = export_history(chat_id)
        if file:
            with open(file,"rb") as f:
                await asyncio.to_thread(bot.send_document, chat_id, f)
        else:
            await asyncio.to_thread(bot.send_message, chat_id, "❌ No history")

    elif " to " in lower:
        res = unit_convert(lower)
        if res:
            await asyncio.to_thread(bot.send_message, chat_id, f"🔄 {res}")

    elif lower.startswith("/plot"):
        buf = plot(text[5:].strip())
        if buf:
            await asyncio.to_thread(bot.send_photo, chat_id, buf)

    elif lower.startswith("/ai"):
        await asyncio.to_thread(bot.send_message, chat_id, ai_chat(text[3:]))

    elif lower.startswith("/solve"):
        res = sp.solve(sp.sympify(text[6:].replace("=","-(")+")"))
        await asyncio.to_thread(bot.send_message, chat_id, f"🧠 `{res}`")

    elif lower == "/dbhistory":
        cursor.execute("SELECT expr,result FROM history WHERE chat_id=?", (chat_id,))
        rows = cursor.fetchall()
        txt = "\n".join([f"{e}={r}" for e,r in rows[-10:]]) if rows else "No history"
        await asyncio.to_thread(bot.send_message, chat_id, txt)

    else:
        result = evaluate(text, chat_id)
        if result is not None:
            save_history(chat_id, text, result)
            await asyncio.to_thread(bot.send_message, chat_id, f"✅ `{result}`")
        else:
            await asyncio.to_thread(bot.send_message, chat_id, "❌ Invalid input")

    return {"ok": True}
