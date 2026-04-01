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
    if not expr:
        return ""

    # trig fix
    expr = re.sub(r'(sin|cos|tan)\s+(\d+)', r'\1(\2)', expr)

    # superscripts → power
    superscripts = {
        "⁰":"0","¹":"1","²":"2","³":"3","⁴":"4",
        "⁵":"5","⁶":"6","⁷":"7","⁸":"8","⁹":"9"
    }

    new_expr = ""
    power_mode = False

    for ch in expr:
        if ch in superscripts:
            if not power_mode:
                new_expr += "**"
                power_mode = True
            new_expr += superscripts[ch]
        else:
            power_mode = False
            new_expr += ch

    expr = new_expr

    # FULL fraction support
    fractions = {
        "½":"1/2","¼":"1/4","¾":"3/4",
        "⅓":"1/3","⅔":"2/3",
        "⅕":"1/5","⅖":"2/5","⅗":"3/5","⅘":"4/5",
        "⅙":"1/6","⅚":"5/6",
        "⅛":"1/8","⅜":"3/8","⅝":"5/8","⅞":"7/8"
    }

    for k,v in fractions.items():
        expr = expr.replace(k, v)

    # mixed numbers (5¾ → 5+3/4)
    expr = re.sub(r'(\d+)\s*(\d+/\d+)', r'(\1+\2)', expr)

    # percent
    expr = re.sub(r'(\d+(\.\d+)?)\s*%', r'(\1/100)', expr)

    # symbols
    expr = expr.replace("×","*").replace("÷","/")

    return expr

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

    if "message" not in data:
        return {"ok": True}

    msg = data["message"]
    chat_id = msg["chat"]["id"]
    text = msg.get("text","").strip()
    lower = text.lower()

    # ===== START (UNCHANGED) =====
    if lower == "/start":
        await asyncio.to_thread(
            bot.send_message,
            chat_id,
            "👋 Welcome to Most Advanced Calculator 🤖\n\nMade by @Sudhakaran12\n\n👉 Use /help to see all features"
        )

    # ===== HELP (UNCHANGED) =====
    elif lower == "/help":
        await asyncio.to_thread(bot.send_message, chat_id, "/help")

    # ===== SHORT URL =====
    elif lower.startswith("/short"):
        url = text[6:]
        res = requests.get(f"http://tinyurl.com/api-create.php?url={url}").text
        await asyncio.to_thread(bot.send_message, chat_id, res)

    # ===== PLOT =====
    elif lower.startswith("/plot"):
        await asyncio.to_thread(bot.send_photo, chat_id, plot(text[5:]))

    # ===== SOLVE =====
    elif lower.startswith("/solve"):
        res = sp.solve(sp.sympify(text[6:].replace("=","-(")+")"))
        await asyncio.to_thread(bot.send_message, chat_id, f"🧠 `{res}`")

    # ===== EXPORT =====
    elif lower == "/export":
        cursor.execute("SELECT expr,result FROM history WHERE chat_id=?", (chat_id,))
        rows = cursor.fetchall()

        if not rows:
            await asyncio.to_thread(bot.send_message, chat_id, "❌ No history")
        else:
            file = f"history_{chat_id}.txt"
            with open(file,"w") as f:
                for e,r in rows:
                    f.write(f"{e} = {r}\n")
            with open(file,"rb") as f:
                await asyncio.to_thread(bot.send_document, chat_id, f)

    # ===== UNIT CONVERTER =====
    elif " to " in lower:
        try:
            v,u1,_,u2 = lower.split()
            v=float(v)
            conv = {
                ("km","m"):1000,
                ("m","km"):0.001,
                ("kg","g"):1000,
                ("g","kg"):0.001
            }
            res = v*conv[(u1,u2)]
            await asyncio.to_thread(bot.send_message, chat_id, f"🔄 {res}")
        except:
            pass

    # ===== CALCULATE =====
    else:
        result = evaluate(text, chat_id)
        if result is not None:
            save_history(chat_id, text, result)
            await asyncio.to_thread(bot.send_message, chat_id, f"✅ `{result}`")
        else:
            await asyncio.to_thread(bot.send_message, chat_id, "❌ Invalid input")

    return {"ok": True}
