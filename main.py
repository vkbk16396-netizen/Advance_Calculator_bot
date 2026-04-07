import logging
import asyncio
import time
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import sympy as sp
import google.generativeai as genai

# ===== CONFIG =====
BOT_TOKEN = "8719916019:AAGBPiuORWMpsotcwA_OwKji_w494dWRUPo"
GENAI_API_KEY = "AIzaSyDl4YaCIulXWEx0Ey5A7fpmhJWEY3yP2Ww"
AI_COOLDOWN_SECONDS = 8

# ===== LOGGING =====
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ===== GEMINI SETUP =====
genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

last_used: dict[int, float] = {}


def can_use_ai(user_id: int) -> bool:
    now = time.time()
    if user_id not in last_used or now - last_used[user_id] > AI_COOLDOWN_SECONDS:
        last_used[user_id] = now
        return True
    return False


def ask_gemini(user_input: str, user_id: int) -> str:
    if not can_use_ai(user_id):
        remaining = int(AI_COOLDOWN_SECONDS - (time.time() - last_used[user_id])) + 1
        return f"⏳ Please wait {remaining}s before using AI again."
    try:
        response = model.generate_content(user_input)
        if response and hasattr(response, "text"):
            return response.text.strip()
        return "⚠️ No response from AI."
    except Exception as e:
        err = str(e)
        logger.warning("Gemini error: %s", err)
        if "429" in err or "quota" in err.lower():
            return "⚠️ AI quota reached. Please try again later."
        return "❌ AI error. Please try again later."


AI_KEYWORDS = {"explain", "what", "who", "how", "write", "why", "when", "describe", "tell"}

def should_use_ai(text: str) -> bool:
    """Return True if the message looks like a natural language query."""
    normalized = text.lower().strip()
    if normalized.replace(" ", "").isdigit():
        return False
    return any(keyword in normalized for keyword in AI_KEYWORDS)


# ===== MATH SOLVER =====
x = sp.symbols("x")

def solve_math(expr: str) -> str | None:
    """Attempt to evaluate a symbolic math expression. Returns None on failure."""
    try:
        sanitized = expr.replace("^", "**")
        result = sp.sympify(sanitized)
        # Avoid returning unevaluated symbols as a "result"
        if result.free_symbols:
            return str(sp.simplify(result))
        return str(result)
    except Exception:
        return None


# ===== COMMAND HANDLERS =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 Welcome to the Advanced Calculator Bot!\n"
        "Send me a math expression or a question.\n"
        "Use /help to see all features."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "📘 *ADVANCED CALCULATOR — HELP*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "🧮 *Basic:* `2+2`, `5*6`, `10/3`\n"
        "📐 *Trig:* `sin(30)`, `cos(pi/4)`\n"
        "📘 *Algebra:* `x^2 + 2*x`\n"
        "📊 *Calculus:* `diff(x^2, x)`\n"
        "🤖 *AI:* `Explain Newton's laws`\n\n"
        "_Tip: The bot auto-detects whether to solve or ask AI._"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")


# ===== MAIN MESSAGE HANDLER =====
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_input = update.message.text.strip()
    user_id = update.message.from_user.id

    if not user_input:
        return

    # 1. Try symbolic math
    math_result = solve_math(user_input)
    if math_result is not None:
        await update.message.reply_text(f"✅ {math_result}")
        return

    # 2. AI fallback for natural language
    if should_use_ai(user_input):
        reply = await asyncio.to_thread(ask_gemini, user_input, user_id)
        await update.message.reply_text(reply)
        return

    # 3. Unrecognised input
    await update.message.reply_text(
        "❌ I couldn't understand that.\n"
        "Try a math expression like `2+2` or ask a question like `What is gravity?`",
        parse_mode="Markdown"
    )


# ===== ENTRY POINT =====
def main() -> None:
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("🚀 Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
