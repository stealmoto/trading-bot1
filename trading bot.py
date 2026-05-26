import os
import asyncio
import logging
from datetime import datetime

import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import Message, BufferedInputFile
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# ─────────────────────────────────────────────
# КОНФИГУРАЦИЯ
# ─────────────────────────────────────────────
BOT_TOKEN = os.environ.get("BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
MY_ID     = int(os.environ.get("MY_ID", "0"))          # твой Telegram user_id

RSI_PERIOD      = 14
RSI_OVERSOLD    = 30   # ниже — сигнал «Покупай»
RSI_OVERBOUGHT  = 70   # выше  — сигнал «Продавай»
CHECK_INTERVAL  = 30   # минут между проверками

# Список тикеров по умолчанию (можно менять командой /add и /remove)
DEFAULT_TICKERS = ["AAPL", "TSLA", "NVDA", "SPY", "MSFT"]

# ─────────────────────────────────────────────
# СОСТОЯНИЕ (in-memory)
# ─────────────────────────────────────────────
watchlist: list[str] = list(DEFAULT_TICKERS)
last_rsi_state: dict[str, str] = {}   # ticker -> "oversold" | "overbought" | "neutral"

# ─────────────────────────────────────────────
# УТИЛИТЫ: RSI
# ─────────────────────────────────────────────

def calculate_rsi(prices: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = prices.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def get_stock_data(ticker: str, period: str = "3mo", interval: str = "1d"):
    """Скачать данные через yfinance. Возвращает DataFrame или None."""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            return None
        df["RSI"] = calculate_rsi(df["Close"].squeeze())
        return df
    except Exception as e:
        logging.error(f"Ошибка загрузки {ticker}: {e}")
        return None


def get_summary(ticker: str, df: pd.DataFrame) -> str:
    """Краткая текстовая сводка по тикеру."""
    price = float(df["Close"].iloc[-1])
    rsi   = float(df["RSI"].iloc[-1])
    prev  = float(df["Close"].iloc[-2]) if len(df) > 1 else price
    trend = "📈 растёт" if price >= prev else "📉 падает"

    if rsi < RSI_OVERSOLD:
        signal = "🟢 Перепроданность — рассмотри покупку"
        zone   = f"RSI {rsi:.1f} (< {RSI_OVERSOLD})"
    elif rsi > RSI_OVERBOUGHT:
        signal = "🔴 Перекупленность — рассмотри продажу / фиксацию"
        zone   = f"RSI {rsi:.1f} (> {RSI_OVERBOUGHT})"
    else:
        signal = "⚪ Нейтральная зона"
        zone   = f"RSI {rsi:.1f}"

    return (
        f"📊 *{ticker}*\n"
        f"💵 Цена: *${price:.2f}*\n"
        f"📉 Тренд: {trend}\n"
        f"📐 {zone}\n"
        f"💡 {signal}\n"
        f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC"
    )


def build_chart(ticker: str, df: pd.DataFrame) -> BytesIO:
    """Строит PNG-график цены + RSI, возвращает BytesIO."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                                   gridspec_kw={"height_ratios": [3, 1]},
                                   sharex=True)
    fig.patch.set_facecolor("#0d1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    close = df["Close"].squeeze()
    rsi   = df["RSI"]

    # — цена
    ax1.plot(df.index, close, color="#58a6ff", linewidth=1.5, label="Close")
    ax1.fill_between(df.index, close, close.min(), alpha=0.15, color="#58a6ff")
    ax1.set_ylabel("Цена ($)", color="#8b949e")
    ax1.set_title(f"{ticker}", color="#e6edf3", fontsize=14, pad=10)
    ax1.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")

    # — RSI
    ax2.plot(df.index, rsi, color="#f0883e", linewidth=1.5)
    ax2.axhline(RSI_OVERBOUGHT, color="#f85149", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.axhline(RSI_OVERSOLD,   color="#3fb950", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.fill_between(df.index, rsi, RSI_OVERBOUGHT,
                     where=(rsi > RSI_OVERBOUGHT), alpha=0.25, color="#f85149")
    ax2.fill_between(df.index, rsi, RSI_OVERSOLD,
                     where=(rsi < RSI_OVERSOLD),   alpha=0.25, color="#3fb950")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI", color="#8b949e")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=30, color="#8b949e")

    plt.tight_layout(pad=1.5)
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf

# ─────────────────────────────────────────────
# АВТОПРОВЕРКА (Scheduler)
# ─────────────────────────────────────────────

async def auto_check(bot: Bot):
    """Проверяет RSI каждого тикера и шлёт сигнал при пересечении порогов."""
    if MY_ID == 0:
        return
    for ticker in watchlist:
        df = get_stock_data(ticker)
        if df is None or df["RSI"].isna().all():
            continue
        rsi = float(df["RSI"].iloc[-1])

        if rsi < RSI_OVERSOLD:
            new_state = "oversold"
        elif rsi > RSI_OVERBOUGHT:
            new_state = "overbought"
        else:
            new_state = "neutral"

        # Шлём уведомление только при смене состояния
        if last_rsi_state.get(ticker) != new_state and new_state != "neutral":
            text = get_summary(ticker, df)
            chart_buf = build_chart(ticker, df)
            await bot.send_photo(
                MY_ID,
                photo=BufferedInputFile(chart_buf.read(), filename=f"{ticker}.png"),
                caption=f"🚨 *Сигнал!*\n{text}",
                parse_mode="Markdown"
            )
        last_rsi_state[ticker] = new_state

# ─────────────────────────────────────────────
# ХЭНДЛЕРЫ
# ─────────────────────────────────────────────

async def cmd_start(message: Message):
    await message.answer(
        "👋 Привет! Я торговый бот-ассистент.\n\n"
        "Команды:\n"
        "• Просто напиши тикер — *NVDA*, *AAPL*, *TSLA*…\n"
        "• /list — список отслеживаемых тикеров\n"
        "• /add ТИКЕР — добавить тикер\n"
        "• /remove ТИКЕР — удалить тикер\n"
        "• /scan — немедленная проверка всех тикеров\n"
        "• /help — справка",
        parse_mode="Markdown"
    )


async def cmd_list(message: Message):
    if not watchlist:
        await message.answer("📋 Список пуст. Добавь тикеры командой /add ТИКЕР")
        return
    text = "📋 *Отслеживаемые тикеры:*\n" + "\n".join(f"• {t}" for t in watchlist)
    await message.answer(text, parse_mode="Markdown")


async def cmd_add(message: Message):
    parts = message.text.strip().split()
    if len(parts) < 2:
        await message.answer("Использование: /add ТИКЕР\nПример: /add GOOGL")
        return
    ticker = parts[1].upper()
    if ticker in watchlist:
        await message.answer(f"✅ {ticker} уже в списке.")
        return
    watchlist.append(ticker)
    await message.answer(f"✅ {ticker} добавлен в список отслеживания.")


async def cmd_remove(message: Message):
    parts = message.text.strip().split()
    if len(parts) < 2:
        await message.answer("Использование: /remove ТИКЕР\nПример: /remove GOOGL")
        return
    ticker = parts[1].upper()
    if ticker not in watchlist:
        await message.answer(f"❌ {ticker} не найден в списке.")
        return
    watchlist.remove(ticker)
    await message.answer(f"🗑 {ticker} удалён из списка.")


async def cmd_scan(message: Message, bot: Bot):
    await message.answer("🔍 Запускаю проверку всех тикеров…")
    await auto_check(bot)
    await message.answer("✅ Проверка завершена.")


async def handle_ticker(message: Message):
    ticker = message.text.strip().upper()
    # Фильтруем — только буквы и цифры (1–5 символов, типичный тикер)
    if not ticker.isalpha() or len(ticker) > 6:
        await message.answer("Введи тикер акции, например: *AAPL* или *NVDA*", parse_mode="Markdown")
        return

    wait_msg = await message.answer(f"⏳ Загружаю данные для *{ticker}*…", parse_mode="Markdown")

    df = get_stock_data(ticker)
    if df is None:
        await wait_msg.delete()
        await message.answer(f"❌ Не удалось найти данные для *{ticker}*. Проверь тикер.", parse_mode="Markdown")
        return

    text      = get_summary(ticker, df)
    chart_buf = build_chart(ticker, df)

    await wait_msg.delete()
    await message.answer_photo(
        photo=BufferedInputFile(chart_buf.read(), filename=f"{ticker}.png"),
        caption=text,
        parse_mode="Markdown"
    )

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

async def main():
    logging.basicConfig(level=logging.INFO)

    bot = Bot(token=BOT_TOKEN)
    dp  = Dispatcher()

    # Регистрация хэндлеров
    dp.message.register(cmd_start,  Command("start"))
    dp.message.register(cmd_start,  Command("help"))
    dp.message.register(cmd_list,   Command("list"))
    dp.message.register(cmd_add,    Command("add"))
    dp.message.register(cmd_remove, Command("remove"))
    dp.message.register(lambda m: cmd_scan(m, bot), Command("scan"))
    dp.message.register(handle_ticker, F.text)

    # Планировщик автопроверки
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        auto_check,
        "interval",
        minutes=CHECK_INTERVAL,
        args=[bot],
        next_run_time=datetime.now()   # первый запуск сразу
    )
    scheduler.start()

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
