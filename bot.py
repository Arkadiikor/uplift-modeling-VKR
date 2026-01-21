from __future__ import annotations
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

sys.path.insert(0, str(SRC_DIR))

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

SESSIONS: dict[int, dict] = {}

def _script_path(name: str) -> Path:
    p = SRC_DIR / "cli" / name
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p

async def _run_cli(cmd: list[str], timeout_sec: int = 120) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return 124, "", f"Timeout after {timeout_sec}s"
    stdout = stdout_b.decode("utf-8", errors="ignore").strip()
    stderr = stderr_b.decode("utf-8", errors="ignore").strip()
    return proc.returncode, stdout, stderr

def _fmt_money(x: float) -> str:
    try:
        return f"{float(x):,.0f}".replace(",", " ")
    except Exception:
        return str(x)

def _tail(text: str, n: int = 20) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:]) if len(lines) > n else text

def _seg_line(seg: dict) -> str:
    return (
        f"SMS+скидка: {int(seg.get('SMS_7', 0))}, "
        f"SMS без скидки: {int(seg.get('SMS_REM', 0))}, "
        f"PUSH: {int(seg.get('PUSH', 0))}, "
        f"NONE: {int(seg.get('NONE', 0))}"
    )

def _plan_block(plans: dict, key: str, title: str) -> str:
    p = plans.get(key, {})
    if not p.get("feasible", False):
        return f"{title}: недоступен ({p.get('reason', '-')})"
    seg = p.get("segments", {})
    return (
        f"{title}:\n"
        f"  ОС={p.get('n_treatment', 0)}, КГ={p.get('n_control', 0)}\n"
        f"  Группы: {_seg_line(seg)}\n"
        f"  Ожидаемая ВП: {_fmt_money(p.get('expected_profit', 0.0))} руб\n"
        f"  Стоимость SMS: {_fmt_money(p.get('sms_cost_estimate', 0.0))} руб"
    )

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Команды:\n/calc <region> <target_profit_rub>\n/status")

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    feat_path = DATA_DIR / "marts" / "features_current.csv"
    nba_root = MODELS_DIR / "nba"
    regions = [p.name.replace("region=", "") for p in nba_root.glob("region=*")] if nba_root.exists() else []
    chat_id = update.effective_chat.id
    s = SESSIONS.get(chat_id)
    last_calc = "-"
    if s:
        last_calc = f"{s.get('region')} / {_fmt_money(s.get('target_profit'))} / {s.get('ts')}"
    await update.message.reply_text(
        "Статус:\n"
        f"features_current: {'OK' if feat_path.exists() else 'нет'}\n"
        f"модели по регионам: {', '.join(regions) if regions else 'нет'}\n"
        f"последний расчёт: {last_calc}"
    )

async def calc_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if msg is None:
        return
    if len(context.args) < 2:
        await msg.reply_text("Формат: /calc <region> <target_profit_rub>")
        return
    region = context.args[0]
    try:
        target_profit = float(context.args[1])
    except ValueError:
        await msg.reply_text("Неверное число target_profit_rub")
        return

    await msg.reply_text("Считаю планы…")

    script = _script_path("recommend.py")
    cmd = [
        sys.executable, str(script),
        "--region", region,
        "--target_profit", str(target_profit),
        "--models_dir", str(MODELS_DIR),
        "--data_dir", str(DATA_DIR),
        "--outputs_dir", str(OUTPUTS_DIR),
        "--dry_run",
    ]

    rc, stdout, stderr = await _run_cli(cmd, timeout_sec=120)
    if rc != 0:
        await msg.reply_text("Ошибка расчёта.\n" + (_tail(stderr) if stderr else ""))
        return

    try:
        result = json.loads(stdout)
    except Exception:
        await msg.reply_text("Не удалось прочитать ответ сервиса.")
        return

    chat_id = update.effective_chat.id
    SESSIONS[chat_id] = {"region": region, "target_profit": target_profit, "ts": datetime.now().isoformat(timespec="seconds")}

    if not result.get("feasible", False):
        await msg.reply_text(f"Кампания невозможна.\nПричина: {result.get('reason','-')}")
        return

    plans = result.get("plans", {})
    text = (
        f"Регион: {region}\n"
        f"Цель: {_fmt_money(result.get('target_profit', 0.0))} руб\n\n"
        f"{_plan_block(plans, 'min', 'Минимальный')}\n\n"
        f"{_plan_block(plans, 'rec', 'Рекомендуемый')}\n\n"
        f"{_plan_block(plans, 'max', 'Максимальный')}"
    )

    keyboard = [
        [InlineKeyboardButton("Сформировать минимальный", callback_data="gen:min")],
        [InlineKeyboardButton("Сформировать рекомендуемый", callback_data="gen:rec")],
        [InlineKeyboardButton("Сформировать максимальный", callback_data="gen:max")],
    ]
    await msg.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None:
        return
    await query.answer()
    data = query.data or ""
    if not data.startswith("gen:"):
        return
    plan = data.split(":", 1)[1]

    chat_id = update.effective_chat.id
    s = SESSIONS.get(chat_id)
    if not s:
        await query.edit_message_text("Нет активного расчёта. Сначала /calc ...")
        return

    script = _script_path("recommend.py")
    cmd = [
        sys.executable, str(script),
        "--region", s["region"],
        "--target_profit", str(s["target_profit"]),
        "--models_dir", str(MODELS_DIR),
        "--data_dir", str(DATA_DIR),
        "--outputs_dir", str(OUTPUTS_DIR),
        "--confirm",
        "--plan", plan,
    ]

    await query.edit_message_text("Формирую базы…")

    rc, stdout, stderr = await _run_cli(cmd, timeout_sec=120)
    if rc != 0:
        await query.edit_message_text("Ошибка формирования баз.\n" + (_tail(stderr) if stderr else ""))
        return

    try:
        result = json.loads(stdout)
    except Exception:
        await query.edit_message_text("Не удалось прочитать ответ сервиса.")
        return

    out_dir = result.get("output_dir")
    if not out_dir:
        await query.edit_message_text(f"Не удалось сформировать базы: {result.get('reason','-')}")
        return

    await query.edit_message_text(
        "Базы сформированы.\n"
        f"campaign_id: {result.get('campaign_id','-')}\n"
        f"план: {plan}\n"
        f"папка: {out_dir}"
    )

def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN env var")
    application = Application.builder().token(token).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("calc", calc_cmd))
    application.add_handler(CommandHandler("status", status_cmd))
    application.add_handler(CallbackQueryHandler(callback_handler))
    logger.info("Bot started")
    application.run_polling(allowed_updates=None)

if __name__ == "__main__":
    main()
