"""Telegram-бот с умной векторной памятью на базе Pinecone (VPg05).

Архитектура памяти:
  - Краткосрочная (short-term): история текущего разговора в оперативной памяти.
  - Долговременная (long-term): векторная база Pinecone — сохраняются только
    сообщения пользователя (не ответы бота).

При каждом сообщении бот:
  1. Ищет релевантные воспоминания в Pinecone.
  2. Формирует системный промпт с контекстом из памяти.
  3. Отправляет запрос в OpenAI вместе с историей.
  4. Сохраняет текст пользователя в Pinecone (с проверкой на дубликат).
  5. Логирует action: inserted / updated / skipped.
"""

from __future__ import annotations

import logging
import os
import uuid

import telebot
from dotenv import load_dotenv
from openai import OpenAI

from pinecone_manager import PineconeManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

load_dotenv()

# ── Конфигурация ─────────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_API", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
SHORT_TERM_LIMIT = 10  # пар (user + assistant) хранится в сессии

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN не задан в .env")

# ── Клиенты ──────────────────────────────────────────────────────────────────
memory = PineconeManager()

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

bot = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode=None)

# Краткосрочная история: {user_id: [{"role": ..., "content": ...}]}
chat_histories: dict[int, list[dict]] = {}


# ── Вспомогательные функции ──────────────────────────────────────────────────

def _user_label(user: telebot.types.User) -> str:
    """Строит имя пользователя из доступных полей — безопасен при отсутствии any."""
    parts = [p for p in (user.first_name, user.last_name) if p]
    return " ".join(parts) if parts else f"пользователь_{user.id}"


def _get_history(user_id: int) -> list[dict]:
    return chat_histories.setdefault(user_id, [])


def _trim_history(user_id: int) -> None:
    h = chat_histories.get(user_id, [])
    max_messages = SHORT_TERM_LIMIT * 2  # пара = 2 записи
    if len(h) > max_messages:
        chat_histories[user_id] = h[-max_messages:]


def _build_system_prompt(user_label: str, memories: list[dict]) -> str:
    base = (
        f"Ты — дружелюбный ассистент с долговременной памятью. "
        f"Ты общаешься с пользователем {user_label}. "
        "Отвечай кратко и по делу. "
        "Если в долговременной памяти есть релевантная информация — используй её в ответе."
    )
    if not memories:
        return base

    facts = "\n".join(
        f"- {m['metadata'].get('text', '')}"
        for m in memories
        if m["metadata"].get("text")
    )
    return base + f"\n\nИз долговременной памяти о пользователе:\n{facts}"


# ── Handlers ─────────────────────────────────────────────────────────────────

@bot.message_handler(commands=["start", "help"])
def start_handler(msg: telebot.types.Message) -> None:
    name = _user_label(msg.from_user)
    bot.reply_to(
        msg,
        f"Привет, {name}! 👋\n\n"
        "Я запоминаю всё важное из нашего разговора.\n"
        "Просто пиши мне — я буду помнить, что ты говорил.\n\n"
        "/reset — очистить историю разговора",
    )


@bot.message_handler(commands=["reset"])
def reset_handler(msg: telebot.types.Message) -> None:
    chat_histories.pop(msg.from_user.id, None)
    bot.reply_to(msg, "История разговора очищена. Начнём заново!")


@bot.message_handler(content_types=["text"])
def message_handler(msg: telebot.types.Message) -> None:
    user_id = msg.from_user.id
    user_label = _user_label(msg.from_user)
    user_text = msg.text.strip()

    try:
        # 1. Ищем релевантные воспоминания из долговременной памяти
        memories = memory.query_by_text(user_text, top_k=5)
        log.info("Найдено воспоминаний: %d для запроса: '%s'", len(memories), user_text[:50])

        # 2. Строим сообщения для LLM
        system_prompt = _build_system_prompt(user_label, memories)
        history = _get_history(user_id)
        messages = (
            [{"role": "system", "content": system_prompt}]
            + history
            + [{"role": "user", "content": user_text}]
        )

        # 3. Запрос к LLM
        response = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            max_tokens=1000,
        )
        answer = response.choices[0].message.content.strip()

        # 4. Сохраняем ТОЛЬКО сообщение пользователя в долговременную память
        #    (служебные данные и ответы бота не сохраняем)
        doc_id = f"{user_id}-{uuid.uuid4().hex[:12]}"
        result = memory.upsert_document(
            doc_id,
            user_text,
            metadata={
                "user_id": str(user_id),
                "user_name": user_label,
            },
        )
        log.info(
            "Память: action=%s | score=%s | '%s'",
            result["action"],
            f"{result['similarity_score']:.3f}" if result["similarity_score"] else "—",
            user_text[:60],
        )

        # 5. Обновляем краткосрочную историю
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": answer})
        _trim_history(user_id)

        bot.reply_to(msg, answer[:4096])

    except Exception as exc:
        log.exception("Ошибка при обработке сообщения")
        bot.reply_to(msg, f"Произошла ошибка: {exc}")


# ── Запуск ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Бот запущен. Ожидаем сообщения...")
    bot.infinity_polling(skip_pending=True)
