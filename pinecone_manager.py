"""Менеджер для чтения и записи в векторную базу данных Pinecone.

Поддерживает фильтрацию дубликатов через косинусное сходство: перед записью
нового сообщения проверяет, есть ли уже похожий вектор в базе. Если сходство
высокое — обновляет существующую запись, если низкое — добавляет новую.
"""

from __future__ import annotations

import os
import time
import uuid
import logging
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

log = logging.getLogger(__name__)

# Глобальный порог косинусного сходства.
# >= SIMILARITY_THRESHOLD → дубликат (action: updated/skipped)
# <  SIMILARITY_THRESHOLD → новая информация (action: inserted)
SIMILARITY_THRESHOLD = 0.85


class PineconeManager:
    """Класс для управления операциями с векторной базой данных Pinecone."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        openai_model: Optional[str] = None,
        dimension: int = 1536,
    ) -> None:
        load_dotenv()

        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME")
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY не задан")
        if not self.index_name:
            raise ValueError("PINECONE_INDEX_NAME не задан")

        self.pc = Pinecone(api_key=self.api_key)
        self.index = self._ensure_index(dimension)

        openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        base_url = openai_base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.openai_model = openai_model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.openai_client = OpenAI(api_key=openai_key, base_url=base_url)

    # ── Внутреннее ──────────────────────────────────────────────────────────

    def _ensure_index(self, dimension: int) -> Any:
        """Создаёт индекс если не существует, возвращает объект индекса."""
        names = set(self.pc.list_indexes().names())
        if self.index_name not in names:
            log.info("Создаю индекс %s...", self.index_name)
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            for _ in range(60):
                desc = self.pc.describe_index(self.index_name)
                if desc.status.ready:
                    break
                time.sleep(2)
            else:
                raise TimeoutError("Индекс не стал готовым за 120 секунд")
            log.info("Индекс %s создан.", self.index_name)
        return self.pc.Index(self.index_name)

    def _check_similarity(self, vector: list[float]) -> Optional[dict[str, Any]]:
        """Возвращает ближайший вектор если score >= SIMILARITY_THRESHOLD, иначе None."""
        results = self.index.query(vector=vector, top_k=1, include_metadata=True)
        if results.matches and results.matches[0].score >= SIMILARITY_THRESHOLD:
            match = results.matches[0]
            return {"id": match.id, "score": float(match.score)}
        return None

    # ── Эмбеддинги ──────────────────────────────────────────────────────────

    def create_embedding(self, text: str) -> list[float]:
        """Создаёт числовой вектор (эмбеддинг) для переданного текста."""
        response = self.openai_client.embeddings.create(
            model=self.openai_model,
            input=text,
        )
        return response.data[0].embedding

    # ── Запись ──────────────────────────────────────────────────────────────

    def upsert_vector(
        self,
        vector_id: str,
        vector: list[float],
        metadata: Optional[dict[str, Any]] = None,
        check_similarity: bool = True,
    ) -> dict[str, Any]:
        """Записывает вектор в Pinecone с опциональной проверкой дубликатов.

        Returns:
            dict с полями:
              action         — 'inserted' | 'updated' | 'skipped'
              similarity_score — float если найден похожий вектор, иначе None
              existing_id    — id существующего вектора если найден
        """
        result: dict[str, Any] = {
            "action": "inserted",
            "similarity_score": None,
            "existing_id": None,
        }

        if check_similarity:
            similar = self._check_similarity(vector)
            if similar:
                existing_id = similar["id"]
                result["action"] = "updated"
                result["similarity_score"] = similar["score"]
                result["existing_id"] = existing_id
                self.index.upsert(vectors=[{
                    "id": existing_id,
                    "values": vector,
                    "metadata": metadata or {},
                }])
                return result

        self.index.upsert(vectors=[{
            "id": vector_id,
            "values": vector,
            "metadata": metadata or {},
        }])
        return result

    def upsert_vectors(
        self,
        records: list[dict[str, Any]],
        check_similarity: bool = True,
    ) -> list[dict[str, Any]]:
        """Записывает несколько векторов. records: [{id, values, metadata?}]"""
        return [
            self.upsert_vector(r["id"], r["values"], r.get("metadata"), check_similarity)
            for r in records
        ]

    def upsert_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[dict[str, Any]] = None,
        check_similarity: bool = True,
    ) -> dict[str, Any]:
        """Создаёт эмбеддинг для текста и записывает в базу."""
        vector = self.create_embedding(text)
        meta = {**(metadata or {}), "text": text}
        return self.upsert_vector(doc_id, vector, meta, check_similarity)

    def upsert_documents(
        self,
        docs: list[dict[str, Any]],
        check_similarity: bool = True,
    ) -> list[dict[str, Any]]:
        """Записывает несколько документов. docs: [{id, text, metadata?}]"""
        return [
            self.upsert_document(d["id"], d["text"], d.get("metadata"), check_similarity)
            for d in docs
        ]

    # ── Поиск ───────────────────────────────────────────────────────────────

    def query_by_vector(self, vector: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        """Поиск по готовому вектору."""
        response = self.index.query(vector=vector, top_k=top_k, include_metadata=True)
        return [
            {"id": m.id, "score": float(m.score), "metadata": m.metadata or {}}
            for m in response.matches
        ]

    def query_by_text(self, text: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Поиск по тексту — автоматически создаёт эмбеддинг."""
        return self.query_by_vector(self.create_embedding(text), top_k)

    # ── Чтение по ID ────────────────────────────────────────────────────────

    def fetch_vectors(self, ids: list[str]) -> dict[str, Any]:
        """Возвращает векторы по их идентификаторам."""
        return dict(self.index.fetch(ids=ids))

    # ── Удаление ────────────────────────────────────────────────────────────

    def delete(self, ids: list[str]) -> None:
        """Удаляет векторы по списку ID."""
        self.index.delete(ids=ids)

    def delete_by_filter(self, filter_expr: dict[str, Any]) -> None:
        """Удаляет векторы по фильтру метаданных."""
        self.index.delete(filter=filter_expr)

    def delete_all(self) -> None:
        """Очищает весь индекс. Безопасен когда namespace ещё пустой."""
        try:
            self.index.delete(delete_all=True)
        except Exception as exc:
            if "404" in str(exc) or "not found" in str(exc).lower():
                pass  # namespace ещё не создан — считается уже пустым
            else:
                raise

    # ── Статистика и обновление ──────────────────────────────────────────────

    def describe_index_stats(self) -> dict[str, Any]:
        """Возвращает статистику индекса."""
        resp = self.index.describe_index_stats()
        return resp.to_dict() if hasattr(resp, "to_dict") else dict(resp)

    def update_metadata(self, vector_id: str, metadata: dict[str, Any]) -> None:
        """Обновляет метаданные существующего вектора."""
        self.index.update(id=vector_id, set_metadata=metadata)


# ── Ручной тест модуля ───────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("=" * 55)
    print("Ручной тест PineconeManager")
    print("=" * 55)

    manager = PineconeManager()

    stats = manager.describe_index_stats()
    print(f"\nИндекс: {manager.index_name}")
    print(f"Статистика: {stats}")
    print(f"Порог сходства (SIMILARITY_THRESHOLD): {SIMILARITY_THRESHOLD}")

    # Очищаем перед тестом для чистого результата
    print("\nОчищаю индекс для чистого теста...")
    manager.delete_all()
    time.sleep(5)  # даём Pinecone время применить удаление

    test_phrases = [
        ("Привет", "Первое сообщение"),
        ("Зачем люди хотят лететь на Марс?", "Новая тема"),
        ("Я бы хотел полететь на Марс", "Вставится как новая запись"),
        ("Я бы не хотел полететь на Марс", "Дубликат — должен быть updated/skipped"),
        ("Я очень люблю пиццу", "Смена темы — должен быть inserted"),
    ]

    print("\n--- Тест записи с проверкой дубликатов ---")
    for phrase, comment in test_phrases:
        doc_id = f"test-{uuid.uuid4().hex[:8]}"
        result = manager.upsert_document(doc_id, phrase, {"source": "test"})
        marker = "[NEW]" if result["action"] == "inserted" else "[DUP]"
        score_str = f"{result['similarity_score']:.3f}" if result["similarity_score"] else "n/a"
        print(f"  {marker} action={result['action']:8s} score={score_str}  | {phrase}")
        print(f"     ({comment})")

    print("\n--- Тест поиска по тексту ---")
    hits = manager.query_by_text("путешествие в космос", top_k=3)
    for hit in hits:
        print(f"  score={hit['score']:.3f} | {hit['metadata'].get('text', '')[:70]}")

    print("\n[OK] Test complete.")
