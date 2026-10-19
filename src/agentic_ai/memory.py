"""Long-term semantic memory.

`ChromaMemory` persists to disk (the original used `chromadb.Client()`, which is
in-memory and forgot everything on exit). Embeddings come from Ollama so the
whole stack stays local; `InMemoryMemory` is a dependency-free store for tests.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text", host: str = "http://localhost:11434") -> None:
        import ollama

        self.model = model
        self._client = ollama.Client(host=host)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(v) for v in self._client.embed(model=self.model, input=texts).embeddings]


@dataclass
class MemoryHit:
    text: str
    score: float  # cosine similarity, higher is closer


class Memory(Protocol):
    def add(self, text: str, kind: str = "interaction") -> str: ...

    def search(self, query: str, k: int = 3) -> list[MemoryHit]: ...

    def count(self) -> int: ...


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class InMemoryMemory:
    def __init__(self, embedder: Embedder, min_score: float = 0.3) -> None:
        self.embedder = embedder
        self.min_score = min_score
        self._items: list[tuple[str, list[float]]] = []

    def add(self, text: str, kind: str = "interaction") -> str:
        self._items.append((text, self.embedder.embed([text])[0]))
        return str(len(self._items))

    def search(self, query: str, k: int = 3) -> list[MemoryHit]:
        if not self._items:
            return []
        q = self.embedder.embed([query])[0]
        hits = sorted((MemoryHit(t, _cosine(q, v)) for t, v in self._items), key=lambda h: h.score, reverse=True)
        return [h for h in hits[:k] if h.score >= self.min_score]

    def count(self) -> int:
        return len(self._items)


class ChromaMemory:
    def __init__(
        self, path: Path, embedder: Embedder, collection: str = "agent_memory", min_score: float = 0.3
    ) -> None:
        import chromadb

        self.embedder = embedder
        self.min_score = min_score
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._col = self._client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    def add(self, text: str, kind: str = "interaction") -> str:
        doc_id = uuid.uuid4().hex  # stable & unique (Python's hash() is randomised per process)
        self._col.add(
            ids=[doc_id],
            documents=[text],
            embeddings=self.embedder.embed([text]),
            metadatas=[{"kind": kind, "created_at": datetime.now(UTC).isoformat()}],
        )
        return doc_id

    def search(self, query: str, k: int = 3) -> list[MemoryHit]:
        n = self.count()
        if n == 0:
            return []
        res = self._col.query(query_embeddings=self.embedder.embed([query]), n_results=min(k, n))
        hits = [MemoryHit(doc, 1.0 - dist) for doc, dist in zip(res["documents"][0], res["distances"][0], strict=False)]
        return [h for h in hits if h.score >= self.min_score]

    def count(self) -> int:
        return self._col.count()
