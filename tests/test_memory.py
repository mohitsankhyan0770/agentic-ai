import pytest
from conftest import BagOfWordsEmbedder

from agentic_ai.memory import InMemoryMemory


def test_empty_search():
    assert InMemoryMemory(BagOfWordsEmbedder()).search("anything") == []


def test_ranking():
    mem = InMemoryMemory(BagOfWordsEmbedder(), min_score=0.0)
    mem.add("the user likes python programming")
    mem.add("the weather in palampur is cold")
    assert "python" in mem.search("python programming", k=1)[0].text


def test_chroma_persists_across_instances(tmp_path):
    pytest.importorskip("chromadb")
    from agentic_ai.memory import ChromaMemory

    ChromaMemory(tmp_path, BagOfWordsEmbedder()).add("user prefers dark mode")
    reopened = ChromaMemory(tmp_path, BagOfWordsEmbedder())
    assert reopened.count() == 1
    assert reopened.search("user prefers dark mode")[0].text == "user prefers dark mode"
    # Duplicate text must not collide (original used hash(text) as the id).
    reopened.add("user prefers dark mode")
    assert reopened.count() == 2
