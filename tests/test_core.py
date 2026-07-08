import numpy as np

from sokrat.ingest import chunk_text
from sokrat.memory import LearnerMemory
from sokrat.models import Chunk
from sokrat.retrieval import Retriever


def test_chunk_text_respects_size_and_covers_all_text():
    text = "\n\n".join(f"Параграф номер {i} з якимось змістом." * 5 for i in range(20))
    chunks = chunk_text(text, target_chars=400, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 400 * 1.5 for c in chunks)
    joined = "\n".join(chunks)
    assert "Параграф номер 0" in joined and "Параграф номер 19" in joined


def test_memory_promotes_topic_out_of_struggling(tmp_path):
    mem = LearnerMemory(tmp_path / "learners")
    profile = mem.load("s1", name="Оля")
    mem.update(profile, struggling=["складні відсотки"])
    assert "складні відсотки" in profile.struggling_topics

    mem.update(profile, mastered=["складні відсотки"])
    assert "складні відсотки" in profile.mastered_topics
    assert "складні відсотки" not in profile.struggling_topics


def test_memory_deduplicates_case_insensitively(tmp_path):
    mem = LearnerMemory(tmp_path / "learners")
    profile = mem.load("s2")
    mem.update(profile, mastered=["Бюджет", "бюджет", "БЮДЖЕТ"])
    assert profile.mastered_topics == ["Бюджет"]


def test_memory_persists_between_loads(tmp_path):
    mem = LearnerMemory(tmp_path / "learners")
    p = mem.load("s3", name="Іван")
    mem.update(p, mastered=["диверсифікація"], note="швидко схопив тему")
    reloaded = mem.load("s3")
    assert reloaded.mastered_topics == ["диверсифікація"]
    assert reloaded.notes == ["швидко схопив тему"]


class _FakeLLM:
    _vocab = {"борг": 0, "відсоток": 1, "бюджет": 2, "ризик": 3}

    def embed(self, texts):
        vecs = []
        for t in texts:
            v = np.zeros(len(self._vocab), dtype=np.float32)
            for word, idx in self._vocab.items():
                v[idx] = float(t.lower().count(word))
            vecs.append(v.tolist())
        return vecs


def test_retriever_ranks_relevant_chunk_first():
    llm = _FakeLLM()
    chunks = [
        Chunk(id=0, source="a.md", text="Бюджет і планування витрат бюджет бюджет"),
        Chunk(id=1, source="b.md", text="Поганий борг має високий відсоток борг борг"),
        Chunk(id=2, source="c.md", text="Ризик і диверсифікація ризик ризик"),
    ]
    vectors = np.array(llm.embed([c.text for c in chunks]), dtype=np.float32)
    from sokrat.retrieval import _normalize

    retriever = Retriever(chunks, _normalize(vectors), llm)
    top_chunk, score = retriever.search("як гасити борг з високим відсотком", k=1)[0]
    assert top_chunk.id == 1
    assert score > 0
