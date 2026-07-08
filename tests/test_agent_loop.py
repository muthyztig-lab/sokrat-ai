"""Verify the agent's tool-calling loop end-to-end with a fake LLM (no network).

This proves the wiring: the agent issues a tool call, the tool runs, the result
is fed back, and the agent produces a final grounded answer.
"""

from types import SimpleNamespace

import numpy as np

from sokrat.agent import TutorAgent
from sokrat.memory import LearnerMemory
from sokrat.models import Chunk
from sokrat.retrieval import Retriever, _normalize


def _msg(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=arguments)
    )


class ScriptedLLM:
    """Returns a pre-scripted sequence of chat turns; embeds deterministically."""

    def __init__(self, turns):
        self._turns = list(turns)
        self.usage = SimpleNamespace(summary=lambda: "fake")

    def chat(self, messages, tools=None, temperature=0.4):
        return self._turns.pop(0)

    def embed(self, texts):
        return [[float(len(t)), 1.0] for t in texts]


def test_agent_calls_tool_then_answers(tmp_path):
    chunks = [Chunk(id=0, source="finance.md", text="Складні відсотки нараховуються на відсотки.")]
    vectors = np.array([[1.0, 1.0]], dtype=np.float32)

    llm = ScriptedLLM(
        turns=[
            # 1st turn: the agent decides to search the materials
            _msg(tool_calls=[_tool_call("c1", "search_materials", '{"query": "складні відсотки"}')]),
            # 2nd turn: with the tool result in context, it answers
            _msg(content="Добре питання! Спробуй сам: чому вклад росте швидше з часом?"),
        ]
    )
    retriever = Retriever(chunks, _normalize(vectors), llm)  # type: ignore[arg-type]
    memory = LearnerMemory(tmp_path / "learners")
    profile = memory.load("stud")

    agent = TutorAgent(
        course_name="Фінанси",
        retriever=retriever,
        memory=memory,
        profile=profile,
        llm=llm,  # type: ignore[arg-type]
    )

    reply = agent.reply("поясни складні відсотки")

    assert "Спробуй сам" in reply
    # transcript recorded both sides; tool round-trip happened (3+ assistant/tool msgs)
    assert agent.transcript[0]["role"] == "student"
    assert agent.transcript[-1]["role"] == "tutor"
    assert any(m.get("role") == "tool" for m in agent.messages)
