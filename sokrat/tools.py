"""The tools the agent can call.

This is what makes Sokrat an *agent* rather than a chatbot: the model decides,
turn by turn, whether to search the material, generate practice, grade an
answer, update what it remembers about the learner, or escalate to a human — and
it sees the result before it replies.

`Toolbox` owns the wiring; `schemas` is the OpenAI function-calling spec and
`call()` dispatches a single tool call and returns a string for the model.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .llm import LLMClient
from .memory import LearnerMemory
from .models import GradingResult, LearnerProfile, PracticeSet
from .retrieval import Retriever

ESCALATIONS_LOG = Path(".sokrat/escalations.jsonl")


class Toolbox:
    def __init__(
        self,
        retriever: Retriever,
        memory: LearnerMemory,
        profile: LearnerProfile,
        llm: LLMClient,
    ) -> None:
        self.retriever = retriever
        self.memory = memory
        self.profile = profile
        self.llm = llm
        self.escalated = False

    # -- OpenAI tool specs ------------------------------------------------
    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [
            _fn(
                "search_materials",
                "Search the course materials for passages relevant to a query. "
                "ALWAYS use this before explaining a concept, so your answer is "
                "grounded in the real course and you can cite the source.",
                {
                    "query": {"type": "string", "description": "What to look up."},
                    "k": {"type": "integer", "description": "How many passages (default 4)."},
                },
                ["query"],
            ),
            _fn(
                "make_practice",
                "Generate grounded practice questions on a topic to check or build "
                "the learner's understanding.",
                {
                    "topic": {"type": "string"},
                    "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                    "n": {"type": "integer", "description": "Number of questions (1-5)."},
                },
                ["topic", "difficulty"],
            ),
            _fn(
                "check_answer",
                "Grade a learner's free-text answer against the course material with "
                "a rubric, returning score, strengths, gaps and feedback.",
                {
                    "question": {"type": "string"},
                    "student_answer": {"type": "string"},
                },
                ["question", "student_answer"],
            ),
            _fn(
                "update_learner_memory",
                "Persist what you learned about this student so future sessions adapt. "
                "Call this when you notice a topic they've mastered or are struggling with.",
                {
                    "mastered": {"type": "array", "items": {"type": "string"}},
                    "struggling": {"type": "array", "items": {"type": "string"}},
                    "note": {"type": "string", "description": "A short observation."},
                },
                [],
            ),
            _fn(
                "escalate_to_human",
                "Flag this conversation for a human teacher/curator. Use for distress, "
                "billing/account issues, repeated confusion after real effort, or anything "
                "outside the course scope.",
                {
                    "reason": {"type": "string"},
                    "urgency": {"type": "string", "enum": ["low", "normal", "high"]},
                },
                ["reason"],
            ),
        ]

    # -- dispatch ---------------------------------------------------------
    def call(self, name: str, args: dict[str, Any]) -> str:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return f"ERROR: unknown tool '{name}'."
        try:
            return handler(**args)
        except TypeError as err:
            return f"ERROR calling {name}: {err}"

    # -- implementations --------------------------------------------------
    def _tool_search_materials(self, query: str, k: int = 4) -> str:
        hits = self.retriever.search(query, k=k)
        if not hits:
            return "No relevant passages found in the course materials."
        lines = []
        for chunk, score in hits:
            lines.append(f"[{chunk.source} · relevance {score:.2f}]\n{chunk.text.strip()}")
        return "\n\n---\n\n".join(lines)

    def _tool_make_practice(self, topic: str, difficulty: str, n: int = 3) -> str:
        n = max(1, min(int(n), 5))
        context = self._tool_search_materials(topic, k=4)
        result = self.llm.parse(
            system=(
                "You are an expert item writer. Create grounded multiple-choice "
                "practice questions using ONLY the provided course context. Each "
                "question must have exactly 4 options, one correct, and a verbatim "
                "source_quote drawn from the context."
            ),
            user=(
                f"Topic: {topic}\nDifficulty: {difficulty}\nNumber of questions: {n}\n\n"
                f"COURSE CONTEXT:\n{context}"
            ),
            schema=PracticeSet,
            temperature=0.4,
        )
        return result.model_dump_json(indent=2)

    def _tool_check_answer(self, question: str, student_answer: str) -> str:
        context = self._tool_search_materials(question, k=4)
        result = self.llm.parse(
            system=(
                "You are a fair, encouraging grader. Grade the student's answer "
                "against the course context using a rubric. Be specific about gaps "
                "but never harsh."
            ),
            user=(
                f"QUESTION:\n{question}\n\nSTUDENT ANSWER:\n{student_answer}\n\n"
                f"COURSE CONTEXT (ground truth):\n{context}"
            ),
            schema=GradingResult,
            temperature=0.2,
        )
        return result.model_dump_json(indent=2)

    def _tool_update_learner_memory(
        self,
        mastered: list[str] | None = None,
        struggling: list[str] | None = None,
        note: str = "",
    ) -> str:
        self.memory.update(
            self.profile, mastered=mastered, struggling=struggling, note=note
        )
        return (
            "Memory updated. "
            f"Mastered: {self.profile.mastered_topics}. "
            f"Struggling: {self.profile.struggling_topics}."
        )

    def _tool_escalate_to_human(self, reason: str, urgency: str = "normal") -> str:
        self.escalated = True
        ESCALATIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with ESCALATIONS_LOG.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "learner": self.profile.learner_id,
                        "reason": reason,
                        "urgency": urgency,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        return (
            "Escalation recorded — a human curator will follow up. "
            "Reassure the student that a person will reach out."
        )


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }
