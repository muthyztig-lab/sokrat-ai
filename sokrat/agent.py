"""The Socratic tutor agent — an OpenAI tool-calling loop with a strong pedagogy
prompt, grounded retrieval, and persistent learner memory.

The agent decides, each turn, whether to search the material, generate practice,
grade an answer, remember something about the learner, or escalate — then answers
the student. It teaches Socratically: it guides with questions and hints, and
only reveals a full answer after the student has genuinely tried.
"""

from __future__ import annotations

import json
from typing import Any

from .llm import LLMClient
from .memory import LearnerMemory
from .models import LearnerProfile
from .tools import Toolbox
from .retrieval import Retriever

SYSTEM_PROMPT = """\
You are Sokrat, an expert, warm AI tutor for the "{course}" course.
You reply in {language}.

## How you teach (non-negotiable)
- Teach Socratically. Do NOT hand over the full answer immediately. Ask a
  guiding question or give a hint, let the student think, and reveal the full
  answer only after they've genuinely attempted it (or explicitly give up after
  real effort).
- Ground everything in the course. Before explaining a concept, call
  `search_materials` and base your explanation on what it returns. If the
  material doesn't cover something, say so honestly instead of inventing.
- Be concise and encouraging. One idea at a time. Never overwhelm.

## Using your tools
- `search_materials`: before any substantive explanation.
- `make_practice`: when the student wants to practice or you want to check a skill.
- `check_answer`: to grade a free-text answer with a rubric before you react to it.
- `update_learner_memory`: whenever you observe a topic the student has mastered
  or is struggling with, so future sessions adapt. Do this proactively.
- `escalate_to_human`: for distress, account/billing issues, or repeated genuine
  confusion after real effort — then reassure the student a human will follow up.

## What you know about this learner
{learner_context}

Adapt difficulty to this profile: go lighter on struggling topics, and don't
re-teach what they've already mastered.
"""


class TutorAgent:
    def __init__(
        self,
        *,
        course_name: str,
        retriever: Retriever,
        memory: LearnerMemory,
        profile: LearnerProfile,
        llm: LLMClient,
        language: str = "Ukrainian",
        max_tool_steps: int = 6,
    ) -> None:
        self.profile = profile
        self.memory = memory
        self.llm = llm
        self.max_tool_steps = max_tool_steps
        self.toolbox = Toolbox(retriever, memory, profile, llm)
        self.transcript: list[dict[str, str]] = []

        system = SYSTEM_PROMPT.format(
            course=course_name,
            language=language,
            learner_context=profile.as_context(),
        )
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system}]

    @property
    def escalated(self) -> bool:
        return self.toolbox.escalated

    def reply(self, user_message: str) -> str:
        """Run one full turn (including any tool calls) and return the reply text."""
        self.messages.append({"role": "user", "content": user_message})
        self.transcript.append({"role": "student", "text": user_message})

        for _ in range(self.max_tool_steps):
            message = self.llm.chat(self.messages, tools=self.toolbox.schemas)
            self.messages.append(_assistant_entry(message))

            if not message.tool_calls:
                reply = message.content or ""
                self.transcript.append({"role": "tutor", "text": reply})
                return reply

            for call in message.tool_calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = self.toolbox.call(call.function.name, args)
                self.messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )

        fallback = "Давай зробимо паузу — сформулюй, будь ласка, питання ще раз."
        self.transcript.append({"role": "tutor", "text": fallback})
        return fallback

    def transcript_text(self) -> str:
        return "\n".join(f"{t['role'].upper()}: {t['text']}" for t in self.transcript)


def _assistant_entry(message) -> dict[str, Any]:
    """Convert an SDK assistant message into a plain dict safe to send back."""
    entry: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        entry["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in message.tool_calls
        ]
    return entry
