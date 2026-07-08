from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

Difficulty = Literal["easy", "medium", "hard"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Chunk(BaseModel):
    id: int
    source: str = Field(description="File the chunk came from.")
    text: str


class Citation(BaseModel):
    source: str
    quote: str


class PracticeQuestion(BaseModel):
    question: str
    options: list[str] = Field(description="Exactly 4 options for a multiple-choice item.")
    correct_index: int = Field(ge=0, le=3, description="0-based index of the correct option.")
    explanation: str
    difficulty: Difficulty
    source_quote: str = Field(
        description="Verbatim quote from the material that supports the correct answer."
    )


class PracticeSet(BaseModel):
    topic: str
    questions: list[PracticeQuestion]


class GradingResult(BaseModel):
    score: int = Field(ge=0, le=100)
    verdict: Literal["correct", "partially_correct", "incorrect"]
    strengths: list[str]
    gaps: list[str] = Field(description="Important things the answer missed or got wrong.")
    feedback: str = Field(description="Warm, constructive, student-facing feedback.")


class LearnerProfile(BaseModel):
    learner_id: str
    name: str = ""
    mastered_topics: list[str] = Field(default_factory=list)
    struggling_topics: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    sessions: int = 0
    created_at: str = Field(default_factory=_now)
    last_seen: str = Field(default_factory=_now)

    def as_context(self) -> str:
        if not (self.mastered_topics or self.struggling_topics or self.notes):
            return "This is a new learner — no history yet."
        parts = [f"Learner: {self.name or self.learner_id} (session #{self.sessions + 1})"]
        if self.mastered_topics:
            parts.append("Already comfortable with: " + ", ".join(self.mastered_topics))
        if self.struggling_topics:
            parts.append("Still struggling with: " + ", ".join(self.struggling_topics))
        if self.notes:
            parts.append("Notes: " + " | ".join(self.notes[-5:]))
        return "\n".join(parts)


class SessionReport(BaseModel):
    learner: str
    summary: str = Field(description="What happened this session, in 2-4 sentences.")
    topics_covered: list[str]
    strengths: list[str]
    struggles: list[str]
    recommended_next_steps: list[str]
    needs_human_attention: bool
    attention_reason: str = Field(default="", description="Why a human should step in, if so.")
