"""Pydantic schemas — the single source of truth for every structured LLM call.

We use OpenAI Structured Outputs (strict JSON schema) against these models, so
responses always have the shape we expect and we never hand-parse JSON.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

Difficulty = Literal["easy", "medium", "hard"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- Retrieval -------------------------------------------------------------


class Chunk(BaseModel):
    """A retrievable slice of the course material."""

    id: int
    source: str = Field(description="File the chunk came from.")
    text: str


class Citation(BaseModel):
    source: str
    quote: str


# --- Practice generation ---------------------------------------------------


class PracticeQuestion(BaseModel):
    """A single practice question grounded in the course material."""

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


# --- Answer grading --------------------------------------------------------


class GradingResult(BaseModel):
    """Rubric-based grading of a student's free-text answer."""

    score: int = Field(ge=0, le=100)
    verdict: Literal["correct", "partially_correct", "incorrect"]
    strengths: list[str]
    gaps: list[str] = Field(description="Important things the answer missed or got wrong.")
    feedback: str = Field(description="Warm, constructive, student-facing feedback.")


# --- Learner memory --------------------------------------------------------


class LearnerProfile(BaseModel):
    """Everything the tutor remembers about one student, persisted between sessions."""

    learner_id: str
    name: str = ""
    mastered_topics: list[str] = Field(default_factory=list)
    struggling_topics: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    sessions: int = 0
    created_at: str = Field(default_factory=_now)
    last_seen: str = Field(default_factory=_now)

    def as_context(self) -> str:
        """A compact, human-readable snapshot the agent can read at the top of a session."""
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


# --- Teacher-facing session report ----------------------------------------


class SessionReport(BaseModel):
    """Auto-generated summary a teacher can skim in 20 seconds."""

    learner: str
    summary: str = Field(description="What happened this session, in 2-4 sentences.")
    topics_covered: list[str]
    strengths: list[str]
    struggles: list[str]
    recommended_next_steps: list[str]
    needs_human_attention: bool
    attention_reason: str = Field(default="", description="Why a human should step in, if so.")
