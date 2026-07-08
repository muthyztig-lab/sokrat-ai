"""Turn a finished tutoring session into a teacher-facing report.

This is the automation payoff: a curator can skim one card per student instead
of reading thousands of transcripts, and instantly see who needs a human.
"""

from __future__ import annotations

from .agent import TutorAgent
from .llm import LLMClient
from .models import SessionReport


def build_report(agent: TutorAgent, llm: LLMClient, language: str = "Ukrainian") -> SessionReport:
    report = llm.parse(
        system=(
            "You are an assistant to a teacher. Read the tutoring transcript and "
            "produce a crisp session report. Be concrete about what the student "
            f"understood and where they struggled. Write text fields in {language}."
        ),
        user=(
            f"LEARNER: {agent.profile.name or agent.profile.learner_id}\n"
            f"ESCALATED TO HUMAN: {agent.escalated}\n\n"
            f"TRANSCRIPT:\n{agent.transcript_text()}"
        ),
        schema=SessionReport,
        temperature=0.2,
    )
    if agent.escalated:
        report.needs_human_attention = True
    return report
