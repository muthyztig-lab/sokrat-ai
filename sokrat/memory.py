"""Persistent per-learner memory.

Each student gets one JSON file. It's intentionally simple and human-readable —
a teacher can open it, and swapping it for a database later means changing only
this file. This is what lets the tutor pick up where it left off and adapt.
"""

from __future__ import annotations

from pathlib import Path

from .models import LearnerProfile, _now

LEARNERS_DIR = Path(".sokrat/learners")


class LearnerMemory:
    def __init__(self, learners_dir: str | Path = LEARNERS_DIR) -> None:
        self.dir = Path(learners_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, learner_id: str) -> Path:
        safe = "".join(c for c in learner_id if c.isalnum() or c in "-_").strip() or "anon"
        return self.dir / f"{safe}.json"

    def load(self, learner_id: str, name: str = "") -> LearnerProfile:
        path = self._path(learner_id)
        if path.exists():
            profile = LearnerProfile.model_validate_json(path.read_text("utf-8"))
            if name and not profile.name:
                profile.name = name
            return profile
        return LearnerProfile(learner_id=learner_id, name=name)

    def save(self, profile: LearnerProfile) -> None:
        profile.last_seen = _now()
        self._path(profile.learner_id).write_text(
            profile.model_dump_json(indent=2), encoding="utf-8"
        )

    def update(
        self,
        profile: LearnerProfile,
        *,
        mastered: list[str] | None = None,
        struggling: list[str] | None = None,
        note: str = "",
    ) -> LearnerProfile:
        """Merge new observations, de-duplicating and resolving mastered vs. struggling."""
        for topic in mastered or []:
            _add(profile.mastered_topics, topic)
            _remove(profile.struggling_topics, topic)  # promoted out of struggling
        for topic in struggling or []:
            if topic.lower() not in {t.lower() for t in profile.mastered_topics}:
                _add(profile.struggling_topics, topic)
        if note:
            profile.notes.append(note.strip())
        self.save(profile)
        return profile

    def end_session(self, profile: LearnerProfile) -> None:
        profile.sessions += 1
        self.save(profile)


def _add(items: list[str], value: str) -> None:
    value = value.strip()
    if value and value.lower() not in {i.lower() for i in items}:
        items.append(value)


def _remove(items: list[str], value: str) -> None:
    lowered = value.strip().lower()
    items[:] = [i for i in items if i.lower() != lowered]
