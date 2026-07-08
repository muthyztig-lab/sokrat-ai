"""Наочне демо БЕЗ ключа OpenAI і без інтернету.

Логіка агента — справжня (той самий цикл, памʼять, звіт). «Несправжній» тут лише
сам AI: замість реальної моделі підставлені заготовлені відповіді, щоб було видно
весь потік роботи.

Запуск:  python examples/demo_offline.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sokrat.agent import TutorAgent
from sokrat.ingest import ingest_path
from sokrat.memory import LearnerMemory
from sokrat.models import SessionReport
from sokrat.retrieval import Retriever, _normalize

import numpy as np

console = Console()


def _msg(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _call(cid, name, args):
    return SimpleNamespace(id=cid, function=SimpleNamespace(name=name, arguments=args))


class DemoLLM:
    """Підставний AI: заготовлені ходи розмови + детерміновані «ембединги»."""

    def __init__(self):
        self.usage = SimpleNamespace(summary=lambda: "демо-режим (0 витрат)")
        self._chat_turns = [
            # Студент: «поясни складні відсотки»
            _call_turn := _msg(tool_calls=[_call("c1", "search_materials",
                                                  '{"query": "складні відсотки"}')]),
            _msg(content="Гарне питання! Спершу подумай сам 🤔 — чому, на твою думку, "
                         "вклад росте з часом **усе швидше**, а не рівномірно?"),
            # Студент відповідає правильно
            _msg(tool_calls=[_call("c2", "update_learner_memory",
                                   '{"mastered": ["складні відсотки"], '
                                   '"note": "сам вивів ідею нарахування на відсотки"}')]),
            _msg(content="Саме так! ✅ Відсотки нараховуються і на вже накопичені відсотки — "
                         "тому й ефект «сніжного кому». Ти це вхопив, рухаємось далі?"),
        ]

    def chat(self, messages, tools=None, temperature=0.4):
        return self._chat_turns.pop(0)

    def embed(self, texts):
        dim = 64
        out = []
        for t in texts:
            v = [0.0] * dim
            for word in t.lower().split():
                v[hash(word) % dim] += 1.0
            out.append(v)
        return out

    def parse(self, *, system, user, schema, temperature=0.3):
        # У цьому демо parse потрібен лише для фінального звіту.
        if schema is SessionReport:
            return SessionReport(
                learner="Олена",
                summary="Студентка розібралася зі складними відсотками, сама вивела ідею "
                        "нарахування відсотків на відсотки.",
                topics_covered=["складні відсотки"],
                strengths=["самостійно дійшла до суті"],
                struggles=[],
                recommended_next_steps=["перейти до теми «подушка безпеки»"],
                needs_human_attention=False,
            )
        raise NotImplementedError(schema)


def run_turn(agent: TutorAgent, text: str) -> None:
    console.print(f"\n[bold green]СТУДЕНТ ›[/] {text}")
    start = len(agent.messages)
    reply = agent.reply(text)
    for m in agent.messages[start:]:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                console.print(
                    f"   [dim]🔧 агент сам вирішив викликати інструмент →[/] "
                    f"[cyan]{tc['function']['name']}[/] [dim]{tc['function']['arguments']}[/]"
                )
    console.print("[bold blue]РЕПЕТИТОР ›[/]")
    console.print(Markdown(reply))


def main() -> None:
    console.print(Rule("[bold]Sokrat — демо (несправжній AI, справжня логіка)[/]"))

    llm = DemoLLM()

    # 1. «Даємо підручник»: читаємо приклад курсу і будуємо памʼять для пошуку.
    course = Path(__file__).resolve().parent / "course"
    chunks = ingest_path(course)
    console.print(f"[green]1.[/] Прочитано матеріали курсу → [bold]{len(chunks)}[/] шматків тексту.")
    vectors = np.array(llm.embed([c.text for c in chunks]), dtype=np.float32)
    retriever = Retriever(chunks, _normalize(vectors), llm)

    # 2. Заводимо учня з памʼяттю.
    memory = LearnerMemory(".sokrat/learners")
    profile = memory.load("olena_demo", name="Олена")
    console.print(f"[green]2.[/] Завантажено профіль учня: [cyan]{profile.name}[/] "
                  f"(сесія #{profile.sessions + 1}).")

    agent = TutorAgent(
        course_name="Фінансова грамотність",
        retriever=retriever, memory=memory, profile=profile, llm=llm,
    )
    console.print("[green]3.[/] Агент готовий. Починається діалог:\n")
    console.print(Rule())

    run_turn(agent, "поясни складні відсотки")
    run_turn(agent, "бо відсотки додаються до суми, і на них теж потім капають відсотки")

    # 3. Що агент запамʼятав про учня.
    console.print(Rule("[bold]Памʼять про учня (збережеться між сесіями)[/]"))
    console.print(Panel(profile.as_context(), border_style="magenta"))

    # 4. Авто-звіт для викладача.
    from sokrat.report import build_report

    report = build_report(agent, llm)
    console.print(Rule("[bold]Авто-звіт викладачу[/]"))
    console.print(Panel(
        f"[bold]{report.summary}[/]\n\n"
        f"Теми: {', '.join(report.topics_covered)}\n"
        f"Сильні сторони: {', '.join(report.strengths)}\n"
        f"Наступний крок: {', '.join(report.recommended_next_steps)}\n"
        f"Потрібен викладач: {'ТАК' if report.needs_human_attention else 'ні'}",
        title=f"Звіт — {report.learner}", border_style="blue",
    ))
    console.print("\n[dim]Це був демо-режим. З реальним ключем OpenAI відповіді генерує "
                  "модель, а не заготовки.[/]")


if __name__ == "__main__":
    main()
