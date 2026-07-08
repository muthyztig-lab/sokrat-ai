"""Command-line interface: `sokrat ingest`, `sokrat chat`, `sokrat report`.

Run `sokrat --help` for everything.
"""

from __future__ import annotations

import os

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from . import __version__
from .agent import TutorAgent
from .llm import LLMClient
from .memory import LearnerMemory
from .report import build_report
from .retrieval import Retriever

app = typer.Typer(add_completion=False, help="Sokrat — a Socratic AI tutor agent.")
console = Console()


def _require_key() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    if not os.getenv("OPENAI_API_KEY"):
        console.print(
            "[bold red]OPENAI_API_KEY is not set.[/] "
            "Copy .env.example to .env and add your key (see README)."
        )
        raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"Sokrat {__version__}")


@app.command()
def ingest(
    path: str = typer.Argument(..., help="File or folder with course materials."),
) -> None:
    """Index course materials so the tutor can teach (and cite) from them."""
    _require_key()
    llm = LLMClient()
    console.print(f"[cyan]Ingesting[/] {path} …")
    retriever = Retriever.build(path, llm)
    retriever.save()
    console.print(
        f"[green]✓[/] Indexed [bold]{len(retriever.chunks)}[/] chunks. "
        f"[dim]{llm.usage.summary()}[/]"
    )


@app.command()
def chat(
    learner: str = typer.Option("student", "--learner", "-l", help="Learner id (for memory)."),
    name: str = typer.Option("", "--name", "-n", help="Learner display name."),
    course: str = typer.Option("this course", "--course", "-c", help="Course name."),
    language: str = typer.Option("Ukrainian", "--lang", help="Reply language."),
) -> None:
    """Start an interactive Socratic tutoring session (type /exit to finish)."""
    _require_key()
    llm = LLMClient()
    try:
        retriever = Retriever.load(llm)
    except FileNotFoundError as err:
        console.print(f"[red]{err}[/]")
        raise typer.Exit(code=1)

    memory = LearnerMemory()
    profile = memory.load(learner, name=name)
    agent = TutorAgent(
        course_name=course,
        retriever=retriever,
        memory=memory,
        profile=profile,
        llm=llm,
        language=language,
    )

    console.print(
        Panel.fit(
            f"[bold]Sokrat[/] — session with [cyan]{profile.name or learner}[/] "
            f"(session #{profile.sessions + 1})\n"
            "[dim]Type your question. Commands: /report, /profile, /exit[/]",
            border_style="cyan",
        )
    )

    while True:
        try:
            user = console.input("[bold green]you ›[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user in {"/exit", "/quit", "/q"}:
            break
        if user == "/profile":
            console.print(Panel(profile.as_context(), title="learner profile"))
            continue
        if user == "/report":
            _print_report(agent, llm, language)
            continue

        with console.status("[dim]thinking…[/]"):
            reply = agent.reply(user)
        console.print("[bold blue]sokrat ›[/]")
        console.print(Markdown(reply))

    # Wrap up: generate the teacher report and persist the session.
    if agent.transcript:
        console.print("\n[dim]Generating session report…[/]")
        _print_report(agent, llm, language)
    memory.end_session(profile)
    console.print(f"[dim]Session cost: {llm.usage.summary()}[/]")


def _print_report(agent: TutorAgent, llm: LLMClient, language: str) -> None:
    report = build_report(agent, llm, language)
    flag = "[red]⚠ needs a human[/]" if report.needs_human_attention else "[green]✓ on track[/]"
    body = (
        f"[bold]{report.summary}[/]\n\n"
        f"[cyan]Topics:[/] {', '.join(report.topics_covered) or '—'}\n"
        f"[green]Strengths:[/] {', '.join(report.strengths) or '—'}\n"
        f"[yellow]Struggles:[/] {', '.join(report.struggles) or '—'}\n"
        f"[magenta]Next steps:[/] {', '.join(report.recommended_next_steps) or '—'}\n"
    )
    if report.needs_human_attention and report.attention_reason:
        body += f"\n[red]Attention:[/] {report.attention_reason}"
    console.print(Panel(body, title=f"Session report — {report.learner}  {flag}", border_style="blue"))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
