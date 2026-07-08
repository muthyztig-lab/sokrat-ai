# 🦉 Sokrat — a Socratic AI tutor **agent**

> An open-source, self-hostable AI tutor **agent** that teaches from *your own*
> course materials, **remembers every learner**, guides with questions instead of
> handing out answers, and writes the teacher a report — escalating to a human
> when a student needs one.

Most "AI tutors" are a single prompt wrapped around ChatGPT. Sokrat is different
in the ways that matter for a real online school:

- **It's an agent, not a chatbot.** Every turn it *decides* whether to search the
  material, generate practice, grade an answer, update what it knows about the
  student, or escalate to a human — using OpenAI tool-calling.
- **It's grounded.** Answers come from *your* materials via retrieval (RAG) with
  citations. If the course doesn't cover something, it says so instead of
  inventing — critical when you're teaching thousands of students.
- **It remembers.** Each learner has a persistent profile (mastered vs. struggling
  topics, notes). Sessions build on each other and difficulty adapts.
- **It's Socratic.** It nudges and asks guiding questions, and only reveals the
  full answer after the student genuinely tries — the pedagogy schools actually
  want, not answer-vending.
- **It closes the loop for teachers.** Every session becomes a short report, and
  hard cases are flagged for a human. That's the automation payoff: one card per
  student instead of thousands of transcripts.

Built with 2026 best practices: **OpenAI Structured Outputs + Pydantic (strict
mode)** for every structured call, a swappable local vector store (no external DB
needed), an on-disk cache, retries, and full token/cost accounting.

---

## Architecture

```
                                  ┌────────────────────────────┐
   course files (.md/.txt/.pdf) ─▶│  ingest → chunk → embed     │─▶ .sokrat/index
                                  └────────────────────────────┘
                                                │  (grounded retrieval, cited)
                                                ▼
 student ──▶  ┌─────────────────────────────────────────────────────────┐
              │                 TutorAgent  (tool-calling loop)          │
              │  Socratic system prompt + per-learner memory context     │
              │                                                          │
              │   tools ▶ search_materials · make_practice · check_answer │
              │           update_learner_memory · escalate_to_human       │
              └─────────────────────────────────────────────────────────┘
                    │                     │                      │
                    ▼                     ▼                      ▼
          .sokrat/learners/*.json   session report        .sokrat/escalations
          (persistent memory)       (for the teacher)     (human-in-the-loop)
```

| Module | Responsibility |
|---|---|
| `models.py` | Pydantic schemas — the contract for every structured LLM call |
| `llm.py` | OpenAI wrapper: structured outputs, embeddings, tool-calling, cache, retries, cost |
| `ingest.py` | Load & chunk `.md/.txt/.pdf` course materials |
| `retrieval.py` | Local embedding vector store + cosine search with citations (RAG) |
| `memory.py` | Persistent per-learner profiles |
| `tools.py` | The agent's tools (function-calling specs + implementations) |
| `agent.py` | The Socratic tool-calling agent loop |
| `report.py` | Auto-generated teacher session report |
| `cli.py` | `sokrat ingest / chat` |

Every seam is swappable: point `retrieval.py` at pgvector/Qdrant, or `memory.py`
at Postgres, and nothing else changes. That's the "scale to thousands" path.

---

## Quickstart

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[pdf]"

# 2. Add your key
cp .env.example .env        # then paste your OPENAI_API_KEY

# 3. Index a course (a sample one ships in examples/)
sokrat ingest examples/course

# 4. Teach
sokrat chat --learner olena --name "Олена" --course "Фінансова грамотність"
```

Then just talk to it:

```
you › поясни складні відсотки
sokrat › Перш ніж пояснювати — як гадаєш, чому вклад росте
        швидше з часом, а не рівномірно? 🤔
you › не знаю
sokrat › Ок, дивись: відсотки нараховуються не лише на початкову суму,
        а й на вже накоплені відсотки… [цитує матеріал курсу]
```

Type `/report` any time to see the teacher report, `/profile` to see what Sokrat
remembers, `/exit` to finish (a report is generated automatically on exit).

---

## Why Socratic + grounded + memory?

- **Socratic** raises actual learning vs. answer-vending (students who are handed
  answers don't retain them).
- **Grounded** means it teaches *your* curriculum, not the open internet — and
  won't hallucinate facts to a paying student.
- **Memory** turns one-off Q&A into a relationship: it stops re-explaining what a
  student already knows and leans in where they struggle.

Together they take a tutor from "cute demo" to "something you can put in front of
thousands of learners and a curator can trust."

---

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | required |
| `SOKRAT_MODEL` | `gpt-4o-mini` | chat/agent model (set to whatever you have access to) |
| `SOKRAT_EMBED_MODEL` | `text-embedding-3-small` | embedding model for retrieval |
| `SOKRAT_PRICE_IN/OUT` | `0.15 / 0.60` | $/1M tokens, used only for the cost read-out |

## Tests

```bash
pip install -e ".[dev]"
pytest            # offline — no API key needed
```

## Roadmap

- [ ] Web/Telegram front-ends over the same agent core
- [ ] pgvector / Qdrant retrieval backends
- [ ] Voice sessions
- [ ] Eval harness (LLM-as-judge) over generated practice quality

## License

MIT — see [LICENSE](LICENSE).
