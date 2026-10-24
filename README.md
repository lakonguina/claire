# Claire

MCP connector that learns how each user likes AI replies (shorter, formatted their way) and tells their assistant before every answer.

Live at [claire.pm](https://claire.pm).

## Goal

AI assistants often answer with the same long, generic format. Claire plugs into ChatGPT, Claude or Grok as an MCP server and adapts replies to each person over time, without an account or an email.

## How it works

1. Before replying, the assistant calls `claire_plan_reply`. Claire picks a **form** (text, bullets, ...) and a **word budget** for this user, then returns an instruction.
2. The assistant writes its answer and displays it with `claire_show_reply`, rendered in an MCP App widget.
3. If the user pushes back on the next message ("shorter", "use a list"), the assistant reports it as a forced field. That signal is the reward.
4. A per-user **Thompson sampling** bandit updates its preferences. A 3% holdout group gets the default policy, so impact can be measured.

Every decision is logged with its propensity, which lets a nightly job replay the log and compare policies offline.

## Features

- MCP server (Streamable HTTP) with OAuth 2.1 and an MCP App widget
- Event log in PostgreSQL, payloads encrypted with AES-GCM
- Nightly job: bandit rebuild, metrics vs holdout, funnel, alerts
- Admin dashboard with KPIs (words saved, correction rate, health)
- Landing, consent and privacy pages in French and English

## Stack

Python, FastAPI, MCP SDK, SQLAlchemy, Alembic, PostgreSQL, Jinja2, Vite (widget), pytest, Docker.

## Project layout

```
app/
├── mcp/        # MCP tools, turn loop, schemas
├── model/      # Thompson sampling, rewards, replay
├── jobs/       # nightly batch
├── admin/      # KPI dashboard
├── auth/       # OAuth 2.1
├── core/       # config, database, encryption, vocabulary
└── templates/  # landing, consent, privacy
widget/         # MCP App source, built into app/static/reply.html
migrations/     # Alembic
tests/
```

## Run locally

```bash
cp .env.example .env
docker compose up --build
```

- App: http://127.0.0.1:8000
- Health check: http://127.0.0.1:8000/health
- MCP Inspector: http://127.0.0.1:6274

Tests: `uv run pytest`
