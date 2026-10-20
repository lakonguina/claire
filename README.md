# Claire

MCP server that makes chat assistants give shorter, better-targeted replies and learns each user's preferences. See `SPEC.md`.

## Run

```sh
cp .env.example .env          # set PUBLIC_URL, ADMIN_TOKEN, POSTGRES_PASSWORD
docker compose up -d --build
```

The app listens on `127.0.0.1:8000`. Put a TLS reverse proxy in front of it. Anyone adds `$PUBLIC_URL/mcp` as a custom connector on claude.ai (or ChatGPT) and clicks **Continue**: no account, each connector gets its own anonymous profile (OAuth, `SPEC.md` section 4.7). Then paste the account instruction from `SPEC.md` section 10.

Admin endpoints take `Authorization: Bearer $ADMIN_TOKEN` (the dashboard asks for it):

- `GET /admin`: the dashboard (live turns, users, learned model, metrics, gold labels)
- `POST /gold_label`: `{"turn_id": "t_31", "outcome": "moved_on"}`
- `GET /metrics`

The `nightly` service runs `python -m app.jobs.nightly` once a day: it refits the noise models, the hyperparameters and the topic clusters, then rebuilds all derived state by replaying the events.

The layout is in `TREE.md`. Texts live in `locales/<language>.json` (English is complete and is the fallback); identifiers live in `app/core/vocab.py`; the schema lives in `app/models.py`, and the app applies the Alembic migrations at start.

## Develop

```sh
uv venv --python 3.12 .venv && uv pip install -p .venv -r requirements-dev.txt
docker run -d --name claire-test-pg -e POSTGRES_USER=claire -e POSTGRES_PASSWORD=claire -e POSTGRES_DB=claire -p 55432:5432 pgvector/pgvector:pg17
.venv/bin/python -m pytest
.venv/bin/pyink app tests migrations && .venv/bin/pylint --rcfile=.pylintrc app tests migrations
.venv/bin/alembic revision --autogenerate -m "what changed"   # after editing app/models.py
(cd widget && npm ci && npm run format && npm run build)   # writes app/static/index.html
```

The system tests drop and recreate the `public` schema of the test database.
The first run downloads the topic embedding model (about 220 MB); the Docker image bakes it in.
