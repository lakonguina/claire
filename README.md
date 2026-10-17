# Lingua Franca

MCP server that makes chat assistants give shorter, better-targeted replies and learns each user's preferences. See `SPEC.md`.

## Run

```sh
cp .env.example .env          # set PUBLIC_URL, ADMIN_TOKEN, POSTGRES_PASSWORD
docker compose up -d --build
docker compose exec app python -m app.admin new-user   # prints the connector URL
```

The app listens on `127.0.0.1:8000`. Put a TLS reverse proxy in front of it, and make the proxy's access log redact `/mcp/<token>`. Add the printed URL as a custom connector on claude.ai (or ChatGPT), then paste the account instruction from `SPEC.md` section 10.

Admin endpoints take `Authorization: Bearer $ADMIN_TOKEN` (or `?key=` for the browser):

- `GET /debug`: recent turns, with a form to add gold labels
- `POST /gold_label`: `{"turn_id": "t_31", "outcome": "moved_on"}`
- `GET /metrics`

The `nightly` service runs `python -m app.nightly` once a day: it refits the noise models and hyperparameters, then rebuilds all derived state by replaying the events.

## Develop

```sh
uv venv --python 3.12 .venv && uv pip install -p .venv -r requirements-dev.txt
docker run -d --name lf-test-pg -e POSTGRES_USER=lf -e POSTGRES_PASSWORD=lf -e POSTGRES_DB=lf -p 55432:5432 postgres:17-alpine
.venv/bin/python -m pytest
(cd widget && npm ci && npm run build)   # writes app/static/index.html
```

The system tests drop and recreate the `public` schema of the test database.
