# claire.pm

MCP connector + landing. Intégration pas à pas : voir [`TODO.md`](TODO.md).

```
app/
├── main.py
├── models.py          # people + events
├── store.py           # append_event (payload AES-GCM)
├── core/              # config, db, crypto, vocab
├── api/               # landing + health
├── mcp/               # Streamable HTTP + MCP Apps widget
├── locales/
├── templates/
└── static/            # app.css + reply.html (widget bundle)
widget/                # MCP App source → built into app/static/reply.html
migrations/            # Alembic
```

## Widget updates

Deploy rebuilds `widget/` into `reply.html`. The MCP resource URI is
`ui://claire/reply-<sha12>.html`, so a new bundle gets a new address on
`tools/list`. Hosts refresh after `CACHE_TTL_MS` (default 1h) or on
reconnect; any old `ui://claire/*` URI still serves the current HTML.

Local widget rebuild: `cd widget && npm ci && npm run build`.

## Run

```bash
cp .env.example .env   # if needed
docker compose up --build
# or: docker compose up -d db && ./dev
```

- App: http://127.0.0.1:8000  
- Health: http://127.0.0.1:8000/health  
- Inspector: http://127.0.0.1:6274  

Langue landing : `Accept-Language`, ou `/?lang=fr` · `/?lang=en`
