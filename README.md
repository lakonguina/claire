# claire.pm

MCP connector + landing. Intégration pas à pas : voir [`TODO.md`](TODO.md).

```
app/
├── main.py
├── models.py          # people + events
├── store.py           # append_event (payload AES-GCM)
├── core/              # config, db, crypto, vocab
├── api/               # landing + health
├── mcp/               # Streamable HTTP (stub → tools next)
├── locales/
├── templates/
└── static/            # styles.css → app.css
migrations/            # Alembic
```

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
