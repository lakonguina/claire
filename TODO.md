# Claire — intégration (pas à pas)

Principe: **libs d’abord**, peu de code maison. On coche une étape à la fois.

Style: **Google Python Style Guide** + docstrings (Args/Returns/Raises) — tech
mais lisibles pour un non-dev. Pas de param nommé `type` (builtin).

- [x] **1. Fondations** — `pydantic-settings`, Postgres (`sqlalchemy`+`alembic`), `cryptography` AES-GCM, `app/core/vocab.py`, table `events`
- [x] **2. Squelette MCP** — `claire_plan_reply` + `claire_show_reply` (schemas Pydantic / MCP), stubs
- [x] **3. Auth** — OAuth 2.1 / bearer sur `/mcp` ; à chaque requête: `touch_person` + `meta.request` + event `token_refresh` au connect
- [x] **4. Boucle turn** — create conv, log (`plan_call`/`decision` avec `meta.request`), close turn, holdout 3 % ; message entier si ≤ 150 mots
- [x] **5. Decision** — Thompson form + buckets words **par person**, log propensity
- [x] **6. Rewards** — `form_forced` / `length_forced` / `words_forced` du plan suivant → update posteriors au `turn_closed`
- [x] **7. Widget `show_reply`** — MCP App + parse markdown (`markdown-it`) + fallback
- [x] **8. Widget events** — `reply_view` / `render_error` via `claire_record_event`
- [x] **9. Nightly** — `python -m app.jobs.nightly`
  - [x] **9a.** replay = online
  - [x] **9b.** métriques vs holdout
  - [x] **9c.** funnel
  - [x] **9d.** alertes champs
- [x] **10. Landing** — instruction `claire_plan_reply` / `claire_show_reply`
- [x] **11. Tests** — `pytest` + `pytest-asyncio` (`tests/`, 18 passed)

**MVP vocab:** pas de `Content` / `Device` / `PastedKind` / `ToolUsed`. UA stocké à la connexion (`token_refresh`), pas comme clé de décision.

**Suivant:** minimisation (si tu veux).
