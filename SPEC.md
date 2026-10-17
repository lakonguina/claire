# Lingua Franca — the communication layer between each person and any AI (MVP)

## 1. Purpose

Lingua Franca is **the communication layer between each person and any AI**: every assistant (Claude, ChatGPT, …, on web, desktop and mobile) talks to the user the way that user wants, in length, format, level and tone, from the first message and on every platform. It learns each user's preferences from their reactions and carries one profile across all of them.

Length is one dimension among others. Guiding principle: Grice's maxim of quantity — say everything that is needed, and nothing more — extended to manner: say it in the form this person takes in best.

**North-star metric: words spent per step forward**, alongside the **total repair rate**. We minimise reply words, including the words the tool itself costs (section 11), *under the constraint* that the user moves forward (no corrections, no missing information, no reformatting requests).

- Neither "fewest tokens" alone (an empty reply is shortest) nor "user satisfaction" alone (flattery, padding) is the goal.
- Button clicks are a **learning signal only**. Never optimise for click-through or engagement.

## 2. How it works

The server never stores the conversation. It receives the structured fields AI sends in tool calls and, in widget mode, the reply text, which it counts and discards at once. AI does the understanding; the server does memory and learning.

An instruction in the user's AI account (section 10) makes AI:

1. Call `plan_reply` **before** every reply. AI sends (a) its verdict on how the user reacted to its previous reply, with a short quote as proof, and (b) facts about the new request.
2. The server records the outcome for the previous turn, updates the user's model instantly, picks a target length and a form (format, level, examples, tone, headings, or a clarifying question first), and returns a short instruction. The instruction also sets the **reply mode** for this turn.
3. **Widget mode (main mode):** AI writes its reply **only** as the `text` argument of `show_reply`, and nothing in the chat. The MCP App renders the reply (as it streams, when the host streams tool input), with 3 buttons underneath in the user's language and a discreet link to the user's profile page (section 4.6); both arrive in the tool result's `structuredContent`, which the host gives to the app but not to the model. The server counts the words exactly, then discards the text.
4. **Fallback mode** (host without MCP Apps support): AI replies normally in the chat and calls no tool after it. Such a host cannot render buttons, so there are none.
5. If the user clicks a button, the UI records the click and sends the matching follow-up message into the chat. Users who type instead are covered by step 1 on the next message.

Widget mode costs no extra tokens (the reply is written once, as a tool argument instead of chat text) and gives an exact length. The reply stays in the conversation history as a tool call, so the assistant keeps full access to what it wrote.

## 3. Stack and constraints

Mandatory stack:

- **Docker** (Docker Compose): one service for the app, one for PostgreSQL.
- **PostgreSQL** for all storage, with the **pgvector** extension for topic embeddings (`pgvector/pgvector:pg17` image).
- **FastAPI** for the HTTP API (internal endpoints, health, metrics, debug page).
- The official **MCP Python SDK** (`mcp` 2.x, `MCPServer`) for the MCP tools, served over Streamable HTTP and mounted inside the FastAPI app. Its built-in MCP Apps extension (`mcp.server.apps`) provides the UI resources. (fastapi_mcp was dropped: its last release, 0.4.0, does not run on `mcp` 2.x.)

Rules:

- **Engineering bar: always the best solution.** "MVP" never justifies a weaker technique, a shortcut or a hand-made approximation. Every feature is built the way it would be built for production at scale; scope is cut by dropping features, never by lowering the quality of the ones kept. In particular: no hand-written keyword or stopword lists where a learned model exists; no identifiers repeated as string literals (they are enums, one source of truth); no user-facing or model-facing text in code (it lives in locale catalogs).
- Python libraries: **SQLAlchemy 2** (async ORM) with **Alembic** for migrations, the PostgreSQL driver `psycopg` 3 and `pgvector` for vectors, the ASGI server (uvicorn), `fastembed` (ONNX, CPU) for topic embeddings, `numpy`, `scikit-learn` and `scipy` for clustering, and the development tools pytest, pylint and pyink. Add a library whenever it is the best tool for the job.
- Widget: built with Vite and `vite-plugin-singlefile` into one self-contained HTML file (all code inlined, no external requests at runtime), using `@modelcontextprotocol/ext-apps` (the official MCP Apps SDK) for the host protocol, `marked` for Markdown and `DOMPurify` for sanitising. The Node build runs in a separate Docker build stage; the running app is Python only.
- The model-facing tools (`plan_reply`, `show_reply`) are `MCPServer` tools; `show_reply` is bound to the `ui://` reply resource with `Apps.tool(resource_uri=...)`. `record_click` is an app-only tool (`visibility=["app"]`). `gold_label`, health and metrics are plain FastAPI routes, never MCP tools.
- The owner self-hosts the server. It **must be reachable over public HTTPS**, because Anthropic's servers call it. TLS termination happens outside the app, via a reverse proxy.
- Auth (MVP): one secret token per user, carried in the connector URL (`/mcp/<token>`). The token identifies the user. No OAuth yet: AI supports authless remote servers, and the user cannot add custom headers to a connector. A small ASGI wrapper around the MCP app moves the token from the path into a request header, which tools read through `ctx.headers`.
- **Both protocol eras.** The server serves MCP `2025-11-25` clients (with the `initialize` handshake and `Mcp-Session-Id` sessions) and `2026-07-28` clients (stateless: protocol version, client capabilities and client info travel in each request's `_meta`, plus `server/discover`). The SDK handles both; application code never depends on either era.
- **No protocol sessions in application logic.** Nothing is keyed on the MCP session. Cross-call state uses server-minted handles passed as ordinary tool arguments (`conversation_id`, `turn_id`), the pattern of SEP-2567. Every handle received is checked against the user behind the token; a handle that belongs to another user is rejected with a generic error.
- Host and mode detection happen **on every call**, from whatever the request carries: `ctx.session.client_params.client_info` gives the host (`unknown` when the client sends none, which 2026-07-28 allows); `mcp.server.apps.client_supports_apps(ctx)` gives MCP Apps support (section 4.1).
- Tool argument validation errors echo the invalid input back to the client. `show_reply` must therefore validate its arguments itself and return a generic error, so the reply text never appears in an error result.
- Store only the structured fields below. Never store message text, except `topic` and `user_quote`. **The reply text received by `show_reply` is counted and discarded immediately:** never written to the database (including `raw_payload`), logs, error reports or traces. The SDK's OpenTelemetry spans record only the tool name; never add tool arguments as span attributes.
- **Token hygiene.** The user token sits in the URL path, so the ASGI server's access log is disabled and replaced by one that logs `/mcp/<redacted>`. The reverse proxy must redact the path the same way.
- **Logging.** The MCP logging feature is deprecated (2026-07-28); the server logs to stderr only.
- **Caching and prompt cache.** Tools are listed in a fixed order with stable descriptions. `tools/list`, `resources/list` and `resources/read` results carry a long `ttlMs` (one hour) with `cacheScope: "private"`. Tool results are compact JSON strings (no indentation), never pretty-printed objects.
- **Code style.** Python follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) to the letter: checked by pylint with Google's `pylintrc` and formatted by pyink at 80 columns. Prefer data-driven, generic code over special cases: a new preference dimension, button or language is a data change, not new logic.
- **Data access.** The schema is defined once, as typed SQLAlchemy models (`app/models.py`); application code queries through the ORM and SQL expression language, never through SQL strings. Alembic migrations are autogenerated from the models and reviewed (the initial one also creates the `vector` extension and the append-only triggers); a test asserts that the migrated schema and the models do not drift. Migrations run at start.
- **Vocabulary and text.** Every closed set (request types, outcomes, format issues, preference dimensions and values, buttons, arms, platforms, reply modes) is a `StrEnum` in one module; the tool schemas, validation and SQL checks derive from it. Every text a person or the model reads (button labels and messages, widget labels, profile page, instruction fragments) lives in JSON locale catalogs (`locales/<language>.json`, keyed by those enums); English is the fallback and the source of the model-facing instructions. Adding a language is adding a file.
- Keep the code focused: as few lines as the features allow.

## 4. Tools

### 4.1 `plan_reply` (MCP tool, called by AI before every reply)

Input:

| Field | Type | Meaning |
| --- | --- | --- |
| `conversation_id` | string, optional | Omit on the first message of a conversation; reuse the value returned by the server afterwards |
| `request_type` | enum: `fact`, `explanation`, `advice`, `decision`, `code`, `writing`, `plan` | Kind of request |
| `topic` | string, max 15 words, in the user's language | What the request is about |
| `language` | ISO 639-1 code | The language of the user's message |
| `needed_words` | int | AI's estimate of the words a complete, no-filler answer needs |
| `last_reply_outcome` | enum (below) | How the user reacted to AI's previous reply in this conversation |
| `user_quote` | string, max 5 words | The user's exact words proving the outcome |
| `last_reply_words` | int, fallback mode only | Approximate length of AI's previous reply (not needed in widget mode, where the server counts exactly) |
| `format_issue` | enum (below), only when `last_reply_outcome` is `wrong_format` | What exactly was wrong with the form |
| `ambiguous` | bool, only when true | The request has several plausible readings |
| `user_message_words` | int | Length of the user's message (rhythm, a stage-2 feature) |

`format_issue` values: `too_technical`, `too_simple`, `want_list`, `want_prose`, `want_example`, `wrong_tone`. Each one feeds the matching preference dimension directly (section 7).

Every field costs output tokens on every turn, so fields that no model uses are not asked for (`pasted_content` and `user_specified` were removed). The cost of what remains is measured (section 11).

**Token economy rule:** every field AI sends costs output tokens, the most expensive and slowest kind. Anything the server can compute itself is never asked from AI:

- the server computes `turn_index`, timestamps, durations, reading time and the control group;
- the server never asks which model answers: it infers how the assistant behaves (section 6.3);
- the server fills defaults for omitted optional fields;
- the **language** is the one field the server cannot compute as well as the assistant: the assistant reads the whole message, the server only a 15-word topic. So the assistant sends it (an ISO code, about 2 tokens), **per turn**, because users switch languages. It selects the locale catalog for the buttons and the widget; a language without a catalog falls back to English;
- the server derives the **topic domain** itself (section 7.1): it embeds `topic` with a local multilingual model and assigns it to the nearest learned topic cluster.

`last_reply_outcome` values:

- `moved_on` — the user progressed: next step, builds on the answer, or new topic
- `too_long`
- `too_short`
- `missing_info` — the reply lacked something the user needed
- `misunderstood` — the reply answered the wrong question
- `wrong_format` — wrong form or level (too technical, unclear, list vs prose)
- `no_signal` — first message, or nothing to judge

Validation: `user_quote` is required for every outcome except `moved_on` and `no_signal`. If it is missing, store the outcome as `no_signal`. Clamp `needed_words` to [5, 2000].

Output (keep it under ~60 tokens):

```json
{
  "conversation_id": "c_8f2k1x",
  "turn_id": "t_31",
  "target_words": 90,
  "instruction": "Reply in about 75 words. No preamble, no recap. Use a bulleted list. Write the reply only in show_reply."
}
```

In widget mode, `instruction` ends with "Write the reply only in show_reply."; in fallback mode it has no tool sentence, and the assistant replies in the chat. **Mode selection, per call:** read the client capabilities carried by the current request (from the `initialize` handshake on 2025-11-25, from the request's `_meta` on 2026-07-28; the SDK's `client_supports_apps(ctx)` covers both). If the host declares MCP Apps support (`io.modelcontextprotocol/ui` with the `text/html;profile=mcp-app` MIME type), the turn uses widget mode; otherwise, fallback mode. Store the mode as `reply_mode` on the turn, and the first value as `ui_supported` on the conversation. Confirm on each host that the declaration matches what is actually rendered, and keep a per-host override in configuration.

`target_words` is the length the user should receive. The word count in `instruction` is the **requested** length, corrected for this assistant's compliance (section 6.3), so that the assistant actually writes about `target_words`.

Keep the output short and in exactly the same format on every call: the tool result stays in the conversation, and a stable, compact result keeps AI's prompt cache efficient.

**Comparison arms: off by default.** Users come for the product, so none is held back: the arm shares default to 0, and savings are reported as **estimated** (section 11). The arms stay for a later opt-in study with consenting users, when a measured figure is needed:

- `control` (`CONTROL_SHARE`): "Reply normally.";
- `concise` (`CONCISE_SHARE`): a fixed "Be concise.", the real competitor (what a user would write in their own instructions). The product must beat it, measurably.

The randomisation unit is set by `CONTROL_UNIT`:

- `conversation` (default while there are fewer than 100 active users): each new conversation goes to an arm with probability equal to its share;
- `user` (from 100 active users): each new user goes to an arm with probability equal to its share. This avoids spillover: a user who learns to be concise in treatment conversations is also more concise in their control conversations, which makes conversation-level comparisons underestimate the effect.

In both arms, `plan_reply` still records everything and still updates the models, but returns `target_words: null` and the arm's fixed instruction (followed by the widget-mode sentence when it applies), and `show_reply` returns no buttons. These arms are what turn estimated savings into measured savings (section 11).

**Clarification.** When `ambiguous` is true, the server decides between asking and answering with a two-arm Thompson bandit per (user, request type): Beta counts of the reward (the turn's P(`moved_on`)) for each arm. If it picks asking, the instruction becomes "Ask one short question to clarify, nothing else." It learns who likes being asked and who hates it.

### 4.2 `show_reply` (MCP tool, widget mode: carries the reply itself)

Input: `turn_id`, `text` (the full reply, in Markdown).

Server side:

- count the words of `text` exactly (split on whitespace, ignoring Markdown syntax characters), store the count as `reply_words_exact` on the turn, then **discard the text immediately** (see section 3);
- record the arrival time: the gap between `plan_reply` and `show_reply` is the writing duration;
- return a minimal result: `content` is `"ok"` (the only part the model sees), and `structuredContent` carries what the app needs: `{"turn_id", "buttons": [{"id", "label", "message", "position"}], "labels", "profile_url"}`. Button labels and messages, and the widget's own labels, are in the turn's language. Buttons are empty in the comparison arms. `structuredContent` is not added to the model's context, so the buttons cost no model tokens and need no extra round trip.

The MCP App (implemented following the current MCP Apps extension specification, `2026-01-26`) must:

- render the reply from the complete tool input (`ui/notifications/tool-input`), which hosts must send. When the host also sends partial tool input (`ui/notifications/tool-input-partial`, optional for hosts), render it progressively as it streams; never depend on it;
- render Markdown properly: headings, lists, emphasis, links, tables, and code blocks with a copy button; plus a "copy reply" button;
- show the 3 buttons under the reply, in the order chosen by the server (section 8), as soon as the tool result (`ui/notifications/tool-result`) arrives; the reply never waits for them;
- on `ui/notifications/tool-cancelled`, keep whatever was rendered, show no buttons, and record nothing (a turn without a `show_reply` call simply has no exact length);
- on click, record the click (call `record_click` through the host's tool-call capability) and wait for it, then send the button's follow-up message into the conversation with `ui/message`;
- use the host-provided style variables and theme, with no external assets;
- stay readable on a phone-width screen;
- show a discreet link to the user's profile page (`profile_url`, opened with `ui/open-link`);
- report the host context once per reply: `platform` (`web`, `desktop`, `mobile`) and `locale` from the host context, through the app-only tool `report_context`.

In `control` conversations, the widget shows the reply without buttons.

### 4.3 Internal endpoints (not exposed to the model)

- `record_click` with `turn_id`, `button`, `position`. Called by the UI only, through the host's tool-call capability. Exposed as an app-only MCP tool (`_meta.ui.visibility: ["app"]`: hosts must not list it to the model and must reject app calls to tools without `"app"`), with a description saying the model must never call it.
- `report_context` with `turn_id`, `platform`, `locale`. App-only tool, like `record_click`. Stored as a `turn_context` event; the conversation's latest platform shapes the next turns (section 6.5), and the locale is a language fallback.
- `POST /gold_label` with `turn_id` and the true `outcome`. Used by the owner to label real turns by hand (a debug page listing recent turns with their `topic`, `user_quote` and inferred outcome). Target a few hundred labels; they anchor the annotator model (section 6.4).

### 4.4 Button catalogue

Labels are short sentences anyone understands, taken from the locale catalog of the turn's language (English and French for now; other languages fall back to English). The follow-up message is sent in the same language.

| Button id | Label (en / fr) | Follow-up message (en / fr) | Outcome it reports | Preference signal |
| --- | --- | --- | --- | --- |
| `shorter` | Make it shorter / Fais plus court | "Shorter, please." / « Plus court, s'il te plaît. » | `too_long` | — |
| `more_detail` | Tell me more / Développe davantage | "Tell me more." / « Développe davantage. » | `too_short` | — |
| `one_line` | Sum it up in one line / Résume en une ligne | "Sum it up in one line." / « Résume en une ligne. » | `too_long` | — |
| `example` | Show me an example / Donne-moi un exemple | "Show me an example." / « Donne-moi un exemple. » | style click | `examples` = `yes` |
| `simpler` | Explain it more simply / Explique plus simplement | "Explain it more simply." / « Explique plus simplement. » | `wrong_format` | `level` = `simple` |
| `as_list` | Make it a list / Mets-le en liste | "Make it a list." / « Mets-le en liste. » | `wrong_format` | `format` = `bullets` |
| `as_prose` | Write it as plain text / Écris-le en texte suivi | "Write it as plain text, no lists." / « Écris-le en texte suivi, sans liste. » | `wrong_format` | `format` = `prose` |
| `not_this` | Not what I wanted / Ce n'est pas ce que je voulais | "That's not what I wanted. Ask me one short question to understand what I need." / « Ce n'est pas ce que je voulais. Pose-moi une seule question courte pour comprendre ce dont j'ai besoin. » | `misunderstood` | — |

A click is one annotator among others (section 6.4), with its own learned reliability, and a preference signal when the table says so.

### 4.5 Several conversations, models and hosts

The same user may run many conversations in parallel, switch models (Opus, Sonnet, Haiku…), and use several hosts (AI, and ChatGPT, which also supports MCP connectors).

- **Parallel conversations:** everything is keyed by `conversation_id`, and durations and reading time are computed per conversation. If the assistant forgets the id, the server starts a new conversation and the previous verdict is lost. That is acceptable, but monitored through tool coverage (section 11).
- **One user everywhere:** the token identifies the user across hosts, so one profile serves all of them.
- **Host detection at zero token cost:** read the client name (`clientInfo`, from the `initialize` handshake on 2025-11-25 or from the request's `_meta` on 2026-07-28) and store it as `host` on the conversation, or `unknown` if the client sends none. Never ask the assistant for it.
- **Model identity is never asked.** Models often do not know their exact version, name an older one, or can be made to say anything. What matters is behaviour, which the server measures itself (section 6.3).
- **Hosts with MCP Apps:** Claude (web and desktop) and ChatGPT both support MCP Apps (official extension support matrix), so widget mode is the normal case on both.
- **Hosts without MCP Apps:** fallback mode (section 4.1): normal chat replies, estimated lengths (section 6.2), and no buttons if the host cannot render them, in which case the other annotators carry the reaction model.

### 4.6 Profile page: "What I learned about you"

`GET /me/<key>`, where `key` is the user's `token_hash` (it grants access to this page only, never to the MCP tools). It lists, in plain words and in the user's latest language, what the server believes: "You prefer bulleted lists", "Short replies for factual questions", "Technical level for code". Each line has a control to correct it. A correction is stored as a `profile_edits` event and is a very strong signal: it adds a large count to the chosen preference value (section 7) or a sharp observation to the length grid (section 6.5). The widget links to it discreetly.

## 5. Learning approach

### 5.1 Online RL or not?

Each decision (the length of this reply, the buttons under it) gets its feedback on the user's next message, and it barely changes what comes after. That is a **contextual bandit**: online learning with exploration, one decision at a time. Full reinforcement learning (PPO, DQN…) solves long sequences of dependent decisions; it is data-hungry and adds nothing here. So: **online learning, yes; full RL, no.**

### 5.2 Model for the MVP: hierarchical Bayesian censored model with Thompson sampling

The product targets as many users as possible. The model family below was chosen because it is the best fit at launch **and** grows into the best fit at scale (section 5.3) without changing its foundations: the same censored likelihood, the same Thompson sampling, the same instant per-user updates.

Why it is the best fit for this problem:

1. **The feedback is censored, and this model uses it exactly as it is.** "Too long" at 300 words does not mean the ideal is 250; it only means the ideal is *below* 300. "Fine" at 300 means 300 was *enough*. These are bounds, not values. A classifier or a standard regressor treats each signal as a point or a class, which either invents information or throws it away. A censored likelihood keeps exactly what the user told us, nothing more.
2. **Every user starts with zero data, at every stage of the product.** Even with millions of users, each new user is a cold start. The hierarchical prior gives a sensible default from the population on the first message, and the user's own posterior moves after 3 to 5 reactions. Tree ensembles and neural networks need thousands of events before they say anything useful about one person.
3. **It explores where it is unsure, and only there.** Thompson sampling draws from the posterior, so it explores a lot on new users and new request types, and almost not at all where it is confident. There is no exploration rate to tune, and it is known to be near-optimal for bandit problems. This matters because every exploration costs the user a slightly worse reply.
4. **It adapts instantly, inside the conversation.** One reaction updates the user's posterior in microseconds. If the user says "shorter" at 11:05, the reply at 11:06 is already shorter. Batch models only adapt at the next retraining.
5. **The trade-off is explicit and auditable.** "Aim for the shortest length with an 80 % chance of being enough" is one readable parameter, not behaviour buried in weights. You can change the product's philosophy by changing one number.
6. **It is readable.** For each user and request type, the posterior answers in plain words: "this person needs about 40 % of what AI would write for explanations." This makes debugging and explaining the product easy.
7. **It is cheap and robust.** Small grids and Gaussian updates, pure Python, no training infrastructure, no GPU.

Known limits, which the next stages solve: it only uses the user, the request type and AI's estimate. It does not share knowledge between *similar* users, only with the population, and it ignores the topic.

### 5.3 Model roadmap by stage

Each step up keeps the censored likelihood, Thompson sampling, instant per-user updates and the measurement models of section 6, and adds what the previous stage could not learn. Switch stages only after the new model wins an offline evaluation on logged data (doubly robust estimator, using the propensities from section 5.5) and then an A/B test on 10 % of traffic.

| Stage | When | Model | What it adds |
| --- | --- | --- | --- |
| 1 — MVP | Launch, until ~200 active users or ~20,000 outcomes | Hierarchical Bayesian censored model (this spec) | Works from the first reaction |
| 2 — Growth | Up to ~5,000 active users | Hierarchical Bayesian censored **regression**: log ratio = features × β + user effect + user × type effect. Features: request type, log of needed words, turn index, topic cluster, rhythm (`user_message_words` and the delay since the previous reply: short, rapid messages suggest a user in a hurry) and platform. β is refit nightly; user effects are updated instantly (Laplace approximation) | Shares knowledge across users through features; uses the topic |
| 3 — Scale | Beyond that | **Neural-linear Thompson sampling**: a two-tower network (user embedding, context embedding from the topic text and features), trained nightly with the censored loss, with a Bayesian linear head updated instantly per user | Learns which users are alike (collaborative filtering) and the meaning of topics, replacing hard request types with a continuous representation |

Stages 2 and 3 will require adding numerical libraries (and an ML framework for stage 3). That decision is taken when the stage starts, not in the MVP.

Buttons and form preferences follow the same path: Beta and Dirichlet counts per (user, scope) in stage 1, then the same features and representations as the length model in stages 2 and 3.

### 5.4 Switching models must stay cheap

The `plan_reply` contract never changes; only the decision logic behind it does. Build it so that a switch is a configuration change, not a rewrite:

- **One policy interface.** Every model implements `decide(context) -> (decision, propensity)` and `update(outcome)`. The active policy and its version are chosen by configuration (an environment variable). Rolling back means pointing the variable at the previous policy.
- **Raw payloads, append-only.** Store the full raw JSON of every tool call (`raw_payload jsonb`) next to the parsed columns, and never update or delete events. A future model can then re-extract any field, even one the current schema ignores.
- **Versions on every row.** Every turn and outcome carries `model_version` (which policy decided) and `schema_version` (which payload format was received).
- **Shadow mode.** Any number of candidate policies can run next to the active one. They receive the same context, and their decisions and propensities are logged in `shadow_decisions`, but never applied. This shows how a candidate would behave on live traffic before it touches a single reply.
- **Replay evaluation.** A script replays the logged events through a candidate policy and computes the doubly robust estimate of words per step forward and repair rate, with confidence intervals. A candidate is promoted only if it beats the active policy in replay, then in an A/B test on 10 % of conversations.

### 5.5 What the MVP must log so later stages work

- For every turn: all `plan_reply` inputs (including `topic`, kept for future embeddings), the turn index, `model_version`, the sampled θ, and the posterior mean and standard deviation.
- **Propensity of the chosen length:** draw 200 samples from the posterior, apply the decision rule of section 6.7, and store the share of samples that give the same rounded `target_words`.
- For every shown button: its position, the shuffled flag, and the Thompson draw.
- For every turn: the form value drawn for each preference dimension and the clarification decision (`style` jsonb).

## 6. Unified probabilistic model

**Principle.** Everything the assistant reports, and everything the server measures, is a **noisy measurement of a hidden truth**. The model states the hidden truths explicitly, gives each measurement a noise model, learns that noise from the data itself, and combines all measurements in one probabilistic model. There are no hand-set weights and no hard labels: uncertainty is carried through to the decision, where it naturally produces more exploration.

Inference runs in two speeds:

- **Online**, on every event: fast approximate updates (Gaussian updates, small grids).
- **Nightly**, an expectation–maximisation (EM) pass re-estimates every noise model and hyperparameter, then rebuilds all derived state by replaying the append-only events.

Notation: Φ is the standard normal CDF (`math.erf`). All lengths are on the log scale and expressed in "words as the assistant counts them". `needed_words` uses the same unit, so any systematic bias in how an assistant counts words cancels out in the ratio length / needed.

### 6.1 Hidden quantities

| Hidden quantity | Level | Meaning |
| --- | --- | --- |
| θ(u,t) | user × request type | Mean log of the **minimal sufficient ratio**: length needed / `needed_words` for this user and type |
| α(u), γ(t) | user; request type | Multilevel decomposition of θ (section 6.5) |
| L | turn | True log length of the reply |
| y | turn | True user reaction: `moved_on`, `too_long`, `too_short`, `missing_info`, `misunderstood`, `wrong_format` |
| π(c, f) | conversation | Probability that the assistant in conversation *c* belongs to behaviour class *f* |
| b(c) | conversation | Assistant's bias when estimating `needed_words` |
| κ(c), v(c) | conversation | Compliance (log written / requested) and writing speed (log words per second) |
| ρ(u), mixture weights | user | Reading speed and reading behaviour (section 6.6) |
| φ(u,s,d) | user × scope × dimension | Distribution of the preferred value of a form dimension (section 7) |

For a given message, the minimal sufficient log ratio is S = θ(u,t) + b(c) + ε, with ε ~ N(0, τ²). τ starts at 0.35 and is refit nightly.

### 6.2 Reply length

Three sensors measure the true length L of a reply:

1. **Exact count** (widget mode): L = log(`reply_words_exact`), with zero variance.
2. **Writing time** (widget mode): s₂ = log(duration − overhead) = L − v(c) + e₂, where duration is the gap between `plan_reply` and `show_reply`, overhead starts at 3 s (refit nightly), and v(c) is the conversation's log writing speed. With the exact count known, this sensor measures the writing speed, which feeds the behaviour-class posterior (section 6.3). Use a heavy-tailed error (Student-t, 4 degrees of freedom, applied as a robust down-weighting of large residuals), because extended thinking, network delays and server load inflate some durations.
3. **Requested length:** s₃ = log(requested words) + κ(c) = L + e₃.
4. **Self-report** (fallback mode): s₁ = log(`last_reply_words`) = L + e₁, sent with the next `plan_reply`. It defines the unit (assistant-counted words), so it has no bias term by construction.

In fallback mode no tool call follows the reply, so only the self-report and the requested length are available.

Online, per turn: fuse the available sensors with a small Gaussian update over (L, κ(c), v(c)). This gives a posterior mean and variance for L, and updates the conversation's compliance and speed. Any sensor may be missing.

Nightly, estimate the writing-time and requested-length noise from widget-mode turns, where the exact count is ground truth. Then estimate the self-report noise from fallback turns: the two sensors are independent, so Var(s₁ − s₃) = σ₁² + σ₃², which gives σ₁² from the already known σ₃². If it comes out negative (sampling noise), floor it at 0.01.

### 6.3 Assistant behaviour

What matters is not *which* model answers but *how it behaves*, and what an assistant says about itself is never trusted, nor even asked. Behaviour follows a hierarchy: population → behaviour class → conversation.

**Behaviour classes, learned.** Nightly, every conversation with measured writing speeds is placed by two numbers the server measures itself: its mean log writing speed (section 6.2) and its compliance κ(c). A Gaussian mixture (scikit-learn, diagonal covariances) finds the classes, its number of components chosen by BIC up to 6, with at least 10 conversations per component; below 10 conversations there is one class, the population (no bias, about 50 words per second, wide spread). Class ids stay stable across nights: new components are matched to the previous ones by distance between means with the Hungarian algorithm. Each class gets the speed and compliance of its component and the estimation bias of its conversations. A class is a family of assistants that behave alike, whatever their brand or version.

**Class as a probability, never a hard label.** For each conversation, keep π(c, f) over the learned classes:

π(c, f) ∝ P(f | host) × Π over turns of N(v̂; μ_v(f), s_v(f)²)

- `P(f | host)`: share of each class per host, learned; uniform until learned.
- v̂: the turn's estimated log writing speed from section 6.2.

π(c, ·) is updated after every turn.

**Estimation bias:** b(c) = Σ_f π(c, f) · b(f), with variance Var b(c) = Σ_f π(c, f) · (b(f)² + w(f)) − b(c)². Var b(c) is added to τ² everywhere S appears. An uncertain class therefore widens the posterior and leads to a little more exploration, instead of a confident wrong answer.

**Compliance and speed:** per conversation, Gaussian priors centred on the class mixture, Σ_f π(c, f) · κ(f) and Σ_f π(c, f) · μ_v(f), with sd 0.3 for compliance and the class spread for speed, updated by the sensor fusion of section 6.2.

### 6.4 User reaction: several annotators, Dawid–Skene

The true reaction y of each turn is hidden. Several independent annotators observe it, each with a **confusion matrix** P(observed | true) that is learned, never assumed:

| Annotator | What it reports | Confusion matrix learned per |
| --- | --- | --- |
| Assistant verdict (`last_reply_outcome`) | One of the outcome values | Behaviour class (soft-weighted by π; pooled until a class has data) |
| Button click | The outcome of the clicked button (section 4.4), or "no click" | Button position |
| Reading behaviour | P(skimmed), from section 6.6 | Population |

Online: P(y | everything) ∝ P(y | user) × Π annotators P(observation | y). P(y | user) is a per-user Dirichlet prior over reactions. Store the resulting probability vector on the outcome.

Nightly: the **Dawid–Skene** algorithm (EM for learning from several noisy annotators) re-estimates every confusion matrix and the true-reaction posteriors. The owner's gold labels (`POST /gold_label`) are clamped to their true value and anchor the solution.

Initial values: assistant verdict 0.8 on the diagonal (0.7 for `moved_on`, because silent dissatisfaction is often read as moving on), buttons 0.95, the rest spread uniformly.

This also covers invented quotes: an assistant that fabricates or misreads reactions agrees less with buttons, reading behaviour and gold labels, so its learned reliability, and therefore its influence, drops.

**Clicks make the assistant verdict dependent.** A click sends the button's follow-up message ("Shorter.") into the chat, and the assistant then reads that message when it judges the turn. Its verdict is no longer an independent observation, which Dawid–Skene assumes. So when a turn has a click, the assistant verdict for that turn is stored but **not used** as an annotator, online or nightly; the click speaks for the turn.

### 6.5 Length likelihood and multilevel structure

**Likelihood of a turn.** With r̂ = L − log(`needed_words`), using the fused length from section 6.2:

- σ² = τ² + Var(L) + Var b(c)
- P(sufficient | θ) = Φ((r̂ − θ − b(c)) / σ)
- likelihood = Σ_y P(y) · ℓ_y(θ), where ℓ_y = P(sufficient) for `moved_on` and `too_long`, ℓ_y = 1 − P(sufficient) for `too_short` and `missing_info`, and ℓ_y = 1 (no length information) for `misunderstood` and `wrong_format`.

There are no hand-set weights: how much a signal counts comes from the annotators' learned reliabilities.

**Multilevel structure for θ.** θ(u,t) = α(u) + γ(t) + η(u,t), with α(u) ~ N(μ₀, σ_α²), γ(t) ~ N(0, σ_γ²) and η(u,t) ~ N(0, σ_η²). Each level is pulled toward the level above in proportion to how little data it has. A request type misclassified by the assistant therefore costs little, because θ for a sparse (user, type) cell stays close to the user's overall α(u).

Online implementation:

- Grid posterior over θ(u,t): θ ∈ [log 0.1, log 2.0], 81 points.
- Grid posterior over α(u) on the same range, updated with each observation using the inflated variance σ² + σ_η² + Var γ(t).
- A new (u,t) cell starts from the prior N(E[α(u)] + γ(t), σ_η² + Var α(u)).
- **Platform.** On a phone, the same request may need fewer words. When the conversation's latest reported platform is `mobile`, the length cell is (u, t@mobile): its own grid, with the same γ(t), learned per user.
- **Profile edits.** A length correction on the profile page (short / medium / detailed) multiplies the cell grid by N(θ; log 0.4 / log 0.7 / log 1.0, 0.15²), a sharp observation.

Nightly: re-estimate μ₀ (initially log 0.6, since assistants tend to over-write), σ_α (0.5), σ_γ (0.3), σ_η (0.3), γ(t) and τ by EM, then rebuild all grids by replay.

### 6.6 Reading behaviour: a mixture per user

After a reply, the time until the next `plan_reply`, minus the estimated writing time and overhead, is the user's reading time. A user either **read fully**, **skimmed**, or **was away** (another conversation, a call, a break). Model it as a mixture, per user:

- **Read fully:** log time ~ N(L − ρ(u), s_r²). ρ(u) is the user's log reading speed: hierarchical prior centred on 4 words per second, refit nightly.
- **Skimmed:** time uniform on the log scale between 1 s and the full-read time.
- **Away:** log time ~ N(μ_away, s_away²), a broad component for long gaps.
- **Mixture weights:** per-user Dirichlet, learned.

Online: compute the posterior responsibility of each component for the turn. P(skimmed) feeds the reaction model as an annotator (section 6.4); P(away) means the reading time carries no information. Nightly: refit all mixture parameters by EM.

### 6.7 Decision (Thompson sampling)

1. Sample θ̃ from the θ(u,t) grid and b̃ from N(b(c), Var b(c)).
2. r* = exp(θ̃ + b̃ + z_q · τ), with z_q the standard normal quantile of q = 0.8 (configurable): the shortest length with an 80 % chance of being enough. This is where the asymmetry lives: a missing piece of information is worse than a few extra words, so q is set high. Do not change the inference to express it.
3. `target_words = needed_words × r*`, clamped to [0.5 × needed, 1.5 × needed], minimum 10.
4. Requested words = `target_words` / exp(E[κ(c)]), then round to the nearest 5, or to the nearest 10 above 50.

Log θ̃, b̃, the posterior mean and standard deviation, `target_words`, requested words and the propensity (section 5.5) for each turn.

5. **Form**, by Thompson sampling too: for each preference dimension, draw from the cell's Dirichlet posterior and take the value with the highest draw (section 7). New users therefore sometimes get another form, which discovers their preference without waiting for a complaint.
6. **Clarification** when `ambiguous` (section 4.1).

## 7. Form preferences

Default instruction for every reply: "No preamble, no recap."

| Dimension | Values (`auto` = no instruction) | Instruction fragments | Scope |
| --- | --- | --- | --- |
| `format` | `auto`, `bullets`, `prose`, `table` | "Use a bulleted list." / "Write in prose, no lists." / "Use a table where it fits." | request type |
| `level` | `auto`, `simple`, `technical` | "Use simple words, no jargon." / "Be technical; skip the basics." | **topic domain** (section 7.1: an expert in code may be a beginner in law) |
| `examples` | `auto`, `yes` | "Include one short example." | request type |
| `tone` | `auto`, `direct`, `warm` | "Be direct." / "Be warm and friendly." | request type |
| `headings` | `auto`, `yes`, `no` | "Use short headings." / "No headings." | request type |

- **Model.** Each dimension is categorical with a Dirichlet posterior (a yes/no dimension is the two-value case, a Beta). Hierarchy as for θ: population → user → user × scope, each level pulled toward the one above with strength 4: α(u,s) = counts(u,s) + 4 · mean(u), mean(u) from counts(u) + 4 · mean(population). The population prior gives `auto` 4 and every other value 0.25: about 11 % of a new user's replies try another form, and one explicit signal makes the requested value win about 90 % of the draws.
- **Decision** (section 6.7): Thompson draw per dimension; the fragment of the drawn value joins the instruction (none for `auto`). Confidence is built in: a value only wins often once its counts dominate.
- **Signals** (count added):
  - explicit, 3: a button with a preference signal (section 4.4), or `format_issue` (`too_technical` → `level`=`simple`; `too_simple` → `technical`; `want_list` → `format`=`bullets`; `want_prose` → `prose`; `want_example` → `examples`=`yes`; `wrong_tone` → every `tone` value except the one used). Like the verdict, `format_issue` is ignored when the turn has a click (section 6.4);
  - implicit, 0.3: each value used on the turn gets 0.3 × P(`moved_on`); each other value of the dimension shares 0.3 × P(`wrong_format`);
  - profile edit, 10.
- **Storage**: `pref_counts(user_id, scope, dimension, value, count)`. User 0 holds the population and scope `*` the user level; buttons (section 8) and the clarification arms (section 4.1) use the same table, as two-value dimensions.
- The instruction fragments in the table above are the English catalog entries (section 3); they are not written in code.

### 7.1 Topic domains, learned

Domains are not a fixed list: they are learned from the topics themselves.

- **Embedding.** At `plan_reply`, the server embeds `topic` with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` through `fastembed` (local, CPU, 384 dimensions, 50+ languages, so a French and an English topic about the same thing land together). The model is downloaded at image build time; topics never leave the server. The vector is stored as an event (`topic_embeddings`).
- **Clustering, nightly.** Spherical k-means (scikit-learn `KMeans` on L2-normalised vectors) over every topic embedding, with k chosen by the best cosine silhouette score in [2, min(30, √n)]. Below 20 topics there is a single domain. Cluster ids stay **stable** across nights: new centroids are matched to the previous ones by cosine similarity with the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`), and only unmatched clusters get new ids. Each cluster is named by its **medoid topic** (the topic closest to its centroid), which the profile page shows ("Topics like « fix a python bug »").
- **Assignment.** A topic's domain is its nearest centroid by cosine distance (pgvector `<=>`), computed online at `plan_reply` and recomputed at replay, so derived state always follows the latest clusters.

## 8. Buttons policy

- Keep Beta(1,1) counts of click-when-shown per (user, request type, button), in `pref_counts` as the dimension `button:<id>` with values `yes` (clicked) and `no`, with the same hierarchy as section 7.
- **Safe button:** the highest posterior mean.
- **Exploration buttons:** draw one sample per remaining button (`random.betavariate`) and take the 2 highest.
- Order: [safe, explore, explore]. With probability 0.2, shuffle randomly instead. Log every shown button with its position and the shuffled flag.
- When the next `plan_reply` arrives in that conversation and no button was clicked, record a "not clicked" for each shown button.
- Position effects are learned through the button annotator's per-position confusion matrix (section 6.4).

## 9. Data model (PostgreSQL)

The schema is the SQLAlchemy models (section 3); Alembic migrations, generated from them, are applied at start. Every event table has a surrogate `id`. Columns holding a vocabulary that evolves (buttons, request types, platforms, observed outcomes) are plain text, so history keeps retired values while the enums validate new writes; stable vocabularies (arms, reply modes, outcome sources) are CHECK-constrained from their enums.

**Events** (append-only: no `UPDATE` or `DELETE`):

- `users(id, token_hash, created_at)`
- `conversations(id, user_id, grp ['treatment'|'control'|'concise'], host, ui_supported bool, protocol_version, created_at)` (legacy `assistant_model` kept for old rows)
- `turns(id, user_id, conversation_id, turn_index, created_at, reply_mode ['widget'|'fallback'], model_version, schema_version, raw_payload jsonb, request_type, topic, needed_words, language, user_message_words, target_words, requested_words, propensity, theta_sample, b_sample, post_mean, post_sd, style jsonb, transport_session)`. `style` holds the drawn form values, the clarification decision, the length cell and the topic domain used.
- `replies(turn_id, reply_words_exact, created_at)`: one row per `show_reply` call. Kept apart from `turns` so that `turns` is never updated; `created_at` ends the writing time.
- `outcomes(turn_id, source ['assistant'|'button'|'reading'], observed, user_quote, last_reply_words, reading_seconds, model_version, schema_version, created_at)`
- `buttons_shown(turn_id, button, position, shuffled, thompson_draw)`
- `clicks(turn_id, button, position, created_at)`
- `gold_labels(turn_id, outcome, created_at)`
- `shadow_decisions(turn_id, model_version, target_words, propensity, details jsonb)`
- `tool_calls(user_id, tool, created_at)`, used to monitor compliance
- `turn_context(turn_id, platform, locale, created_at)`, reported by the widget
- `profile_edits(user_id, scope, dimension, value, created_at)`: corrections from the profile page (`dimension` = `length` for a length cell)
- `topic_embeddings(turn_id, model, embedding vector(384))`: the embedding of each turn's `topic`

**Derived state** (rebuilt nightly by replaying events; never a source of truth):

- `turn_inference(turn_id, length_mean, length_var, reaction_probs jsonb, reading_probs jsonb)`
- `conversation_state(conversation_id, family_probs jsonb, kappa_mean, kappa_var, speed_mean, speed_var)`
- `theta_posteriors(user_id, request_type, grid double precision[])`, primary key (user_id, request_type); `request_type` is the length cell, `t` or `t@mobile`
- `alpha_posteriors(user_id, grid double precision[])`
- `user_reading(user_id, rho_mean, rho_var, mixture jsonb, reaction_prior jsonb)`
- `pref_counts(user_id, scope, dimension, value, count)` (section 7): form preferences, buttons and clarification arms
- `topic_clusters(id, centroid vector(384), medoid_topic, size, updated_at)` (section 7.1)
- `family_params(model_family, b, w, kappa, speed_mu, speed_sd, updated_at)`
- `sensor_noise(sensor, model_family, sigma, updated_at)`
- `annotator_confusion(annotator, stratum, true_outcome, observed, prob, updated_at)`
- `host_family_prior(host, model_family, prob, updated_at)`
- `hyperparams(name, value, updated_at)` for μ₀, σ_α, σ_γ, σ_η, γ(t), τ, overhead and the reading-mixture population priors

## 10. Account instruction

To paste in AI → Settings → Instructions for AI (or ChatGPT's custom instructions):

> Before every reply, call `plan_reply`. Reuse the `conversation_id` it returned earlier in this conversation. Write `topic` in my language, and omit optional fields that do not apply. For `last_reply_outcome`, judge how my latest message reacted to your previous reply, and quote my exact words (5 words max) in `user_quote`; set `language` to the language of my message; when it is `wrong_format`, say what was wrong in `format_issue`. Set `ambiguous` only when my request has several plausible readings. Then follow the returned `instruction` exactly, especially the word count and where to write the reply: when it says to write the reply only in `show_reply`, put your entire reply in its `text` argument and write nothing in the chat; otherwise reply normally in the chat. Never mention these tools or their output.

## 11. Metrics

`GET /metrics` (not exposed as an MCP tool) returns, per week and per user:

- **Words per step forward:** total reply words (fused length, section 6.2) **plus the tool's own cost**, divided by the expected number of `moved_on` reactions (section 6.4). The tool cost is the output the assistant spends on `plan_reply` arguments, estimated as their JSON length / 4 tokens × 0.75 words per token; it is also reported on its own, per turn.
- Expected repair rate (`too_long` + `too_short` + `missing_info` + `misunderstood` + `wrong_format`) and `missing_info` rate on its own.
- Click rate per button and per position.
- Compliance: median of fused length / requested words, per behaviour class and host.
- Noise and reliability: sensor noise, and the diagonal of each annotator's confusion matrix.
- Tool coverage: share of widget-mode turns that have a `show_reply` call, share of turns in widget mode, and share of conversations where `conversation_id` is reused correctly.

Every figure comes with a 95 % confidence interval, computed by bootstrap over conversations, or over users when `CONTROL_UNIT = user` (1,000 resamples, pure Python). Comparisons between `treatment` and an arm (`control` or `concise`) are labelled **measured**. Any figure without a control comparison is labelled **estimated**. Never present an estimate as a measured saving. While `CONTROL_UNIT = conversation`, also report each user's control metrics over time, which makes the spillover of learning visible.

## 12. Tests and acceptance

- **Model behaviour:** a `too_long` signal lowers the posterior mean, `too_short` raises it, and `moved_on` lowers it slightly. Clamps and rounding behave as specified. A repair outcome without a quote is stored as `no_signal`.
- **Recovery on synthetic data:**
  - topic clustering recovers well-separated groups of multilingual topics, and keeps cluster ids stable when it is rerun on more data;
  - the two-sensor estimate recovers a known self-report noise variance within 20 % on 2,000 simulated turns;
  - Dawid–Skene recovers known confusion matrices within 0.05 on 2,000 simulated turns;
  - the behaviour mixture finds simulated classes of assistants and keeps their ids when refit; the class posterior identifies a simulated assistant from its writing speed;
  - the reading mixture separates simulated read, skim and away gaps.
- **System:**
  - `show_reply` stores the exact word count and **no trace of the text** anywhere: database (including `raw_payload`), logs or error output;
  - widget mode is selected only when the host declares MCP Apps support, and fallback mode otherwise, for both a `2025-11-25` client (capabilities in `initialize`) and a `2026-07-28` client (capabilities in each request's `_meta`);
  - a `turn_id` or `conversation_id` belonging to another user is rejected, and the token never appears in logs;
  - when a turn has a click, the assistant verdict for that turn does not enter the reaction posterior;
  - with an arm share of 0.1, about 10 % of new conversations (or users) land in that arm, and get its fixed instruction and no buttons;
  - preferences: a `format_issue`, a button with a preference signal and a profile edit each move the matching dimension; Thompson draws follow the posterior; buttons come in the turn's language; `ambiguous` can turn the instruction into one question; a `mobile` report moves the next turns to the `@mobile` length cell;
  - the profile page lists the preferences in plain words and applies a correction;
  - the metrics include the tool cost;
  - rebuilding all derived state from scratch by replaying the events gives the same state as the online path, up to the nightly refit;
  - swapping the active policy through configuration changes decisions without any other code change.
- **Integration:** `docker compose up` starts the app and PostgreSQL. All tools are exercised with the MCP Inspector over Streamable HTTP, including a full turn sequence: plan, reply, click, next plan.
- **Manual check:** add the connector on claude.ai, then confirm on web and on the AI mobile app:
  - in widget mode, the reply renders correctly (streaming, Markdown, code, copy), the buttons appear under it, and AI writes nothing outside `show_reply`;
  - on a host without MCP Apps, fallback mode gives normal chat replies with no tool call after them.

## 13. Out of scope for the MVP

Prompting tips ("help me ask better"), suggested follow-up questions (rejected: they cost output tokens on every reply and push engagement), a quick-start questionnaire (preferences are learned fast enough), several profiles per user, a public benchmark of how well each AI listens, topic-exploration buttons, collaborative filtering across users, OAuth, a server-side classifier of `user_quote` as an extra annotator, locale catalogs beyond English and French, stages 2 and 3 of section 5.3, and the replay-evaluation script (needed before the first model switch, not at launch; the MVP logs everything it requires).

**Planned: a timing tool for fallback mode.** Fallback-mode lengths are less precise than widget-mode ones, which works against the purpose of the product. Add it back as soon as a host without MCP Apps matters:

- `show_feedback(turn_id)`, a plain tool **not bound to the UI** (no `_meta.ui`), so it can never render a second widget. The `instruction` asks for it only in fallback mode ("Then call show_feedback."); in widget mode, `show_reply` already records the same time.
- Its arrival time restores, for fallback turns: the writing-time sensor (section 6.2), hence three independent length sensors and the **three-cornered hat** estimate of the self-report noise (Var(sᵢ − sⱼ) = σᵢ² + σⱼ² for the three pairs); the writing-speed evidence of the family posterior (section 6.3); and a measured, rather than estimated, start of the reading time (section 6.6).
- Its arrival is stored as a `replies` row with no word count. A second presentation call for the same turn is ignored: the first `replies` row wins.

## 14. Build order

1. Docker Compose (app + PostgreSQL with pgvector), SQLAlchemy models and Alembic migrations, health endpoint.
2. Pure-Python model core with unit tests and synthetic-data recovery tests:
   1. sensor fusion and sensor-noise estimates;
   2. behaviour classes and their posterior;
   3. Dawid–Skene;
   4. reading mixture;
   5. multilevel θ grids;
   6. decision rule.
3. `plan_reply` as an `MCPServer` tool mounted in FastAPI, tested with the MCP Inspector.
4. `show_reply` with its MCP App (Markdown reply, streamed when the host allows, and buttons from `structuredContent`), per-call mode selection, `record_click`, and `gold_label` with its debug page.
5. Nightly job: EM refits, then replay to rebuild derived state. Then the `/metrics` endpoint.
6. Form preferences, learned topic domains, clarification, localized sentence buttons, platform context, the profile page and the comparison arms.
7. Deploy behind HTTPS, add the connector on claude.ai, paste the account instruction, and test on mobile.
