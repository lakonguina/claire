# Lingua Franca — MCP server for shorter, better-targeted AI replies accross any chatbot (ChatGPT/AI/Gemini..) (MVP)

## 1. Purpose

Make AI Chat (web, desktop and mobile apps) give each user **what they need to move forward, in as few words as possible**, and learn each user's preferences from their reactions.

Guiding principle: Grice's maxim of quantity — say everything that is needed, and nothing more.

**North-star metric: words spent per step forward.** We minimise reply words *under the constraint* that the user moves forward (no corrections, no missing information).

- Neither "fewest tokens" alone (an empty reply is shortest) nor "user satisfaction" alone (flattery, padding) is the goal.
- Button clicks are a **learning signal only**. Never optimise for click-through or engagement.

## 2. How it works

The server never stores the conversation. It receives the structured fields AI sends in tool calls and, in widget mode, the reply text, which it counts and discards at once. AI does the understanding; the server does memory and learning.

An instruction in the user's AI account (section 10) makes AI:

1. Call `plan_reply` **before** every reply. AI sends (a) its verdict on how the user reacted to its previous reply, with a short quote as proof, and (b) facts about the new request.
2. The server records the outcome for the previous turn, updates the user's model instantly, picks a target length and style, and returns a short instruction. The instruction also sets the **reply mode** for this turn.
3. **Widget mode (main mode):** AI writes its reply **only** as the `text` argument of `show_reply`, and nothing in the chat. The MCP App renders the reply (as it streams, when the host streams tool input), with 3 buttons underneath; the buttons arrive in the tool result's `structuredContent`, which the host gives to the app but not to the model. The server counts the words exactly, then discards the text.
4. **Fallback mode** (host without MCP Apps support): AI replies normally in the chat and calls no tool after it. Such a host cannot render buttons, so there are none.
5. If the user clicks a button, the UI records the click and sends the matching follow-up message into the chat. Users who type instead are covered by step 1 on the next message.

Widget mode costs no extra tokens (the reply is written once, as a tool argument instead of chat text) and gives an exact length. The reply stays in the conversation history as a tool call, so the assistant keeps full access to what it wrote.

## 3. Stack and constraints

Mandatory stack:

- **Docker** (Docker Compose): one service for the app, one for PostgreSQL.
- **PostgreSQL** for all storage.
- **FastAPI** for the HTTP API (internal endpoints, health, metrics, debug page).
- The official **MCP Python SDK** (`mcp` 2.x, `MCPServer`) for the MCP tools, served over Streamable HTTP and mounted inside the FastAPI app. Its built-in MCP Apps extension (`mcp.server.apps`) provides the UI resources. (fastapi_mcp was dropped: its last release, 0.4.0, does not run on `mcp` 2.x.)

Rules:

- Python: do not add libraries beyond these, a PostgreSQL driver (`psycopg` 3 and its pool), the ASGI server (uvicorn) and pytest. The models in sections 5 and 6 are implemented in **pure Python** with the standard library (`math`, `random`, `statistics.NormalDist` for the normal CDF and quantile; small Gaussian updates written by hand). No numpy, scipy or ML framework.
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
- Keep the code focused: target about 1,500 lines of Python excluding tests, plus about 150 lines for the widget.

## 4. Tools

### 4.1 `plan_reply` (MCP tool, called by AI before every reply)

Input:

| Field | Type | Meaning |
| --- | --- | --- |
| `conversation_id` | string, optional | Omit on the first message of a conversation; reuse the value returned by the server afterwards |
| `request_type` | enum: `fact`, `explanation`, `advice`, `decision`, `code`, `writing`, `plan` | Kind of request |
| `topic` | string, max 15 words, in the user's language | What the request is about |
| `needed_words` | int | AI's estimate of the words a complete, no-filler answer needs |
| `last_reply_outcome` | enum (below) | How the user reacted to AI's previous reply in this conversation |
| `user_quote` | string, max 5 words | The user's exact words proving the outcome |
| `last_reply_words` | int, fallback mode only | Approximate length of AI's previous reply (not needed in widget mode, where the server counts exactly) |
| `user_message_words` | int | Length of the user's message |
| `pasted_content` | enum: `code`, `data`, `text`, `file`; omit when nothing was pasted | What the user pasted or attached |
| `user_specified` | list of enum: `audience`, `goal`, `format`, `length`, `context`; omit when empty | What the user's message states explicitly (facts, not opinions) |
| `assistant_model` | string, first call of a conversation only | The model answering (e.g. `claude-opus-5-5`, `gpt-6-astra`), if the assistant knows it |

The last four fields are not used by the MVP model. They are logged from day one because later models (section 5.3) and later heads (clarification, tips) will need them, and a field that is not logged today will never exist in the history.

**Token economy rule:** every field AI sends costs output tokens, the most expensive and slowest kind. Anything the server can compute itself is never asked from AI:

- the server computes `turn_index`, timestamps, durations, reading time and the control group;
- the server stores `assistant_model` per conversation after the first call;
- the server fills defaults for omitted optional fields;
- the server detects the language from `topic`, which AI must write in the user's language. Use a small pure-Python stopword heuristic for `fr`, `en`, `es`, `de`, `it` and `pt`, and fall back to `unknown`. Store the result as `language`.

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
  "instruction": "Reply in about 75 words. No preamble, no recap. Use simple words. Write the reply only in show_reply."
}
```

In widget mode, `instruction` ends with "Write the reply only in show_reply."; in fallback mode it has no tool sentence, and the assistant replies in the chat. **Mode selection, per call:** read the client capabilities carried by the current request (from the `initialize` handshake on 2025-11-25, from the request's `_meta` on 2026-07-28; the SDK's `client_supports_apps(ctx)` covers both). If the host declares MCP Apps support (`io.modelcontextprotocol/ui` with the `text/html;profile=mcp-app` MIME type), the turn uses widget mode; otherwise, fallback mode. Store the mode as `reply_mode` on the turn, and the first value as `ui_supported` on the conversation. Confirm on each host that the declaration matches what is actually rendered, and keep a per-host override in configuration.

`target_words` is the length the user should receive. The word count in `instruction` is the **requested** length, corrected for this assistant's compliance (section 6.3), so that the assistant actually writes about `target_words`.

Keep the output short and in exactly the same format on every call: the tool result stays in the conversation, and a stable, compact result keeps AI's prompt cache efficient.

**Control group.** Randomisation unit is set by configuration (`CONTROL_UNIT`):

- `conversation` (default while there are fewer than 100 active users): each new conversation goes to `control` with probability 0.1;
- `user` (from 100 active users): each new user goes to `control` with probability 0.1. This avoids spillover: a user who learns to be concise in treatment conversations is also more concise in their control conversations, which makes conversation-level comparisons underestimate the effect.

In `control`, `plan_reply` still records everything and still updates the models, but returns `target_words: null` and `instruction: "Reply normally."` (followed by the widget-mode sentence when it applies), and `show_reply` returns no buttons. This group is what turns estimated savings into measured savings (section 11).

### 4.2 `show_reply` (MCP tool, widget mode: carries the reply itself)

Input: `turn_id`, `text` (the full reply, in Markdown).

Server side:

- count the words of `text` exactly (split on whitespace, ignoring Markdown syntax characters), store the count as `reply_words_exact` on the turn, then **discard the text immediately** (see section 3);
- record the arrival time: the gap between `plan_reply` and `show_reply` is the writing duration;
- return a minimal result: `content` is `"ok"` (the only part the model sees), and `structuredContent` carries the buttons for the app (`{"buttons": [{"id", "label", "message", "position"}]}`, empty in `control`). `structuredContent` is not added to the model's context, so the buttons cost no model tokens and need no extra round trip.

The MCP App (implemented following the current MCP Apps extension specification, `2026-01-26`) must:

- render the reply from the complete tool input (`ui/notifications/tool-input`), which hosts must send. When the host also sends partial tool input (`ui/notifications/tool-input-partial`, optional for hosts), render it progressively as it streams; never depend on it;
- render Markdown properly: headings, lists, emphasis, links, tables, and code blocks with a copy button; plus a "copy reply" button;
- show the 3 buttons under the reply, in the order chosen by the server (section 8), as soon as the tool result (`ui/notifications/tool-result`) arrives; the reply never waits for them;
- on `ui/notifications/tool-cancelled`, keep whatever was rendered, show no buttons, and record nothing (a turn without a `show_reply` call simply has no exact length);
- on click, record the click (call `record_click` through the host's tool-call capability) and wait for it, then send the button's follow-up message into the conversation with `ui/message`;
- use the host-provided style variables and theme, with no external assets;
- stay readable on a phone-width screen.

In `control` conversations, the widget shows the reply without buttons.

### 4.3 Internal endpoints (not exposed to the model)

- `record_click` with `turn_id`, `button`, `position`. Called by the UI only, through the host's tool-call capability. Exposed as an app-only MCP tool (`_meta.ui.visibility: ["app"]`: hosts must not list it to the model and must reject app calls to tools without `"app"`), with a description saying the model must never call it.
- `POST /gold_label` with `turn_id` and the true `outcome`. Used by the owner to label real turns by hand (a debug page listing recent turns with their `topic`, `user_quote` and inferred outcome). Target a few hundred labels; they anchor the annotator model (section 6.4).

### 4.4 Button catalogue

| Button id | Label | Follow-up message sent | Outcome it reports |
| --- | --- | --- | --- |
| `shorter` | Shorter | "Shorter." | `too_long` |
| `more_detail` | More detail | "More detail." | `too_short` |
| `example` | Example | "Give an example." | style: `example` +1 |
| `simpler` | Simpler | "Simpler, please." | `wrong_format`; style: `simple` +1 |
| `one_line` | One-line summary | "Summarize in one line." | `too_long` |
| `rephrase_request` | Rephrase my request | "Rewrite my last request as a clearer prompt, and tell me in one line what was missing." | `misunderstood`; at most once per conversation |

A click is one annotator among others (section 6.4), with its own learned reliability.

### 4.5 Several conversations, models and hosts

The same user may run many conversations in parallel, switch models (Opus, Sonnet, Haiku…), and use several hosts (AI, and ChatGPT, which also supports MCP connectors).

- **Parallel conversations:** everything is keyed by `conversation_id`, and durations and reading time are computed per conversation. If the assistant forgets the id, the server starts a new conversation and the previous verdict is lost. That is acceptable, but monitored through tool coverage (section 11).
- **One user everywhere:** the token identifies the user across hosts, so one profile serves all of them.
- **Host detection at zero token cost:** read the client name (`clientInfo`, from the `initialize` handshake on 2025-11-25 or from the request's `_meta` on 2026-07-28) and store it as `host` on the conversation, or `unknown` if the client sends none. Never ask the assistant for it.
- **Model identity is never trusted as a label.** Models often do not know their exact version, or name an older one. The model family is a latent variable inferred from behaviour (section 6.3). The self-report `assistant_model` is one noisy clue among others.
- **Hosts with MCP Apps:** Claude (web and desktop) and ChatGPT both support MCP Apps (official extension support matrix), so widget mode is the normal case on both.
- **Hosts without MCP Apps:** fallback mode (section 4.1): normal chat replies, estimated lengths (section 6.2), and no buttons if the host cannot render them, in which case the other annotators carry the reaction model.

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
| 2 — Growth | Up to ~5,000 active users | Hierarchical Bayesian censored **regression**: log ratio = features × β + user effect + user × type effect. Features: request type, log of needed words, turn index, topic cluster. β is refit nightly; user effects are updated instantly (Laplace approximation) | Shares knowledge across users through features; uses the topic |
| 3 — Scale | Beyond that | **Neural-linear Thompson sampling**: a two-tower network (user embedding, context embedding from the topic text and features), trained nightly with the censored loss, with a Bayesian linear head updated instantly per user | Learns which users are alike (collaborative filtering) and the meaning of topics, replacing hard request types with a continuous representation |

Stages 2 and 3 will require adding numerical libraries (and an ML framework for stage 3). That decision is taken when the stage starts, not in the MVP.

Buttons follow the same path: Beta-Bernoulli per (user, type, button) in stage 1, then the same features and representations as the length model in stages 2 and 3.

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
| π(c, f) | conversation | Probability that the assistant in conversation *c* belongs to model family *f* |
| b(c) | conversation | Assistant's bias when estimating `needed_words` |
| κ(c), v(c) | conversation | Compliance (log written / requested) and writing speed (log words per second) |
| ρ(u), mixture weights | user | Reading speed and reading behaviour (section 6.6) |

For a given message, the minimal sufficient log ratio is S = θ(u,t) + b(c) + ε, with ε ~ N(0, τ²). τ starts at 0.35 and is refit nightly.

### 6.2 Reply length

Three sensors measure the true length L of a reply:

1. **Exact count** (widget mode): L = log(`reply_words_exact`), with zero variance.
2. **Writing time** (widget mode): s₂ = log(duration − overhead) = L − v(c) + e₂, where duration is the gap between `plan_reply` and `show_reply`, overhead starts at 3 s (refit nightly), and v(c) is the conversation's log writing speed. With the exact count known, this sensor measures the writing speed, which feeds the family posterior (section 6.3). Use a heavy-tailed error (Student-t, 4 degrees of freedom, applied as a robust down-weighting of large residuals), because extended thinking, network delays and server load inflate some durations.
3. **Requested length:** s₃ = log(requested words) + κ(c) = L + e₃.
4. **Self-report** (fallback mode): s₁ = log(`last_reply_words`) = L + e₁, sent with the next `plan_reply`. It defines the unit (assistant-counted words), so it has no bias term by construction.

In fallback mode no tool call follows the reply, so only the self-report and the requested length are available.

Online, per turn: fuse the available sensors with a small Gaussian update over (L, κ(c), v(c)). This gives a posterior mean and variance for L, and updates the conversation's compliance and speed. Any sensor may be missing.

Nightly, estimate the writing-time and requested-length noise from widget-mode turns, where the exact count is ground truth. Then estimate the self-report noise from fallback turns: the two sensors are independent, so Var(s₁ − s₃) = σ₁² + σ₃², which gives σ₁² from the already known σ₃². If it comes out negative (sampling noise), floor it at 0.01.

### 6.3 Assistant behaviour

What matters is not *which* model answers but *how it behaves*. Behaviour follows a hierarchy: population → model family → conversation.

**Family as a probability, never a hard label.** For each conversation, keep π(c, f) over f ∈ {`claude-opus`, `claude-sonnet`, `claude-haiku`, `gpt`, `other`}:

π(c, f) ∝ P(f | host) × P(self-report | f) × Π over turns of N(v̂; μ_v(f), s_v(f)²)

- `P(f | host)`: share of each family per host, learned.
- `P(self-report | f)`: a **reliability matrix** for self-reports, learned. Example: "when a model says Opus, it is Opus 85 % of the time." A missing self-report contributes 1.
- v̂: the turn's estimated log writing speed from section 6.2.

π(c, ·) is updated after every turn.

**Estimation bias:** b(c) = Σ_f π(c, f) · b(f), with variance Var b(c) = Σ_f π(c, f) · (b(f)² + w(f)) − b(c)². Var b(c) is added to τ² everywhere S appears. An uncertain family therefore widens the posterior and leads to a little more exploration, instead of a confident wrong answer.

**Compliance and speed:** per conversation, Gaussian priors centred on the family mixture, Σ_f π(c, f) · κ(f) and Σ_f π(c, f) · μ_v(f), with sd 0.3 for compliance and the family spread for speed, updated by the sensor fusion of section 6.2.

### 6.4 User reaction: several annotators, Dawid–Skene

The true reaction y of each turn is hidden. Several independent annotators observe it, each with a **confusion matrix** P(observed | true) that is learned, never assumed:

| Annotator | What it reports | Confusion matrix learned per |
| --- | --- | --- |
| Assistant verdict (`last_reply_outcome`) | One of the outcome values | Model family (soft-weighted by π) |
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

## 7. Style

Default instruction for every reply: "No preamble, no recap."

Per (user, request type), keep Beta(1,1) counts for two modifiers: `simple` ("Use simple words.") and `example` ("Include one short example."). Update them from button clicks: `simpler` and `example` add a success; when either button is shown but not clicked, add a failure. Include a modifier in the instruction when its posterior mean is above 0.5 and it has at least 2 signals.

## 8. Buttons policy

- Keep Beta(1,1) counts of click-when-shown per (user, request type, button).
- **Safe button:** the highest posterior mean.
- **Exploration buttons:** draw one sample per remaining button (`random.betavariate`) and take the 2 highest.
- Order: [safe, explore, explore]. With probability 0.2, shuffle randomly instead. Log every shown button with its position and the shuffled flag.
- When the next `plan_reply` arrives in that conversation and no button was clicked, record a "not clicked" for each shown button.
- Position effects are learned through the button annotator's per-position confusion matrix (section 6.4).

## 9. Data model (PostgreSQL)

Manage the schema with a plain SQL migration file applied at container start.

**Events** (append-only: no `UPDATE` or `DELETE`):

- `users(id, token_hash, created_at)`
- `conversations(id, user_id, grp ['treatment'|'control'], host, ui_supported bool, assistant_model, protocol_version, created_at)`
- `turns(id, user_id, conversation_id, turn_index, created_at, reply_mode ['widget'|'fallback'], model_version, schema_version, raw_payload jsonb, request_type, topic, needed_words, language, user_message_words, pasted_content, user_specified text[], assistant_model, target_words, requested_words, propensity, theta_sample, b_sample, post_mean, post_sd, style jsonb)`
- `replies(turn_id, reply_words_exact, created_at)`: one row per `show_reply` call. Kept apart from `turns` so that `turns` is never updated; `created_at` ends the writing time.
- `outcomes(turn_id, source ['assistant'|'button'|'reading'], observed, user_quote, last_reply_words, reading_seconds, model_version, schema_version, created_at)`
- `buttons_shown(turn_id, button, position, shuffled, thompson_draw)`
- `clicks(turn_id, button, position, created_at)`
- `gold_labels(turn_id, outcome, created_at)`
- `shadow_decisions(turn_id, model_version, target_words, propensity, details jsonb)`
- `tool_calls(user_id, tool, created_at)`, used to monitor compliance

**Derived state** (rebuilt nightly by replaying events; never a source of truth):

- `turn_inference(turn_id, length_mean, length_var, reaction_probs jsonb, reading_probs jsonb)`
- `conversation_state(conversation_id, family_probs jsonb, kappa_mean, kappa_var, speed_mean, speed_var)`
- `theta_posteriors(user_id, request_type, grid double precision[])`, primary key (user_id, request_type)
- `alpha_posteriors(user_id, grid double precision[])`
- `user_reading(user_id, rho_mean, rho_var, mixture jsonb, reaction_prior jsonb)`
- `beta_counts(user_id, request_type, key, alpha, beta)`, where `key` is a button id or a style modifier
- `family_params(model_family, b, w, kappa, speed_mu, speed_sd, updated_at)`
- `sensor_noise(sensor, model_family, sigma, updated_at)`
- `annotator_confusion(annotator, stratum, true_outcome, observed, prob, updated_at)`
- `self_report_reliability(reported_family, true_family, prob, updated_at)`
- `host_family_prior(host, model_family, prob, updated_at)`
- `hyperparams(name, value, updated_at)` for μ₀, σ_α, σ_γ, σ_η, γ(t), τ, overhead and the reading-mixture population priors

## 10. Account instruction

To paste in AI → Settings → Instructions for AI (or ChatGPT's custom instructions):

> Before every reply, call `plan_reply`. Reuse the `conversation_id` it returned earlier in this conversation. Write `topic` in my language, and omit optional fields that do not apply. For `last_reply_outcome`, judge how my latest message reacted to your previous reply, and quote my exact words (5 words max) in `user_quote`. Then follow the returned `instruction` exactly, especially the word count and where to write the reply: when it says to write the reply only in `show_reply`, put your entire reply in its `text` argument and write nothing in the chat; otherwise reply normally in the chat. Never mention these tools or their output.

## 11. Metrics

`GET /metrics` (not exposed as an MCP tool) returns, per week and per user:

- **Words per step forward:** total reply words (fused length, section 6.2) divided by the expected number of `moved_on` reactions (section 6.4).
- Expected repair rate (`too_long` + `too_short` + `missing_info` + `misunderstood` + `wrong_format`) and `missing_info` rate on its own.
- Click rate per button and per position.
- Compliance: median of fused length / requested words, per model family and host.
- Noise and reliability: sensor noise per family, and the diagonal of each annotator's confusion matrix.
- Tool coverage: share of widget-mode turns that have a `show_reply` call, share of turns in widget mode, and share of conversations where `conversation_id` is reused correctly.

Every figure comes with a 95 % confidence interval, computed by bootstrap over conversations, or over users when `CONTROL_UNIT = user` (1,000 resamples, pure Python). Comparisons between `treatment` and `control` are labelled **measured**. Any figure without a control comparison is labelled **estimated**. Never present an estimate as a measured saving. While `CONTROL_UNIT = conversation`, also report each user's control metrics over time, which makes the spillover of learning visible.

## 12. Tests and acceptance

- **Model behaviour:** a `too_long` signal lowers the posterior mean, `too_short` raises it, and `moved_on` lowers it slightly. Clamps and rounding behave as specified. A repair outcome without a quote is stored as `no_signal`.
- **Recovery on synthetic data:**
  - the two-sensor estimate recovers a known self-report noise variance within 20 % on 2,000 simulated turns;
  - Dawid–Skene recovers known confusion matrices within 0.05 on 2,000 simulated turns;
  - the family posterior identifies a simulated model whose self-report is wrong 30 % of the time;
  - the reading mixture separates simulated read, skim and away gaps.
- **System:**
  - `show_reply` stores the exact word count and **no trace of the text** anywhere: database (including `raw_payload`), logs or error output;
  - widget mode is selected only when the host declares MCP Apps support, and fallback mode otherwise, for both a `2025-11-25` client (capabilities in `initialize`) and a `2026-07-28` client (capabilities in each request's `_meta`);
  - a `turn_id` or `conversation_id` belonging to another user is rejected, and the token never appears in logs;
  - when a turn has a click, the assistant verdict for that turn does not enter the reaction posterior;
  - about 10 % of new conversations (or users) land in `control`, and those get no length instruction and no buttons;
  - rebuilding all derived state from scratch by replaying the events gives the same state as the online path, up to the nightly refit;
  - swapping the active policy through configuration changes decisions without any other code change.
- **Integration:** `docker compose up` starts the app and PostgreSQL. All tools are exercised with the MCP Inspector over Streamable HTTP, including a full turn sequence: plan, feedback, click, next plan.
- **Manual check:** add the connector on claude.ai, then confirm on web and on the AI mobile app:
  - in widget mode, the reply renders correctly (streaming, Markdown, code, copy), the buttons appear under it, and AI writes nothing outside `show_reply`;
  - on a host without MCP Apps, fallback mode gives normal chat replies with no tool call after them.

## 13. Out of scope for the MVP

"I understood: …" clarification lines, prompting tips, topic-exploration buttons, collaborative filtering across users, OAuth, a server-side classifier of `user_quote` as an extra annotator, localisation of button labels, stages 2 and 3 of section 5.3, and the replay-evaluation script (needed before the first model switch, not at launch; the MVP logs everything it requires).

## 14. Build order

1. Docker Compose (app + PostgreSQL), SQL migration, health endpoint.
2. Pure-Python model core with unit tests and synthetic-data recovery tests:
   1. sensor fusion and sensor-noise estimates;
   2. family posterior;
   3. Dawid–Skene;
   4. reading mixture;
   5. multilevel θ grids;
   6. decision rule.
3. `plan_reply` as an `MCPServer` tool mounted in FastAPI, tested with the MCP Inspector.
4. `show_reply` with its MCP App (Markdown reply, streamed when the host allows, and buttons from `structuredContent`), per-call mode selection, `record_click`, and `gold_label` with its debug page.
5. Nightly job: EM refits, then replay to rebuild derived state. Then the `/metrics` endpoint.
6. Deploy behind HTTPS, add the connector on claude.ai, paste the account instruction, and test on mobile.
