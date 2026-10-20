# Claire — design doc

Status: built · Updated: 2026-10-20

**Audience:** anyone changing Claire. **Covers:** what Claire optimises, how each axis is measured and acted on, what is stored, and the rules every change keeps. **Does not cover:** formulas and constants (docstrings in `app/model/`), running the app (`README.md`), file layout (`TREE.md`).

## Summary

Claire sits between a person and their assistant (Claude, ChatGPT) as an MCP connector. Before every reply it tells the assistant how to answer this person: length, form, whether to ask first, which blocking point to raise. It learns from how the person reacts.

One objective: **the person reaches their goal with the fewest reply words.** Two counter-metrics keep it honest: repairs (the reply had to be fixed) and questions asked (asking must not become free).

This revision changes three things compared with the current system: reply text and some user messages are stored, hallucination becomes a first-class axis, and the buttons are removed.

## Context

- Assistants answer too long, in the wrong form, sometimes wrongly, and a static custom instruction ("Be concise") treats every person and request alike.
- MCP lets a server speak to the assistant at every turn; MCP Apps let it render the reply.
- **Observed: nobody clicks the buttons under the reply.** Every signal built on clicks is dead weight.
- **Observed: people readily answer multiple-choice questions** (pick a number, or type a precision).
- Every new user is a cold start: the model must be useful from the first message and adapt within a few reactions.

## Goals

1. Minimise **reply words per goal reached**, without raising repairs or questions per goal.
2. Fit length and form (prose, list, table, or a mix) to the person, the topic and the device, precisely.
3. Detect hallucinations, from the person's corrections.
4. Scope ambiguous requests with a short multiple-choice question, only when needed.
5. Raise the expertise of each chat: the one blocking point the person had not seen, with the field's reference works that develop it.
6. Keep an anonymised corpus of conversations, so a new model can be tested on real exchanges.
7. Setup stays one connector URL and one pasted sentence; no account.

## Non-goals

- Optimising clicks, engagement or time in chat.
- Knowing which model answers: the server measures how the assistant behaves.
- Accounts, a shared profile across apps, a profile page.
- Trivia or "you may also like" content: an expertise point appears only if it blocks the goal.
- Checking facts during the turn: verification is offline and shapes later turns.
- Importing users' chat exports: asking people to upload their whole history is not acceptable to them.

## KPIs

| Role | KPI | Definition |
| --- | --- | --- |
| Objective | **Words per goal reached** | Reply words in a chat, plus the tool's own token cost, divided by the goals reached in it |
| Counter-power | **Repair rate** | Share of turns whose reply had to be fixed: `too_long`, `too_short`, `missing_info`, `misunderstood`, `wrong_format`, `wrong_fact`, or a rephrased question |
| Counter-power | **Questions per goal** | Clarifying questions asked per goal reached |

Every other figure is a diagnostic of one axis below. Every KPI comes with a 95 % bootstrap interval. A **permanent holdout** (3 % of conversations, the assistant's normal behaviour) runs from day one: it is the only baseline that cannot be rebuilt later, and the only way to label a saving **measured** rather than estimated. It is drawn per conversation while there are fewer than 100 active people, then per person: a person who learns brevity in their Claire chats keeps it in their holdout chats, which makes a per-conversation comparison underestimate the gain once there is enough traffic to afford a per-person one.

**Ground truth from the person.** On a small sample of goals marked reached (10 %), the widget asks one multiple-choice question: "Did you get what you needed? 1. Yes 2. Partly 3. No". The answer stays out of the chat. It calibrates every other signal (an annotator of the nightly refit and replay), and counts in questions per goal.

**Goals and end of chat.** The assistant sends `goal` (max 10 words) on the first message and whenever the person starts a new goal, and `goal_reached` when the person shows it is met. A chat **ends** after T days without a call, T learned as the gap after which chats are almost never reopened (about 7 days today). A goal still open at the end counts as not reached. The KPIs are reported on ended chats, and on all chats for a live view.

## Design

### The turn loop

```
user message
  └─ assistant calls plan_reply(verdict on its previous reply + quote, goal, facts on the request,
                                and the user's message when the server asked for it)
       server: closes the previous turn, updates the person's model, decides
       returns: conversation_id, turn_id, instruction
  └─ widget mode: assistant writes the reply only in show_reply(text)
       server: stores the text and its exact word count
       widget: renders the reply; a multiple-choice question when the turn asks one;
               a small privacy link
  └─ fallback mode (host without MCP Apps): plain chat reply; the reply itself is not stored
```

Mode is decided per call from the host's declared capabilities. State across calls travels in server-minted handles (`conversation_id`, `turn_id`), never in the MCP session.

### Length

The exact word count of every widget-mode reply is stored with the reply. The hidden quantity is the shortest sufficient length, as a ratio of the assistant's `needed_words`. "Too long at 300" says only *below 300*, so the likelihood is censored. A Bayesian regression on learned features (topic domain, phone) with a population prior; Thompson sampling picks the length with an 80 % chance of being enough. The requested length is corrected for how much this assistant over- or under-writes.

Diagnostics: target, actual, and their gap per assistant behaviour class.

### Form

Values: prose, list, table, and their mixes (prose with one short list, prose with one table); plus level, examples, tone, headings. Learned per (person, topic domain, device), since a table reads badly on a phone, shrunk toward the person's overall preference and the population's. Thompson draws per dimension. Signals: `wrong_format` with its `format_issue`, and the implicit evidence of turns that moved on.

Diagnostic: `wrong_format` rate, per value and per device.

### Ambiguity

The assistant sends `ambiguity`, its confidence (0 to 1) that the request has several readings. Assistants are poorly calibrated, so the server recalibrates it against what followed: a rephrased question or a `misunderstood` verdict. When the calibrated ambiguity reaches 0.5, a bandit per (person, topic domain, device) decides between asking and answering. Asking means one multiple-choice question: numbered options plus a free field, in the widget (a numbered list in fallback mode). The answer goes back into the chat as the person's message.

**Rephrasing is measured, not asked.** Two consecutive topics closer than a learned distance count as a rephrase. An edited or regenerated message is detected through `last_turn_id`, the `turn_id` plan_reply last returned in this chat: when it is not the conversation's latest turn, the person went back, and the verdict goes to the turn it names. The learned distance is the one that best tells the turns judged `misunderstood`; a rephrase is an annotator of the reaction model.

Diagnostic: calibration curve of `ambiguity`, and repairs after asking vs after answering.

### Hallucination

The person is the signal: a verdict, `wrong_fact`, with the person's quote; on such turns the assistant copies the person's message, so the correction is stored. It catches the errors the person can see, not plausible ones, and it fires falsely when the assistant gives in to a person who is wrong. A person's correction settles the point for their own chat, where it stays in the assistant's context, never for others.

A check against reference works (claims of stored replies, judged offline against public summaries, a trust learned per source, cautions to other people) comes later: see Open questions.

Diagnostic: `wrong_fact` rate.

### Expertise

On each turn, a bandit per (person, domain, device) decides whether the instruction asks for **one** point the person had not raised that blocks their goal, naming the 2 or 3 reference works of the field that develop it, from the assistant's own knowledge: it leads the person towards the field's best ideas. Never a curiosity. The length counter-power keeps it short: a point that costs words without moving the goal shows up in words per goal reached.

Diagnostic: moved-on rate (a raised point taken up) and goals reached, with vs without a raised point.

### Cost and compliance of tool fields

Every field the assistant sends costs output tokens and is one more thing it may forget. Each field is treated as a launch that must pay for itself:

- **Budget.** `plan_reply` arguments stay under 60 output tokens on a normal turn; costly fields (the user's message) are asked only when the instruction requests them. The objective already counts the tool's cost, so a field that adds tokens without helping lowers it.
- **Ablation.** In 10 % of conversations, one optional field (`ambiguity`, `last_turn_id`, `messages_since_last_call`, `user_message_words`) is never asked: the instruction says so, and the server ignores it if sent. The randomised comparison measures what each field is worth on the objective against its cost. A field that does not pay is dropped. Share of turns over the budget is a diagnostic.
- **Compliance per field.** For each field and each assistant behaviour class: how often it is sent when it is due (`goal` on the first turn, `user_message` and `choices` when asked, `last_turn_id` and `messages_since_last_call` on later turns); invalid values are refused by the schema. Where a class sends what the instruction asks less than 20 % of the time, its instruction stops asking for it.
- **Instruction changes are launches.** A new prompt catalog (`locales/candidate.json`) first runs on scripted conversations against the main assistants (`python -m app.evals.instruction`, Claude and ChatGPT through their APIs), then on a share of live conversations, compared with the current catalog on the KPIs.

### Learning core

Everything the assistant reports and everything the server measures is a noisy measurement of a hidden truth, with a noise model learned from the data. Online, closed-form updates at every event; nightly, refits of noise models and priors, then a replay of all events rebuilds the derived state. The reaction of each turn fuses the assistant's verdict, reading time, rephrasing, and, from the nightly replay on, the ground-truth answer, each with a learned reliability (Dawid–Skene). The assistant's ambiguity is Platt-scaled nightly against misunderstood replies and rephrased requests.

## Data and privacy

**Stored text:**

- every widget-mode reply, with its word count;
- the person's message, copied by the assistant only when the instruction asks for it: the first message of a chat (it states the goal), repair turns (they carry the correction), and a random 10 % of conversations, decided by the server. Only messages up to 150 words are copied; longer ones (pastes, documents) are stored as their length only;
- `goal`, `topic`, `user_quote`, and the raw payload of each call.

When a message is stored, the server computes its language, length and embedding itself instead of asking for them.

**Rules:**

- Consent: the consent page states that Claire stores conversations, encrypted, for 3 years, and that they may hold sensitive data (health, opinions); Continue accepts it, and the version of the wording accepted is recorded. The page and the policy state the service is for adults.
- Retention: 3 years for text, then deleted; aggregates and learned parameters are kept.
- Encryption: every text a person or a reply wrote is encrypted by the application (AES-256-GCM, `DATA_KEY` from the environment): replies, messages, topics, quotes, goals, raw payloads, cluster names. A database dump or backup reveals none of them. Only the app and the nightly job hold the key; the dashboard decrypts them behind the admin token.
- Breach: the supervisory authority is notified within 72 hours (GDPR article 33), and the people concerned when the risk is high.
- Deletion: a small "My data" link in the widget opens an email to the owner holding the person's code, an HMAC of their internal number: unforgeable, and it names no one, so the request needs no other proof of identity. The owner pastes it on the dashboard, which erases everything about the person (turns, texts, learned state) and revokes their connector; the request is recorded.
- A privacy policy page states all of the above.
- Text never appears in logs, error reports or traces. Nothing names a person.

**Storage.** PostgreSQL with pgvector. Events are append-only, keep their raw payload, and are the only source of truth; derived state is rebuilt from them. The only deletions are retention and deletion requests.

### What we collect

The rule: what can be recomputed later need not be stored; what is not recorded when it happens is lost for good.

| Lost if not recorded now | Recomputable later | Deliberately not collected |
| --- | --- | --- |
| The propensity of **every** randomised decision (length, form, clarification, expertise point, holdout, ablated field, catalog), for offline evaluation of a new workflow | Retention cohorts, durations, lengths, embeddings, reaction posteriors | User messages outside the sample: a token-cost choice that limits what can be retrained |
| The exact instruction sent, and the catalog and policy versions behind it | Anything derived from stored text and raw payloads, by replay | Attachments and images |
| `messages_since_last_call`: messages the assistant answered without calling the tool, for true coverage and correct attribution of verdicts | | The assistant's model name |
| Widget events: reply visible on screen (seconds), "copy reply", choice question shown and answer (the option, or a free answer, which reaches the chat), ground-truth answer, privacy link, render errors, cancellations | | |
| Host context once per chat: host and version, declared capabilities, platform, locale, time zone when given | | |
| The version of every model that produces derived data (embedding, length policy) | | |
| Acquisition funnel: referrer and UTM on the landing and consent pages, connector registered, consent given, first call, account instruction pasted (calls on every turn) | | |
| Churn: last token refresh of each connector | | |
| Legal record: consent version and date, deletion requests, retention purges | | |

## Invariants

A change that needs to break one starts by changing this doc.

- **Token cost.** Never ask the assistant for what the server can compute; ask for costly fields only when the server needs them. Tool results are compact and stable, for the prompt cache.
- **Privacy.** Text only in the tables above, never in logs. No one's text reaches another person.
- **Compatibility.** Hosts cache tool schemas and the widget: a tool field is never renamed or removed; old widget addresses still serve.
- **Security.** Every handle is checked against the person behind the token; tokens are stored hashed and never logged.
- **Replay.** Online state equals a replay of the events, up to the nightly refit. A turn never explains itself.
- **One source of truth.** Identifiers are enums (`app/core/vocab.py`); every text a person or the model reads lives in `locales/`.
- **Quality.** Scope is cut by dropping features, never by a weaker technique for the ones kept.

## Changes from the current system

No live data needs keeping: the code is rewritten to this design and redeployed from an empty database. The compatibility invariant applies from that deploy on.

| Change | Why |
| --- | --- |
| Remove the buttons, the button bandit and the frequent follow-up questions offered through them | Nobody clicks |
| Store reply text and selected user messages | Experiments with new models, and the later reference check |
| Objective: words per goal reached, instead of per step forward | A chat serves goals, not turns |
| New fields: `goal`, `goal_reached`, `ambiguity` (replaces `ambiguous`), `last_turn_id`, `user_message`, `messages_since_last_call` | Goals, calibrated ambiguity, edit detection, stored messages, coverage |
| New verdict: `wrong_fact` | The person's hallucination signal |
| Clarification becomes a multiple-choice question | People answer them |
| A permanent 3 % holdout instead of arms off by default | A baseline cannot be rebuilt later |
| Propensities, instruction versions, widget events, funnel and legal records | Collected now or lost (see What we collect) |

## Alternatives considered

| Alternative | Why not |
| --- | --- |
| A static "Be concise" instruction | Same for everyone; kept as a comparison arm the product must beat |
| Copy every user message | Output tokens, the costliest kind, on every turn, and huge pastes |
| Import users' chat exports | Asking for a whole history repels people |
| Fact-check during the turn | Latency on every reply; the error is already delivered when found |
| One person's correction propagated as a fact | Opens the door to poisoning |
| Classifier or point regression on reactions | Feedback is a bound; a point invents or discards information |
| Full RL (PPO, DQN) | Each decision barely affects later ones: a contextual bandit fits, with far less data |
| Accounts | Friction; anonymous OAuth now, accounts can attach to the same users later |

## Open questions

- **Reference check, later.** Check stored replies offline against the field's reference works: sentences as claims, judged once each by a local multilingual NLI model against public summaries (Wikipedia, Grokipedia), a trust learned per source (Knowledge-Based Trust, Google, 2015), strong contradictions arbitrated by the owner, and a contested claim reaching other people only as a caution once independent people confirm it (the bridging rule of Community Notes), named by the reference concept, never by someone's sentence. No paid model.
- Should form be decided per part of a reply (prose then a table), rather than once per reply?
- T (end of chat) per person, or one population value?
- Fallback mode stores no reply: acceptable while widget hosts dominate?
