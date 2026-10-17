-- Events: append-only, enforced by the trigger at the end of this file.

CREATE TABLE users (
    id bigserial PRIMARY KEY,
    token_hash text NOT NULL UNIQUE,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE conversations (
    id bigserial PRIMARY KEY,
    user_id bigint NOT NULL REFERENCES users(id),
    grp text NOT NULL CHECK (grp IN ('treatment', 'control')),
    host text NOT NULL,
    ui_supported boolean NOT NULL,
    assistant_model text,
    protocol_version text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE turns (
    id bigserial PRIMARY KEY,
    user_id bigint NOT NULL REFERENCES users(id),
    conversation_id bigint NOT NULL REFERENCES conversations(id),
    turn_index int NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    reply_mode text NOT NULL CHECK (reply_mode IN ('widget', 'fallback')),
    model_version text NOT NULL,
    schema_version int NOT NULL,
    raw_payload jsonb NOT NULL,
    request_type text NOT NULL,
    topic text NOT NULL,
    needed_words int NOT NULL,
    language text NOT NULL,
    user_message_words int,
    pasted_content text,
    user_specified text[],
    assistant_model text,
    target_words int,
    requested_words int,
    propensity double precision,
    theta_sample double precision,
    b_sample double precision,
    post_mean double precision,
    post_sd double precision,
    style jsonb NOT NULL DEFAULT '[]'
);
CREATE INDEX turns_conversation ON turns (conversation_id, turn_index);

CREATE TABLE replies (
    turn_id bigint NOT NULL REFERENCES turns(id),
    tool text NOT NULL CHECK (tool IN ('show_reply', 'show_feedback')),
    reply_words_exact int,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX replies_turn ON replies (turn_id);

CREATE TABLE outcomes (
    turn_id bigint NOT NULL REFERENCES turns(id),
    source text NOT NULL CHECK (source IN ('assistant', 'button', 'reading')),
    observed text NOT NULL,
    user_quote text,
    last_reply_words int,
    reading_seconds double precision,
    model_version text NOT NULL,
    schema_version int NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX outcomes_turn ON outcomes (turn_id);

CREATE TABLE buttons_shown (
    turn_id bigint NOT NULL REFERENCES turns(id),
    button text NOT NULL,
    position int NOT NULL,
    shuffled boolean NOT NULL,
    thompson_draw double precision
);
CREATE INDEX buttons_shown_turn ON buttons_shown (turn_id);

CREATE TABLE clicks (
    turn_id bigint NOT NULL REFERENCES turns(id),
    button text NOT NULL,
    position int NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX clicks_turn ON clicks (turn_id);

CREATE TABLE gold_labels (
    turn_id bigint NOT NULL REFERENCES turns(id),
    outcome text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE shadow_decisions (
    turn_id bigint NOT NULL REFERENCES turns(id),
    model_version text NOT NULL,
    target_words int,
    propensity double precision,
    details jsonb NOT NULL DEFAULT '{}'
);

CREATE TABLE tool_calls (
    user_id bigint NOT NULL REFERENCES users(id),
    tool text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE FUNCTION forbid_change() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'table % is append-only', TG_TABLE_NAME;
END $$;

DO $$
DECLARE t text;
BEGIN
    FOREACH t IN ARRAY ARRAY['users', 'conversations', 'turns', 'replies', 'outcomes', 'buttons_shown',
                             'clicks', 'gold_labels', 'shadow_decisions', 'tool_calls'] LOOP
        EXECUTE format('CREATE TRIGGER %I_append_only BEFORE UPDATE OR DELETE OR TRUNCATE ON %I '
                       'FOR EACH STATEMENT EXECUTE FUNCTION forbid_change()', t, t);
    END LOOP;
END $$;

-- Derived state: rebuilt nightly by replaying events; never a source of truth.

CREATE TABLE turn_inference (
    turn_id bigint PRIMARY KEY,
    length_mean double precision,
    length_var double precision,
    reaction_probs jsonb,
    reading_probs jsonb
);

CREATE TABLE conversation_state (
    conversation_id bigint PRIMARY KEY,
    family_probs jsonb NOT NULL,
    kappa_mean double precision NOT NULL,
    kappa_var double precision NOT NULL,
    speed_mean double precision NOT NULL,
    speed_var double precision NOT NULL
);

CREATE TABLE theta_posteriors (
    user_id bigint NOT NULL,
    request_type text NOT NULL,
    grid double precision[] NOT NULL,
    PRIMARY KEY (user_id, request_type)
);

CREATE TABLE alpha_posteriors (
    user_id bigint PRIMARY KEY,
    grid double precision[] NOT NULL
);

CREATE TABLE user_reading (
    user_id bigint PRIMARY KEY,
    rho_mean double precision NOT NULL,
    rho_var double precision NOT NULL,
    mixture jsonb NOT NULL,
    reaction_prior jsonb NOT NULL
);

CREATE TABLE beta_counts (
    user_id bigint NOT NULL,
    request_type text NOT NULL,
    key text NOT NULL,
    alpha double precision NOT NULL,
    beta double precision NOT NULL,
    PRIMARY KEY (user_id, request_type, key)
);

CREATE TABLE family_params (
    model_family text PRIMARY KEY,
    b double precision NOT NULL,
    w double precision NOT NULL,
    kappa double precision NOT NULL,
    speed_mu double precision NOT NULL,
    speed_sd double precision NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE sensor_noise (
    sensor text NOT NULL,
    model_family text NOT NULL,
    sigma double precision NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (sensor, model_family)
);

CREATE TABLE annotator_confusion (
    annotator text NOT NULL,
    stratum text NOT NULL,
    true_outcome text NOT NULL,
    observed text NOT NULL,
    prob double precision NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (annotator, stratum, true_outcome, observed)
);

CREATE TABLE self_report_reliability (
    reported_family text NOT NULL,
    true_family text NOT NULL,
    prob double precision NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (reported_family, true_family)
);

CREATE TABLE host_family_prior (
    host text NOT NULL,
    model_family text NOT NULL,
    prob double precision NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (host, model_family)
);

CREATE TABLE hyperparams (
    name text PRIMARY KEY,
    value double precision NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);
