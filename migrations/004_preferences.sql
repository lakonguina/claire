-- Form preferences, learned topic domains, comparison arms, platform context
-- and profile edits (SPEC sections 4, 7 and 9).

CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE conversations DROP CONSTRAINT conversations_grp_check;
ALTER TABLE conversations ADD CONSTRAINT conversations_grp_check
    CHECK (grp IN ('treatment', 'control', 'concise'));

-- pasted_content and user_specified are no longer sent; old rows keep them.
ALTER TABLE outcomes ADD COLUMN format_issue text;

CREATE TABLE topic_embeddings (
    turn_id bigint NOT NULL REFERENCES turns(id),
    model text NOT NULL,
    embedding vector(384) NOT NULL
);
CREATE INDEX topic_embeddings_turn ON topic_embeddings (turn_id);

CREATE TABLE turn_context (
    turn_id bigint NOT NULL REFERENCES turns(id),
    platform text,
    locale text,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX turn_context_turn ON turn_context (turn_id);

CREATE TABLE profile_edits (
    user_id bigint NOT NULL REFERENCES users(id),
    scope text NOT NULL,
    dimension text NOT NULL,
    value text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

DO $$
DECLARE t text;
BEGIN
    FOREACH t IN ARRAY ARRAY['turn_context', 'profile_edits', 'topic_embeddings'] LOOP
        EXECUTE format('CREATE TRIGGER %I_append_only BEFORE UPDATE OR DELETE OR TRUNCATE ON %I '
                       'FOR EACH STATEMENT EXECUTE FUNCTION forbid_change()', t, t);
    END LOOP;
END $$;

-- Derived: replaces beta_counts. User 0 is the population, scope '*' the user.
DROP TABLE beta_counts;
CREATE TABLE pref_counts (
    user_id bigint NOT NULL,
    scope text NOT NULL,
    dimension text NOT NULL,
    value text NOT NULL,
    count double precision NOT NULL,
    PRIMARY KEY (user_id, scope, dimension, value)
);

-- Derived (SPEC 7.1): refit nightly, ids kept stable across refits.
CREATE TABLE topic_clusters (
    id bigint PRIMARY KEY,
    centroid vector(384) NOT NULL,
    medoid_topic text NOT NULL,
    size int NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);
