-- MCP transport session of the plan_reply call (2025-11-25 clients only). Observed for diagnosis, never used as a key.
ALTER TABLE turns ADD COLUMN transport_session text;
