/**
 * MCP App for claire_show_reply (claire og wire shape).
 *
 * Args: flat {turn_id, text} — the written reply.
 * Result: stop signal only (no essay in structuredContent).
 */
import { App } from "@modelcontextprotocol/ext-apps";
import DOMPurify from "dompurify";
import { marked } from "marked";

const replyEl = document.getElementById("reply");
const errEl = document.getElementById("err");
const app = new App({ name: "claire", version: "0.1.0" });

let turnId = null;
let viewed = false;
let text = "";

async function record(kind, value = null) {
  if (!turnId) return;
  try {
    await app.callServerTool({
      name: "claire_record_event",
      arguments: { turn_id: turnId, kind, value },
    });
  } catch {
    /* reporting must never break the reply */
  }
}

function render() {
  try {
    replyEl.innerHTML = DOMPurify.sanitize(marked.parse(text || ""));
  } catch (e) {
    errEl.hidden = false;
    errEl.textContent = "Could not render reply.";
    record("render_error", String(e?.message || e));
  }
  if (turnId && !viewed && text) {
    viewed = true;
    record("reply_view", 1);
  }
}

/** Flat args preferred; unwrap legacy ``{ show: {…} }`` / ``parts``. */
function applyArgs(raw) {
  const args = raw?.arguments || raw || {};
  const body = args.show || args;
  turnId = body.turn_id || turnId;
  if (typeof body.text === "string") {
    text = body.text;
    render();
    return;
  }
  if (Array.isArray(body.parts)) {
    text = body.parts
      .map((p) => p?.text || "")
      .filter(Boolean)
      .join("\n\n");
    render();
  }
}

app.ontoolinput = (params) => applyArgs(params);
app.ontoolinputpartial = (params) => applyArgs(params);

app.ontoolresult = (result) => {
  if (result?.isError) {
    text = "";
    turnId = null;
    render();
    return;
  }
  const sc = result?.structuredContent || {};
  turnId = sc.turn_id || turnId;
};

app.onerror = (err) => {
  errEl.hidden = false;
  errEl.textContent = "Widget error.";
  record("render_error", String(err?.message || err));
};

app.connect().catch((err) => {
  errEl.hidden = false;
  errEl.textContent = "Could not connect widget.";
  record("render_error", String(err?.message || err));
});
