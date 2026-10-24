/**
 * MCP App for claire_show_reply.
 *
 * Contract: args = human reply (ontoolinput / partial); result = model stop
 * signal; optional _meta.claire.parts = widget-only mirror if input was skipped.
 * Never read the essay from structuredContent (model-visible; hosts may echo).
 */
import { App } from "@modelcontextprotocol/ext-apps";
import DOMPurify from "dompurify";
import { marked } from "marked";

const replyEl = document.getElementById("reply");
const errEl = document.getElementById("err");
const app = new App({ name: "claire", version: "0.1.0" });

let turnId = null;
let viewed = false;

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

function render(text) {
  try {
    replyEl.innerHTML = DOMPurify.sanitize(marked.parse(text || ""));
  } catch (e) {
    errEl.hidden = false;
    errEl.textContent = "Could not render reply.";
    record("render_error", String(e?.message || e));
  }
}

function showParts(parts) {
  const text = (parts || [])
    .map((p) => p?.text || "")
    .filter(Boolean)
    .join("\n\n");
  render(text);
  if (turnId && !viewed && text) {
    viewed = true;
    record("reply_view", 1);
  }
}

/** Host streams / delivers the tool call arguments (the written reply). */
app.ontoolinput = (params) => {
  const args = params?.arguments || params || {};
  turnId = args.turn_id || turnId;
  showParts(args.parts);
};

app.ontoolinputpartial = (params) => {
  const args = params?.arguments || params || {};
  turnId = args.turn_id || turnId;
  if (args.parts) showParts(args.parts);
};

/** Ack only in structuredContent; parts may arrive under _meta.claire. */
app.ontoolresult = (result) => {
  const sc = result?.structuredContent || {};
  const meta = result?._meta || result?.meta || {};
  turnId = sc.turn_id || turnId;
  const parts = meta?.claire?.parts;
  if (parts?.length) showParts(parts);
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
