/**
 * MCP App for claire_show_reply: display parts and report view / errors.
 */
import { App } from "@modelcontextprotocol/ext-apps";

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
    replyEl.textContent = text || "";
  } catch (e) {
    errEl.hidden = false;
    errEl.textContent = "Could not render reply.";
    record("render_error", String(e?.message || e));
  }
}

function fromResult(result) {
  const sc = result?.structuredContent || {};
  turnId = sc.turn_id || turnId;
  const parts = sc.parts || [];
  const text = parts.map((p) => p?.text || "").filter(Boolean).join("\n\n");
  render(text);
  if (turnId && !viewed) {
    viewed = true;
    record("reply_view", 1);
  }
}

app.ontoolresult = fromResult;
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
