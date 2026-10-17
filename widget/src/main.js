import {
  App,
  applyDocumentTheme,
  applyHostFonts,
  applyHostStyleVariables,
} from "@modelcontextprotocol/ext-apps";
import DOMPurify from "dompurify";
import { marked } from "marked";
import "./style.css";

const replyEl = document.getElementById("reply");
const buttonsEl = document.getElementById("buttons");
const copyReplyEl = document.getElementById("copy-reply");
const app = new App({ name: "lingua-franca", version: "1.0.0" });

let text = "";
let turnId = null;
let pending = false;

function render(final) {
  pending = false;
  replyEl.hidden = !text;
  replyEl.innerHTML = DOMPurify.sanitize(marked.parse(text));
  if (final) {
    for (const pre of replyEl.querySelectorAll("pre")) {
      pre.append(copyButton(() => pre.querySelector("code")?.innerText ?? pre.innerText));
    }
    copyReplyEl.hidden = !text;
  }
}

function scheduleRender() {
  if (!pending) {
    pending = true;
    requestAnimationFrame(() => render(false));
  }
}

async function copy(value) {
  try {
    await navigator.clipboard.writeText(value);
  } catch {
    const area = Object.assign(document.createElement("textarea"), { value });
    document.body.append(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }
}

function onCopy(button, getValue) {
  const label = button.textContent;
  button.onclick = async () => {
    await copy(getValue());
    button.textContent = "Copied ✓";
    button.classList.add("copied");
    setTimeout(() => {
      button.textContent = label;
      button.classList.remove("copied");
    }, 1500);
  };
  return button;
}

function copyButton(getValue) {
  return onCopy(Object.assign(document.createElement("button"), { className: "copy ghost", textContent: "Copy" }), getValue);
}

function renderButtons(buttons) {
  buttonsEl.replaceChildren(
    ...buttons.map((b) => {
      const el = Object.assign(document.createElement("button"), { textContent: b.label });
      el.onclick = () => click(b);
      return el;
    }),
  );
}

async function click(b) {
  for (const el of buttonsEl.querySelectorAll("button")) el.disabled = true;
  try {
    await app.callServerTool({
      name: "record_click",
      arguments: { turn_id: turnId, button: b.id, position: b.position },
    });
  } finally {
    await app.sendMessage({ role: "user", content: [{ type: "text", text: b.message }] });
  }
}

function applyContext(ctx) {
  if (ctx?.theme) applyDocumentTheme(ctx.theme);
  if (ctx?.styles?.variables) applyHostStyleVariables(ctx.styles.variables);
  if (ctx?.styles?.css?.fonts) applyHostFonts(ctx.styles.css.fonts);
}

app.ontoolinputpartial = ({ arguments: args }) => {
  if (typeof args?.text === "string") {
    text = args.text;
    scheduleRender();
  }
};
app.ontoolinput = ({ arguments: args }) => {
  text = typeof args?.text === "string" ? args.text : "";
  turnId = args?.turn_id ?? null;
  render(true);
};
app.ontoolresult = (result) => {
  const data = result.structuredContent ?? {};
  turnId = data.turn_id ?? turnId;
  if (!result.isError) renderButtons(data.buttons ?? []);
};
app.ontoolcancelled = () => {
  render(true);
  buttonsEl.replaceChildren();
};
app.onhostcontextchanged = applyContext;

replyEl.addEventListener("click", (event) => {
  const link = event.target.closest("a[href]");
  if (link) {
    event.preventDefault();
    app.openLink({ url: link.href });
  }
});
onCopy(copyReplyEl, () => text);

await app.connect();
applyContext(app.getHostContext());
