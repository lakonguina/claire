/**
 * @fileoverview The reply widget (SPEC 4.2): renders the reply as it streams,
 * then the buttons (follow-up questions first), copy actions and profile
 * link sent by the server. Every label comes from the server.
 */
import {
  App,
  applyDocumentTheme,
  applyHostFonts,
  applyHostStyleVariables,
} from '@modelcontextprotocol/ext-apps';
import DOMPurify from 'dompurify';
import {marked} from 'marked';
import './style.css';

const COPIED_MS = 1500;
const replyEl = document.getElementById('reply');
const buttonsEl = document.getElementById('buttons');
const copyReplyEl = document.getElementById('copy-reply');
const profileEl = document.getElementById('profile');
const app = new App({name: 'lingua-franca', version: '2.0.0'});

let text = '';
let turnId = null;
let pending = false;
let reported = false;

/** Renders the Markdown reply. */
function render() {
  pending = false;
  replyEl.hidden = !text;
  replyEl.innerHTML = DOMPurify.sanitize(marked.parse(text));
}

/** Renders at most once per animation frame while the reply streams. */
function scheduleRender() {
  if (!pending) {
    pending = true;
    requestAnimationFrame(render);
  }
}

/**
 * Copies text to the clipboard.
 * @param {string} value
 */
async function copy(value) {
  try {
    await navigator.clipboard.writeText(value);
  } catch {
    const area = Object.assign(document.createElement('textarea'), {value});
    document.body.append(area);
    area.select();
    document.execCommand('copy');
    area.remove();
  }
}

/**
 * Makes a button copy a value, with a short confirmation.
 * @param {!HTMLButtonElement} button
 * @param {function(): string} getValue
 * @param {!Object<string, string>} labels
 * @return {!HTMLButtonElement}
 */
function copyAction(button, getValue, labels) {
  const label = button.textContent;
  button.onclick = async () => {
    await copy(getValue());
    button.textContent = labels.copied;
    button.classList.add('copied');
    setTimeout(() => {
      button.textContent = label;
      button.classList.remove('copied');
    }, COPIED_MS);
  };
  return button;
}

/**
 * Adds the copy actions and the profile link, in the user's language.
 * @param {!Object<string, string>} labels
 * @param {?string} profileUrl
 */
function decorate(labels, profileUrl) {
  for (const pre of replyEl.querySelectorAll('pre')) {
    const button = Object.assign(document.createElement('button'), {
      className: 'copy ghost',
      textContent: labels.copy,
    });
    const code = () => pre.querySelector('code')?.innerText ?? pre.innerText;
    pre.append(copyAction(button, code, labels));
  }
  copyReplyEl.textContent = labels.copy_reply;
  copyAction(copyReplyEl, () => text, labels);
  copyReplyEl.hidden = !text;
  if (profileUrl) {
    profileEl.textContent = labels.profile;
    profileEl.href = profileUrl;
    profileEl.hidden = false;
  }
}

/**
 * Records a click, then sends the button's message. Buttons stay clickable:
 * a second click corrects a mistaken first one.
 * @param {{id: string, position: number, message: string}} button
 * @param {!HTMLButtonElement} el
 */
async function click(button, el) {
  for (const other of document.querySelectorAll('.clicked')) {
    other.classList.remove('clicked');
  }
  el.classList.add('clicked');
  try {
    await app.callServerTool({
      name: 'record_click',
      arguments: {
        turn_id: turnId,
        button: button.id,
        position: button.position,
      },
    });
  } finally {
    await app.sendMessage({
      role: 'user',
      content: [{type: 'text', text: button.message}],
    });
  }
}

/**
 * Renders the buttons chosen by the server, follow-up questions first.
 * @param {!Array<{id: string, position: number, label: string,
 *     message: string}>} buttons
 */
function renderButtons(buttons) {
  buttonsEl.replaceChildren(
    ...buttons.map((b) => {
      const el = Object.assign(document.createElement('button'), {
        textContent: b.label,
      });
      el.onclick = () => click(b, el);
      return el;
    }),
  );
}

/** Reports the host's platform and locale once (SPEC 4.3). */
function reportContext() {
  const context = app.getHostContext();
  if (reported || !turnId || !context) return;
  reported = true;
  app.callServerTool({
    name: 'report_context',
    arguments: {
      turn_id: turnId,
      platform: context.platform ?? null,
      locale: context.locale ?? null,
    },
  });
}

/** @param {?Object} context The host context. */
function applyContext(context) {
  if (context?.theme) applyDocumentTheme(context.theme);
  if (context?.styles?.variables) {
    applyHostStyleVariables(context.styles.variables);
  }
  if (context?.styles?.css?.fonts) applyHostFonts(context.styles.css.fonts);
}

app.ontoolinputpartial = ({arguments: args}) => {
  if (typeof args?.text === 'string') {
    text = args.text;
    scheduleRender();
  }
};
app.ontoolinput = ({arguments: args}) => {
  text = typeof args?.text === 'string' ? args.text : '';
  turnId = args?.turn_id ?? null;
  render();
  reportContext();
};
app.ontoolresult = (result) => {
  if (result.isError) return;
  const data = result.structuredContent ?? {};
  turnId = data.turn_id ?? turnId;
  renderButtons(data.buttons ?? []);
  if (data.labels) decorate(data.labels, data.profile_url ?? null);
};
app.ontoolcancelled = () => {
  render();
  renderButtons([]);
};
app.onhostcontextchanged = applyContext;

for (const container of [replyEl, profileEl.parentElement]) {
  container.addEventListener('click', (event) => {
    const link = event.target.closest('a[href]');
    if (link) {
      event.preventDefault();
      app.openLink({url: link.href});
    }
  });
}

await app.connect();
applyContext(app.getHostContext());
