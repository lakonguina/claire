/**
 * @fileoverview The reply widget: renders the reply as it streams, then what
 * the server sends with it: the options of a clarifying question, the
 * "did you get what you needed?" question, copy actions and the data link.
 * It reports what it sees (time on screen, copies, answers) to the server.
 * Every label comes from the server.
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
const FREE_CHOICE = 'free';
const replyEl = document.getElementById('reply');
const choicesEl = document.getElementById('choices');
const optionsEl = document.getElementById('options');
const freeEl = document.getElementById('free');
const sendEl = document.getElementById('send');
const surveyEl = document.getElementById('survey');
const copyReplyEl = document.getElementById('copy-reply');
const privacyEl = document.getElementById('privacy');
const app = new App({name: 'claire', version: '3.0.0'});

let text = '';
let turnId = null;
let accepted = false;
let pending = false;
let reported = false;
let visibleSince = null;
let visibleSeconds = 0;

/**
 * Reports an event about the shown reply; never blocks the person.
 * @param {string} kind
 * @param {(string|number|null)=} value
 * @return {!Promise<void>}
 */
async function record(kind, value = null) {
  if (!turnId) return;
  try {
    await app.callServerTool({
      name: 'record_event',
      arguments: {turn_id: turnId, kind, value},
    });
  } catch {
    // Reporting must never break the reply.
  }
}

/** Renders the Markdown reply. */
function render() {
  pending = false;
  replyEl.hidden = !text;
  try {
    replyEl.innerHTML = DOMPurify.sanitize(marked.parse(text));
  } catch {
    replyEl.textContent = text;
    record('render_error');
  }
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
 * @param {function()=} onCopy
 * @return {!HTMLButtonElement}
 */
function copyAction(button, getValue, labels, onCopy = () => {}) {
  const label = button.textContent;
  button.onclick = async () => {
    await copy(getValue());
    onCopy();
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
 * Adds the copy actions, in the user's language.
 * @param {!Object<string, string>} labels
 */
function decorate(labels) {
  for (const pre of replyEl.querySelectorAll('pre')) {
    const button = Object.assign(document.createElement('button'), {
      className: 'copy ghost',
      textContent: labels.copy,
    });
    const code = () => pre.querySelector('code')?.innerText ?? pre.innerText;
    pre.append(copyAction(button, code, labels));
  }
  copyReplyEl.textContent = labels.copy_reply;
  copyAction(
    copyReplyEl,
    () => text,
    labels,
    () => record('copy'),
  );
  copyReplyEl.hidden = !text;
}

/**
 * Sends the person's answer into the chat, as their next message.
 * @param {string} message
 */
async function answer(message) {
  for (const el of choicesEl.querySelectorAll('button, input')) {
    el.disabled = true;
  }
  await app.sendMessage({
    role: 'user',
    content: [{type: 'text', text: message}],
  });
}

/**
 * Renders the options of a clarifying question and a free answer field.
 * @param {!Array<string>} choices
 * @param {!Object<string, string>} labels
 */
function renderChoices(choices, labels) {
  choicesEl.hidden = !choices.length;
  optionsEl.replaceChildren(
    ...choices.map((option, index) => {
      const button = Object.assign(document.createElement('button'), {
        type: 'button',
        textContent: `${index + 1}. ${option}`,
      });
      button.onclick = async () => {
        button.classList.add('clicked');
        await record('choice', index);
        await answer(option);
      };
      const item = document.createElement('li');
      item.append(button);
      return item;
    }),
  );
  freeEl.placeholder = labels.free;
  sendEl.textContent = labels.send;
  choicesEl.onsubmit = async (event) => {
    event.preventDefault();
    const value = freeEl.value.trim();
    if (!value) return;
    await record('choice', FREE_CHOICE);
    await answer(value);
  };
}

/**
 * Renders the ground-truth question; its answer stays out of the chat.
 * @param {!Object} labels
 */
function renderSurvey(labels) {
  surveyEl.hidden = false;
  document.getElementById('survey-question').textContent = labels.survey;
  const answers = document.getElementById('survey-answers');
  answers.replaceChildren(
    ...Object.entries(labels.satisfaction).map(([value, label]) => {
      const button = Object.assign(document.createElement('button'), {
        textContent: label,
      });
      button.onclick = () => {
        record('satisfaction', value);
        answers.replaceChildren(document.createTextNode(labels.thanks));
      };
      return button;
    }),
  );
}

/**
 * Shows the data link: an email to the owner with the person's code.
 * @param {{email: string, code: string}} privacy
 * @param {!Object<string, string>} labels
 */
function renderPrivacy(privacy, labels) {
  privacyEl.hidden = false;
  privacyEl.textContent = labels.privacy;
  privacyEl.onclick = () => {
    record('privacy_link');
    const query = new URLSearchParams({
      subject: labels.privacy_subject,
      body: labels.privacy_body.replace('{code}', privacy.code),
    });
    app.openLink({
      url: `mailto:${privacy.email}?${query.toString().replace(/\+/g, '%20')}`,
    });
  };
}

/** Starts or stops counting the time the reply is on screen. */
function trackVisibility(visible) {
  if (visible && visibleSince === null) {
    visibleSince = performance.now();
  } else if (!visible && visibleSince !== null) {
    visibleSeconds += (performance.now() - visibleSince) / 1000;
    visibleSince = null;
    // The running total: the server keeps each turn's largest.
    record('visible', Math.round(visibleSeconds * 10) / 10);
  }
}

/** Reports the host's platform, locale and time zone once. */
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
      time_zone:
        context.timeZone ?? Intl.DateTimeFormat().resolvedOptions().timeZone,
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
};
app.ontoolresult = (result) => {
  if (result.isError) {
    // A refused call (a repeat, or an unknown turn) must not show its text.
    text = '';
    turnId = null;
    render();
    return;
  }
  const data = result.structuredContent ?? {};
  // Only a turn the server accepted: the model may have invented the input's.
  turnId = data.turn_id ?? turnId;
  accepted = true;
  reportContext();
  const labels = data.labels ?? {};
  decorate(labels);
  renderChoices(data.choices ?? [], labels);
  if (data.survey) renderSurvey(labels);
  if (data.privacy) renderPrivacy(data.privacy, labels);
  trackVisibility(document.visibilityState === 'visible');
};
app.ontoolcancelled = () => {
  render();
  record('cancelled');
};
app.onhostcontextchanged = applyContext;

replyEl.addEventListener('click', (event) => {
  const link = event.target.closest('a[href]');
  if (link) {
    event.preventDefault();
    app.openLink({url: link.href});
  }
});
new IntersectionObserver(([entry]) => {
  if (accepted) trackVisibility(entry.isIntersecting);
}).observe(replyEl);
document.addEventListener('visibilitychange', () => {
  if (accepted) trackVisibility(document.visibilityState === 'visible');
});
addEventListener('pagehide', () => trackVisibility(false));

await app.connect();
applyContext(app.getHostContext());
