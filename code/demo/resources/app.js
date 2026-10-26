// The shared demo page. It loads /api/bootstrap, posts the inputs to the run url and draws the blocks it gets back
// (demo/model.py lists them). Nothing here is specific to any demo.
const $ = (selector) => document.querySelector(selector);
function h(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key.startsWith("aria-") && typeof value === "boolean") node.setAttribute(key, String(value));
    else if (value !== false && value != null) node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) if (child != null) node.append(child);
  return node;
}

const state = {
  boot: null, values: {}, caseId: null, enabled: new Set(), result: null, index: 0, focus: null,
  choices: {},  // switcher name -> option id
  frames: {},   // timeline path -> frame index
  actions: {},  // action path -> response
  fields: {},   // action path + field -> number
};

async function api(url, body) {
  const response = await fetch(url, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : undefined);
  return response.json();
}

function request(extra = {}) {
  return { inputs: { ...state.values, case: state.caseId }, steps: [...state.enabled], ...extra };
}

// --- blocks ------------------------------------------------------------------------------------------------------

const GUTTER = { added: "+", removed: "−", changed: "~" };

function pane(title, meta, body) {
  const head = title || meta ? h("h4", {}, h("span", {}, title || ""), h("span", {}, meta || "")) : null;
  return h("div", { class: "pane" }, head, body);
}

function codeLine(line, index, marks) {
  const kind = marks && marks.has(index) ? marks.get(index) : line.kind;
  const text = h("span", {});
  let at = 0;
  for (const [start, end] of line.spans) {
    if (start > at) text.append(line.text.slice(at, start));
    text.append(h("span", { class: "hl-span" }, line.text.slice(start, end)));
    at = end;
  }
  text.append(line.text.slice(at) || (line.spans.length ? "" : " "));
  return h("div", { class: `line ${kind}`, "data-key": line.key || null },
    h("span", { class: "no" }, index + 1), h("span", { class: "gut" }, GUTTER[kind] || ""), text);
}

function codePane(block, marks) {
  const body = block.lines.length
    ? h("div", { class: "code" }, block.lines.map((line, i) => codeLine(line, i, marks)))
    : h("div", { class: "empty" }, block.empty || "Nothing to show.");
  return pane(block.title, block.meta, body);
}

const SWATCH_TONES = new Set(["good", "bad", "accent", "muted", "warn", "series-1", "series-2"]);

function legend(items) {
  // a tone the page knows is a colour square; any other tone is a column kind, drawn as a bar in its colour
  return h("div", { class: "legend" }, items.map((item) =>
    h("span", {}, item.mark ? `${item.mark} ` : SWATCH_TONES.has(item.tone)
      ? h("i", { class: `swatch tone-${item.tone}` }) : h("i", { class: `bar-swatch k-${item.tone}` }), item.label)));
}

const BLOCKS = {
  code: (b) => codePane(b),

  diff: (b) => {
    const parts = [h("div", { class: b.before ? "panes" : "panes single" }, b.before ? codePane(b.before) : null, codePane(b.after))];
    if (b.before) parts.push(legend([
      { label: "added", tone: "good" }, { label: "changed", tone: "warn" }, { label: "removed", tone: "bad" },
    ]));
    return h("div", {}, parts);
  },

  metrics: (b) => h("div", { class: "metrics", style: b.columns ? `grid-template-columns: repeat(${b.columns}, minmax(0, 1fr))` : null },
    b.cards.map((c) => h("div", { class: `metric ${c.tone ? "tone-" + c.tone : ""}`, "data-key": c.key || null },
      h("div", { class: "label" }, c.label),
      h("div", { class: `value ${c.tone === "good" || c.tone === "bad" ? c.tone : ""}` }, c.value),
      h("div", { class: "sub" }, c.sub)))),

  stats: (b) => {
    const table = h("table", { class: "stats" },
      h("thead", {}, h("tr", {}, b.headers.map((text) => h("th", {}, text)))),
      h("tbody", {}, b.rows.map((row) => h("tr", {},
        h("td", {}, h("b", {}, row.label), row.note ? h("div", { class: "cell-note" }, row.note) : null),
        row.cells.map((cell) => h("td", { class: `${cell.bars.length ? "bars" : ""} ${cell.tone ? "tone-" + cell.tone : ""}` },
          cell.text, cell.note ? h("div", { class: "cell-note" }, cell.note) : null,
          cell.bars.map((bar) => h("i", { class: `bar tone-${bar.tone}`, style: `width:${Math.round(100 * bar.fraction)}%` }))))))));
    return h("div", {}, table, b.legend.length ? legend(b.legend) : null);
  },

  switcher: (b, ctx) => {
    const usable = b.options.filter((o) => !o.disabled);
    let current = state.choices[b.name];
    if (!usable.some((o) => o.id === current)) current = (usable[0] || b.options[0]).id;
    const chosen = b.options.find((o) => o.id === current);
    return h("div", {},
      h("div", { class: "seg", role: "group" }, b.options.map((o) =>
        h("button", { "aria-pressed": o.id === current, disabled: Boolean(o.disabled), title: o.disabled || null,
                      onclick: () => { state.choices[b.name] = o.id; renderStages(); } }, o.label))),
      renderBlocks(chosen.blocks, `${ctx.path}:${chosen.id}`));
  },

  grid: (b) => {
    if (!b.columns.length) return pane(b.title, "", h("div", { class: "empty" }, b.empty || "Nothing to show."));
    const head = h("tr", {}, h("th", {}), b.columns.map((c) =>
      h("th", { class: `v k-${c.kind}`, title: c.tip, "data-key": c.key || null }, h("div", {}, c.label))));
    const rows = b.rows.map((row, r) => h("tr", {},
      h("td", { class: "label", title: row.label }, h("span", { class: "no" }, r + 1), row.label),
      row.cells.map((cell, c) => h("td", { class: `cell k-${b.columns[c].kind} ${cell.state}`, title: cell.tip, "data-key": b.columns[c].key || null },
        cell.mark ? h("span", { class: "mark" }, cell.mark) : null))));
    return h("div", { style: `margin-left:${b.indent * 14}px` }, pane(b.title, "",
      h("div", {}, h("div", { class: "grid-scope" }, h("table", { class: "grid" }, h("thead", {}, head), h("tbody", {}, rows))),
        b.legend.length ? legend(b.legend) : null)));
  },

  timeline: (b, ctx) => {
    if (!b.frames.length) return h("div", { class: "empty" }, b.empty || "Nothing to step through.");
    const total = b.frames.length;
    const key = ctx.path;
    const dynamic = h("div", {});
    const slider = h("input", { type: "range", min: 0, max: total - 1, "aria-label": b.title });
    const back = h("button", { "aria-label": "Previous" }, "‹");
    const forward = h("button", { "aria-label": "Next" }, "›");
    const where = h("span", { class: "where" });
    function refresh() {
      const at = Math.min(Math.max(state.frames[key] || 0, 0), total - 1);
      state.frames[key] = at;
      slider.value = at;
      back.disabled = at <= 0;
      forward.disabled = at >= total - 1;
      const frame = b.frames[at];
      where.textContent = frame.label || `${at + 1} of ${total}`;
      const parts = [renderBlocks(frame.blocks, `${ctx.path}@${at}`)];
      if (b.panes.length) {
        parts.push(h("div", { class: "panes block" }, b.panes.map((p, i) => {
          const marks = new Map((frame.marks[i] || []).map(([line, kind]) => [line, kind]));
          return codePane(p, marks);
        })));
      }
      dynamic.replaceChildren(...parts);
      dynamic.querySelectorAll(".code").forEach((box) => {
        const now = box.querySelector(".line.now");
        if (now) box.scrollTop = Math.max(now.offsetTop - box.clientHeight / 2, 0);
      });
    }
    slider.addEventListener("input", () => { state.frames[key] = parseInt(slider.value, 10); refresh(); });
    back.addEventListener("click", () => { state.frames[key] -= 1; refresh(); });
    forward.addEventListener("click", () => { state.frames[key] += 1; refresh(); });
    refresh();
    return h("div", {}, h("div", { class: "frame-ctl" }, back, slider, forward, where), dynamic);
  },

  note: (b) => h("div", { class: `note ${b.tone}` }, b.text, b.items.length ? h("ul", {}, b.items.map((t) => h("li", {}, t))) : null),

  action: (b, ctx) => {
    const key = ctx.path + JSON.stringify(b.payload);
    const response = state.actions[key];
    const inputs = b.fields.map((f) => {
      const fieldKey = key + f.name;
      return h("label", { class: "muted" }, `${f.label} `, h("input", {
        type: "number", min: f.low, max: f.high, value: state.fields[fieldKey] ?? f.default, "aria-label": f.label,
        oninput: (event) => { state.fields[fieldKey] = parseInt(event.target.value, 10) || f.default; },
      }));
    });
    const button = h("button", { class: "run", onclick: async () => {
      button.disabled = true;
      button.textContent = "Running…";
      const payload = { ...b.payload };
      for (const f of b.fields) payload[f.name] = state.fields[key + f.name] ?? f.default;
      state.actions[key] = await api(state.boot.urls.actions[b.action], request({ payload }));
      renderStages();
    } }, response ? "Run again" : b.label);
    return h("div", {},
      h("div", { class: "action-bar" }, button, inputs, b.hint ? h("span", { class: "muted" }, b.hint) : null),
      response ? h("div", { class: "action-result" }, renderBlocks(response.blocks || [], key)) : null);
  },

  diagram: (b) => pane(b.title, b.meta, h("div", { class: "diagram-wrap" }, circuitDiagram(b.picture))),

  columns: (b, ctx) => h("div", { class: "panes", style: `grid-template-columns: repeat(${b.children.length}, minmax(0, 1fr))` },
    b.children.map((blocks, i) => h("div", {}, renderBlocks(blocks, `${ctx.path}/${i}`)))),
};

function renderBlocks(blocks, path) {
  return h("div", {}, blocks.map((block, i) => {
    const draw = BLOCKS[block.type];
    const node = draw ? draw(block, { path: `${path}.${i}` }) : h("div", { class: "note bad" }, `Unknown block: ${block.type}`);
    return h("div", { class: "block" }, node);
  }));
}

// --- stages and the workflow -------------------------------------------------------------------------------------

function stageView(stage, path) {
  return [h("div", { class: "stage-head" }, h("h3", {}, stage.name), stage.summary ? h("p", {}, stage.summary) : null),
          renderBlocks(stage.blocks, path)];
}

function renderStages() {
  const result = state.result;
  const error = $("#input-error");
  const root = $("#stages");
  const stepper = $("#stepper");
  if (!result) return;
  if (error) {
    error.hidden = !result.error;
    if (result.error) {
      const where = result.error.line ? ` (line ${result.error.line}, column ${result.error.column})` : "";
      error.textContent = result.error.message + where;
    }
  }
  const stages = result.stages || [];
  if (result.error) {
    stepper.replaceChildren();
    root.replaceChildren(h("div", { class: "note bad" }, "Fix the input on the left to see the result."));
    return;
  }
  if (!state.boot.workflow) {
    stepper.replaceChildren();
    root.replaceChildren(...stages.map((stage) => h("section", { class: "panel" }, stageView(stage, stage.id))));
    return;
  }
  state.index = Math.min(Math.max(state.index, 0), Math.max(stages.length - 1, 0));
  stepper.replaceChildren();
  if (stages.length > 1) {
    stages.forEach((stage, i) => {
      if (i) stepper.append(h("span", { class: "arrow" }, "→"));
      stepper.append(h("button", { class: `step${stage.failed ? " failed" : ""}`, "aria-current": i === state.index ? "step" : null, onclick: () => go(i) },
        h("span", { class: "n" }, i === 0 ? "•" : i), stage.name));
    });
    const prev = h("button", { "aria-label": "Previous stage", onclick: () => go(state.index - 1) }, "←");
    const next = h("button", { "aria-label": "Next stage", onclick: () => go(state.index + 1) }, "→");
    prev.disabled = state.index <= 0;
    next.disabled = state.index >= stages.length - 1;
    stepper.append(h("div", { class: "nav" }, prev, next));
  }
  const stage = stages[state.index];
  root.replaceChildren(...(stage ? stageView(stage, stage.id) : []));
}

function go(index) {
  const stages = (state.result && state.result.stages) || [];
  if (index < 0 || index >= stages.length) return;
  state.index = index;
  renderStages();
}

let timer;
let runs = 0;
function schedule() { clearTimeout(timer); timer = setTimeout(run, 250); }

async function run() {
  const mine = ++runs;
  state.actions = {};  // a result belongs to the inputs it was run for
  $("#stages").classList.add("busy");
  const result = await api(state.boot.urls.run, request());
  if (mine !== runs) return;
  $("#stages").classList.remove("busy");
  state.result = result;
  const focused = (result.stages || []).findIndex((s) => s.id === state.focus);
  if (focused >= 0) state.index = focused;
  state.focus = null;
  renderStages();
}

// --- inputs ------------------------------------------------------------------------------------------------------

function renderCases() {
  const boot = state.boot;
  if (!boot.cases.length) return;
  const active = boot.cases.find((c) => c.id === state.caseId);
  $("#cases").replaceChildren(
    h("div", { class: "chips" }, boot.cases.map((c) => h("button", { class: "chip", "aria-pressed": c.id === state.caseId, onclick: () => chooseCase(c) }, c.name))),
    h("p", { class: "hint" }, active ? active.description : "Your own input."));
}

function chooseCase(c) {
  state.caseId = c.id;
  for (const [id, value] of Object.entries(c.values)) {
    state.values[id] = value;
    const element = document.getElementById(`input-${id}`);
    if (element) element.value = value;
  }
  state.index = 0;
  state.frames = {};
  renderCases();
  run();
}

function buildInputs() {
  const boot = state.boot;
  const root = $("#inputs");
  const panels = [];
  if (boot.cases.length) panels.push(h("section", { class: "panel" }, h("h2", {}, boot.cases_label), h("div", { id: "cases" })));
  const numbers = boot.inputs.filter((i) => i.kind === "number");
  if (numbers.length) {
    panels.push(h("section", { class: "panel" }, h("h2", {}, "Settings"), numbers.map((spec) =>
      h("label", { class: "field" }, spec.label, h("input", {
        type: "number", id: `input-${spec.id}`, min: spec.low, max: spec.high, step: 1, value: state.values[spec.id], "aria-label": spec.label,
        oninput: (event) => {
          const n = parseInt(event.target.value, 10);
          if (Number.isFinite(n)) { state.values[spec.id] = Math.min(Math.max(n, spec.low), spec.high); schedule(); }
        },
      }), spec.hint ? h("span", {}, spec.hint) : null))));
  }
  for (const spec of boot.inputs.filter((i) => i.kind === "text")) {
    panels.push(h("section", { class: "panel" }, h("h2", {}, spec.label),
      h("textarea", { id: `input-${spec.id}`, rows: spec.rows, spellcheck: "false", "aria-label": spec.label, oninput: (event) => {
        state.values[spec.id] = event.target.value;
        const active = boot.cases.find((c) => c.id === state.caseId);
        if (active && spec.id in active.values) { state.caseId = null; renderCases(); }
        schedule();
      } }, state.values[spec.id]),
      spec.hint ? h("p", { class: "hint" }, spec.hint) : null));
  }
  panels.push(h("div", { class: "error", id: "input-error", hidden: true }));
  root.replaceChildren(...panels);
  renderCases();
}

function renderSteps() {
  const steps = state.boot.steps;
  const panel = $("#steps-panel");
  if (!steps.length) return;
  panel.hidden = false;
  const nameOf = (id) => (steps.find((s) => s.id === id) || {}).name || id;
  panel.replaceChildren(
    h("div", { class: "panel-head" }, h("h2", {}, state.boot.steps_label),
      state.boot.max_steps ? null : h("div", { class: "tools" }, h("button", { onclick: () => setAll(true) }, "All on"), h("button", { onclick: () => setAll(false) }, "All off"))),
    h("div", { class: "step-grid" }, steps.map((step, i) => {
      const on = state.enabled.has(step.id);
      const missing = on ? step.requires.filter((r) => !state.enabled.has(r)).map(nameOf) : [];
      return h("button", { class: "toggle", "aria-pressed": on, onclick: () => toggle(step.id) },
        h("span", { class: "box" }, on ? "✓" : ""),
        h("span", { class: "name" }, `${i + 1}. ${step.name}`),
        h("span", { class: "what" }, step.summary),
        missing.length ? h("span", { class: "warn" }, `Expects ${missing.join(" and ")} first. Without it this may fail or give wrong results.`) : null);
    })));
}

function toggle(id) {
  if (state.enabled.has(id)) state.enabled.delete(id); else { state.enabled.add(id); state.focus = id; }
  const limit = state.boot.max_steps;
  while (limit && state.enabled.size > limit) state.enabled.delete(state.enabled.values().next().value);  // the oldest goes
  renderSteps();
  run();
}

function setAll(on) {
  state.enabled = new Set(on ? state.boot.steps.map((s) => s.id) : []);
  state.index = 0;
  renderSteps();
  run();
}

// --- highlighting and keys -----------------------------------------------------------------------------------------

function highlight(key, on) {
  if (key == null) return;
  document.querySelectorAll(`[data-key="${CSS.escape(key)}"]`).forEach((el) => el.classList.toggle("hl", on));
}
document.addEventListener("mouseover", (event) => { const el = event.target.closest("[data-key]"); if (el) highlight(el.dataset.key, true); });
document.addEventListener("mouseout", (event) => { const el = event.target.closest("[data-key]"); if (el) highlight(el.dataset.key, false); });
document.addEventListener("keydown", (event) => {
  if (["TEXTAREA", "INPUT"].includes(event.target.tagName) || event.metaKey || event.ctrlKey || event.altKey) return;
  if (!state.boot || !state.boot.workflow) return;
  if (event.key === "ArrowRight") go(state.index + 1);
  if (event.key === "ArrowLeft") go(state.index - 1);
});

// A demo's theme is a set of CSS variables per scheme; they are written after app.css so they win.
function applyTheme(theme) {
  if (!theme) return;
  const rules = (variables) => Object.entries(variables).map(([name, value]) => `${name}: ${value};`).join(" ");
  const style = document.createElement("style");
  style.textContent = `:root { ${rules(theme.light)} } @media (prefers-color-scheme: dark) { :root { ${rules(theme.dark)} } }`;
  document.head.append(style);
}

(async function init() {
  state.boot = await api("/api/bootstrap");
  const boot = state.boot;
  applyTheme(boot.theme);
  document.title = boot.title;
  $("#title").textContent = boot.title;
  $("#description").textContent = boot.description;
  for (const spec of boot.inputs) state.values[spec.id] = spec.default;
  state.enabled = new Set(boot.steps.filter((s) => s.enabled).map((s) => s.id));
  buildInputs();
  if (boot.steps_first) $("#stepper").before($("#steps-panel"));
  renderSteps();
  if (boot.cases.length) chooseCase(boot.cases[0]); else run();
})();
