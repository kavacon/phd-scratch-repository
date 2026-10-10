// Drawing a circuit as SVG, shared by the pipeline explorer and the comparison app. The data comes from
// allocation_poc/circuits.py (diagram): one label per wire, and gates already placed in layers.
(function () {
  const style = document.createElement("style");
  style.textContent = `
.diagram-wrap { overflow: auto; max-height: 640px; padding: 6px 10px 10px; background: var(--panel); }
.diagram text { font: 10px var(--mono); fill: var(--muted); }
.diagram text.odd { fill: var(--accent); }
.diagram text.glyph { fill: var(--text); text-anchor: middle; dominant-baseline: central; font-size: 9px; font-weight: 700; }
.diagram .wire { stroke: var(--border); stroke-width: 1; }
.diagram .link { stroke: var(--text); stroke-width: 1.5; }
.diagram .ctl { fill: var(--text); stroke: var(--text); stroke-width: 1.5; }
.diagram .ctl.neg { fill: var(--panel); }
.diagram .tgt { fill: var(--panel); stroke: var(--text); stroke-width: 1.5; }
.diagram .meter { fill: var(--chg); stroke: var(--text); stroke-width: 1.2; }
.diagram .dial { stroke: var(--text); stroke-width: 1.2; }
.diagram .box { fill: var(--accent-soft); stroke: var(--accent); stroke-width: 1.2; }
`;
  document.head.append(style);
})();

const SVG_NS = "http://www.w3.org/2000/svg";
function svg(tag, props = {}, ...children) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(props)) node.setAttribute(key, value);
  for (const child of children.flat()) if (child != null) node.append(child);
  return node;
}

function circuitDiagram(picture) {
  const row = 22, col = 22, left = 84, top = 14;
  const width = left + (picture.layers + 1) * col + 16;
  const height = top * 2 + picture.wires.length * row;
  const y = (q) => top + q * row + row / 2;
  const x = (layer) => left + (layer - 0.5) * col;
  const root = svg("svg", { class: "diagram", width, height, viewBox: `0 0 ${width} ${height}`, role: "img",
                            "aria-label": "The circuit, one wire per qubit with time running left to right" });
  picture.wires.forEach((label, q) => {
    const odd = !label.startsWith("scratch") && Math.floor(q / picture.register_width) % 2 === 1;
    root.append(svg("line", { class: "wire", x1: left - 6, x2: width - 8, y1: y(q), y2: y(q) }));
    const text = svg("text", { class: odd ? "odd" : "", x: left - 10, y: y(q) + 3, "text-anchor": "end" });
    text.textContent = label;
    root.append(text);
  });
  for (const gate of picture.gates) {
    const gx = x(gate.layer);
    const wires = [gate.target, ...gate.controls.map(([q]) => q)];
    const low = Math.min(...wires), high = Math.max(...wires);
    const part = svg("g", {}, svg("title", {}, gate.detail || `${gate.label}${gate.controls.length ? `, ${gate.controls.length} control${gate.controls.length === 1 ? "" : "s"}` : ""}`));
    if (high > low) part.append(svg("line", { class: "link", x1: gx, x2: gx, y1: y(low), y2: y(high) }));
    for (const [q, positive] of gate.controls) part.append(svg("circle", { class: `ctl${positive ? "" : " neg"}`, cx: gx, cy: y(q), r: 3.5 }));
    if (gate.kind === "meter") {
      const cy = y(gate.target);
      part.append(svg("rect", { class: "meter", x: gx - 8, y: cy - 7, width: 16, height: 14, rx: 2 }),
                  svg("path", { class: "dial", d: `M ${gx - 5} ${cy + 3} A 5 5 0 0 1 ${gx + 5} ${cy + 3}`, fill: "none" }),
                  svg("line", { class: "dial", x1: gx, y1: cy + 3, x2: gx + 3, y2: cy - 3 }));
    } else if (gate.kind === "box") {
      part.append(svg("rect", { class: "box", x: gx - 8, y: y(gate.target) - 7, width: 16, height: 14, rx: 3 }));
      const text = svg("text", { class: "glyph", x: gx, y: y(gate.target) });
      text.textContent = gate.label;
      part.append(text);
    } else {
      const cy = y(gate.target);
      part.append(svg("circle", { class: "tgt", cx: gx, cy, r: 7 }), svg("line", { class: "link", x1: gx - 7, x2: gx + 7, y1: cy, y2: cy }), svg("line", { class: "link", x1: gx, x2: gx, y1: cy - 7, y2: cy + 7 }));
    }
    root.append(part);
  }
  return root;
}
