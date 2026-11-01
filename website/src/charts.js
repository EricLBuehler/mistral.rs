// Bars are drawn from the HTML tables next to them, so the tables stay the single source of numbers.
const COLUMN_TICK = 1000;
const ARROW_STEPS = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 };

const integer = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const decimal = new Intl.NumberFormat("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const compact = new Intl.NumberFormat("en-US", { notation: "compact" });

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function formatValue(value) {
  return value >= 100 ? integer.format(value) : decimal.format(value);
}

function gain(value, baseline) {
  const pct = (value / baseline - 1) * 100;
  return `${pct >= 0 ? "+" : ""}${decimal.format(pct)}%`;
}

function readTable(table) {
  const series = [...table.tHead.rows[0].cells].slice(1).map((cell) => cell.textContent.trim());
  const rows = [...table.tBodies[0].rows].map((row) => ({
    label: row.cells[0].textContent.trim(),
    values: [...row.cells].slice(1).map((cell) => Number(cell.textContent.replace(/,/g, ""))),
  }));
  return { title: table.caption.textContent.trim(), category: table.tHead.rows[0].cells[0].textContent.trim(), series, rows };
}

function makeBar(data, row, index, size) {
  const value = row.values[index];
  const bar = el("span", `bar series-${index + 1}`);
  bar.setAttribute("role", "img");
  bar.tabIndex = -1;
  bar.dataset.value = `${formatValue(value)} tokens/s`;
  bar.dataset.label = `${data.series[index]}, ${row.label} ${data.category.toLowerCase()}`;
  bar.setAttribute("aria-label", `${bar.dataset.label}: ${bar.dataset.value}`);
  bar.style.setProperty("--size", `${size}%`);
  return bar;
}

function renderColumns(panel, data) {
  const peak = Math.max(...data.rows.flatMap((row) => row.values));
  const max = Math.ceil(peak / COLUMN_TICK) * COLUMN_TICK;
  const plot = el("div", "columns");
  const grid = el("div", "grid");
  for (let tick = 0; tick <= max; tick += COLUMN_TICK) {
    const line = el("div", "grid-line");
    line.style.bottom = `${(tick / max) * 100}%`;
    line.append(el("span", "tick", compact.format(tick)));
    grid.append(line);
  }
  const groups = el("div", "column-groups");
  for (const row of data.rows) {
    const group = el("div", "column-group");
    const bars = el("div", "column-bars");
    row.values.forEach((value, index) => bars.append(makeBar(data, row, index, (value / max) * 100)));
    const top = (Math.max(...row.values) / max) * 100;
    const label = el("span", "column-gain", gain(row.values[0], row.values[1]));
    label.style.bottom = `${top}%`;
    bars.append(label);
    group.append(bars, el("span", "column-label", row.label));
    groups.append(group);
  }
  plot.append(grid, groups);
  panel.replaceChildren(
    el("p", "chart-title", `${data.title}. Labels show the gain over ${data.series[1]}.`),
    plot,
    el("p", "axis-title", data.category),
  );
}

function attachInteraction(figure) {
  const bars = [...figure.querySelectorAll(".bar")];
  if (bars.length === 0) return;
  bars[0].tabIndex = 0;

  const tooltip = el("div", "chart-tooltip");
  tooltip.hidden = true;
  figure.append(tooltip);

  const show = (bar) => {
    tooltip.replaceChildren(el("strong", "", bar.dataset.value), el("span", "", bar.dataset.label));
    tooltip.hidden = false;
    const frame = figure.getBoundingClientRect();
    const rect = bar.getBoundingClientRect();
    tooltip.style.left = `${rect.left + rect.width / 2 - frame.left}px`;
    tooltip.style.top = `${rect.top - frame.top}px`;
  };
  const hide = () => {
    tooltip.hidden = true;
  };

  figure.addEventListener("pointerover", (event) => {
    const bar = event.target.closest(".bar");
    if (bar) show(bar);
  });
  figure.addEventListener("pointerout", (event) => {
    if (event.target.closest(".bar")) hide();
  });
  figure.addEventListener("focusin", (event) => {
    const bar = event.target.closest(".bar");
    if (bar) show(bar);
  });
  figure.addEventListener("focusout", hide);
  figure.addEventListener("keydown", (event) => {
    const step = ARROW_STEPS[event.key];
    const current = bars.indexOf(document.activeElement);
    if (!step || current === -1) return;
    event.preventDefault();
    const next = bars[(current + step + bars.length) % bars.length];
    bars[current].tabIndex = -1;
    next.tabIndex = 0;
    next.focus();
  });
}

export function renderCharts(root = document) {
  for (const figure of root.querySelectorAll("figure[data-chart]")) {
    for (const panel of figure.querySelectorAll(".chart-panel[data-table]")) {
      const table = root.getElementById(panel.dataset.table);
      if (table) renderColumns(panel, readTable(table));
    }
    attachInteraction(figure);
  }
}
