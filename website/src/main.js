import { renderCharts } from "./charts.js";
import { copyText } from "./clipboard.js";

const COPY_RESET_MS = 1800;
// break narrow-screen lines only between arguments, never inside a flag, URL, or pipe
const ARGUMENT_BREAK = / (?=https:\/\/|\| |-)/;

const commands = {
  unix: "curl -fsSL https://mistralrs.dev/install.sh | sh",
  windows: "irm https://mistralrs.dev/install.ps1 | iex",
};

const tabs = [...document.querySelectorAll('[role="tab"]')];
const panel = document.querySelector("#install-command");
const command = panel.querySelector("code");
const copyButton = panel.querySelector(".copy-button");
let platform = "unix";

function renderCommand(code, text) {
  const parts = text.split(ARGUMENT_BREAK).map((part) => {
    const span = document.createElement("span");
    span.className = "nowrap";
    span.textContent = part;
    return span;
  });
  code.replaceChildren(...parts.flatMap((part, index) => (index ? [" ", part] : [part])));
}

function selectPlatform(nextPlatform, moveFocus = false) {
  platform = nextPlatform;

  for (const tab of tabs) {
    const selected = tab.dataset.platform === platform;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;

    if (selected) {
      panel.setAttribute("aria-labelledby", tab.id);
      if (moveFocus) tab.focus();
    }
  }

  renderCommand(command, commands[platform]);
  copyButton.textContent = "Copy";
  copyButton.setAttribute(
    "aria-label",
    `Copy ${platform === "unix" ? "macOS and Linux" : "Windows"} install command`,
  );
}

for (const [index, tab] of tabs.entries()) {
  tab.addEventListener("click", () => selectPlatform(tab.dataset.platform));
  tab.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;

    event.preventDefault();
    let nextIndex = index;

    if (event.key === "ArrowLeft") nextIndex = (index - 1 + tabs.length) % tabs.length;
    if (event.key === "ArrowRight") nextIndex = (index + 1) % tabs.length;
    if (event.key === "Home") nextIndex = 0;
    if (event.key === "End") nextIndex = tabs.length - 1;

    selectPlatform(tabs[nextIndex].dataset.platform, true);
  });
}

async function copyWithFeedback(button, text) {
  const copied = await copyText(text);
  button.textContent = copied ? "Copied" : "Copy failed";
  window.clearTimeout(button.resetTimer);
  button.resetTimer = window.setTimeout(() => {
    button.textContent = "Copy";
  }, COPY_RESET_MS);
}

copyButton.addEventListener("click", () => copyWithFeedback(copyButton, commands[platform]));

for (const button of document.querySelectorAll(".copy-button[data-copy]")) {
  button.addEventListener("click", () => copyWithFeedback(button, button.dataset.copy));
}

renderCharts();

for (const code of document.querySelectorAll(".command-row code")) {
  renderCommand(code, code.textContent);
}
