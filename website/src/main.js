const COPY_RESET_MS = 1800;

const commands = {
  unix: "curl -fsSL https://mistralrs.dev/install.sh | sh",
  windows: "irm https://mistralrs.dev/install.ps1 | iex",
};

const tabs = [...document.querySelectorAll('[role="tab"]')];
const panel = document.querySelector("#install-command");
const command = panel.querySelector("code");
const copyButton = panel.querySelector(".copy-button");
let platform = "unix";
let copyResetTimer;

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

  command.textContent = commands[platform];
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

copyButton.addEventListener("click", async () => {
  await navigator.clipboard.writeText(commands[platform]);
  copyButton.textContent = "Copied";
  window.clearTimeout(copyResetTimer);
  copyResetTimer = window.setTimeout(() => {
    copyButton.textContent = "Copy";
  }, COPY_RESET_MS);
});
