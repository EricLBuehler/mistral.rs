export async function copyText(
  text,
  { clipboard = globalThis.navigator?.clipboard, document = globalThis.document } = {},
) {
  if (clipboard?.writeText) {
    try {
      await clipboard.writeText(text);
      return true;
    } catch {}
  }

  if (!document?.body || typeof document.execCommand !== "function") return false;

  try {
    const textarea = document.createElement("textarea");
    const handleCopy = (event) => {
      if (!event.clipboardData) return;
      event.clipboardData.setData("text/plain", text);
      event.preventDefault();
    };
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    textarea.style.pointerEvents = "none";
    document.body.append(textarea);
    document.addEventListener("copy", handleCopy);
    textarea.select();

    try {
      return document.execCommand("copy");
    } finally {
      document.removeEventListener("copy", handleCopy);
      textarea.remove();
    }
  } catch {
    return false;
  }
}
