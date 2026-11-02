// Shared class strings so repeated controls look identical everywhere.
export const ICON_BUTTON =
  "inline-flex h-8 w-8 items-center justify-center rounded-md text-muted hover:bg-surface-raised hover:text-foreground disabled:pointer-events-none disabled:opacity-40";
export const MESSAGE_ACTION =
  "inline-flex h-7 items-center gap-1 rounded-md px-1.5 text-xs text-muted hover:bg-surface-raised hover:text-foreground disabled:pointer-events-none disabled:opacity-40";
export const BUTTON_PRIMARY =
  "inline-flex items-center justify-center gap-1.5 rounded-md bg-fill px-3 py-1.5 text-sm font-medium text-on-fill hover:opacity-90 disabled:opacity-40";
export const BUTTON_SECONDARY =
  "inline-flex items-center justify-center gap-1.5 rounded-md border border-line-strong px-3 py-1.5 text-sm text-foreground hover:bg-surface-raised disabled:opacity-40";
export const PANEL_HEADING = "text-xs font-medium uppercase tracking-wide text-muted";
// Revealed on hover or keyboard focus; always shown on touch screens where hover doesn't exist.
export const HOVER_REVEAL =
  "opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100 [@media(hover:none)]:opacity-100";
export const TOOL_CARD = "overflow-hidden rounded-md border border-line-strong bg-surface";
export const FIELD =
  "rounded-md border border-line-strong bg-background text-foreground outline-none placeholder:text-muted focus:border-accent disabled:opacity-50";
