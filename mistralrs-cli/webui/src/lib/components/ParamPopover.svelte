<script lang="ts">
  interface Props {
    label: string;
    value: number | null;
    min?: number;
    max?: number;
    step?: number;
    defaultValue?: number | null;
    onCommit: (v: number | null) => void;
    onClose: () => void;
    /** When true, allow `null` (param unset / use server default). */
    nullable?: boolean;
  }

  let { label, value, min, max, step, defaultValue, onCommit, onClose, nullable = false }: Props = $props();

  let editing = $state<number | string>("");
  let rootEl = $state<HTMLDivElement | null>(null);

  function commitFromInput(raw: string) {
    if (raw === "" && nullable) {
      onCommit(null);
      return;
    }
    const n = Number(raw);
    if (!Number.isFinite(n)) return;
    onCommit(n);
  }

  function reset() {
    onCommit(defaultValue ?? null);
    editing = defaultValue ?? "";
  }

  function onKey(e: KeyboardEvent) {
    if (e.key === "Enter") {
      commitFromInput(String(editing));
      onClose();
    } else if (e.key === "Escape") {
      onClose();
    }
  }

  function onDocMouseDown(e: MouseEvent) {
    if (rootEl && !rootEl.contains(e.target as Node)) onClose();
  }

  $effect(() => {
    // Defer to the next tick so the click that opened the popover doesn't immediately close it.
    const id = setTimeout(
      () => document.addEventListener("mousedown", onDocMouseDown),
      0,
    );
    return () => {
      clearTimeout(id);
      document.removeEventListener("mousedown", onDocMouseDown);
    };
  });

  $effect(() => {
    editing = value ?? "";
  });

  let useSlider = $derived(min != null && max != null);
</script>

<div
  bind:this={rootEl}
  class="absolute top-full left-0 z-50 mt-1 w-64 rounded-lg border border-gray-200 bg-white p-3 shadow-lg dark:border-gray-700 dark:bg-gray-900"
>
  <div class="mb-2 flex items-center justify-between">
    <span class="font-mono text-[11px] uppercase tracking-wide text-gray-500 dark:text-gray-400">{label}</span>
    {#if defaultValue != null}
      <button
        class="font-mono text-[10px] text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
        onclick={reset}
      >
        reset {defaultValue}
      </button>
    {/if}
  </div>

  {#if useSlider}
    <input
      type="range"
      class="mb-2 w-full accent-blue-500"
      min={min}
      max={max}
      step={step ?? 0.01}
      value={typeof editing === "number" ? editing : Number(editing) || min}
      oninput={(e) => {
        const n = Number((e.target as HTMLInputElement).value);
        editing = n;
        onCommit(n);
      }}
    />
  {/if}

  <input
    type="number"
    class="w-full rounded border border-gray-300 bg-white px-2 py-1 font-mono text-sm text-gray-900 outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-400 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-100"
    {min}
    {max}
    {step}
    placeholder={nullable ? "(unset)" : ""}
    bind:value={editing}
    onkeydown={onKey}
    onblur={() => commitFromInput(String(editing))}
  />
</div>
