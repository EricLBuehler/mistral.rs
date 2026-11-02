import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { after, before, test } from "node:test";
import { createServer } from "vite";

let server;
let directory;

before(async () => {
  directory = await mkdtemp(`${process.cwd()}/tests/.svelte-`);
  server = await createServer({
    server: { middlewareMode: true, ws: false, watch: null },
  });
});

after(async () => {
  await server?.close();
  if (directory) await rm(directory, { recursive: true });
});

test("compiles TypeScript reactive stores for the browser", async () => {
  const result = await server.transformRequest("/src/lib/stores/chat.svelte.ts");
  assert.match(result.code, /svelte_internal_client/);
  assert.doesNotMatch(result.code, /\$state(?:<[^>]+>)?\(/);
});

test("supports component HMR when CSS changes, is removed, and is restored", async () => {
  const file = `${directory}/Counter.svelte`;
  const url = file.slice(process.cwd().length);
  const source = (color) => `<script lang="ts">let count = $state<number>(0);</script>
<button onclick={() => count++}>{count}</button>
${color ? `<style>button { color: ${color}; }</style>` : ""}`;
  await writeFile(file, source("red"));
  const component = await server.transformRequest(url);
  assert.match(component.code, /import\.meta\.hot\.accept/);
  const cssUrl = url.replace(/\.svelte$/, ".css");
  const originalCss = await server.transformRequest(cssUrl);
  assert.match(originalCss.code, /color: red/);

  await writeFile(file, source("blue"));
  const module = await server.moduleGraph.getModuleByUrl(url);
  const plugin = server.config.plugins.find((plugin) => plugin.name === "svelte");
  const affected = await plugin.handleHotUpdate({ file, modules: [module], server });
  assert.ok(affected.some((module) => module.url === cssUrl));
  const updatedCss = await server.transformRequest(cssUrl);
  assert.match(updatedCss.code, /color: blue/);
  assert.doesNotMatch(updatedCss.code, /color: red/);

  await writeFile(file, source(null));
  const removed = await plugin.handleHotUpdate({ file, modules: [module], server });
  assert.deepEqual(removed, [module]);
  const withoutCss = await server.transformRequest(url);
  assert.doesNotMatch(withoutCss.code, /Counter\.css/);
  const css = server.moduleGraph.getModuleById(file.replace(/\.svelte$/, ".css"));
  assert.equal(css.importers.size, 0);

  await writeFile(file, source("green"));
  const restored = await plugin.handleHotUpdate({ file, modules: [module], server });
  assert.ok(restored.includes(css));
  const restoredCss = await server.transformRequest(cssUrl);
  assert.match(restoredCss.code, /color: green/);
  assert.doesNotMatch(restoredCss.code, /color: blue/);
});
