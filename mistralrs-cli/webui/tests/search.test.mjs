import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { createServer } from "vite";

let server;
let highlightSegments;

before(async () => {
  server = await createServer({ server: { middlewareMode: true, ws: false, watch: null }, logLevel: "error" });
  ({ highlightSegments } = await server.ssrLoadModule("/src/lib/utils/search.ts"));
});

after(async () => {
  await server?.close();
});

test("marks every term case-insensitively", () => {
  assert.deepEqual(highlightSegments("KV cache and kv", "kv"), [
    { text: "KV", match: true },
    { text: " cache and ", match: false },
    { text: "kv", match: true },
  ]);
});

test("treats regex characters literally and handles empty queries", () => {
  assert.deepEqual(highlightSegments("use c++ (fast)", "c++ (fast)"), [
    { text: "use ", match: false },
    { text: "c++", match: true },
    { text: " ", match: false },
    { text: "(fast)", match: true },
  ]);
  assert.deepEqual(highlightSegments("abc", "  "), [{ text: "abc", match: false }]);
});
