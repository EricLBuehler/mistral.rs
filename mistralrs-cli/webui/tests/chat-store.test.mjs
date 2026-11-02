import assert from "node:assert/strict";
import { after, before, beforeEach, test } from "node:test";
import { createServer } from "vite";

const TEST_TIMEOUT_MS = 10_000;

let server;
let chatStore;
let modelStore;
let fake;

class FakeServer {
  constructor() {
    this.chats = new Map();
    this.nextChat = 1;
    this.failAppends = 0;
    this.streams = [];
    this.requests = [];
  }

  chat(id) {
    return this.chats.get(id);
  }

  sse(signal) {
    const encoder = new TextEncoder();
    let controller;
    const body = new ReadableStream({ start: (c) => (controller = c) });
    const stream = {
      send: (obj) => controller.enqueue(encoder.encode(`data: ${JSON.stringify(obj)}\n\n`)),
      content: (text) => stream.send({ choices: [{ index: 0, delta: { content: text }, finish_reason: null }] }),
      finish: (usage) => {
        stream.send({ choices: [{ index: 0, delta: {}, finish_reason: "stop" }], usage });
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        controller.close();
      },
    };
    signal?.addEventListener("abort", () =>
      controller.error(new DOMException("aborted", "AbortError")),
    );
    this.streams.push(stream);
    return new Response(body, { status: 200 });
  }

  async fetch(url, init = {}) {
    const path = String(url);
    const body = init.body ? JSON.parse(init.body) : undefined;
    this.requests.push({ path, body });
    const json = (v) => new Response(JSON.stringify(v), { status: 200 });
    if (path.endsWith("v1/chat/completions")) return this.sse(init.signal);
    if (path.endsWith("api/new_chat")) {
      const id = `chat_${this.nextChat++}`;
      this.chats.set(id, { title: null, model: body.model, kind: "text", created_at: new Date().toISOString(), messages: [], tail: null });
      return json({ id });
    }
    if (path.endsWith("api/append_message")) {
      if (this.failAppends > 0) {
        this.failAppends--;
        return new Response("disk full", { status: 500 });
      }
      const chat = this.chat(body.chat_id);
      const i = chat.messages.findIndex((m) => m.id === body.message.id);
      if (i >= 0) chat.messages[i] = body.message;
      else chat.messages.push(body.message);
      chat.tail = body.message.id;
      return new Response("ok");
    }
    if (path.endsWith("api/set_tail")) {
      this.chat(body.id).tail = body.tail;
      return new Response("ok");
    }
    if (path.endsWith("api/load_chat")) return json(this.chat(body.id));
    if (path.endsWith("api/list_chats")) {
      return json({ chats: [...this.chats].map(([id, c]) => ({ id, title: c.title, model: c.model, created_at: c.created_at, updated_at: c.created_at, preview: null })) });
    }
    if (path.includes("api/search_chats")) {
      const q = new URL(path, "http://x").searchParams.get("q");
      if (this.searchDelays?.[q]) await new Promise((r) => setTimeout(r, this.searchDelays[q]));
      const hits = [...this.chats]
        .filter(([, c]) => c.messages.some((m) => m.content.toLowerCase().includes(q.toLowerCase())))
        .map(([id, c]) => ({ id, title: c.title, model: c.model, created_at: c.created_at, updated_at: c.created_at, preview: null, snippet: q }));
      return json({ chats: hits });
    }
    if (path.endsWith("api/delete_chat")) {
      this.chats.delete(body.id);
      return new Response("ok");
    }
    if (path.endsWith("api/restore_chat_session")) return json({ session_id: null });
    if (path.endsWith("api/fork_session") || path.endsWith("api/save_chat_session")) return new Response("ok");
    return new Response("not found", { status: 404 });
  }
}

const tick = () => new Promise((r) => setTimeout(r, 0));

async function until(cond, label) {
  for (let i = 0; i < 200; i++) {
    if (cond()) return;
    await new Promise((r) => setTimeout(r, 10));
  }
  assert.fail(`timed out waiting for ${label}`);
}

before(async () => {
  globalThis.window = { innerWidth: 1024, location: { origin: "http://x", hostname: "x", port: "", protocol: "http:" } };
  globalThis.document = { querySelector: () => null };
  const mem = new Map();
  globalThis.localStorage = {
    getItem: (k) => mem.get(k) ?? null,
    setItem: (k, v) => mem.set(k, v),
    removeItem: (k) => mem.delete(k),
  };
  globalThis.fetch = (url, init) => fake.fetch(url, init);
  server = await createServer({ server: { middlewareMode: true, ws: false, watch: null }, logLevel: "error" });
  ({ chatStore } = await server.ssrLoadModule("/src/lib/stores/chat.svelte.ts"));
  ({ modelStore } = await server.ssrLoadModule("/src/lib/stores/models.svelte.ts"));
});

after(async () => {
  await server?.close();
});

beforeEach(async () => {
  fake = new FakeServer();
  await chatStore.newChat();
  chatStore.saveError = null;
  chatStore.lastError = null;
  modelStore.selectedModel = "m";
});

async function sendAndReply(text, reply, usage, attachments) {
  const sending = chatStore.sendMessage(text, attachments);
  await until(() => fake.streams.length > 0 && chatStore.isStreaming, "stream start");
  const stream = fake.streams.at(-1);
  stream.content(reply);
  stream.finish(usage);
  await sending;
  await chatStore.flushSaves();
}

test("assistant reply is saved after its user message with server usage", { timeout: TEST_TIMEOUT_MS }, async () => {
  await sendAndReply("hi", "hello", {
    prompt_tokens: 12,
    completion_tokens: 3,
    total_tokens: 15,
    prompt_tokens_details: { cached_tokens: 8 },
    avg_compl_tok_per_sec: 42.5,
  });
  const saved = fake.chat("chat_1").messages;
  assert.deepEqual(saved.map((m) => m.role), ["user", "assistant"]);
  assert.equal(saved[1].parent_id, saved[0].id);
  assert.equal(saved[1].tokens, 3);
  assert.equal(saved[1].cached_prompt_tokens, 8);
  assert.equal(saved[1].decode_tok_per_sec, 42.5);
});

test("switching chats mid-stream saves the partial reply to the chat it started in", { timeout: TEST_TIMEOUT_MS }, async () => {
  await sendAndReply("first", "ok");
  const sending = chatStore.sendMessage("second");
  await until(() => fake.streams.length === 2 && chatStore.isStreaming, "second stream");
  fake.streams[1].content("partial");
  await tick();

  await chatStore.newChat();
  await sending;
  await chatStore.flushSaves();

  assert.equal(chatStore.currentChatId, null);
  assert.equal(chatStore.messages.length, 0);
  const saved = fake.chat("chat_1").messages;
  assert.equal(saved.length, 4);
  assert.equal(saved[3].content, "partial");
  assert.equal(saved[3].finish_reason, "stopped");
});

test("a failed save pauses the queue, keeps order, and resumes on retry", { timeout: TEST_TIMEOUT_MS }, async () => {
  fake.failAppends = 3;
  await sendAndReply("hi", "hello");
  assert.match(chatStore.saveError, /Couldn't save message: .*HTTP 500 disk full/);
  assert.equal(fake.chat("chat_1").messages.length, 0);

  chatStore.retrySaves();
  await until(() => chatStore.saveError == null && fake.chat("chat_1").messages.length === 2, "retry");
  assert.deepEqual(fake.chat("chat_1").messages.map((m) => m.role), ["user", "assistant"]);
});

test("editing a user message adds a branch and keeps the original", { timeout: TEST_TIMEOUT_MS }, async () => {
  await sendAndReply("v1", "a1");
  const original = chatStore.messages[0];

  const editing = chatStore.editMessage(original.id, "v2");
  await until(() => fake.streams.length === 2 && chatStore.isStreaming, "edit stream");
  fake.streams[1].content("a2");
  fake.streams[1].finish();
  await editing;
  await chatStore.flushSaves();

  assert.deepEqual(chatStore.messages.map((m) => m.content), ["v2", "a2"]);
  const sib = chatStore.siblingInfo(chatStore.messages[0].id);
  assert.equal(sib.total, 2);
  chatStore.switchBranch(original.id);
  assert.deepEqual(chatStore.messages.map((m) => m.content), ["v1", "a1"]);
  assert.equal(fake.chat("chat_1").messages.length, 4);
});

test("request errors are shown but never sent back to the model", { timeout: TEST_TIMEOUT_MS }, async () => {
  const realFetch = fake.fetch.bind(fake);
  fake.fetch = async (url, init) =>
    String(url).endsWith("v1/chat/completions") && fake.requests.every((r) => !r.path.endsWith("completions"))
      ? (fake.requests.push({ path: String(url) }), new Response("boom", { status: 500 }))
      : realFetch(url, init);

  await chatStore.sendMessage("first");
  await chatStore.flushSaves();
  const failed = chatStore.messages.at(-1);
  assert.equal(failed.content, "");
  assert.match(failed.error, /HTTP 500: boom/);

  await sendAndReply("second", "ok");
  const lastRequest = fake.requests.filter((r) => r.path.endsWith("completions")).at(-1);
  assert.deepEqual(lastRequest.body.messages, [{ role: "user", content: "first\n\nsecond" }]);
  assert.ok(!JSON.stringify(lastRequest.body.messages).includes("boom"));
});

test("text files and audio reach the model and survive a reload", { timeout: TEST_TIMEOUT_MS }, async () => {
  const audio = "data:audio/wav;base64,UklGRg==";
  const files = [{ name: "notes.md", text: "has ``` inside" }];
  await sendAndReply("summarize", "ok", undefined, { audios: [audio], files });

  const request = fake.requests.filter((r) => r.path.endsWith("completions")).at(-1);
  assert.deepEqual(request.body.messages[0].content, [
    { type: "text", text: "notes.md:\n````\nhas ``` inside\n````\n\nsummarize" },
    { type: "audio_url", audio_url: { url: audio } },
  ]);

  await chatStore.loadChat("chat_1");
  const user = chatStore.messages[0];
  assert.equal(user.content, "summarize");
  assert.deepEqual(user.audios, [audio]);
  assert.deepEqual(user.files, files);
});

test("failing to create a chat reports it and lets the caller keep the draft", { timeout: TEST_TIMEOUT_MS }, async () => {
  fake.fetch = async (url) =>
    String(url).endsWith("api/new_chat") ? new Response("nope", { status: 500 }) : new Response("{}");
  const sent = await chatStore.sendMessage("draft");
  assert.equal(sent, false);
  assert.match(chatStore.lastError, /Couldn't create chat/);
  assert.equal(chatStore.messages.length, 0);
});

test("search debounces, ignores stale responses, and clears", { timeout: TEST_TIMEOUT_MS }, async () => {
  await sendAndReply("tell me about llamas", "ok");
  await chatStore.newChat();
  await sendAndReply("tell me about alpacas", "ok");

  fake.searchDelays = { l: 200 };
  chatStore.setSearchQuery("l");
  await until(() => fake.requests.some((r) => r.path.includes("q=l")), "first search sent");
  chatStore.setSearchQuery("alpaca");
  assert.equal(fake.requests.filter((r) => r.path.includes("search_chats")).length, 1);
  await until(() => chatStore.searchResults?.length === 1, "alpaca results");
  await new Promise((r) => setTimeout(r, 250));
  assert.deepEqual(chatStore.searchResults.map((c) => c.id), ["chat_2"]);

  chatStore.setSearchQuery("  ");
  assert.equal(chatStore.searchResults, null);
});
