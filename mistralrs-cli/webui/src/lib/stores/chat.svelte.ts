import type {
  DisplayMessage,
  AgenticToolCallProgress,
  ChatCompletionMessage,
  Attachments,
  MessageContent,
  TextAttachment,
  ChatFile,
  ChatMessageRecord,
  ChatSummary,
  ChatSearchHit,
  StreamOptions,
  StreamBlock,
  File as ProducedFile,
  AgentToolApprovalBlock,
  Usage,
} from "../types";
import {
  resolveAgentApproval as postAgentApprovalDecision,
  streamChatCompletion,
} from "../services/streaming";
import * as api from "../services/api";
import { settingsStore } from "./settings.svelte";
import { modelStore } from "./models.svelte";

const UI_UPLOAD_URL =
  /^uploads\/(?:[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})\.(?:jpg|jpeg|png|gif|webp|bmp|svg|mp4|avi|mov|mkv|webm|m4v)$/;
const SAVE_RETRY_DELAYS_MS = [300, 1000];
const CHARS_PER_TOKEN_ESTIMATE = 4;
const SEARCH_DEBOUNCE_MS = 150;
export const STOPPED_FINISH_REASON = "stopped";

const MIN_FENCE_LENGTH = 3;

function fenced(file: TextAttachment): string {
  const longestRun = Math.max(0, ...(file.text.match(/`+/g) ?? []).map((run) => run.length));
  const fence = "`".repeat(Math.max(MIN_FENCE_LENGTH, longestRun + 1));
  return `${file.name}:\n${fence}\n${file.text}\n${fence}`;
}

export function promptText(content: string, files: TextAttachment[] | undefined): string {
  return [...(files ?? []).map(fenced), content].filter(Boolean).join("\n\n");
}

function inferenceMediaUrl(url: string): string {
  if (!UI_UPLOAD_URL.test(url)) return url;
  return `mistralrs-upload:${url.slice("uploads/".length)}`;
}

function newId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function toRecord(m: DisplayMessage): ChatMessageRecord {
  return {
    id: m.id,
    parent_id: m.parentId,
    role: m.role,
    content: m.content,
    images: m.images,
    videos: m.videos,
    audios: m.audios,
    files: m.files,
    blocks: m.blocks,
    finish_reason: m.finishReason ?? undefined,
    elapsed_ms: m.elapsedMs,
    ttft_ms: m.ttftMs,
    tokens: m.tokens,
    model: m.model,
    session_id: m.sessionId,
    prompt_tokens: m.promptTokens,
    cached_prompt_tokens: m.cachedPromptTokens,
    decode_tok_per_sec: m.decodeTokPerSec,
    error: m.error,
  };
}

function fromRecord(m: ChatMessageRecord, id: string, parentId: string | null): DisplayMessage {
  return {
    id,
    parentId,
    role: m.role as DisplayMessage["role"],
    content: m.content,
    images: m.images,
    videos: m.videos,
    audios: m.audios,
    files: m.files,
    blocks: m.blocks,
    finishReason: m.finish_reason,
    elapsedMs: m.elapsed_ms,
    ttftMs: m.ttft_ms,
    tokens: m.tokens,
    model: m.model,
    sessionId: m.session_id,
    promptTokens: m.prompt_tokens,
    cachedPromptTokens: m.cached_prompt_tokens,
    decodeTokPerSec: m.decode_tok_per_sec,
    error: m.error,
  };
}

async function withRetry(run: () => Promise<void>): Promise<void> {
  for (let attempt = 0; ; attempt++) {
    try {
      await run();
      return;
    } catch (e) {
      if (attempt >= SAVE_RETRY_DELAYS_MS.length) throw e;
      await new Promise((r) => setTimeout(r, SAVE_RETRY_DELAYS_MS[attempt]));
    }
  }
}

interface PendingSave {
  chatId: string;
  label: string;
  run: () => Promise<void>;
}

interface StreamCtx {
  chatId: string;
  assistantId: string;
  parentId: string | null;
}

class ChatStore {
  /** Full tree of messages keyed by id. Map preserves insertion order. */
  allNodes = $state<Map<string, DisplayMessage>>(new Map());
  /** Active leaf id. The current conversation path = walk parentId from here back to root. */
  tailId = $state<string | null>(null);
  /** Derived view: messages on the active path, root to tail. Updated by `rebuildPath()`. */
  messages = $state<DisplayMessage[]>([]);

  chats = $state<ChatSummary[]>([]);
  searchQuery = $state("");
  /** Null when no search is active; otherwise hits for `searchQuery`. */
  searchResults = $state<ChatSearchHit[] | null>(null);
  currentChatId = $state<string | null>(null);
  /** Agentic session ID; survives turns and (via sidecar) server restarts. */
  currentSessionId = $state<string | null>(null);
  isStreaming = $state(false);

  /** Set when a save exhausted its retries; the queue is paused until `retrySaves()`. */
  saveError = $state<string | null>(null);
  /** Last non-save failure (load, delete, create), dismissible. */
  lastError = $state<string | null>(null);

  /** Ordered blocks so the model can interleave content, reasoning, and tool calls. */
  streamingBlocks = $state<StreamBlock[]>([]);
  streamingFinishReason = $state<string | null>(null);
  streamingStart = $state<number | null>(null);
  streamingFirstTokenAt = $state<number | null>(null);
  streamingTokens = $state(0);

  private abortController: AbortController | null = null;
  private streamDone: Promise<void> | null = null;
  private stopRequested = false;
  private streamingModel: string | null = null;
  private streamingCharCount = 0;
  private streamingError: string | null = null;
  private streamingUsage: Usage | null = null;

  private searchSeq = 0;
  private searchTimer: ReturnType<typeof setTimeout> | null = null;

  private saveQueue: PendingSave[] = [];
  private drainPromise: Promise<void> | null = null;

  // ===== tree helpers =====

  private rebuildPath() {
    const path: DisplayMessage[] = [];
    let curId: string | null = this.tailId;
    while (curId) {
      const node = this.allNodes.get(curId);
      if (!node) break;
      path.push(node);
      curId = node.parentId;
    }
    path.reverse();
    this.messages = path;
  }

  private insertNode(msg: DisplayMessage) {
    this.allNodes.set(msg.id, msg);
    this.allNodes = new Map(this.allNodes);
    this.tailId = msg.id;
    this.rebuildPath();
  }

  private childrenOf(parentId: string | null): DisplayMessage[] {
    const out: DisplayMessage[] = [];
    for (const [, node] of this.allNodes) {
      if (node.parentId === parentId) out.push(node);
    }
    return out;
  }

  /** Walk forward from `nodeId` picking the most recently inserted child each step. */
  private deepestDescendant(nodeId: string): string {
    let cur = nodeId;
    while (true) {
      const kids = this.childrenOf(cur);
      if (kids.length === 0) return cur;
      cur = kids[kids.length - 1].id;
    }
  }

  /** Sibling info for branch navigation UI. */
  siblingInfo(messageId: string): { index: number; total: number; siblings: string[] } | null {
    const node = this.allNodes.get(messageId);
    if (!node) return null;
    const sibs = this.childrenOf(node.parentId);
    if (sibs.length <= 1) return null;
    const idx = sibs.findIndex((s) => s.id === messageId);
    return { index: idx, total: sibs.length, siblings: sibs.map((s) => s.id) };
  }

  // ===== persistence =====

  private enqueueSave(chatId: string, label: string, run: () => Promise<void>) {
    this.saveQueue.push({ chatId, label, run });
    void this.drainSaves();
  }

  private saveMessage(chatId: string, msg: DisplayMessage) {
    const record = toRecord(msg);
    this.enqueueSave(chatId, "message", () => api.appendMessage(chatId, record));
  }

  private saveTail(chatId: string, tail: string | null) {
    this.enqueueSave(chatId, "branch selection", () => api.setTail(chatId, tail));
  }

  private drainSaves(): Promise<void> {
    this.drainPromise ??= this.runSaves().finally(() => {
      this.drainPromise = null;
      if (this.saveQueue.length && !this.saveError) void this.drainSaves();
    });
    return this.drainPromise;
  }

  private async runSaves() {
    while (this.saveQueue.length) {
      const op = this.saveQueue[0];
      try {
        await withRetry(op.run);
      } catch (e) {
        if (this.saveQueue.includes(op)) {
          this.saveError = `Couldn't save ${op.label}: ${errorText(e)}`;
          return;
        }
      }
      this.saveQueue = this.saveQueue.filter((o) => o !== op);
    }
    await this.refreshChats();
  }

  retrySaves() {
    this.saveError = null;
    void this.drainSaves();
  }

  /** Resolves once queued saves are written, or the queue paused on an error. */
  async flushSaves() {
    if (this.drainPromise) await this.drainPromise;
  }

  async refreshChats() {
    try {
      this.chats = (await api.listChats()).chats;
    } catch (e) {
      console.error("Failed to list chats:", e);
    }
    if (this.searchQuery.trim()) await this.runSearch();
  }

  setSearchQuery(query: string) {
    this.searchQuery = query;
    if (this.searchTimer) clearTimeout(this.searchTimer);
    if (!query.trim()) {
      this.searchSeq++;
      this.searchResults = null;
      return;
    }
    this.searchTimer = setTimeout(() => void this.runSearch(), SEARCH_DEBOUNCE_MS);
  }

  private async runSearch() {
    const seq = ++this.searchSeq;
    const query = this.searchQuery;
    try {
      const { chats } = await api.searchChats(query);
      if (seq === this.searchSeq) this.searchResults = chats;
    } catch (e) {
      if (seq === this.searchSeq) this.lastError = `Search failed: ${errorText(e)}`;
    }
  }

  // ===== send/generate =====

  /** Returns false when nothing was sent, so the caller can keep the user's draft. */
  async sendMessage(content: string, attachments: Attachments = {}): Promise<boolean> {
    if (!content.trim() && !Object.values(attachments).some((list) => list?.length)) return false;
    if (this.isStreaming) return false;
    const model = modelStore.selectedModel;
    if (!model) return false;

    if (!this.currentChatId) {
      try {
        const { id } = await api.newChat(model);
        this.currentChatId = id;
      } catch (e) {
        this.lastError = `Couldn't create chat: ${errorText(e)}`;
        return false;
      }
    }
    const chatId = this.currentChatId;

    const userMsg: DisplayMessage = {
      id: newId(),
      parentId: this.tailId,
      role: "user",
      content,
      images: attachments.images,
      videos: attachments.videos,
      audios: attachments.audios,
      files: attachments.files,
    };
    this.insertNode(userMsg);
    this.saveMessage(chatId, userMsg);

    await this.generateAssistant(model, chatId);
    return true;
  }

  private buildApiMessages(): ChatCompletionMessage[] {
    const apiMessages: ChatCompletionMessage[] = [];
    if (settingsStore.systemPrompt.trim()) {
      apiMessages.push({ role: "system", content: settingsStore.systemPrompt });
    }
    const turns: { role: "user" | "assistant"; text: string; media: MessageContent[] }[] = [];
    for (const msg of this.messages) {
      if (msg.role === "system") continue;
      if (msg.error && !msg.content) continue;
      const text = msg.role === "user" ? promptText(msg.content, msg.files) : msg.content;
      const media: MessageContent[] = [
        ...(msg.images ?? []).map((url) => ({ type: "image_url" as const, image_url: { url: inferenceMediaUrl(url) } })),
        ...(msg.videos ?? []).map((url) => ({ type: "video_url" as const, video_url: { url: inferenceMediaUrl(url) } })),
        ...(msg.audios ?? []).map((url) => ({ type: "audio_url" as const, audio_url: { url } })),
      ];
      const prev = turns.at(-1);
      // A failed reply is skipped above; merging keeps roles alternating for strict chat templates.
      if (msg.role === "user" && prev?.role === "user") {
        prev.text = [prev.text, text].filter(Boolean).join("\n\n");
        prev.media.push(...media);
        continue;
      }
      turns.push({ role: msg.role, text, media });
    }
    for (const turn of turns) {
      apiMessages.push({
        role: turn.role,
        content: turn.media.length ? [{ type: "text", text: turn.text }, ...turn.media] : turn.text,
      });
    }
    return apiMessages;
  }

  /** Streams a new assistant message under the current tail. Used by send, edit, and regenerate. */
  private async generateAssistant(model: string, chatId: string) {
    const apiMessages = this.buildApiMessages();
    const ctx: StreamCtx = { chatId, assistantId: newId(), parentId: this.tailId };

    this.isStreaming = true;
    this.stopRequested = false;
    this.streamingBlocks = [];
    this.streamingFinishReason = null;
    this.streamingStart = performance.now();
    this.streamingFirstTokenAt = null;
    this.streamingTokens = 0;
    this.streamingCharCount = 0;
    this.streamingError = null;
    this.streamingUsage = null;
    this.streamingModel = model;
    this.abortController = new AbortController();

    const options: StreamOptions = {
      model: model || "default",
      temperature: settingsStore.temperature,
      top_p: settingsStore.topP,
      top_k: settingsStore.topK,
      max_tokens: settingsStore.maxTokens,
      repetition_penalty: settingsStore.repetitionPenalty,
      reasoning_effort: settingsStore.reasoningEffort || undefined,
      agent_permission:
        settingsStore.agentPermission !== "auto" ? settingsStore.agentPermission : undefined,
      session_id: this.currentSessionId ?? undefined,
      abortSignal: this.abortController.signal,
    };

    if (settingsStore.enableSearch && modelStore.capabilities.search_enabled) {
      options.web_search_options = { search_context_size: "medium" };
    }
    if (settingsStore.enableCodeExecution && modelStore.capabilities.code_execution_enabled) {
      options.enable_code_execution = true;
    }
    if (settingsStore.enableShell && modelStore.capabilities.shell_enabled) {
      options.enable_shell = true;
    }

    const accountTokens = (text: string) => {
      if (this.streamingFirstTokenAt == null) this.streamingFirstTokenAt = performance.now();
      this.streamingCharCount += text.length;
      this.streamingTokens = Math.max(
        1,
        Math.round(this.streamingCharCount / CHARS_PER_TOKEN_ESTIMATE),
      );
    };

    const appendText = (type: "content" | "reasoning", text: string) => {
      accountTokens(text);
      const last = this.streamingBlocks[this.streamingBlocks.length - 1];
      if (last?.type === type) {
        last.content += text;
        this.streamingBlocks = [...this.streamingBlocks];
      } else {
        this.streamingBlocks = [...this.streamingBlocks, { type, content: text }];
      }
    };

    let resolve!: () => void;
    this.streamDone = new Promise<void>((r) => (resolve = r));
    try {
      await streamChatCompletion(apiMessages, options, {
        onContent: (text) => appendText("content", text),
        onReasoning: (text) => appendText("reasoning", text),
        onToolCallProgress: (event) => {
          const idx = this.streamingBlocks.findIndex(
            (b) =>
              b.type === "tool_call" &&
              b.data.round === event.round &&
              b.data.tool_name === event.tool_name,
          );
          if (idx >= 0) {
            const existing = this.streamingBlocks[idx] as {
              type: "tool_call";
              data: AgenticToolCallProgress;
            };
            if (event.phase === "complete") {
              const existingData = existing.data.data;
              const newData = event.data;
              if (
                newData.tool_type === "code_execution" &&
                existingData.tool_type === "code_execution" &&
                !newData.code &&
                existingData.code
              ) {
                newData.code = existingData.code;
              }
              if (
                newData.tool_type === "web_search" &&
                existingData.tool_type === "web_search" &&
                !newData.query &&
                existingData.query
              ) {
                newData.query = existingData.query;
              }
              if (
                newData.tool_type === "shell" &&
                existingData.tool_type === "shell" &&
                newData.commands.length === 0 &&
                existingData.commands.length > 0
              ) {
                newData.commands = existingData.commands;
              }
            }
            existing.data = event;
            this.streamingBlocks = [...this.streamingBlocks];
          } else {
            this.streamingBlocks = [...this.streamingBlocks, { type: "tool_call", data: event }];
          }
        },
        onApprovalRequired: (event) => {
          const idx = this.streamingBlocks.findIndex(
            (b) => b.type === "approval" && b.data.approval_id === event.approval_id,
          );
          const block: AgentToolApprovalBlock = { ...event, status: "pending" };
          if (idx >= 0) {
            (this.streamingBlocks[idx] as { type: "approval"; data: AgentToolApprovalBlock }).data =
              block;
            this.streamingBlocks = [...this.streamingBlocks];
          } else {
            this.streamingBlocks = [...this.streamingBlocks, { type: "approval", data: block }];
          }
        },
        onFile: (file: ProducedFile) => {
          const idx = this.streamingBlocks.findIndex(
            (b) => b.type === "file" && b.data.id === file.id,
          );
          if (idx >= 0) {
            (this.streamingBlocks[idx] as { type: "file"; data: ProducedFile }).data = file;
            this.streamingBlocks = [...this.streamingBlocks];
          } else {
            this.streamingBlocks = [...this.streamingBlocks, { type: "file", data: file }];
          }
        },
        onFinishReason: (reason) => {
          this.streamingFinishReason = reason;
        },
        onSessionId: (id) => {
          this.currentSessionId = id;
          this.enqueueSave(chatId, "agent session", () => api.saveChatSession(chatId, id));
        },
        onUsage: (usage) => {
          this.streamingUsage = usage;
          this.streamingTokens = usage.completion_tokens;
        },
        onDone: () => this.finalizeStreaming(ctx),
        onError: (error) => {
          this.streamingError = error;
          this.finalizeStreaming(ctx);
        },
      });
    } finally {
      this.streamDone = null;
      resolve();
    }
  }

  async resolveAgentApproval(
    approvalId: string,
    decision: "approve" | "deny",
    rememberForSession: boolean,
    message?: string,
  ) {
    const idx = this.streamingBlocks.findIndex(
      (b) => b.type === "approval" && b.data.approval_id === approvalId,
    );
    if (idx < 0) return;

    const block = this.streamingBlocks[idx] as { type: "approval"; data: AgentToolApprovalBlock };
    block.data = {
      ...block.data,
      status: "submitting",
      remember_for_session: rememberForSession,
      message,
      error: undefined,
    };
    this.streamingBlocks = [...this.streamingBlocks];

    try {
      await postAgentApprovalDecision(approvalId, decision, rememberForSession, message);
      block.data = {
        ...block.data,
        status: decision === "approve" ? "approved" : "denied",
        remember_for_session: rememberForSession,
        message,
      };
    } catch (e) {
      block.data = { ...block.data, status: "error", error: String(e) };
    }
    this.streamingBlocks = [...this.streamingBlocks];
  }

  private finalizeStreaming(ctx: StreamCtx) {
    const now = performance.now();
    const elapsedMs =
      this.streamingStart != null ? Math.max(0, now - this.streamingStart) : undefined;
    const ttftMs =
      this.streamingStart != null && this.streamingFirstTokenAt != null
        ? Math.max(0, this.streamingFirstTokenAt - this.streamingStart)
        : undefined;
    const usage = this.streamingUsage;

    if (this.streamingBlocks.length || this.streamingError) {
      const content = this.streamingBlocks
        .filter((b): b is { type: "content"; content: string } => b.type === "content")
        .map((b) => b.content)
        .join("");
      const assistantMsg: DisplayMessage = {
        id: ctx.assistantId,
        parentId: ctx.parentId,
        role: "assistant",
        content,
        blocks: [...this.streamingBlocks],
        finishReason:
          this.streamingFinishReason ?? (this.stopRequested ? STOPPED_FINISH_REASON : undefined),
        elapsedMs,
        ttftMs,
        tokens: usage?.completion_tokens ?? (this.streamingTokens || undefined),
        promptTokens: usage?.prompt_tokens,
        cachedPromptTokens: usage?.prompt_tokens_details?.cached_tokens,
        decodeTokPerSec: usage?.avg_compl_tok_per_sec,
        model: this.streamingModel ?? undefined,
        sessionId: this.currentSessionId ?? undefined,
        error: this.streamingError ?? undefined,
      };
      this.insertNode(assistantMsg);
      this.saveMessage(ctx.chatId, assistantMsg);
    }

    this.streamingBlocks = [];
    this.streamingFinishReason = null;
    this.streamingStart = null;
    this.streamingFirstTokenAt = null;
    this.streamingTokens = 0;
    this.streamingCharCount = 0;
    this.streamingError = null;
    this.streamingUsage = null;
    this.streamingModel = null;
    this.isStreaming = false;
    this.abortController = null;
  }

  stopStreaming() {
    this.stopRequested = true;
    this.abortController?.abort();
  }

  /** Stops any in-flight stream and waits until its partial reply is saved to the chat it started in. */
  private async stopAndWait() {
    if (!this.streamDone) return;
    const done = this.streamDone;
    this.stopStreaming();
    await done;
  }

  // ===== edit / regenerate / branch =====

  /** Fork the agentic session so a new branch diverges cleanly from the turns on the current path. */
  private async forkSessionForPath(srcSessionId: string | null) {
    const priorTurns = this.messages.filter((m) => m.role === "assistant" && !m.error).length;
    if (!srcSessionId || priorTurns === 0) {
      this.currentSessionId = null;
      return;
    }
    const newSessionId = newId();
    try {
      await api.forkSession(srcSessionId, newSessionId, priorTurns);
      this.currentSessionId = newSessionId;
    } catch (e) {
      console.error("Failed to fork session, starting a fresh one:", e);
      this.currentSessionId = null;
    }
  }

  /** Session id stamped on the most recent assistant message on the active path. */
  private lastAssistantSessionId(): string | null {
    for (let i = this.messages.length - 1; i >= 0; i--) {
      const m = this.messages[i];
      if (m.role === "assistant" && m.sessionId) return m.sessionId;
    }
    return null;
  }

  /** Adds the edited text as a sibling branch of `messageId` and generates a reply; the original branch is kept. */
  async editMessage(messageId: string, newContent: string) {
    if (this.isStreaming) return;
    const node = this.allNodes.get(messageId);
    const model = modelStore.selectedModel;
    const chatId = this.currentChatId;
    if (!node || node.role !== "user" || !model || !chatId) return;

    this.tailId = node.parentId;
    this.rebuildPath();
    await this.forkSessionForPath(this.lastAssistantSessionId());

    const edited: DisplayMessage = {
      id: newId(),
      parentId: node.parentId,
      role: "user",
      content: newContent,
      images: node.images,
      videos: node.videos,
      audios: node.audios,
      files: node.files,
    };
    this.insertNode(edited);
    this.saveMessage(chatId, edited);
    await this.generateAssistant(model, chatId);
  }

  /** Create a new assistant sibling under the same user message. The old assistant stays in the tree. */
  async regenerateAssistant(assistantId: string) {
    if (this.isStreaming) return;
    const node = this.allNodes.get(assistantId);
    const model = modelStore.selectedModel;
    const chatId = this.currentChatId;
    if (!node || node.role !== "assistant" || !node.parentId || !model || !chatId) return;

    this.tailId = node.parentId;
    this.rebuildPath();
    this.saveTail(chatId, node.parentId);
    await this.forkSessionForPath(node.sessionId ?? this.currentSessionId);
    await this.generateAssistant(model, chatId);
  }

  /** Switch the active path to follow `siblingId`'s subtree. Navigates to that branch's deepest leaf. */
  switchBranch(siblingId: string) {
    if (this.isStreaming || !this.allNodes.has(siblingId) || !this.currentChatId) return;
    const leaf = this.deepestDescendant(siblingId);
    this.tailId = leaf;
    this.rebuildPath();
    this.currentSessionId = this.lastAssistantSessionId();
    this.saveTail(this.currentChatId, leaf);
  }

  // ===== load / new / delete =====

  async loadChat(id: string) {
    await this.stopAndWait();
    await this.flushSaves();
    let chat: ChatFile;
    try {
      chat = await api.loadChat(id);
    } catch (e) {
      this.lastError = `Couldn't open chat: ${errorText(e)}`;
      return;
    }
    this.resetView();
    this.currentChatId = id;

    // Legacy chats may lack ids/parent_ids; treat the array as a linear chain.
    const nodes = new Map<string, DisplayMessage>();
    let prevId: string | null = null;
    for (const m of chat.messages) {
      const nodeId = m.id ?? newId();
      nodes.set(nodeId, fromRecord(m, nodeId, m.parent_id ?? prevId));
      prevId = nodeId;
    }
    this.allNodes = nodes;

    let tail: string | null = chat.tail ?? null;
    if (!tail || !nodes.has(tail)) tail = prevId;
    this.tailId = tail;
    this.rebuildPath();

    let restored: { session_id: string | null } | null = null;
    try {
      restored = await api.restoreChatSession(id);
    } catch (e) {
      console.error("Failed to restore chat session:", e);
    }
    // Prefer the active branch's per-message session id; legacy chats only have the sidecar.
    this.currentSessionId = this.lastAssistantSessionId() ?? restored?.session_id ?? null;
  }

  private resetView() {
    this.currentChatId = null;
    this.currentSessionId = null;
    this.allNodes = new Map();
    this.tailId = null;
    this.messages = [];
  }

  async newChat() {
    await this.stopAndWait();
    this.resetView();
  }

  async renameChat(id: string, title: string) {
    try {
      await api.renameChat(id, title);
    } catch (e) {
      this.lastError = `Couldn't rename chat: ${errorText(e)}`;
    }
    await this.refreshChats();
  }

  async deleteChat(id: string) {
    if (this.currentChatId === id) await this.stopAndWait();
    const pausedOnThisChat = this.saveError != null && this.saveQueue[0]?.chatId === id;
    this.saveQueue = this.saveQueue.filter((op) => op.chatId !== id);
    if (pausedOnThisChat) this.retrySaves();
    try {
      await api.deleteChat(id);
    } catch (e) {
      this.lastError = `Couldn't delete chat: ${errorText(e)}`;
      return;
    }
    if (this.currentChatId === id) this.resetView();
    await this.refreshChats();
  }
}

export const chatStore = new ChatStore();
