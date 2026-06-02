import type {
  DisplayMessage,
  AgenticToolCallProgress,
  ChatCompletionMessage,
  StreamOptions,
  StreamBlock,
  CodeExecutionData,
  File as ProducedFile,
  AgentToolApprovalRequired,
  AgentToolApprovalBlock,
} from "../types";
import {
  resolveAgentApproval as postAgentApprovalDecision,
  streamChatCompletion,
} from "../services/streaming";
import * as api from "../services/api";
import { settingsStore } from "./settings.svelte";
import { modelStore } from "./models.svelte";

function newId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

class ChatStore {
  /** Full tree of messages keyed by id. Map preserves insertion order. */
  allNodes = $state<Map<string, DisplayMessage>>(new Map());
  /** Active leaf id. The current conversation path = walk parentId from here back to root. */
  tailId = $state<string | null>(null);
  /** Derived view: messages on the active path, root → tail. Updated by `rebuildPath()`. */
  messages = $state<DisplayMessage[]>([]);

  currentChatId = $state<string | null>(null);
  /** Agentic session ID; survives turns and (via sidecar) server restarts. */
  currentSessionId = $state<string | null>(null);
  isStreaming = $state(false);

  /** Ordered blocks so the model can interleave content, reasoning, and tool calls. */
  streamingBlocks = $state<StreamBlock[]>([]);
  streamingFinishReason = $state<string | null>(null);
  streamingStart = $state<number | null>(null);
  streamingFirstTokenAt = $state<number | null>(null);
  streamingTokens = $state(0);
  streamingTokRate = $state(0);

  private abortController: AbortController | null = null;
  private streamingModel: string | null = null;
  private streamingCharCount = 0;
  private tokRateTimer: ReturnType<typeof setInterval> | null = null;

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

  private insertNode(msg: DisplayMessage, makeTail = true) {
    this.allNodes.set(msg.id, msg);
    this.allNodes = new Map(this.allNodes);
    if (makeTail) this.tailId = msg.id;
    this.rebuildPath();
  }

  private removeSubtree(rootId: string) {
    const toRemove = new Set<string>([rootId]);
    let changed = true;
    while (changed) {
      changed = false;
      for (const [id, node] of this.allNodes) {
        if (!toRemove.has(id) && node.parentId && toRemove.has(node.parentId)) {
          toRemove.add(id);
          changed = true;
        }
      }
    }
    for (const id of toRemove) this.allNodes.delete(id);
    this.allNodes = new Map(this.allNodes);
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

  // ===== send/generate =====

  async sendMessage(content: string, imageUrls?: string[], videoUrls?: string[]) {
    if (!content.trim() && !imageUrls?.length && !videoUrls?.length) return;
    if (this.isStreaming) return;

    const model = modelStore.selectedModel;
    if (!model) return;

    try {
      if (!this.currentChatId) {
        const { id } = await api.newChat(model);
        this.currentChatId = id;
      }
    } catch (e) {
      console.error("Failed to create chat:", e);
      return;
    }

    const userMsg: DisplayMessage = {
      id: newId(),
      parentId: this.tailId,
      role: "user",
      content,
      images: imageUrls,
      videos: videoUrls,
    };
    this.insertNode(userMsg);

    if (this.currentChatId) {
      api
        .appendMessage(
          this.currentChatId,
          "user",
          content,
          imageUrls,
          videoUrls,
          undefined,
          undefined,
          undefined,
          userMsg.id,
          userMsg.parentId,
        )
        .catch((e) => console.error("Failed to persist user message:", e));
    }

    await this.generateAssistant(model);
  }

  /** Streams a new assistant message under the current tail. Used by send, edit-of-user, and regenerate. */
  private async generateAssistant(model: string) {
    const apiMessages: ChatCompletionMessage[] = [];

    if (settingsStore.systemPrompt.trim()) {
      apiMessages.push({ role: "system", content: settingsStore.systemPrompt });
    }

    for (const msg of this.messages) {
      if (msg.role === "system") continue;
      const hasMedia = msg.images?.length || msg.videos?.length;
      if (hasMedia) {
        const contentParts: ChatCompletionMessage["content"] = [
          { type: "text", text: msg.content },
        ];
        if (msg.images?.length) {
          for (const url of msg.images) {
            (contentParts as Array<{ type: string; image_url?: { url: string } }>).push({
              type: "image_url",
              image_url: { url },
            });
          }
        }
        if (msg.videos?.length) {
          for (const url of msg.videos) {
            (contentParts as Array<{ type: string; video_url?: { url: string } }>).push({
              type: "video_url",
              video_url: { url },
            });
          }
        }
        apiMessages.push({ role: msg.role, content: contentParts });
      } else {
        apiMessages.push({ role: msg.role, content: msg.content });
      }
    }

    const assistantParent = this.tailId;
    const assistantId = newId();

    this.isStreaming = true;
    this.streamingBlocks = [];
    this.streamingFinishReason = null;
    this.streamingStart = performance.now();
    this.streamingFirstTokenAt = null;
    this.streamingTokens = 0;
    this.streamingTokRate = 0;
    this.streamingCharCount = 0;
    this.streamingModel = model;
    this.abortController = new AbortController();

    this.tokRateTimer = setInterval(() => {
      if (this.streamingFirstTokenAt == null) return;
      const decodeSec = (performance.now() - this.streamingFirstTokenAt) / 1000;
      if (decodeSec <= 0) return;
      this.streamingTokRate = this.streamingTokens / decodeSec;
    }, 250);

    const options: StreamOptions = {
      model: model || "default",
      temperature: settingsStore.temperature,
      top_p: settingsStore.topP,
      top_k: settingsStore.topK,
      max_tokens: settingsStore.maxTokens,
      repetition_penalty: settingsStore.repetitionPenalty,
      enable_thinking: settingsStore.enableThinking || undefined,
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

    const accountTokens = (text: string) => {
      if (this.streamingFirstTokenAt == null) this.streamingFirstTokenAt = performance.now();
      this.streamingCharCount += text.length;
      this.streamingTokens = Math.max(1, Math.round(this.streamingCharCount / 4));
    };

    await streamChatCompletion(apiMessages, options, {
      onContent: (text) => {
        accountTokens(text);
        const last = this.streamingBlocks[this.streamingBlocks.length - 1];
        if (last?.type === "content") {
          last.content += text;
          this.streamingBlocks = [...this.streamingBlocks];
        } else {
          this.streamingBlocks = [...this.streamingBlocks, { type: "content", content: text }];
        }
      },
      onReasoning: (text) => {
        accountTokens(text);
        const last = this.streamingBlocks[this.streamingBlocks.length - 1];
        if (last?.type === "reasoning") {
          last.content += text;
          this.streamingBlocks = [...this.streamingBlocks];
        } else {
          this.streamingBlocks = [...this.streamingBlocks, { type: "reasoning", content: text }];
        }
      },
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
              existingData.tool_type === "code_execution"
            ) {
              if (!newData.code && existingData.code) {
                newData.code = existingData.code;
              }
            }
            if (newData.tool_type === "web_search" && existingData.tool_type === "web_search") {
              if (!newData.query && existingData.query) {
                newData.query = existingData.query;
              }
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
        const block: AgentToolApprovalBlock = {
          ...event,
          status: "pending",
        };
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
        if (this.currentChatId) {
          api
            .saveChatSession(this.currentChatId, id)
            .catch((e) => console.error("Failed to save chat session:", e));
        }
      },
      onDone: () => {
        this.finalizeStreaming(assistantId, assistantParent);
      },
      onError: (error) => {
        const errorText = `**Error:** ${error}`;
        const last = this.streamingBlocks[this.streamingBlocks.length - 1];
        if (last?.type === "content") {
          last.content += `\n\n${errorText}`;
          this.streamingBlocks = [...this.streamingBlocks];
        } else {
          this.streamingBlocks = [...this.streamingBlocks, { type: "content", content: errorText }];
        }
        this.finalizeStreaming(assistantId, assistantParent);
      },
    });
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
      block.data = {
        ...block.data,
        status: "error",
        error: String(e),
      };
    }
    this.streamingBlocks = [...this.streamingBlocks];
  }

  private async finalizeStreaming(assistantId: string, parentId: string | null) {
    const now = performance.now();
    const elapsedMs =
      this.streamingStart != null ? Math.max(0, now - this.streamingStart) : undefined;
    const ttftMs =
      this.streamingStart != null && this.streamingFirstTokenAt != null
        ? Math.max(0, this.streamingFirstTokenAt - this.streamingStart)
        : undefined;
    const tokens = this.streamingTokens || undefined;
    const model = this.streamingModel ?? undefined;

    if (this.streamingBlocks.length) {
      const fullContent = this.streamingBlocks
        .filter((b): b is { type: "content"; content: string } => b.type === "content")
        .map((b) => b.content)
        .join("");

      const sessionId = this.currentSessionId ?? undefined;
      const assistantMsg: DisplayMessage = {
        id: assistantId,
        parentId,
        role: "assistant",
        content: fullContent,
        blocks: [...this.streamingBlocks],
        finishReason: this.streamingFinishReason ?? undefined,
        elapsedMs,
        ttftMs,
        tokens,
        model,
        sessionId,
      };
      this.insertNode(assistantMsg);

      if (this.currentChatId) {
        api
          .appendMessage(
            this.currentChatId,
            "assistant",
            fullContent,
            undefined,
            undefined,
            assistantMsg.blocks,
            assistantMsg.finishReason ?? undefined,
            { elapsed_ms: elapsedMs, ttft_ms: ttftMs, tokens, model, session_id: sessionId },
            assistantId,
            parentId,
          )
          .catch((e) => console.error("Failed to persist assistant message:", e));
      }
    }

    if (this.tokRateTimer) {
      clearInterval(this.tokRateTimer);
      this.tokRateTimer = null;
    }
    this.streamingBlocks = [];
    this.streamingFinishReason = null;
    this.streamingStart = null;
    this.streamingFirstTokenAt = null;
    this.streamingTokens = 0;
    this.streamingTokRate = 0;
    this.streamingCharCount = 0;
    this.streamingModel = null;
    this.isStreaming = false;
    this.abortController = null;
  }

  stopStreaming() {
    this.abortController?.abort();
  }

  // ===== edit / regenerate / branch =====

  /** Rewrite a message's content in place. Drops every descendant. If it's a user message, re-runs generation. */
  async editMessage(messageId: string, newContent: string) {
    if (this.isStreaming) return;
    const node = this.allNodes.get(messageId);
    if (!node) return;

    // Drop all descendants of `messageId`.
    const descendants = new Set<string>();
    let changed = true;
    while (changed) {
      changed = false;
      for (const [id, n] of this.allNodes) {
        if (id === messageId) continue;
        if (descendants.has(id)) continue;
        if (n.parentId === messageId || (n.parentId && descendants.has(n.parentId))) {
          descendants.add(id);
          changed = true;
        }
      }
    }
    for (const id of descendants) this.allNodes.delete(id);

    node.content = newContent;
    this.allNodes.set(node.id, node);
    this.allNodes = new Map(this.allNodes);
    this.tailId = messageId;
    this.rebuildPath();

    if (this.currentChatId) {
      try {
        await api.editMessage(this.currentChatId, messageId, newContent);
      } catch (e) {
        console.error("Failed to persist edit:", e);
      }
    }

    if (node.role === "user") {
      const model = modelStore.selectedModel;
      if (!model) return;

      // Fork the agentic session so the edited branch diverges cleanly from the prior turns.
      const priorTurns = this.messages.filter(
        (m) => m.role === "assistant" && m.parentId !== messageId,
      ).length;
      const srcSessionId = this.lastAssistantSessionId();
      if (srcSessionId && priorTurns > 0) {
        const newSessionId = newId();
        try {
          await api.forkSession(srcSessionId, newSessionId, priorTurns);
          this.currentSessionId = newSessionId;
        } catch (e) {
          console.error("Failed to fork session:", e);
        }
      } else {
        this.currentSessionId = null;
      }

      await this.generateAssistant(model);
    }
  }

  /** Find the session id stamped on the most recent assistant message on the active path. */
  private lastAssistantSessionId(): string | null {
    for (let i = this.messages.length - 1; i >= 0; i--) {
      const m = this.messages[i];
      if (m.role === "assistant" && m.sessionId) return m.sessionId;
    }
    return null;
  }

  /** Create a new assistant sibling under the same user message. The old assistant stays in the tree. */
  async regenerateAssistant(assistantId: string) {
    if (this.isStreaming) return;
    const node = this.allNodes.get(assistantId);
    if (!node || node.role !== "assistant" || !node.parentId) return;
    const model = modelStore.selectedModel;
    if (!model) return;

    this.tailId = node.parentId;
    this.rebuildPath();

    if (this.currentChatId) {
      try {
        await api.setTail(this.currentChatId, node.parentId);
      } catch (e) {
        console.error("Failed to set tail:", e);
      }
    }

    // Fork the agentic session so branches have independent state. Count complete turns BEFORE the
    // regenerated assistant: those are the prior user/assistant pairs that survive the fork.
    const priorTurns = this.messages.filter(
      (m) => m.role === "assistant" && m.id !== assistantId,
    ).length;
    const srcSessionId = node.sessionId ?? this.currentSessionId;
    if (srcSessionId && priorTurns > 0) {
      const newSessionId = newId();
      try {
        await api.forkSession(srcSessionId, newSessionId, priorTurns);
        this.currentSessionId = newSessionId;
      } catch (e) {
        console.error("Failed to fork session:", e);
      }
    } else {
      this.currentSessionId = null;
    }

    await this.generateAssistant(model);
  }

  /** Switch the active path to follow `siblingId`'s subtree. Navigates to that branch's deepest leaf. */
  async switchBranch(siblingId: string) {
    if (this.isStreaming) return;
    if (!this.allNodes.has(siblingId)) return;
    const leaf = this.deepestDescendant(siblingId);
    this.tailId = leaf;
    this.rebuildPath();
    this.currentSessionId = this.lastAssistantSessionId();
    if (this.currentChatId) {
      try {
        await api.setTail(this.currentChatId, leaf);
      } catch (e) {
        console.error("Failed to set tail:", e);
      }
    }
  }

  // ===== load / new / delete =====

  async loadChat(id: string) {
    const chat = await api.loadChat(id);
    this.currentChatId = id;

    // Build the tree. Legacy chats may lack ids/parent_ids; treat the array as a linear chain.
    const nodes = new Map<string, DisplayMessage>();
    let prevId: string | null = null;
    for (const m of chat.messages) {
      const id = m.id ?? newId();
      const parentId = m.parent_id ?? prevId;
      nodes.set(id, {
        id,
        parentId,
        role: m.role as "user" | "assistant",
        content: m.content,
        images: m.images,
        videos: m.videos,
        blocks: m.blocks,
        finishReason: m.finish_reason,
        elapsedMs: m.elapsed_ms,
        ttftMs: m.ttft_ms,
        tokens: m.tokens,
        model: m.model,
        sessionId: m.session_id,
      });
      prevId = id;
    }
    this.allNodes = nodes;

    // Prefer the server-stored tail; otherwise fall back to the last message.
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
    // Prefer the active branch's per-message session id; fall back to the sidecar
    // if the branch carries no assistant-stamped session yet (e.g. legacy chats).
    this.currentSessionId = this.lastAssistantSessionId() ?? restored?.session_id ?? null;
  }

  async newChat() {
    this.currentChatId = null;
    this.currentSessionId = null;
    this.allNodes = new Map();
    this.tailId = null;
    this.messages = [];
    this.streamingBlocks = [];
    this.isStreaming = false;
  }

  async deleteChat(id: string) {
    await api.deleteChat(id);
    if (this.currentChatId === id) {
      this.newChat();
    }
  }
}

export const chatStore = new ChatStore();
