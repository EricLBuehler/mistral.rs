import type {
  DisplayMessage,
  AgenticToolCallProgress,
  ChatCompletionMessage,
  StreamOptions,
  StreamBlock,
  CodeExecutionData,
  File as ProducedFile,
} from "../types";
import { streamChatCompletion } from "../services/streaming";
import * as api from "../services/api";
import { settingsStore } from "./settings.svelte";
import { modelStore } from "./models.svelte";

class ChatStore {
  messages = $state<DisplayMessage[]>([]);
  currentChatId = $state<string | null>(null);
  /** Agentic session ID — preserves tool history, code execution variables,
   * and accumulated images across turns. Persisted in a sidecar file so it
   * survives server restarts. */
  currentSessionId = $state<string | null>(null);
  isStreaming = $state(false);

  /** Ordered blocks so the model can interleave content, reasoning, and tool calls. */
  streamingBlocks = $state<StreamBlock[]>([]);
  streamingFinishReason = $state<string | null>(null);

  /** Wall-clock start of the current streaming response (performance.now()). */
  streamingStart = $state<number | null>(null);
  /** Timestamp of the first token (content or reasoning) for TTFT. */
  streamingFirstTokenAt = $state<number | null>(null);
  /** Live token count for the current streaming response (4-char estimate). */
  streamingTokens = $state(0);
  /** Live decode tokens-per-second over the last 1.5s window. */
  streamingTokRate = $state(0);

  private abortController: AbortController | null = null;
  private streamingModel: string | null = null;
  private streamingCharCount = 0;
  private tokRateTimer: ReturnType<typeof setInterval> | null = null;

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
      role: "user",
      content,
      images: imageUrls,
      videos: videoUrls,
    };
    this.messages.push(userMsg);

    api
      .appendMessage(this.currentChatId, "user", content, imageUrls, videoUrls)
      .catch((e) => console.error("Failed to persist user message:", e));

    const apiMessages: ChatCompletionMessage[] = [];

    if (settingsStore.systemPrompt.trim()) {
      apiMessages.push({
        role: "system",
        content: settingsStore.systemPrompt,
      });
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
            (contentParts as Array<{type: string; image_url?: {url: string}}>).push({
              type: "image_url",
              image_url: { url },
            });
          }
        }
        if (msg.videos?.length) {
          for (const url of msg.videos) {
            (contentParts as Array<{type: string; video_url?: {url: string}}>).push({
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

    // Decode tok/s since first token. Same definition as the per-message footer, so the
    // live counter and the final number match. Tracks real aggregate behavior with no
    // smoothing constants; variance shrinks naturally as the sample grows.
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
      session_id: this.currentSessionId ?? undefined,
      abortSignal: this.abortController.signal,
    };

    if (settingsStore.enableSearch && modelStore.capabilities.search_enabled) {
      options.web_search_options = { search_context_size: "medium" };
    }
    if (
      settingsStore.enableCodeExecution &&
      modelStore.capabilities.code_execution_enabled
    ) {
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
          this.streamingBlocks = [
            ...this.streamingBlocks,
            { type: "content", content: text },
          ];
        }
      },
      onReasoning: (text) => {
        accountTokens(text);
        const last = this.streamingBlocks[this.streamingBlocks.length - 1];
        if (last?.type === "reasoning") {
          last.content += text;
          this.streamingBlocks = [...this.streamingBlocks];
        } else {
          this.streamingBlocks = [
            ...this.streamingBlocks,
            { type: "reasoning", content: text },
          ];
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
          // The complete phase often lacks fields set in the calling phase; preserve them.
          if (event.phase === "complete") {
            const existingData = existing.data.data;
            const newData = event.data;
            if (newData.tool_type === "code_execution" && existingData.tool_type === "code_execution") {
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
          this.streamingBlocks = [
            ...this.streamingBlocks,
            { type: "tool_call", data: event },
          ];
        }
      },
      onFile: (file: ProducedFile) => {
        // Replace any existing block with the same id (server may emit updated metadata).
        const idx = this.streamingBlocks.findIndex(
          (b) => b.type === "file" && b.data.id === file.id,
        );
        if (idx >= 0) {
          (this.streamingBlocks[idx] as { type: "file"; data: ProducedFile }).data = file;
          this.streamingBlocks = [...this.streamingBlocks];
        } else {
          this.streamingBlocks = [
            ...this.streamingBlocks,
            { type: "file", data: file },
          ];
        }
      },
      onFinishReason: (reason) => {
        this.streamingFinishReason = reason;
      },
      onSessionId: (id) => {
        this.currentSessionId = id;
        // Write a sidecar so agentic state survives a server restart.
        if (this.currentChatId) {
          api
            .saveChatSession(this.currentChatId, id)
            .catch((e) => console.error("Failed to save chat session:", e));
        }
      },
      onDone: () => {
        this.finalizeStreaming();
      },
      onError: (error) => {
        const errorText = `**Error:** ${error}`;
        const last = this.streamingBlocks[this.streamingBlocks.length - 1];
        if (last?.type === "content") {
          last.content += `\n\n${errorText}`;
          this.streamingBlocks = [...this.streamingBlocks];
        } else {
          this.streamingBlocks = [
            ...this.streamingBlocks,
            { type: "content", content: errorText },
          ];
        }
        this.finalizeStreaming();
      },
    });
  }

  private async finalizeStreaming() {
    const now = performance.now();
    const elapsedMs = this.streamingStart != null
      ? Math.max(0, now - this.streamingStart)
      : undefined;
    const ttftMs = this.streamingStart != null && this.streamingFirstTokenAt != null
      ? Math.max(0, this.streamingFirstTokenAt - this.streamingStart)
      : undefined;
    const tokens = this.streamingTokens || undefined;
    const model = this.streamingModel ?? undefined;

    if (this.streamingBlocks.length) {
      // Concatenate content blocks for the API conversation history's `content` field.
      // (Blocks remain the source of truth for display.)
      const fullContent = this.streamingBlocks
        .filter((b): b is { type: "content"; content: string } => b.type === "content")
        .map((b) => b.content)
        .join("");

      const assistantMsg: DisplayMessage = {
        role: "assistant",
        content: fullContent,
        blocks: [...this.streamingBlocks],
        finishReason: this.streamingFinishReason ?? undefined,
        elapsedMs,
        ttftMs,
        tokens,
        model,
      };
      this.messages.push(assistantMsg);

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
            { elapsed_ms: elapsedMs, ttft_ms: ttftMs, tokens, model },
          )
          .catch((e) =>
            console.error("Failed to persist assistant message:", e),
          );
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

  async loadChat(id: string) {
    const chat = await api.loadChat(id);
    this.currentChatId = id;
    this.messages = chat.messages.map((m) => ({
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
    }));

    // Restore the agentic session into the engine's in-memory store so the
    // next turn can reuse tool history, code execution variables, etc.
    try {
      const result = await api.restoreChatSession(id);
      this.currentSessionId = result.session_id;
    } catch (e) {
      console.error("Failed to restore chat session:", e);
      this.currentSessionId = null;
    }
  }

  async newChat() {
    this.currentChatId = null;
    this.currentSessionId = null;
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
