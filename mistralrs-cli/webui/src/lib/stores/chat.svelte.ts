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
  // Agentic session ID — preserves tool history, code execution variables,
  // and accumulated images across turns. Persisted in a sidecar file so it
  // survives server restarts.
  currentSessionId = $state<string | null>(null);
  isStreaming = $state(false);

  // Accumulated streaming state. All output (content, reasoning, tool calls)
  // is captured as ordered blocks so the model can interleave them.
  streamingBlocks = $state<StreamBlock[]>([]);
  streamingFinishReason = $state<string | null>(null);

  private abortController: AbortController | null = null;

  async sendMessage(content: string, imageUrls?: string[], videoUrls?: string[]) {
    if (!content.trim() && !imageUrls?.length && !videoUrls?.length) return;
    if (this.isStreaming) return;

    const model = modelStore.selectedModel;
    if (!model) return;

    // Create chat if needed
    try {
      if (!this.currentChatId) {
        const { id } = await api.newChat(model);
        this.currentChatId = id;
      }
    } catch (e) {
      console.error("Failed to create chat:", e);
      return;
    }

    // Add user message
    const userMsg: DisplayMessage = {
      role: "user",
      content,
      images: imageUrls,
      videos: videoUrls,
    };
    this.messages.push(userMsg);

    // Persist user message (fire-and-forget)
    api
      .appendMessage(this.currentChatId, "user", content, imageUrls, videoUrls)
      .catch((e) => console.error("Failed to persist user message:", e));

    // Build the messages array for the API
    const apiMessages: ChatCompletionMessage[] = [];

    // System prompt
    if (settingsStore.systemPrompt.trim()) {
      apiMessages.push({
        role: "system",
        content: settingsStore.systemPrompt,
      });
    }

    // History
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

    // Start streaming
    this.isStreaming = true;
    this.streamingBlocks = [];
    this.streamingFinishReason = null;
    this.abortController = new AbortController();

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

    await streamChatCompletion(apiMessages, options, {
      onContent: (text) => {
        // Append to last content block, or start a new one. Content is a block
        // so the model can interleave content with tool calls.
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
        // Append to last reasoning block, or start a new one
        const last = this.streamingBlocks[this.streamingBlocks.length - 1];
        if (last?.type === "reasoning") {
          last.content += text;
          // Trigger reactivity
          this.streamingBlocks = [...this.streamingBlocks];
        } else {
          this.streamingBlocks = [
            ...this.streamingBlocks,
            { type: "reasoning", content: text },
          ];
        }
      },
      onToolCallProgress: (event) => {
        // Find existing tool call block for same round+tool_name
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
          // Merge: preserve fields from calling phase that complete phase may lack
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
        // Replace any existing block with the same id (server may emit
        // updated metadata), otherwise append a new file block.
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
        // Server-side: export session from in-memory store and write a sidecar
        // file so the agentic state survives a server restart.
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
        // Add the error as a content block so it shows in the chat
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
    if (this.streamingBlocks.length) {
      // Concatenate content blocks for the API conversation history `content` field.
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
      };
      this.messages.push(assistantMsg);

      // Persist (blocks are the source of truth; content is derived for API history)
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
          )
          .catch((e) =>
            console.error("Failed to persist assistant message:", e),
          );
      }
    }

    this.streamingBlocks = [];
    this.streamingFinishReason = null;
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
