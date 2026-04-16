import type {
  DisplayMessage,
  AgenticToolCallProgress,
  ChatCompletionMessage,
  StreamOptions,
  StreamBlock,
  CodeExecutionData,
} from "../types";
import { streamChatCompletion } from "../services/streaming";
import * as api from "../services/api";
import { settingsStore } from "./settings.svelte";
import { modelStore } from "./models.svelte";

class ChatStore {
  messages = $state<DisplayMessage[]>([]);
  currentChatId = $state<string | null>(null);
  isStreaming = $state(false);

  // Accumulated streaming state
  streamingContent = $state("");
  streamingBlocks = $state<StreamBlock[]>([]);

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
    this.streamingContent = "";
    this.streamingBlocks = [];
    this.abortController = new AbortController();

    const options: StreamOptions = {
      model: model || "default",
      temperature: settingsStore.temperature,
      top_p: settingsStore.topP,
      top_k: settingsStore.topK,
      max_tokens: settingsStore.maxTokens,
      repetition_penalty: settingsStore.repetitionPenalty,
      enable_thinking: settingsStore.enableThinking || undefined,
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
        this.streamingContent += text;
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
      onDone: () => {
        this.finalizeStreaming();
      },
      onError: (error) => {
        if (this.streamingContent) {
          this.streamingContent += `\n\n**Error:** ${error}`;
        } else {
          this.streamingContent = `**Error:** ${error}`;
        }
        this.finalizeStreaming();
      },
    });
  }

  private async finalizeStreaming() {
    if (this.streamingContent || this.streamingBlocks.length) {
      const assistantMsg: DisplayMessage = {
        role: "assistant",
        content: this.streamingContent,
        blocks: this.streamingBlocks.length
          ? [...this.streamingBlocks]
          : undefined,
      };
      this.messages.push(assistantMsg);

      // Persist (including blocks for UI display on reload)
      if (this.currentChatId) {
        api
          .appendMessage(
            this.currentChatId,
            "assistant",
            this.streamingContent,
            undefined,
            assistantMsg.blocks,
          )
          .catch((e) =>
            console.error("Failed to persist assistant message:", e),
          );
      }
    }

    this.streamingContent = "";
    this.streamingBlocks = [];
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
    }));
  }

  async newChat() {
    this.currentChatId = null;
    this.messages = [];
    this.streamingContent = "";
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
