import type {
  DisplayMessage,
  AgenticToolCallProgress,
  ChatCompletionMessage,
  StreamOptions,
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
  streamingReasoning = $state("");
  streamingToolCalls = $state<AgenticToolCallProgress[]>([]);

  private abortController: AbortController | null = null;

  async sendMessage(content: string, imageUrls?: string[]) {
    if (!content.trim() && !imageUrls?.length) return;
    if (this.isStreaming) return;

    const model = modelStore.selectedModel;
    if (!model) return;

    // Create chat if needed
    if (!this.currentChatId) {
      const { id } = await api.newChat(model);
      this.currentChatId = id;
    }

    // Add user message
    const userMsg: DisplayMessage = {
      role: "user",
      content,
      images: imageUrls,
    };
    this.messages.push(userMsg);

    // Persist user message
    await api.appendMessage(
      this.currentChatId,
      "user",
      content,
      imageUrls,
    );

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
      if (msg.images?.length) {
        apiMessages.push({
          role: msg.role,
          content: [
            { type: "text", text: msg.content },
            ...msg.images.map((url) => ({
              type: "image_url" as const,
              image_url: { url },
            })),
          ],
        });
      } else {
        apiMessages.push({ role: msg.role, content: msg.content });
      }
    }

    // Start streaming
    this.isStreaming = true;
    this.streamingContent = "";
    this.streamingReasoning = "";
    this.streamingToolCalls = [];
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
        this.streamingReasoning += text;
      },
      onToolCallProgress: (event) => {
        // Merge with existing by round+tool_name, or append
        const idx = this.streamingToolCalls.findIndex(
          (tc) =>
            tc.round === event.round && tc.tool_name === event.tool_name,
        );
        if (idx >= 0) {
          this.streamingToolCalls[idx] = event;
        } else {
          this.streamingToolCalls.push(event);
        }
        // Trigger reactivity
        this.streamingToolCalls = [...this.streamingToolCalls];
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
    if (this.streamingContent || this.streamingReasoning) {
      const assistantMsg: DisplayMessage = {
        role: "assistant",
        content: this.streamingContent,
        reasoning: this.streamingReasoning || undefined,
        toolCalls: this.streamingToolCalls.length
          ? [...this.streamingToolCalls]
          : undefined,
      };
      this.messages.push(assistantMsg);

      // Persist
      if (this.currentChatId) {
        await api.appendMessage(
          this.currentChatId,
          "assistant",
          this.streamingContent,
        );
      }
    }

    this.streamingContent = "";
    this.streamingReasoning = "";
    this.streamingToolCalls = [];
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
    }));
  }

  async newChat() {
    this.currentChatId = null;
    this.messages = [];
    this.streamingContent = "";
    this.streamingReasoning = "";
    this.streamingToolCalls = [];
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
