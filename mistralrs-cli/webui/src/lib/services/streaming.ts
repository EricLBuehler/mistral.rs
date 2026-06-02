import type {
  ChatCompletionMessage,
  ChatCompletionChunk,
  AgenticToolCallProgress,
  AgentToolApprovalRequired,
  File as ProducedFile,
  StreamOptions,
  StreamCallbacks,
} from "../types";

function getApiBase(): string {
  const base = document.querySelector("base")?.getAttribute("href") ?? "/ui/";
  // Go from /ui/ up to /
  try {
    return new URL("../", new URL(base, window.location.origin)).pathname;
  } catch {
    return "/";
  }
}

export async function streamChatCompletion(
  messages: ChatCompletionMessage[],
  options: StreamOptions,
  callbacks: StreamCallbacks,
): Promise<void> {
  const apiBase = getApiBase();
  const url = `${apiBase}v1/chat/completions`;

  const body: Record<string, unknown> = {
    model: options.model || "default",
    messages,
    stream: true,
  };

  if (options.temperature != null) body.temperature = options.temperature;
  if (options.top_p != null) body.top_p = options.top_p;
  if (options.top_k != null) body.top_k = options.top_k;
  if (options.max_tokens != null) body.max_tokens = options.max_tokens;
  if (options.repetition_penalty != null)
    body.repetition_penalty = options.repetition_penalty;
  if (options.enable_thinking) body.enable_thinking = true;
  if (options.web_search_options)
    body.web_search_options = options.web_search_options;
  if (options.enable_code_execution) body.enable_code_execution = true;
  if (options.agent_permission) body.agent_permission = options.agent_permission;
  if (options.session_id) body.session_id = options.session_id;
  if (options.files?.length) body.files = options.files;

  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: options.abortSignal,
    });
  } catch (e: unknown) {
    if (e instanceof DOMException && e.name === "AbortError") {
      callbacks.onDone();
      return;
    }
    callbacks.onError(`Network error: ${e}`);
    return;
  }

  if (!response.ok) {
    let errorText: string;
    try {
      errorText = await response.text();
    } catch {
      errorText = response.statusText;
    }
    callbacks.onError(`HTTP ${response.status}: ${errorText}`);
    return;
  }

  const reader = response.body?.getReader();
  if (!reader) {
    callbacks.onError("No response body");
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";
  let currentEventType = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      // Keep incomplete last line in buffer
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        const trimmed = line.trim();

        if (trimmed === "") {
          // Empty line resets event type
          currentEventType = "";
          continue;
        }

        if (trimmed.startsWith("event:")) {
          currentEventType = trimmed.slice(6).trim();
          continue;
        }

        if (trimmed.startsWith(":")) {
          // SSE comment (keep-alive), ignore
          continue;
        }

        if (trimmed.startsWith("data:")) {
          const data = trimmed.slice(5).trim();

          if (data === "[DONE]") {
            callbacks.onDone();
            return;
          }

          try {
            if (currentEventType === "agentic_tool_call_progress") {
              const event = JSON.parse(data) as AgenticToolCallProgress;
              callbacks.onToolCallProgress(event);
            } else if (currentEventType === "agentic_tool_approval_required") {
              const event = JSON.parse(data) as AgentToolApprovalRequired;
              callbacks.onApprovalRequired(event);
            } else if (currentEventType === "file_produced") {
              const file = JSON.parse(data) as ProducedFile;
              callbacks.onFile(file);
            } else {
              const chunk = JSON.parse(data) as ChatCompletionChunk;
              const choice = chunk.choices?.[0];
              if (choice?.delta) {
                if (choice.delta.content) {
                  callbacks.onContent(choice.delta.content);
                }
                if (choice.delta.reasoning_content) {
                  callbacks.onReasoning(choice.delta.reasoning_content);
                }
              }
              if (choice?.finish_reason) {
                callbacks.onFinishReason(choice.finish_reason);
              }
              if (chunk.session_id) {
                callbacks.onSessionId(chunk.session_id);
              }
            }
          } catch {
            // Skip malformed JSON
          }
        }
      }
    }
  } catch (e: unknown) {
    if (!(e instanceof DOMException && e.name === "AbortError")) {
      callbacks.onError(`Stream error: ${e}`);
      return;
    }
    // Fall through to onDone() so the UI finalizes any buffered chunks.
  }

  // If stream ended without [DONE] (or was aborted), still finalize.
  callbacks.onDone();
}

export async function resolveAgentApproval(
  approvalId: string,
  decision: "approve" | "deny",
  rememberForSession: boolean,
  message?: string,
): Promise<{ status: "resolved" | "queued" | "not_found" }> {
  const apiBase = getApiBase();
  const body: Record<string, unknown> = {
    decision,
    remember_for_session: rememberForSession,
  };
  if (message) body.message = message;

  const response = await fetch(
    `${apiBase}v1/agent/approvals/${encodeURIComponent(approvalId)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  );

  if (!response.ok) {
    let text = response.statusText;
    try {
      text = await response.text();
    } catch {}
    throw new Error(`Approval ${approvalId}: HTTP ${response.status} ${text}`);
  }

  return response.json();
}
