export interface ChatCompletionMessage {
  role: "system" | "user" | "assistant";
  content: string | MessageContent[];
  images?: string[];
}

export interface MessageContent {
  type: "text" | "image_url" | "video_url";
  text?: string;
  image_url?: { url: string };
  video_url?: { url: string };
}

export interface ChatCompletionChunk {
  id: string;
  choices: ChunkChoice[];
  model: string;
  usage?: Usage;
  /** Returned in the final SSE chunk; the agentic session ID for follow-up turns. */
  session_id?: string;
}

export interface ChunkChoice {
  delta: Delta;
  finish_reason: string | null;
  index: number;
}

export interface Delta {
  role?: string;
  content?: string | null;
  reasoning_content?: string | null;
  tool_calls?: ToolCallDelta[];
}

export interface ToolCallDelta {
  index: number;
  id?: string;
  type?: string;
  function?: { name?: string; arguments?: string };
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface AgenticToolCallProgress {
  type: "agentic_tool_call_progress";
  round: number;
  tool_name: string;
  phase: "calling" | "complete";
  data: AgenticToolCallData;
}

export interface AgentToolMetadata {
  source: "built_in" | "user" | "mcp" | "external";
  kind: "code_execution" | "web_search" | "file" | "custom" | "external";
  label: string;
}

export interface AgentToolApprovalRequired {
  type: "agentic_tool_approval_required";
  approval_id: string;
  session_id: string;
  round: number;
  tool: AgentToolMetadata;
  arguments: Record<string, unknown>;
}

export type AgentToolApprovalStatus =
  | "pending"
  | "submitting"
  | "approved"
  | "denied"
  | "error";

export interface AgentToolApprovalBlock extends AgentToolApprovalRequired {
  status: AgentToolApprovalStatus;
  remember_for_session?: boolean;
  message?: string;
  error?: string;
}

export type AgentPermission = "auto" | "ask" | "deny";

export type AgenticToolCallData =
  | CodeExecutionData
  | WebSearchData
  | CustomToolData;

export interface CodeExecutionData {
  tool_type: "code_execution";
  code?: string;
  stdout?: string;
  stderr?: string;
  exception?: string;
  images_base64?: string[];
  video_frames_base64?: string[];
  video_frame_count?: number;
  working_directory?: string;
  execution_time_ms?: number;
}

export interface WebSearchData {
  tool_type: "web_search";
  query?: string;
  results_count?: number;
}

export interface CustomToolData {
  tool_type: "custom";
  arguments?: string;
  content?: string;
}

export interface FileSource {
  tool: string;
  round: number;
  turn?: number;
}

/**
 * Mirrors `mistralrs_core::File`. Body fields (`text`, `preview`,
 * `data_base64`, `error`) are flattened from `FileContent` on the wire.
 */
export interface File {
  id: string;
  name: string;
  format?: string;
  mime_type?: string;
  bytes: number;
  source: FileSource;
  /** Text body; present for text files when not elided. */
  text?: string;
  /** Short preview for text files (first ~1KB on a UTF-8 boundary). */
  preview?: string;
  /** Base64 body; present for binary files when not elided. */
  data_base64?: string;
  /** Set when the file is an error placeholder. */
  error?: { code: string; message: string };
  /** URL to fetch the body via `GET /v1/files/{id}`. */
  url?: string;
  metadata?: Record<string, string>;
}

export interface UiModelInfo {
  name: string;
  kind: string;
  input_modalities?: string[];
  output_modalities?: string[];
  generation_defaults: GenerationParams;
}

export interface GenerationParams {
  temperature: number | null;
  top_p: number | null;
  top_k: number | null;
  max_tokens: number | null;
  repetition_penalty: number | null;
  system_prompt: string | null;
}

export interface ChatFile {
  id?: string;
  title: string | null;
  model: string;
  kind: string;
  created_at: string;
  messages: ChatMessageRecord[];
  /** Server-side agentic session ID. The serialized state lives in a sidecar file. */
  session_id?: string | null;
  /** Active leaf id; walking `parent_id` from here yields the current path. */
  tail?: string | null;
}

export interface ChatMessageRecord {
  id?: string;
  parent_id?: string;
  role: string;
  content: string;
  images?: string[];
  videos?: string[];
  blocks?: StreamBlock[];
  finish_reason?: string;
  elapsed_ms?: number;
  ttft_ms?: number;
  tokens?: number;
  model?: string;
  session_id?: string;
}

export interface Capabilities {
  search_enabled: boolean;
  code_execution_enabled: boolean;
  tool_dispatch_url: string | null;
}

export interface Settings {
  defaults: GenerationParams;
  model: string | null;
  search_enabled: boolean;
  search_embedding_model: string | null;
}

export type StreamBlock =
  | { type: "reasoning"; content: string }
  | { type: "tool_call"; data: AgenticToolCallProgress }
  | { type: "approval"; data: AgentToolApprovalBlock }
  | { type: "content"; content: string }
  | { type: "file"; data: File };

export interface DisplayMessage {
  id: string;
  parentId: string | null;
  role: "user" | "assistant" | "system";
  content: string;
  blocks?: StreamBlock[];
  images?: string[];
  videos?: string[];
  finishReason?: string | null;
  /** Wall-clock duration of the streaming response in milliseconds (TTFT + decode). */
  elapsedMs?: number;
  /** Time from request send to first token in milliseconds. */
  ttftMs?: number;
  /** Approximate token count (4 chars per token client-side estimate). */
  tokens?: number;
  /** Server-reported model that produced this message. */
  model?: string;
  /** Agentic session id this assistant message was generated under. Branches each pin their own. */
  sessionId?: string;
}

export interface StreamOptions {
  model: string;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  max_tokens?: number;
  repetition_penalty?: number;
  enable_thinking?: boolean;
  web_search_options?: WebSearchOptions;
  enable_code_execution?: boolean;
  agent_permission?: AgentPermission;
  /** If set, server reuses the agentic session (tool history, code execution state). */
  session_id?: string;
  /** Required output files to declare on the request. */
  files?: RequestedFile[];
  abortSignal?: AbortSignal;
}

export interface RequestedFile {
  name: string;
  format?: string;
  description?: string;
}

export interface WebSearchOptions {
  search_context_size?: "low" | "medium" | "high";
}

export interface StreamCallbacks {
  onContent: (text: string) => void;
  onReasoning: (text: string) => void;
  onToolCallProgress: (event: AgenticToolCallProgress) => void;
  onApprovalRequired: (event: AgentToolApprovalRequired) => void;
  onFile: (file: File) => void;
  onFinishReason: (reason: string) => void;
  onSessionId: (id: string) => void;
  onDone: () => void;
  onError: (error: string) => void;
}
