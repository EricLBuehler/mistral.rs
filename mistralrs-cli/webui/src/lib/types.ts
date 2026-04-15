// === Chat Completion API Types ===

export interface ChatCompletionMessage {
  role: "system" | "user" | "assistant";
  content: string | MessageContent[];
  images?: string[];
}

export interface MessageContent {
  type: "text" | "image_url";
  text?: string;
  image_url?: { url: string };
}

export interface ChatCompletionChunk {
  id: string;
  choices: ChunkChoice[];
  model: string;
  usage?: Usage;
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

// === Agentic Tool Call Progress ===

export interface AgenticToolCallProgress {
  type: "agentic_tool_call_progress";
  round: number;
  tool_name: string;
  phase: "calling" | "complete";
  data: AgenticToolCallData;
}

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

// === UI State Types ===

export interface UiModelInfo {
  name: string;
  kind: string;
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
}

export interface ChatMessageRecord {
  role: string;
  content: string;
  images?: string[];
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

// === Display message (enriched for UI rendering) ===

export type StreamBlock =
  | { type: "reasoning"; content: string }
  | { type: "tool_call"; data: AgenticToolCallProgress };

export interface DisplayMessage {
  role: "user" | "assistant" | "system";
  content: string;
  blocks?: StreamBlock[];
  images?: string[];
}

// === Streaming options ===

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
  abortSignal?: AbortSignal;
}

export interface WebSearchOptions {
  search_context_size?: "low" | "medium" | "high";
}

export interface StreamCallbacks {
  onContent: (text: string) => void;
  onReasoning: (text: string) => void;
  onToolCallProgress: (event: AgenticToolCallProgress) => void;
  onDone: () => void;
  onError: (error: string) => void;
}
