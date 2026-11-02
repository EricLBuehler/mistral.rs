import type {
  UiModelInfo,
  ChatFile,
  ChatMessageRecord,
  ChatSummary,
  ChatSearchHit,
  Capabilities,
  Settings,
} from "../types";

function getBase(): string {
  return document.querySelector("base")?.getAttribute("href") ?? "/ui/";
}

function apiUrl(path: string): string {
  return `${getBase()}api/${path}`;
}

async function httpError(method: string, path: string, res: Response): Promise<Error> {
  const body = await res.text().catch(() => "");
  return new Error(`${method} ${path}: HTTP ${res.status}${body ? ` ${body}` : ""}`);
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(apiUrl(path));
  if (!res.ok) throw await httpError("GET", path, res);
  return res.json();
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(apiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body != null ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw await httpError("POST", path, res);
  return res.json();
}

async function postVoid(path: string, body?: unknown): Promise<void> {
  const res = await fetch(apiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body != null ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw await httpError("POST", path, res);
}

// === Model endpoints ===

export async function listModels(): Promise<{
  models: UiModelInfo[];
}> {
  return get("list_models");
}

export async function selectModel(name: string): Promise<void> {
  await postVoid("select_model", { name });
}

// === Chat endpoints ===

export async function listChats(): Promise<{ chats: ChatSummary[] }> {
  return get("list_chats");
}

export async function searchChats(query: string): Promise<{ chats: ChatSearchHit[] }> {
  return get(`search_chats?q=${encodeURIComponent(query)}`);
}

export async function newChat(model: string): Promise<{ id: string }> {
  return post("new_chat", { model });
}

export async function deleteChat(id: string): Promise<void> {
  await postVoid("delete_chat", { id });
}

export async function loadChat(id: string): Promise<ChatFile> {
  return post("load_chat", { id });
}

export async function renameChat(
  id: string,
  title: string,
): Promise<void> {
  await postVoid("rename_chat", { id, title });
}

export async function appendMessage(chatId: string, message: ChatMessageRecord): Promise<void> {
  await postVoid("append_message", { chat_id: chatId, message });
}

export async function setTail(chatId: string, tail: string | null): Promise<void> {
  await postVoid("set_tail", { id: chatId, tail });
}

export async function forkSession(
  srcSessionId: string,
  destSessionId: string,
  numTurns: number,
): Promise<void> {
  await postVoid("fork_session", {
    src_session_id: srcSessionId,
    dest_session_id: destSessionId,
    num_turns: numTurns,
  });
}

// === Settings & Capabilities ===

export async function getSettings(): Promise<Settings> {
  return get("settings");
}

export async function getCapabilities(): Promise<Capabilities> {
  return get("capabilities");
}

export interface McpToolInfo {
  name: string;
  description: string | null;
}

export async function listMcpTools(): Promise<{ tools: McpToolInfo[] }> {
  return get("mcp_tools");
}

// === File uploads ===

async function uploadFile(
  endpoint: string,
  file: File,
): Promise<{ path: string; url: string }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(apiUrl(endpoint), {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`);
  return res.json();
}

export function uploadImage(file: File) {
  return uploadFile("upload_image", file);
}

export function uploadText(file: File) {
  return uploadFile("upload_text", file);
}

export function uploadAudio(file: File) {
  return uploadFile("upload_audio", file);
}

export function uploadVideo(file: File) {
  return uploadFile("upload_video", file);
}

// === Speech ===

export async function generateSpeech(
  text: string,
): Promise<{ url: string }> {
  return post("generate_speech", { text });
}

// === Agentic session persistence ===

export async function saveChatSession(
  chat_id: string,
  session_id: string,
): Promise<void> {
  await postVoid("save_chat_session", { chat_id, session_id });
}

export async function restoreChatSession(
  chat_id: string,
): Promise<{ session_id: string | null }> {
  return post("restore_chat_session", { chat_id });
}
