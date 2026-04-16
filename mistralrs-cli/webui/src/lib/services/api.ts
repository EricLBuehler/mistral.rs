import type {
  UiModelInfo,
  ChatFile,
  Capabilities,
  Settings,
} from "../types";

function getBase(): string {
  return document.querySelector("base")?.getAttribute("href") ?? "/ui/";
}

function apiUrl(path: string): string {
  return `${getBase()}api/${path}`;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(apiUrl(path));
  if (!res.ok) throw new Error(`GET ${path}: ${res.statusText}`);
  return res.json();
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(apiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body != null ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`POST ${path}: ${res.statusText}`);
  return res.json();
}

async function postVoid(path: string, body?: unknown): Promise<void> {
  const res = await fetch(apiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body != null ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`POST ${path}: ${res.statusText}`);
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

export async function listChats(): Promise<{ chats: ChatFile[] }> {
  return get("list_chats");
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

export async function appendMessage(
  id: string,
  role: string,
  content: string,
  images?: string[],
  blocks?: unknown[],
): Promise<void> {
  await postVoid("append_message", { id, role, content, images, blocks });
}

// === Settings & Capabilities ===

export async function getSettings(): Promise<Settings> {
  return get("settings");
}

export async function getCapabilities(): Promise<Capabilities> {
  return get("capabilities");
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

// === Speech ===

export async function generateSpeech(
  text: string,
): Promise<{ url: string }> {
  return post("generate_speech", { text });
}
