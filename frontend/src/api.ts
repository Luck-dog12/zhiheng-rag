import type { Citation, Turn } from './types';

let csrfToken = '';
export function setCsrf(value: string) { csrfToken = value; }

export class ApiError extends Error {
  constructor(message: string, public code: string, public status: number) { super(message); }
}

async function check(response: Response): Promise<Response> {
  if (response.ok) return response;
  const body = await response.json().catch(() => ({}));
  const message = body.error?.message ?? (Array.isArray(body.detail) ? body.detail.map((d: { msg: string }) => d.msg).join('；') : '请求失败，请稍后重试');
  if (response.status === 401) window.dispatchEvent(new Event('rag:unauthorized'));
  throw new ApiError(message, body.error?.code ?? 'request_failed', response.status);
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body && !(init.body instanceof FormData)) headers.set('Content-Type', 'application/json');
  if (csrfToken) headers.set('X-CSRF-Token', csrfToken);
  const response = await check(await fetch('/api/v1' + path, { ...init, headers, credentials: 'include' }));
  return response.status === 204 ? undefined as T : await response.json();
}

export function parseSSE(buffer: string): { frames: { event: string; data: unknown }[]; rest: string } {
  const normalized = buffer.replace(/\r\n/g, '\n');
  const segments = normalized.split('\n\n');
  const rest = segments.pop() ?? '';
  const frames = segments.flatMap(segment => {
    const lines = segment.split('\n');
    const event = lines.find(l => l.startsWith('event:'))?.slice(6).trim() ?? 'message';
    const data = lines.filter(l => l.startsWith('data:')).map(l => l.slice(5).trimStart()).join('\n');
    return data ? [{ event, data: JSON.parse(data) as unknown }] : [];
  });
  return { frames, rest };
}

export function applyChatEvent(turn: Turn, event: string, data: Record<string, unknown>): Turn {
  if (event === 'delta') return { ...turn, answer: turn.answer + String(data.text) };
  if (event === 'reset') return { ...turn, answer: String(data.text ?? ''), draft_answer: '', citations: [], status: 'running' };
  if (event === 'citations') return { ...turn, citations: data.citations as Citation[] };
  if (event === 'done') return { ...turn, status: 'completed', draft_answer: '', can_resume: false };
  if (event === 'error') return { ...turn, status: 'failed', citations: [], error_code: String(data.code), can_resume: data.can_resume === true };
  return turn;
}

export async function chatStream(body: unknown, signal: AbortSignal, onEvent: (event: string, data: Record<string, unknown>) => void) {
  const response = await check(await fetch('/api/v1/chat', { method: 'POST', credentials: 'include', signal,
    headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': csrfToken }, body: JSON.stringify(body) }));
  const reader = response.body?.getReader();
  if (!reader) throw new Error('浏览器不支持流式响应');
  const decoder = new TextDecoder();
  let buffer = '', completed = false;
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      const parsed = parseSSE(buffer);
      buffer = parsed.rest;
      for (const frame of parsed.frames) {
        const data = frame.data as Record<string, unknown>;
        onEvent(frame.event, data);
        if (frame.event === 'error') throw new ApiError(String(data.message), String(data.code), 502);
        if (frame.event === 'done') completed = true;
      }
      if (done) break;
    }
    if (!completed) throw new Error('连接中断，本次回答尚未完成');
  } finally { await reader.cancel().catch(() => undefined); reader.releaseLock(); }
}
