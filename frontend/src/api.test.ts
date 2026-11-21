import { describe, expect, it } from 'vitest';
import { applyChatEvent, parseSSE } from './api';
import type { Turn } from './types';

describe('SSE frame parsing', () => {
  it('retains incomplete frames and parses multiple events', () => {
    const input = 'event: delta\ndata: {"text":"你好"}\n\nevent: done\ndata: {"id":';
    const first = parseSSE(input);
    expect(first.frames).toEqual([{ event: 'delta', data: { text: '你好' } }]);
    const second = parseSSE(first.rest + '"123"}\n\n');
    expect(second.frames[0].event).toBe('done');
    expect(second.rest).toBe('');
  });
  it('handles CRLF and keepalive comments', () => {
    expect(parseSSE(': ping\r\n\r\nevent: status\r\ndata: {"stage":"retrieving"}\r\n\r\n').frames)
      .toEqual([{ event: 'status', data: { stage: 'retrieving' } }]);
  });
});

it('replaces a truncated attempt instead of duplicating it during recovery', () => {
  let turn: Turn = { id: 'test', question: '指导我', answer: '', citations: [], status: 'running' };
  turn = applyChatEvent(turn, 'delta', { text: '未完成的第一段' });
  turn = applyChatEvent(turn, 'reset', { text: '', reason: 'length' });
  turn = applyChatEvent(turn, 'delta', { text: '完整答案 [1]' });
  turn = applyChatEvent(turn, 'done', {});
  expect(turn.answer).toBe('完整答案 [1]');
  expect(turn.status).toBe('completed');
  expect(turn.can_resume).toBe(false);
});

it('preserves the explicit recovery flag when a streamed answer fails', () => {
  const turn: Turn = { id: 'test', question: '指导我', answer: '草稿', citations: [], status: 'running' };
  const failed = applyChatEvent(turn, 'error', { code: 'generation_incomplete', can_resume: true });
  expect(failed.status).toBe('failed');
  expect(failed.can_resume).toBe(true);
  expect(failed.citations).toEqual([]);
});
