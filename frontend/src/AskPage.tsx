import { useEffect, useRef, useState } from 'react';
import { App, Alert, Button, Drawer, Input, Modal, Select, Spin, Tag, Tooltip } from 'antd';
import { ArrowUp, ArrowUpRight, BookOpenText, ChatCircleDots, Check, FileText, Plus, Quotes, Stop, Trash } from '@phosphor-icons/react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { api, applyChatEvent, chatStream } from './api';
import { LoadError, useLoad } from './components';
import type { Chunk, Citation, Conversation, Turn } from './types';

const stageLabels: Record<string, string> = { understanding: '结合上下文思考问题', retrieving: '选择策略查找资料', assessing: '评估证据是否回答问题', checking_sources: '核验此前资料', generating: '思考并归纳回答', recovering: '正在恢复未完成的回答', resuming: '重新完成上次的回答' };

export function AskPage({ kbIds, hasKnowledge }: { kbIds: string[]; hasKnowledge: boolean }) {
  const history = useLoad<Conversation[]>('/conversations', []);
  const [conversation, setConversation] = useState<string | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]), [question, setQuestion] = useState('');
  const [busy, setBusy] = useState(false), [stage, setStage] = useState('');
  const [loading, setLoading] = useState(false), [error, setError] = useState('');
  const [source, setSource] = useState<Citation | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<Conversation | null>(null);
  const [deleting, setDeleting] = useState(false);
  const interactionLocked = busy || loading || deleting || !!deleteTarget;
  const controller = useRef<AbortController | null>(null);
  const thread = useRef<HTMLDivElement | null>(null);
  const { message } = App.useApp();
  useEffect(() => () => controller.current?.abort(), []);
  useEffect(() => { thread.current?.scrollTo({ top: thread.current.scrollHeight, behavior: 'auto' }); }, [turns, stage]);
  async function loadConversation(id: string) {
    if (interactionLocked) return;
    setLoading(true); setError('');
    try { const result = await api<{ turns: Turn[] }>(`/conversations/${id}`); setConversation(id); setTurns(result.turns); setSource(null); }
    catch (e) { setError((e as Error).message); } finally { setLoading(false); }
  }
  function newConversation() {
    if (interactionLocked) return;
    setConversation(null); setTurns([]); setQuestion(''); setSource(null); setError('');
  }
  async function deleteConversation() {
    if (!deleteTarget || deleting) return;
    setDeleting(true);
    try {
      await api(`/conversations/${deleteTarget.id}`, { method: 'DELETE' });
      history.setData(previous => previous.filter(c => c.id !== deleteTarget.id));
      if (conversation === deleteTarget.id) {
        setConversation(null); setTurns([]); setQuestion(''); setSource(null); setError('');
      }
      setDeleteTarget(null); history.reload(); message.success('对话已删除');
    } catch (e) { message.error((e as Error).message); }
    finally { setDeleting(false); }
  }
  async function openSource(citation: Citation) {
    try {
      const current = await api<Chunk>(`/chunks/${citation.chunk_id}`);
      setSource({ ...citation, sources: current.sources, heading_path: current.heading_path });
    } catch (e) { setSource(null); message.error((e as Error).message); }
  }
  async function send(value = question) {
    if (!value.trim() || interactionLocked) return;
    if (!kbIds.length) { message.warning('请在右上方选择知识库'); return; }
    setBusy(true); setError(''); setStage('understanding'); setQuestion('');
    const pendingId = crypto.randomUUID();
    const abort = new AbortController(); controller.current = abort;
    try {
      let id = conversation;
      if (!id) { const created = await api<Conversation>('/conversations', { method: 'POST' }); id = created.id; setConversation(id); }
      setTurns(previous => [...previous, { id: pendingId, question: value, answer: '', citations: [], status: 'running' }]);
      const update = (apply: (turn: Turn) => Turn) => setTurns(previous => previous.map(t => t.id === pendingId ? apply(t) : t));
      await chatStream({ question: value, conversation_id: id, kb_ids: kbIds }, abort.signal, (event, data) => {
        if (event === 'status') {
          const label = stageLabels[String(data.stage)] || '正在处理';
          const strategy = ({ hybrid: '混合检索', dense: '语义检索', sparse: '关键词检索' } as Record<string, string>)[String(data.strategy)];
          setStage(data.stage === 'retrieving' && data.search_number ? `第 ${data.search_number} 轮 · ${strategy || label}` : label);
        }
        update(t => applyChatEvent(t, event, data));
      });
    } catch (e) {
      const detail = abort.signal.aborted ? '已停止生成，本次回答未完成。' : (e as Error).message;
      setError(detail);
      setTurns(previous => previous.map(t => t.id === pendingId ? { ...t, status: 'failed', citations: [], can_resume: t.can_resume || abort.signal.aborted } : t));
    } finally { history.reload(); setBusy(false); setStage(''); controller.current = null; }
  }
  return <div className="ask-layout"><aside className="conversation-panel"><Button block icon={<Plus size={17} />} disabled={interactionLocked} onClick={newConversation}>新对话</Button><div className="eyebrow recent-label">最近对话</div><LoadError error={history.error} retry={history.reload} />
    {history.data.length === 0 && <p className="muted small">你的对话会保存在这里。</p>}{history.data.map(c => <div key={c.id} className={`conversation-row ${c.id === conversation ? 'selected' : ''}`}>
      <button className="conversation-item" title={c.title} aria-current={c.id === conversation ? 'true' : undefined} onClick={() => void loadConversation(c.id)} disabled={interactionLocked}><ChatCircleDots size={17} /><span>{c.title}</span></button>
      <Tooltip title="删除对话"><Button type="text" className="conversation-delete" size="small" aria-label={`删除对话：${c.title}`} icon={<Trash size={16} />} disabled={interactionLocked || history.loading} onClick={() => setDeleteTarget(c)} /></Tooltip>
    </div>)}<div className="conversation-foot"><Quotes size={24} weight="duotone" /><p>有依据的回答，<br />来自沉淀下来的知识。</p></div>
    </aside><div className="chat-workspace"><div className="mobile-conversations"><Select aria-label="最近对话" placeholder="最近对话" value={conversation || undefined} disabled={interactionLocked} options={history.data.map(c => ({ value: c.id, label: c.title }))} onChange={id => void loadConversation(id)} /><Button icon={<Plus size={16} />} disabled={interactionLocked} onClick={newConversation}>新对话</Button><Tooltip title="删除当前对话"><Button aria-label="删除当前对话" icon={<Trash size={16} />} disabled={interactionLocked || !conversation || history.loading} onClick={() => { const selected = history.data.find(c => c.id === conversation); if (selected) setDeleteTarget(selected); }} /></Tooltip></div><div className="chat-thread" ref={thread} aria-live="polite">
      {loading ? <div className="chat-loading"><Spin /></div> : turns.length === 0 ? <div className="chat-welcome"><span className="welcome-icon"><BookOpenText size={37} weight="duotone" /></span><div className="eyebrow">YOUR TEAM'S KNOWLEDGE, CONNECTED</div><h1>从一个问题，<br /><span>找到团队的答案。</span></h1><p>在授权的知识库中查找、理解与整合资料。<br />每个回答，都可以回到原文核验。</p>
        <div className="suggestion-list">{['概括这份资料中的关键结论', '查找相关规定，并列出原文依据', '这些文档中有哪些需要关注的时间节点？'].map(text => <button key={text} onClick={() => setQuestion(text)} disabled={!hasKnowledge}><span>{text}</span><ArrowUpRight size={18} /></button>)}</div>
        {!hasKnowledge && <Alert type="info" message="还没有可访问的知识库，请先创建知识库或联系管理员授权。" showIcon />}
      </div> : turns.map((turn, index) => <article key={turn.id} className="chat-turn"><div className="user-question"><span>你</span><p>{turn.question}</p></div>{turn.status === 'redacted' ? <Alert type="info" message="相关资料已更新、删除或撤销访问权限。" /> : <div className="assistant-answer"><div className="answer-label"><span className="answer-logo"><BooksIcon /></span><strong>知衡</strong>{turn.status === 'completed' && <span className="verified"><Check size={13} />已完成</span>}{turn.status === 'failed' && <Tag color="warning">未完成</Tag>}</div>
        <div className="answer-body"><ReactMarkdown remarkPlugins={[remarkGfm]} components={{ a: ({ children }) => <span>{children}</span>, img: () => null }}>{turn.answer || turn.draft_answer || ''}</ReactMarkdown>{turn.status === 'running' && !turn.answer && <div className="thinking"><Spin size="small" />{stageLabels[stage] || stage || '正在处理'}</div>}</div>
        {turn.citations.length > 0 && <div className="citations"><div className="citations-label">回答依据 · {turn.citations.length} 个片段</div>{turn.citations.map(c => <button key={c.number} className="citation-chip" onClick={() => void openSource(c)}><span>{c.number}</span><FileText size={16} /><strong>{c.title}</strong><ArrowUpRight size={15} /></button>)}</div>}
        {turn.status === 'failed' && <Alert type="warning" message={turn.can_resume ? '本次回答未完成或未通过校验，已保留问题与资料线索。草稿尚未核验，可重新完成回答。' : '本次回答未通过校验，请重试或补充问题。'} showIcon action={turn.can_resume && index === turns.length - 1 ? <Button size="small" disabled={interactionLocked} onClick={() => void send('继续完成上一条未完成的回答')}>重新完成</Button> : undefined} />}
        {turn.status === 'completed' && index === turns.length - 1 && <Button type="text" disabled={interactionLocked} onClick={() => void send('继续')}>继续展开</Button>}
      </div>}</article>)}
    </div><div className="composer-area">{error && <Alert type="error" message={error} closable onClose={() => setError('')} className="form-alert" />}{busy && <div className="generation-status"><span className="status-dot" />{stageLabels[stage] || stage || '处理中'}</div>}
      <form className="composer" onSubmit={e => { e.preventDefault(); void send(); }}><Input.TextArea aria-label="输入你的问题" placeholder="向团队知识库提问…" value={question} onChange={e => setQuestion(e.target.value)} autoSize={{ minRows: 2, maxRows: 6 }} maxLength={4000} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); void send(); } }} /><div className="composer-bottom"><span><span className="scope-dot" />{kbIds.length ? `已选择 ${kbIds.length} 个知识库` : '请先选择知识库'}</span><Tooltip title={busy ? '停止生成' : '发送问题'}><Button type="primary" shape="circle" aria-label={busy ? '停止生成' : '发送问题'} htmlType={busy ? 'button' : 'submit'} disabled={!busy && (interactionLocked || !question.trim() || !kbIds.length)} icon={busy ? <Stop size={17} weight="fill" /> : <ArrowUp size={21} />} onClick={busy ? () => controller.current?.abort() : undefined} /></Tooltip></div></form><div className="composer-note">回答基于所选资料生成，重要信息请结合原文核验。<span>Enter 发送 · Shift + Enter 换行</span></div>
    </div></div><Drawer title="查看回答依据" width={620} open={!!source} onClose={() => setSource(null)}>{source && <><Tag color="success">引用 [{source.number}]</Tag><h2>{source.title}</h2><p className="muted">{source.heading_path.join(' / ')}</p>{source.sources.map((s, i) => <div className="source-excerpt" key={i}><div className="eyebrow">{s.page ? `第 ${s.page} 页` : `第 ${s.paragraph} 段`}</div><pre className="text-content">{s.original}</pre></div>)}<Button href={`/api/v1/documents/${source.document_id}/download?version_id=${source.version_id}`}>下载引用原文</Button></>}</Drawer>
    <Modal title="删除对话" open={!!deleteTarget} okText="删除对话" cancelText="取消" okButtonProps={{ danger: true }} confirmLoading={deleting} cancelButtonProps={{ disabled: deleting }} closable={!deleting} maskClosable={!deleting} keyboard={!deleting} onOk={() => void deleteConversation()} onCancel={() => { if (!deleting) setDeleteTarget(null); }}>
      <p>确定删除“{deleteTarget?.title}”及其中的全部问答记录吗？</p><p className="muted">删除后无法恢复，知识库中的文档不受影响。</p>
    </Modal>
  </div>;
}

function BooksIcon() { return <BookOpenText size={19} weight="fill" />; }
