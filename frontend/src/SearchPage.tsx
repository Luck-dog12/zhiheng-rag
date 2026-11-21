import { useState } from 'react';
import { Alert, Button, Collapse, Empty, Input, Space, Tag } from 'antd';
import { MagnifyingGlass, ArrowRight } from '@phosphor-icons/react';
import { api } from './api';
import { PageHeading } from './components';
import type { Hit } from './types';

type SearchResult = {
  hits: Hit[]; candidates: Hit[]; discarded: number; query: string;
  sparse_mode: 'bm25' | 'model'; top_k: number;
};

export function SearchPage({ kbIds }: { kbIds: string[] }) {
  const [question, setQuestion] = useState(''), [busy, setBusy] = useState(false), [error, setError] = useState('');
  const [result, setResult] = useState<SearchResult | null>(null);
  const selectedIds = new Set(result?.hits.map(hit => hit.chunk_id));
  const remaining = result?.candidates.filter(hit => !selectedIds.has(hit.chunk_id)) ?? [];

  return <><PageHeading eyebrow="RETRIEVAL EXPLORER" title="看见答案之前的证据" description="检查单次检索的两路召回、融合排名和重排序结果，定位知识库的检索表现。" />
    <div className="retrieval-flow"><span>稠密语义</span><b>+</b><span>{result ? (result.sparse_mode === 'model' ? '模型稀疏向量' : 'BM25 关键词') : '稀疏检索'}</span><ArrowRight size={16} /><span>RRF 融合</span><ArrowRight size={16} /><span>重排序</span><ArrowRight size={16} /><strong>{result ? `最多选取 ${result.top_k} 个片段` : '选取相关片段'}</strong></div>
    <form className="surface search-form" onSubmit={async e => {
      e.preventDefault();
      if (busy || !question.trim() || !kbIds.length) return;
      setBusy(true); setError(''); setResult(null);
      try { setResult(await api<SearchResult>('/retrieval/search', { method: 'POST', body: JSON.stringify({ question, kb_ids: kbIds }) })); }
      catch (e) { setError((e as Error).message); }
      finally { setBusy(false); }
    }}><label htmlFor="search-question">检索问题</label><div className="search-input-row"><Input id="search-question" value={question} onChange={e => setQuestion(e.target.value)} placeholder="输入具体问题、产品编号或业务关键词" size="large" maxLength={4000} /><Button type="primary" size="large" htmlType="submit" loading={busy} disabled={!kbIds.length || !question.trim()} icon={<MagnifyingGlass size={17} />}>开始检索</Button></div></form>
    {error && <Alert type="error" message={error} showIcon />}
    {!result && !busy && <div className="search-empty"><MagnifyingGlass size={36} weight="light" /><h3>从一次检索开始</h3><p>选择知识库并输入问题，比较参与重排序的候选与本次选中的片段。</p></div>}
    {result && <>
      <Alert type="info" showIcon message={`按重排序得分从高到低选中 ${result.hits.length} 个片段，共 ${result.candidates.length} 个候选`} description={`以重排序得分决定本次前 ${result.top_k} 名；稠密、稀疏的召回排名和 RRF 融合分数是前序指标，不能与重排序得分直接比较。知识问答的 Agent 可结合多轮证据进一步筛选，最终最多使用 ${result.top_k} 个片段。`} />
      <div className="section-toolbar result-toolbar"><h2>本次选中片段 <span className="muted">{result.hits.length} / {result.top_k} 个</span></h2><Tag>{result.discarded} 个无效版本已过滤</Tag></div>
      {result.hits.length === 0 ? <Empty description="没有找到可用证据" /> : result.hits.map((hit, i) => <article className="search-result" key={hit.chunk_id}>
        <div className="result-rank">{String(i + 1).padStart(2, '0')}</div><div className="result-content">
          <div className="result-title"><h3>{hit.title}</h3></div><p>{hit.summary}</p>
          <div className="result-scores"><span className="rerank-score" title={`重排序得分：${hit.rerank_score}`}>重排序第 {i + 1} 名 · 得分 {hit.rerank_score.toFixed(4)}</span><Tag>RRF 融合 {hit.rrf_score.toFixed(5)}</Tag>{Object.entries(hit.paths).map(([name, path]) => <Tag key={name}>{name === 'dense' ? '稠密' : result.sparse_mode === 'model' ? '模型稀疏' : 'BM25'} 召回第 {path.rank} 名 · {path.score.toFixed(4)}</Tag>)}</div>
          <Collapse ghost items={[{ key: 'body', label: '查看分块正文与原文', children: <><pre className="text-content">{hit.body}</pre><h4>原始证据</h4>{hit.sources.map((source, index) => <div key={index}><Tag>{source.page ? `第 ${source.page} 页` : `第 ${source.paragraph} 段`}</Tag><pre className="text-content source-text">{source.original}</pre></div>)}</> }]} />
        </div>
      </article>)}
      {remaining.length > 0 && <Collapse items={[{ key: 'all', label: `查看其余 ${remaining.length} 个候选（本次未选中）`, children: remaining.map((hit, i) => <div className="candidate-row" key={hit.chunk_id}><Space><span className="status-dot" />{hit.title}</Space><span>重排序第 {result.hits.length + i + 1} 名 · 得分 {hit.rerank_score.toFixed(4)}</span></div>) }]} />}
    </>}
  </>;
}
