import { useEffect, useState } from 'react';
import { App, Alert, Button, Collapse, Drawer, Empty, Pagination, Popconfirm, Select, Space, Table, Tabs, Tag, Upload } from 'antd';
import { UploadSimple, FileText, Funnel } from '@phosphor-icons/react';
import { api } from './api';
import { date, EmptyKnowledge, LoadError, PageHeading, Refresh, Status, useLoad } from './components';
import type { Chunk, Doc, Job, KB, User } from './types';

export function DocumentsPage({ user, kbs, initialKb, onChange }: { user: User; kbs: KB[]; initialKb?: string; onChange: () => void }) {
  const { message } = App.useApp();
  const [kb, setKb] = useState<string | undefined>(initialKb), [page, setPage] = useState(1);
  const [uploading, setUploading] = useState(false), [opened, setOpened] = useState<Doc | null>(null);
  const docs = useLoad<{ total: number; items: Doc[] }>(`/documents?offset=${(page - 1) * 20}&limit=20${kb ? '&kb_id=' + kb : ''}`, { total: 0, items: [] });
  const jobs = useLoad<Job[]>('/jobs', [], user.role === 'admin');
  useEffect(() => { setKb(initialKb); setPage(1); }, [initialKb]);
  useEffect(() => {
    if (!docs.data.items.some(d => !['ready', 'failed', 'superseded'].includes(d.status))) return;
    const interval = setInterval(() => { docs.reload(); jobs.reload(); }, 4000);
    return () => clearInterval(interval);
  }, [docs.data.items, docs.reload, jobs.reload]);
  async function upload(file: File, document?: Doc) {
    if (!(document?.kb_id || kb)) { message.warning('请先选择上传到哪个知识库'); return false; }
    setUploading(true);
    try {
      const body = new FormData(); body.append('file', file); body.append('kb_id', document?.kb_id || kb!);
      if (document) body.append('document_id', document.id);
      const result = await api<{ deduplicated: boolean }>('/documents', { method: 'POST', body });
      message.success(result.deduplicated ? '相同文档已存在，可查看当前处理状态' : '文档已提交，正在后台处理');
      docs.reload(); jobs.reload(); onChange();
    } catch (e) { message.error((e as Error).message); } finally { setUploading(false); }
    return false;
  }
  return <><PageHeading eyebrow="DOCUMENT COLLECTION" title="让资料，成为团队的积累" description="上传文档后自动整理、分块并建立索引，随时查看处理结果与原始依据。" action={user.role === 'admin' && <Upload beforeUpload={file => upload(file)} showUploadList={false} accept=".txt,.md,.markdown,.docx,.pdf"><Button type="primary" icon={<UploadSimple size={17} />} loading={uploading} disabled={!kb}>上传文档</Button></Upload>} />
    {kbs.length === 0 ? <EmptyKnowledge /> : <Tabs items={[{ key: 'documents', label: '全部文档', children: <div className="surface document-surface"><div className="section-toolbar"><Space><Funnel size={18} /><Select aria-label="筛选知识库" value={kb} onChange={v => { setKb(v); setPage(1); }} allowClear placeholder="全部知识库" style={{ minWidth: 200 }} options={kbs.map(k => ({ label: k.name, value: k.id }))} /><span className="muted">共 {docs.data.total} 份文档</span></Space><Refresh onClick={docs.reload} loading={docs.loading} /></div><LoadError error={docs.error} retry={docs.reload} />
      <Table<Doc> rowKey="id" dataSource={docs.data.items} loading={docs.loading} pagination={false} scroll={{ x: 760 }} locale={{ emptyText: <Empty description="这里还没有文档，上传第一份团队资料吧" /> }} columns={[
        { title: '文档名称', dataIndex: 'title', render: (v, d) => <div className="document-name"><span className="file-icon"><FileText size={22} /></span><div><button className="text-link" onClick={() => setOpened(d)}>{v}</button><div className="muted">{kbs.find(k => k.id === d.kb_id)?.name}{d.warnings.length > 0 && <Tag color="warning">{d.warnings.length} 项解析提示</Tag>}</div></div></div> },
        { title: '最新版本状态', dataIndex: 'status', render: (v, d) => <><Status value={v} />{d.current_version_id && d.current_version_id !== d.latest_version_id && <div className="muted">当前仍使用上一版本</div>}</> },
        { title: '分块', dataIndex: 'chunk_count', width: 70 }, { title: '上传时间', dataIndex: 'created_at', render: date },
        { title: '操作', render: (_, d) => <Space size={0} wrap><Button type="link" onClick={() => setOpened(d)}>查看</Button>{user.role === 'admin' && <><Upload showUploadList={false} beforeUpload={file => upload(file, d)} accept=".txt,.md,.markdown,.docx,.pdf"><Button type="link" disabled={uploading}>更新</Button></Upload><Popconfirm title="删除文档后将立即停止用于问答" onConfirm={async () => { try { await api(`/documents/${d.id}`, { method: 'DELETE' }); docs.reload(); jobs.reload(); onChange(); message.success('文档已下线，文件正在后台清理'); } catch (e) { message.error((e as Error).message); } }}><Button type="text" danger>删除</Button></Popconfirm></>}</Space> },
      ]} /><div className="pagination-row"><span className="muted">支持 TXT、Markdown、DOCX、文本型 PDF · 单文件不超过 50 MB</span><Pagination current={page} pageSize={20} total={docs.data.total} showSizeChanger={false} onChange={setPage} /></div>
    </div> }, ...(user.role === 'admin' ? [{ key: 'jobs', label: '处理任务', children: <div className="surface"><div className="section-toolbar"><span>后台任务与恢复</span><Refresh onClick={jobs.reload} loading={jobs.loading} /></div><LoadError error={jobs.error} retry={jobs.reload} /><Table<Job> rowKey="id" dataSource={jobs.data} loading={jobs.loading} columns={[
      { title: '任务', dataIndex: 'kind', render: (v: string, j: Job) => <div>{v === 'ingest' ? '文档入库' : v === 'delete' ? '文件清理' : '索引清理'}<div className="muted">{j.id.slice(0, 8)}</div></div> },
      { title: '状态', dataIndex: 'status', render: (v: string) => <Status value={v} /> }, { title: '处理阶段', dataIndex: 'stage', render: (v: string) => <Status value={v} /> },
      { title: '尝试次数', dataIndex: 'attempts' }, { title: '错误信息', dataIndex: 'error_message', render: (v: string) => v || '—' },
      { title: '操作', render: (_: unknown, j: Job) => j.status === 'failed' && <Button onClick={async () => { try { await api(`/jobs/${j.id}/retry`, { method: 'POST' }); jobs.reload(); docs.reload(); } catch (e) { message.error((e as Error).message); } }}>重试任务</Button> },
    ]} scroll={{ x: 700 }} /></div> }] : [])]} />}
    {opened && <ChunkDrawer doc={opened} user={user} close={() => setOpened(null)} />}
  </>;
}

function ChunkDrawer({ doc, user, close }: { doc: Doc; user: User; close: () => void }) {
  const versions = useLoad<{ id: string; status: string; current: boolean; created_at: string; warnings: string[] }[]>(`/documents/${doc.id}/versions`, []);
  const [version, setVersion] = useState(doc.current_version_id ?? doc.latest_version_id), [page, setPage] = useState(1);
  const chunks = useLoad<{ total: number; items: Chunk[] }>(`/chunks?document_id=${doc.id}&version_id=${version}&offset=${(page - 1) * 10}&limit=10`, { total: 0, items: [] });
  const warnings = versions.data.find(v => v.id === version)?.warnings ?? [];
  return <Drawer width={760} title={doc.title} open onClose={close}><div className="drawer-toolbar"><Select aria-label="文档版本" value={version} onChange={v => { setVersion(v); setPage(1); }} options={versions.data.map(v => ({ value: v.id, label: `${date(v.created_at)}${v.current ? ' · 当前版本' : ''}` }))} style={{ minWidth: 280 }} /><Button href={`/api/v1/documents/${doc.id}/download?version_id=${version}`} disabled={user.role !== 'admin' && version !== doc.current_version_id}>下载原文</Button></div>
    {warnings.map(w => <Alert key={w} message={w} type="warning" showIcon className="form-alert" />)}<LoadError error={chunks.error || versions.error} retry={() => { chunks.reload(); versions.reload(); }} />
    {chunks.data.items.length === 0 && <Empty description={chunks.loading ? '正在加载分块…' : '此版本尚未生成可查看的分块'} />}
    {chunks.data.items.map(chunk => <article className="chunk-card" key={chunk.chunk_id}><div className="chunk-heading"><span>CHUNK {String(chunk.ordinal + 1).padStart(3, '0')}</span><code>{chunk.chunk_id.slice(0, 8)}</code></div><h3>{chunk.heading_path.join(' / ') || '正文片段'}</h3><p className="chunk-summary">{chunk.metadata.summary}</p><div className="keyword-list">{chunk.metadata.keywords.map(k => <Tag key={k}>{k}</Tag>)}</div>
      {(chunk.metadata.time_mentions.length > 0 || chunk.metadata.locations.length > 0) && <div className="mention-list">{chunk.metadata.time_mentions.map((t, i) => <Tag key={'t' + i}>时间 · {t.raw}</Tag>)}{chunk.metadata.locations.map((t, i) => <Tag key={'l' + i}>地点 · {t.raw}</Tag>)}</div>}
      <Collapse ghost items={[{ key: 'body', label: '处理后的分块正文', children: <pre className="text-content">{chunk.body}</pre> }, { key: 'source', label: '原始证据与定位', children: chunk.sources.map((s, i) => <div key={i}><Tag>{s.page ? `第 ${s.page} 页` : `第 ${s.paragraph} 段`}</Tag><pre className="text-content">{s.original}</pre></div>) }, { key: 'edits', label: `清洗与指代替换记录 · ${chunk.edits.length}`, children: chunk.edits.length ? <pre className="code-block">{JSON.stringify(chunk.edits, null, 2)}</pre> : '本片段未发生文本修改' }]} />
    </article>)}<Pagination current={page} pageSize={10} total={chunks.data.total} showSizeChanger={false} onChange={setPage} />
  </Drawer>;
}
