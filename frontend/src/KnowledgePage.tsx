import { useState } from 'react';
import { App, Alert, Button, Checkbox, Form, Input, Modal, Select, Space, Table, Tabs, Tag } from 'antd';
import { ArrowUpRight, Books, Plus, ShieldCheck } from '@phosphor-icons/react';
import { api } from './api';
import { date, EmptyKnowledge, LoadError, Loading, PageHeading, Refresh, useLoad } from './components';
import type { KB, User } from './types';

export function KnowledgePage({ user, knowledge, onOpen }: { user: User; knowledge: ReturnType<typeof useLoad<KB[]>>; onOpen: (id: string) => void }) {
  const { message } = App.useApp();
  const [editing, setEditing] = useState<KB | 'new' | null>(null);
  const [grantKb, setGrantKb] = useState<KB | null>(null);
  const [form] = Form.useForm();
  const [busy, setBusy] = useState(false);
  return <><PageHeading eyebrow="KNOWLEDGE LIBRARY" title="团队知识，井然有序" description="按业务主题整理文档，为不同团队开放合适的知识。" action={user.role === 'admin' && <Button type="primary" icon={<Plus size={17} />} onClick={() => { form.resetFields(); setEditing('new'); }}>新建知识库</Button>} />
    <LoadError error={knowledge.error} retry={knowledge.reload} />
    <div className="library-summary"><span><strong>{knowledge.data.length.toString().padStart(2, '0')}</strong> 个可访问知识库</span><span>{knowledge.data.reduce((n, k) => n + k.document_count, 0)} 份文档</span></div>
    {knowledge.loading ? <Loading /> : knowledge.data.length === 0 ? <EmptyKnowledge /> : <div className="knowledge-grid">{knowledge.data.map((kb, i) => <article className="knowledge-card" key={kb.id}>
      <div className="knowledge-card-top"><span className="kb-icon"><Books size={25} weight="duotone" /></span><span className="collection-number">COLLECTION / {String(i + 1).padStart(2, '0')}</span></div>
      <h2>{kb.name}</h2><p>{kb.description || '集中整理与这个主题有关的文档和团队经验。'}</p><div className="knowledge-card-bottom"><span>{kb.document_count} 份文档</span><Button type="text" onClick={() => onOpen(kb.id)}>查看文档 <ArrowUpRight size={16} /></Button></div>
      {user.role === 'admin' && <div className="knowledge-actions"><Button size="small" type="text" onClick={() => { form.setFieldsValue(kb); setEditing(kb); }}>编辑</Button><Button size="small" type="text" icon={<ShieldCheck size={15} />} onClick={() => setGrantKb(kb)}>成员授权</Button></div>}
    </article>)}</div>}
    <Modal title={editing === 'new' ? '新建知识库' : '编辑知识库'} open={!!editing} onCancel={() => setEditing(null)} confirmLoading={busy} onOk={() => form.submit()} destroyOnClose><Form form={form} layout="vertical" onFinish={async values => {
      setBusy(true); try { await api(editing === 'new' ? '/knowledge-bases' : `/knowledge-bases/${(editing as KB).id}`, { method: editing === 'new' ? 'POST' : 'PATCH', body: JSON.stringify(values) }); setEditing(null); knowledge.reload(); message.success('知识库已保存'); } catch (e) { message.error((e as Error).message); } finally { setBusy(false); }
    }}><Form.Item name="name" label="知识库名称" rules={[{ required: true, message: '请输入名称' }]}><Input maxLength={160} /></Form.Item><Form.Item name="description" label="用途说明" initialValue=""><Input.TextArea rows={3} maxLength={4000} /></Form.Item></Form></Modal>
    {grantKb && <GrantModal kb={grantKb} close={() => setGrantKb(null)} />}
  </>;
}

function GrantModal({ kb, close }: { kb: KB; close: () => void }) {
  const users = useLoad<User[]>('/users', []);
  const grants = useLoad<{ user_ids: string[] }>(`/knowledge-bases/${kb.id}/grants`, { user_ids: [] });
  const [selected, setSelected] = useState<string[] | null>(null);
  const [busy, setBusy] = useState(false);
  const { message } = App.useApp();
  return <Modal title={`知识库授权 · ${kb.name}`} open onCancel={close} confirmLoading={busy} okButtonProps={{ disabled: grants.loading || users.loading || !!grants.error || !!users.error }} onOk={async () => {
    setBusy(true); try { await api(`/knowledge-bases/${kb.id}/grants`, { method: 'PUT', body: JSON.stringify({ user_ids: selected ?? grants.data.user_ids }) }); message.success('授权已更新'); close(); } catch (e) { message.error((e as Error).message); } finally { setBusy(false); }
  }}><p className="muted">管理员可以访问所有知识库。勾选需要访问此知识库的团队成员。</p><LoadError error={users.error || grants.error} retry={() => { users.reload(); grants.reload(); }} />
    <Checkbox.Group className="grant-list" value={selected ?? grants.data.user_ids} onChange={v => setSelected(v as string[])}>{users.data.filter(u => u.role === 'member').map(u => <Checkbox key={u.id} value={u.id} disabled={!u.active}>{u.display_name} <span className="muted">{u.username}{!u.active && ' · 已停用'}</span></Checkbox>)}</Checkbox.Group>
  </Modal>;
}

type Audit = { id: string; action: string; user_id: string | null; target_id: string | null; created_at: string; trace_id: string; detail: Record<string, unknown> };
export function TeamPage() {
  const users = useLoad<User[]>('/users', []), audit = useLoad<Audit[]>('/audit', []);
  const { message } = App.useApp();
  const [open, setOpen] = useState(false), [busy, setBusy] = useState(false);
  const [form] = Form.useForm();
  return <><PageHeading eyebrow="ACCESS & ACTIVITY" title="团队协作，权限清晰" description="管理工作台成员，并追溯资料与授权的关键操作。" action={<Button type="primary" icon={<Plus size={17} />} onClick={() => { form.resetFields(); setOpen(true); }}>创建成员</Button>} />
    <Tabs items={[{ key: 'members', label: '团队成员', children: <div className="surface"><LoadError error={users.error} retry={users.reload} /><Table<User> rowKey="id" loading={users.loading} dataSource={users.data} columns={[
      { title: '成员', dataIndex: 'display_name', render: (v, u) => <div><strong>{v}</strong><div className="muted">{u.username}</div></div> },
      { title: '角色', dataIndex: 'role', render: v => <Tag>{v === 'admin' ? '管理员' : '团队成员'}</Tag> },
      { title: '状态', dataIndex: 'active', render: v => <Tag color={v ? 'success' : 'default'}>{v ? '已启用' : '已停用'}</Tag> },
      { title: '操作', render: (_, u) => <Button type="text" danger={u.active} onClick={async () => { try { await api(`/users/${u.id}`, { method: 'PATCH', body: JSON.stringify({ active: !u.active }) }); users.reload(); audit.reload(); } catch (e) { message.error((e as Error).message); } }}>{u.active ? '停用' : '启用'}</Button> },
    ]} /></div> }, { key: 'audit', label: '操作审计', children: <div className="surface"><div className="section-toolbar"><span>最近 100 条操作记录</span><Refresh onClick={audit.reload} loading={audit.loading} /></div><LoadError error={audit.error} retry={audit.reload} /><Table<Audit> rowKey="id" dataSource={audit.data} loading={audit.loading} scroll={{ x: 650 }} columns={[
      { title: '操作', dataIndex: 'action' }, { title: '时间', dataIndex: 'created_at', render: date },
      { title: '操作者', dataIndex: 'user_id', render: id => users.data.find(u => u.id === id)?.display_name ?? (id ? id.slice(0, 8) : '系统任务') },
    ]} expandable={{ expandedRowRender: row => <pre className="code-block">{JSON.stringify({ target_id: row.target_id, trace_id: row.trace_id, ...row.detail }, null, 2)}</pre> }} /></div> }]} />
    <Modal title="创建团队成员" open={open} onCancel={() => setOpen(false)} onOk={() => form.submit()} confirmLoading={busy} destroyOnClose><Alert type="info" message="创建后需在知识库中为成员分配访问权限。" showIcon /><Form form={form} layout="vertical" className="modal-form" onFinish={async values => { setBusy(true); try { await api('/users', { method: 'POST', body: JSON.stringify(values) }); setOpen(false); users.reload(); message.success('成员已创建'); } catch (e) { message.error((e as Error).message); } finally { setBusy(false); } }}>
      <Form.Item name="username" label="用户名" rules={[{ required: true, message: '请输入用户名' }]}><Input autoComplete="off" /></Form.Item><Form.Item name="display_name" label="显示名称" rules={[{ required: true, message: '请输入显示名称' }]}><Input /></Form.Item><Form.Item name="password" label="初始密码" rules={[{ required: true, min: 12, message: '密码至少 12 个字符' }]}><Input.Password autoComplete="new-password" /></Form.Item><Form.Item name="role" label="角色" initialValue="member"><Select options={[{ value: 'member', label: '团队成员' }, { value: 'admin', label: '管理员' }]} /></Form.Item>
    </Form></Modal>
  </>;
}
