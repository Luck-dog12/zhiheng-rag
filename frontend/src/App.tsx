import { useEffect, useState } from 'react';
import { App as AntApp, Alert, Avatar, Button, Form, Input, Select, Skeleton, Space, Tooltip } from 'antd';
import { Books, ChatCircleText, Files, ListMagnifyingGlass, ShieldCheck, SignOut, SquaresFour, UsersThree } from '@phosphor-icons/react';
import { api, setCsrf } from './api';
import { KnowledgePage, TeamPage } from './KnowledgePage';
import { DocumentsPage } from './DocumentsPage';
import { AskPage } from './AskPage';
import { SearchPage } from './SearchPage';
import { useLoad } from './components';
import type { KB, User } from './types';

type Page = 'ask' | 'knowledge' | 'documents' | 'search' | 'team';
const navigation = [
  { key: 'ask' as const, label: '知识问答', icon: ChatCircleText },
  { key: 'knowledge' as const, label: '知识库', icon: SquaresFour },
  { key: 'documents' as const, label: '文档管理', icon: Files },
  { key: 'search' as const, label: '检索调试', icon: ListMagnifyingGlass },
  { key: 'team' as const, label: '成员与审计', icon: UsersThree },
];

export default function App() {
  const [user, setUser] = useState<User | null>(null);
  const [checking, setChecking] = useState(true);
  const [startupError, setStartupError] = useState('');
  const { message } = AntApp.useApp();
  useEffect(() => {
    let alive = true;
    const clear = () => { setUser(null); setCsrf(''); };
    window.addEventListener('rag:unauthorized', clear);
    api<{ user: User; csrf_token: string }>('/auth/me').then(r => {
      if (alive) { setUser(r.user); setCsrf(r.csrf_token); }
    }).catch(e => { if (alive && e.status !== 401) setStartupError(e.message); })
      .finally(() => { if (alive) setChecking(false); });
    return () => { alive = false; window.removeEventListener('rag:unauthorized', clear); };
  }, []);
  if (checking) return <div className="boot"><Books size={40} /><Skeleton active /></div>;
  if (!user) return <Login error={startupError} onLogin={(u, token) => { setUser(u); setCsrf(token); setStartupError(''); }} />;
  return <Workspace user={user} logout={async () => {
    try { await api('/auth/logout', { method: 'POST' }); setUser(null); setCsrf(''); }
    catch (e) { message.error((e as Error).message); }
  }} />;
}

function Login({ onLogin, error }: { onLogin: (user: User, token: string) => void; error: string }) {
  const [busy, setBusy] = useState(false), [failure, setFailure] = useState(error);
  return <div className="login-layout">
    <section className="login-story"><div className="brand"><span className="brand-mark"><Books size={26} /></span><strong>知衡</strong><span>KNOWLEDGE WORKSPACE</span></div>
      <div className="login-statement"><div className="eyebrow">让知识连接每一次决策</div><h1>团队的知识，<br />成为有据可循的答案。</h1><p>让分散的文档回到同一个工作台。<br />从资料到答案，每一步都有来源。</p></div>
      <div className="document-art" aria-hidden="true"><div className="art-sheet back" /><div className="art-sheet front"><span>团队知识手册</span><i /><i /><i /><div className="art-highlight">从文档中找到可靠依据</div><i /><i /><small>DOCUMENT / EVIDENCE / ANSWER</small></div></div>
      <footer>企业知识工作台 <span>01 / KNOWLEDGE, CONNECTED.</span></footer>
    </section>
    <section className="login-form"><div><div className="eyebrow">欢迎回来</div><h2>登录工作台</h2><p className="muted">使用管理员为你创建的账号继续。</p>
      {failure && <Alert type="error" showIcon message={failure} className="form-alert" />}
      <Form layout="vertical" onFinish={async values => {
        setBusy(true); setFailure('');
        try { const r = await api<{ user: User; csrf_token: string }>('/auth/login', { method: 'POST', body: JSON.stringify(values) }); onLogin(r.user, r.csrf_token); }
        catch (e) { setFailure((e as Error).message); } finally { setBusy(false); }
      }}><Form.Item label="用户名" name="username" rules={[{ required: true, message: '请输入用户名' }]}><Input size="large" autoComplete="username" /></Form.Item>
        <Form.Item label="密码" name="password" rules={[{ required: true, message: '请输入密码' }]}><Input.Password size="large" autoComplete="current-password" /></Form.Item>
        <Button type="primary" htmlType="submit" size="large" block loading={busy}>进入工作台</Button></Form>
      <div className="login-footnote"><ShieldCheck size={18} />资料访问遵循企业知识库授权</div>
    </div></section>
  </div>;
}

function Workspace({ user, logout }: { user: User; logout: () => void }) {
  const [page, setPage] = useState<Page>('ask');
  const [selected, setSelected] = useState<string[]>([]);
  const knowledge = useLoad<KB[]>('/knowledge-bases', []);
  useEffect(() => {
    setSelected(current => {
      const valid = current.filter(id => knowledge.data.some(k => k.id === id));
      return valid.length ? valid : knowledge.data.slice(0, 1).map(k => k.id);
    });
  }, [knowledge.data]);
  return <div className="workspace">
    <aside className="sidebar"><a className="brand" href="#" onClick={e => { e.preventDefault(); setPage('ask'); }}><span className="brand-mark"><Books size={25} /></span><strong>知衡</strong><small>企业知识工作台</small></a>
      <div className="sidebar-section">WORKSPACE</div><nav>{navigation.filter(n => n.key !== 'team' || user.role === 'admin').map(n => <button key={n.key} aria-label={n.label} title={n.label} className={page === n.key ? 'nav-item active' : 'nav-item'} onClick={() => setPage(n.key)}><n.icon size={20} weight={page === n.key ? 'fill' : 'regular'} /><span>{n.label}</span>{page === n.key && <span className="active-dot" />}</button>)}</nav>
      <div className="sidebar-note"><ShieldCheck size={22} /><strong>知识有边界，协作有依据</strong><p>只检索你有权访问的资料。</p></div>
      <div className="profile"><Avatar style={{ background: '#dce9e1', color: '#176b55' }}>{user.display_name[0]}</Avatar><div><strong>{user.display_name}</strong><small>{user.role === 'admin' ? '工作台管理员' : '团队成员'}</small></div><Tooltip title="退出登录"><Button aria-label="退出登录" type="text" icon={<SignOut size={18} />} onClick={logout} /></Tooltip></div>
    </aside>
    <div className="main-frame"><header className="topbar"><div className="breadcrumb">工作空间 <span>/</span> <strong>{navigation.find(n => n.key === page)?.label}</strong></div><Space><span className="scope-label">检索范围</span><Select aria-label="选择知识库" mode="multiple" className="scope-select" maxTagCount={1} value={selected} onChange={setSelected} options={knowledge.data.map(k => ({ label: k.name, value: k.id }))} placeholder="选择知识库" loading={knowledge.loading} /></Space></header>
      <main className={page === 'ask' ? 'main-content ask-main' : 'main-content'}>
        {knowledge.error && <Alert type="error" message={knowledge.error} action={<Button onClick={knowledge.reload}>重试</Button>} />}
        {page === 'ask' && <AskPage kbIds={selected} hasKnowledge={knowledge.data.length > 0} />}
        {page === 'knowledge' && <KnowledgePage user={user} knowledge={knowledge} onOpen={id => { setSelected([id]); setPage('documents'); }} />}
        {page === 'documents' && <DocumentsPage user={user} kbs={knowledge.data} initialKb={selected[0]} onChange={knowledge.reload} />}
        {page === 'search' && <SearchPage kbIds={selected} />}
        {page === 'team' && user.role === 'admin' && <TeamPage />}
      </main><div className="workspace-footer">知衡 / 将团队经验，转化为可追溯的知识。</div>
    </div>
  </div>;
}
