import { useCallback, useEffect, useState } from 'react';
import { Alert, Button, Empty, Skeleton, Tag } from 'antd';
import { ArrowClockwise, Books } from '@phosphor-icons/react';
import { api } from './api';

export function useLoad<T>(path: string, initial: T, enabled = true) {
  const [data, setData] = useState<T>(initial);
  const [loading, setLoading] = useState(enabled);
  const [error, setError] = useState('');
  const [revision, setRevision] = useState(0);
  const reload = useCallback(() => setRevision(n => n + 1), []);
  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    setLoading(true);
    api<T>(path, { signal: controller.signal }).then(value => {
      setData(value); setError('');
    }).catch(e => { if (!controller.signal.aborted) setError(e.message); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [path, revision, enabled]);
  return { data, setData, loading, error, reload };
}

export function LoadError({ error, retry }: { error: string; retry: () => void }) {
  return error ? <Alert type="error" showIcon message={error} action={<Button size="small" onClick={retry}>重试</Button>} /> : null;
}

export function PageHeading({ eyebrow, title, description, action }: { eyebrow: string; title: string; description: string; action?: React.ReactNode }) {
  return <div className="page-heading"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>{action}</div>;
}

export function EmptyKnowledge({ description = '先创建知识库，再上传团队的文档资料。' }: { description?: string }) {
  return <div className="empty-state"><Books size={48} weight="light" /><Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={description} /></div>;
}

const stages: Record<string, string> = { queued: '等待处理', running: '处理中', parsing: '解析文档', coreference: '清洗与指代消解', metadata: '生成元数据', indexing: '建立索引', ready: '已就绪', done: '已完成', failed: '处理失败', superseded: '已有新版本' };
export function Status({ value }: { value: string }) {
  const color = ['ready', 'done'].includes(value) ? 'success' : value === 'failed' ? 'error' : value === 'superseded' ? 'default' : 'processing';
  return <Tag color={color}>{stages[value] ?? value}</Tag>;
}

export function Refresh({ onClick, loading }: { onClick: () => void; loading?: boolean }) {
  return <Button icon={<ArrowClockwise size={16} />} onClick={onClick} loading={loading}>刷新</Button>;
}

export function Loading() { return <div className="surface"><Skeleton active paragraph={{ rows: 5 }} /></div>; }
export const date = (value: string) => new Date(value).toLocaleString('zh-CN', { hour12: false });
