import React from 'react';
import ReactDOM from 'react-dom/client';
import '@ant-design/v5-patch-for-react-19';
import { App as AntApp, ConfigProvider } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import App from './App';
import './styles.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode><ConfigProvider locale={zhCN} theme={{ token: { colorPrimary: '#176b55', colorInfo: '#176b55',
    borderRadius: 8, fontFamily: '"Segoe UI", "Microsoft YaHei", sans-serif', colorText: '#24352f', colorBgLayout: '#f4f6f3' } }}>
    <AntApp><App /></AntApp>
  </ConfigProvider></React.StrictMode>,
);
