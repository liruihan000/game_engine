"use client";

import { useEffect, useState } from 'react';
import { CopilotKit } from '@copilotkit/react-core';

interface DynamicCopilotProviderProps {
  children: React.ReactNode;
}

export default function DynamicCopilotProvider({ children }: DynamicCopilotProviderProps) {
  const [threadId, setThreadId] = useState<string>('default');

  useEffect(() => {
    // 统一的 threadId 加载函数
    const loadThreadId = () => {
      const gameContext = sessionStorage.getItem('gameContext');
      if (gameContext) {
        try {
          const context = JSON.parse(gameContext);
          if (context.threadId) {
            console.log('🧵 Setting threadId for CopilotKit:', context.threadId);
            setThreadId(context.threadId);
          }
        } catch (error) {
          console.error('Failed to parse gameContext:', error);
        }
      }
    };

    // 初始加载
    loadThreadId();

    // 监听自定义房间切换事件
    const handleRoomChanged = (event: CustomEvent) => {
      console.log('🏠 Room changed event received:', event.detail);
      loadThreadId(); // 重新加载 threadId
    };

    // 添加事件监听器
    window.addEventListener('roomChanged', handleRoomChanged as EventListener);
    
    // 清理函数
    return () => {
      window.removeEventListener('roomChanged', handleRoomChanged as EventListener);
    };
  }, []); // 只在组件挂载时执行一次

  return (
    <CopilotKit
      runtimeUrl="/api/copilotkit"
      agent="sample_agent"
      showDevConsole={false}
      publicApiKey={process.env.NEXT_PUBLIC_COPILOT_CLOUD_PUBLIC_API_KEY}
      headers={{
        'X-Thread-ID': threadId, // 🔑 传递房间特定的 threadId
      }}
      // Disable all UI components
      // chatComponentsConfig={{
      //   showPopup: false,
      //   showSidebar: false,
      //   showChat: false,
      // }}
    >
      {children}
    </CopilotKit>
  );
}