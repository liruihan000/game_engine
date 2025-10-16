"use client";

import { useEffect, useState } from 'react';
import { CopilotKit } from '@copilotkit/react-core';

interface DynamicCopilotProviderProps {
  children: React.ReactNode;
}

export default function DynamicCopilotProvider({ children }: DynamicCopilotProviderProps) {
  const [threadId, setThreadId] = useState<string>(() => {
    if (typeof window === 'undefined') return 'default';
    try {
      const gameContext = sessionStorage.getItem('gameContext');
      if (gameContext) {
        const context = JSON.parse(gameContext);
        if (context?.threadId && typeof context.threadId === 'string') {
          return context.threadId;
        }
      }
    } catch {
      // ignore and fallback
    }
    return 'default';
  });

  useEffect(() => {
    // Unified loader for threadId
    const loadThreadId = () => {
      const gameContext = sessionStorage.getItem('gameContext');
      if (gameContext) {
        try {
          const context = JSON.parse(gameContext);
          if (context.threadId && context.threadId !== threadId) {
            console.log('🧵 Setting threadId for CopilotKit:', context.threadId);
            setThreadId(context.threadId);
          }
        } catch (error) {
          console.error('Failed to parse gameContext:', error);
        }
      }
    };

    // Initial load
    loadThreadId();

    // Listen for custom room change events
    const handleRoomChanged = (event: CustomEvent) => {
      console.log('🏠 Room changed event received:', event.detail);
      loadThreadId(); // Reload threadId
    };

    // Add event listener
    window.addEventListener('roomChanged', handleRoomChanged as EventListener);
    
    // Cleanup
    return () => {
      window.removeEventListener('roomChanged', handleRoomChanged as EventListener);
    };
  }, [threadId]); // Depend on current threadId to avoid unnecessary resets

  return (
    <CopilotKit
      runtimeUrl="/api/copilotkit"
      agent="sample_agent"
      showDevConsole={false}
      publicApiKey={process.env.NEXT_PUBLIC_COPILOT_CLOUD_PUBLIC_API_KEY}
      headers={{
        'X-Thread-ID': threadId, // 🔑 Pass room-specific threadId
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