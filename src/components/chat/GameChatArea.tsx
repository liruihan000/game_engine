"use client";

import React, { useState, useRef, useEffect } from 'react';
import { Send, MessageCircle, Users } from 'lucide-react';

interface ChatMessage {
  id: string;
  playerId: string;
  playerName: string;
  message: string;
  timestamp: number;
  type?: 'message' | 'system' | 'action';
}

interface BotPlayer {
  id: string;
  name: string;
  role?: string;
  isAlive: boolean;
}

interface GameChatAreaProps {
  messages: ChatMessage[];
  currentPlayerId?: string | null;
  currentPlayerName?: string;
  onSendMessage: (message: string, targetBotId?: string) => void;
  playerCount?: number;
  availableBots?: BotPlayer[];
}

export function GameChatArea({ 
  messages, 
  currentPlayerId, 
  currentPlayerName,
  onSendMessage,
  playerCount = 0,
  availableBots = []
}: GameChatAreaProps) {
  const [inputMessage, setInputMessage] = useState('');
  const [selectedBot, setSelectedBot] = useState<string>('');
  const [isClient, setIsClient] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Fix hydration by ensuring we know when we're on the client
  useEffect(() => {
    setIsClient(true);
  }, []);

  // Auto scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = () => {
    const trimmed = inputMessage.trim();
    if (trimmed && currentPlayerId) {
      onSendMessage(trimmed, selectedBot || undefined);
      setInputMessage('');
      inputRef.current?.focus();
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const formatTime = (timestamp: number) => {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  const getMessageStyle = (msg: ChatMessage) => {
    if (msg.type === 'system') {
      return 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800';
    }
    if (msg.type === 'action') {
      return 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800';
    }
    if (msg.playerId === currentPlayerId) {
      return 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800 ml-8';
    }
    return 'bg-gray-50 dark:bg-gray-800 border-gray-200 dark:border-gray-700';
  };

  return (
    <div className="flex flex-col h-full bg-white dark:bg-gray-900 border-l border-gray-200 dark:border-gray-700">
      {/* Chat Header */}
      <div className="flex-shrink-0 p-4 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
        <div className="flex items-center gap-2">
          <MessageCircle className="w-5 h-5 text-blue-600 dark:text-blue-400" />
          <h3 className="font-semibold text-gray-800 dark:text-gray-200">
            游戏聊天
          </h3>
          <div className="flex items-center gap-1 ml-auto text-sm text-gray-500 dark:text-gray-400">
            <Users className="w-4 h-4" />
            <span>{playerCount}</span>
          </div>
        </div>
        {currentPlayerName && (
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            当前玩家: {currentPlayerName}
          </p>
        )}
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.length === 0 ? (
          <div className="text-center py-8">
            <MessageCircle className="w-12 h-12 text-gray-300 dark:text-gray-600 mx-auto mb-3" />
            <p className="text-gray-500 dark:text-gray-400">
              还没有消息，开始聊天吧！
            </p>
          </div>
        ) : (
          <>
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`p-3 rounded-lg border ${getMessageStyle(msg)}`}
              >
                <div className="flex items-center gap-2 mb-1">
                  <span className="font-medium text-sm text-gray-800 dark:text-gray-200">
                    {msg.type === 'system' ? '🤖 系统' : 
                     msg.type === 'action' ? '⚡ 游戏' : 
                     msg.playerName}
                  </span>
                  <span className="text-xs text-gray-500 dark:text-gray-400">
                    {formatTime(msg.timestamp)}
                  </span>
                </div>
                <p className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap">
                  {msg.message}
                </p>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Input Area */}
      <div className="flex-shrink-0 p-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
        {isClient && currentPlayerId ? (
          <div className="space-y-2">
            {/* Bot选择器 */}
            {availableBots.length > 0 && (
              <div className="flex items-center gap-2 text-sm">
                <label className="text-gray-600 dark:text-gray-400 whitespace-nowrap">
                  回复对象:
                </label>
                <select
                  value={selectedBot}
                  onChange={(e) => setSelectedBot(e.target.value)}
                  className="px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-800 dark:text-gray-200 text-sm"
                >
                  <option value="">随机Bot回复</option>
                  {availableBots
                    .filter(bot => bot.isAlive)
                    .map(bot => (
                      <option key={bot.id} value={bot.id}>
                        @{bot.name} {bot.role ? `(${bot.role})` : ''}
                      </option>
                    ))}
                </select>
                {selectedBot && (
                  <button
                    onClick={() => setSelectedBot('')}
                    className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                    title="取消指定"
                  >
                    ✕
                  </button>
                )}
              </div>
            )}
            
            {/* 消息输入框 */}
            <div className="flex gap-2">
              <input
                ref={inputRef}
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={selectedBot 
                  ? `向 ${availableBots.find(b => b.id === selectedBot)?.name} 说话...` 
                  : "输入消息..."
                }
                className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-800 dark:text-gray-200 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                maxLength={500}
              />
              <button
                onClick={handleSend}
                disabled={!inputMessage.trim()}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center gap-1"
              >
                <Send className="w-4 h-4" />
                <span className="hidden sm:inline">发送</span>
              </button>
            </div>
          </div>
        ) : (
          <div className="text-center py-2">
            <p className="text-sm text-gray-500 dark:text-gray-400">
              请先加入游戏才能聊天
            </p>
          </div>
        )}
      </div>
    </div>
  );
}