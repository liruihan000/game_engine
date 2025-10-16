"use client";

import React, { useState, useRef, useEffect } from 'react';
import { Send, MessageCircle, Users } from 'lucide-react';
import type { ChatMessage } from '@/lib/canvas/types';

// Use shared ChatMessage type from lib to avoid divergence

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
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Fix hydration by ensuring we know when we're on the client
  useEffect(() => {
    setIsClient(true);
  }, []);

  // Auto scroll to bottom when new messages arrive
  useEffect(() => {
    const scrollToBottom = () => {
      const container = messagesContainerRef.current;
      if (container) {
        container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' });
        return;
      }
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'start' });
    };
    
    // Immediate scroll for initial load, delayed scroll for new messages
    if (messages.length > 0) {
      setTimeout(scrollToBottom, 100);
    }
  }, [messages]);

  const handleSend = () => {
    const trimmed = inputMessage.trim();
    if (trimmed && currentPlayerId) {
      onSendMessage(trimmed, selectedBot || undefined);
      setInputMessage('');
      inputRef.current?.focus();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
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
    <div className="flex flex-col h-full min-h-0 overflow-hidden bg-card">
      {/* Chat Header */}
      <div className="flex-shrink-0 p-4 border-b border-sidebar-border bg-accent/10">
        <div className="flex items-center gap-2">
          <h3 className="font-semibold text-sidebar-foreground">
            Room Chat
          </h3>
          <div className="flex items-center gap-1 ml-auto text-sm text-muted-foreground">
            <Users className="w-4 h-4" />
            <span suppressHydrationWarning>{isClient ? playerCount : 0}</span>
          </div>
        </div>
        {currentPlayerName && (
          <p className="text-sm text-muted-foreground mt-1">
            Current player: {currentPlayerName}
          </p>
        )}
      </div>

      {/* Messages Area */}
      <div ref={messagesContainerRef} className="flex-1 min-h-0 max-h-full overflow-y-auto scroll-smooth overscroll-contain p-4 space-y-3">
        {messages.length === 0 ? (
          <div className="text-center py-8">
            <MessageCircle className="w-12 h-12 text-muted-foreground/40 mx-auto mb-3" />
            <p className="text-muted-foreground">
              No messages yet — start the conversation.
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
                  <span className="font-medium text-sm text-foreground">
                    {msg.type === 'system' ? '🤖 System' : 
                     msg.type === 'action' ? '⚡ Game' : 
                     msg.playerName}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {formatTime(msg.timestamp)}
                  </span>
                </div>
                <p className="text-sm text-foreground/80 whitespace-pre-wrap break-words break-all">
                  {msg.message}
                </p>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Input Area */}
      <div className="flex-shrink-0 p-4 border-t border-sidebar-border bg-accent/10">
        {isClient && currentPlayerId ? (
          <div className="space-y-2">
            {/* Bot selector */}
            {availableBots.length > 0 && (
              <div className="flex items-center gap-2 text-sm">
                <label className="text-muted-foreground whitespace-nowrap">
                  Reply to:
                </label>
                <select
                  value={selectedBot}
                  onChange={(e) => setSelectedBot(e.target.value)}
                  className="px-2 py-1 border border-border rounded bg-background text-foreground text-sm"
                >
                  <option value="">Random bot</option>
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
                    className="text-muted-foreground hover:text-foreground"
                    title="Clear selection"
                  >
                    ✕
                  </button>
                )}
              </div>
            )}
            
            {/* Message input */}
            <div className="flex gap-2">
              <input
                ref={inputRef}
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={selectedBot 
                  ? `Message @${availableBots.find(b => b.id === selectedBot)?.name}…` 
                  : "Type a message…"
                }
                className="flex-1 px-3 py-2 border border-border rounded-lg bg-background text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                maxLength={500}
              />
              <button
                onClick={handleSend}
                disabled={!inputMessage.trim()}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-muted disabled:text-muted-foreground disabled:cursor-not-allowed transition-colors flex items-center gap-1"
              >
                <Send className="w-4 h-4" />
                <span className="hidden sm:inline">Send</span>
              </button>
            </div>
          </div>
        ) : (
          <div className="text-center py-2">
            <p className="text-sm text-muted-foreground">
              Join the game to start chatting.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}