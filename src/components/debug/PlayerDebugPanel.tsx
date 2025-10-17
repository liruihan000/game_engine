"use client";

import React from 'react';
import { AgentState } from '@/lib/canvas/types';
import { getPlayersFromStates } from '@/lib/player-utils';

interface PlayerDebugPanelProps {
  viewState: AgentState;
  isVisible: boolean;
}

export function PlayerDebugPanel({ viewState, isVisible }: PlayerDebugPanelProps) {
  if (!isVisible) return null;

  const players = viewState.player_states ? getPlayersFromStates(viewState.player_states) : [];
  const deadPlayers = viewState.deadPlayers || [];
  const playerActions = viewState.playerActions || {};

  const getPlayerStatus = (playerId: string) => {
    const isDead = deadPlayers.includes(playerId);
    const playerState = viewState.player_states?.[playerId] || {};
    const actions = playerActions[playerId];
    
    return {
      isDead,
      state: playerState,
      lastAction: actions?.actions || 'No actions',
      lastPhase: actions?.phase || 'Unknown',
      lastTimestamp: actions?.timestamp ? new Date(actions.timestamp * 1000).toLocaleTimeString() : 'N/A'
    };
  };

  const formatStateValue = (value: unknown): string => {
    if (value === null || value === undefined) return 'null';
    if (typeof value === 'boolean') return value ? 'true' : 'false';
    if (typeof value === 'object') return JSON.stringify(value);
    return String(value);
  };

  return (
    <div className="fixed top-4 right-4 w-96 max-h-[80vh] bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg z-50 overflow-hidden">
      <div className="bg-blue-50 dark:bg-blue-900 p-3 border-b border-gray-200 dark:border-gray-700">
        <h3 className="font-semibold text-sm text-blue-800 dark:text-blue-200">
          🐛 Player Debug Panel
        </h3>
        <p className="text-xs text-blue-600 dark:text-blue-300 mt-1">
          Real-time player status • {players.length} players
        </p>
      </div>
      
      <div className="overflow-y-auto max-h-[60vh]">
        {players.length === 0 ? (
          <div className="p-4 text-center text-gray-500 dark:text-gray-400">
            <div className="text-2xl mb-2">👻</div>
            <p>No players found</p>
          </div>
        ) : (
          <div className="space-y-3 p-3">
            {players.map((player) => {
              const status = getPlayerStatus(player.id);
              return (
                <div 
                  key={player.id} 
                  className={`p-3 rounded-lg border ${
                    status.isDead 
                      ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800' 
                      : 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800'
                  }`}
                >
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-lg">
                      {status.isDead ? '💀' : '😊'}
                    </span>
                    <div className="flex-1">
                      <p className="font-medium text-sm">
                        {player.name}
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        ID: {player.id} • {status.isDead ? 'Dead' : 'Alive'}
                      </p>
                    </div>
                  </div>
                  
                  {/* Player State */}
                  <div className="text-xs space-y-1 mb-2">
                    <div className="font-medium text-gray-700 dark:text-gray-300">State:</div>
                    {Object.keys(status.state).length === 0 ? (
                      <div className="text-gray-500 dark:text-gray-400 ml-2">No state data</div>
                    ) : (
                      <div className="bg-white dark:bg-gray-700 p-2 rounded border ml-2">
                        {Object.entries(status.state).map(([key, value]) => (
                          <div key={key} className="flex justify-between text-xs">
                            <span className="font-mono text-blue-600 dark:text-blue-400">
                              {key}:
                            </span>
                            <span className="font-mono text-gray-600 dark:text-gray-300 ml-2 break-all">
                              {formatStateValue(value)}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                  
                  {/* Last Action */}
                  <div className="text-xs">
                    <div className="font-medium text-gray-700 dark:text-gray-300 mb-1">Last Action:</div>
                    <div className="bg-white dark:bg-gray-700 p-2 rounded border text-xs">
                      <div className="text-gray-800 dark:text-gray-200 mb-1">{status.lastAction}</div>
                      <div className="flex justify-between text-gray-500 dark:text-gray-400">
                        <span>Phase: {status.lastPhase}</span>
                        <span>{status.lastTimestamp}</span>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
      
      {/* Footer with game info */}
      <div className="bg-gray-50 dark:bg-gray-700 p-2 border-t border-gray-200 dark:border-gray-600 text-xs">
        <div className="flex justify-between items-center">
          <span className="text-gray-600 dark:text-gray-400">
            Game: {viewState.gameName || 'Unknown'}
          </span>
          <span className="text-gray-600 dark:text-gray-400">
            Votes: {viewState.vote?.length || 0}
          </span>
        </div>
      </div>
    </div>
  );
}