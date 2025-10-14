/**
 * 通用玩家信息获取工具
 * 支持多种数据源获取当前房间的玩家列表
 */

export interface PlayerInfo {
  id: string;
  name: string;
}

/**
 * 从 player_states 获取玩家信息 (前端优先方式)
 */
export function getPlayersFromStates(
  playerStates: Record<string, Record<string, unknown>>
): PlayerInfo[] {
  return Object.entries(playerStates).map(([id, data]) => ({
    id,
    name: (data as any)?.name || `Player ${id}`
  }));
}

/**
 * 从 sessionStorage 获取当前房间ID
 */
export function getCurrentRoomId(): string | null {
  if (typeof window === 'undefined') return null;
  
  const gameContext = sessionStorage.getItem('gameContext');
  if (gameContext) {
    try {
      const context = JSON.parse(gameContext);
      return context.roomId || null;
    } catch {
      return null;
    }
  }
  return null;
}

/**
 * 通过 API 获取房间玩家信息 (备用方式)
 */
export async function getPlayersFromAPI(roomId: string): Promise<PlayerInfo[]> {
  try {
    const response = await fetch(`/api/rooms/${roomId}/players`);
    if (!response.ok) return [];
    
    const data = await response.json();
    return data.players?.map((player: any) => ({
      id: String(player.id),
      name: player.name || `Player ${player.id}`
    })) || [];
  } catch {
    return [];
  }
}

/**
 * 通用玩家信息获取函数
 * 优先级: player_states > API > 默认空数组
 */
export async function getPlayers(
  playerStates?: Record<string, Record<string, unknown>>
): Promise<PlayerInfo[]> {
  // 方式1: 从 player_states 获取 (最优先)
  if (playerStates && Object.keys(playerStates).length > 0) {
    return getPlayersFromStates(playerStates);
  }
  
  // 方式2: 从 API 获取 (备用)
  const roomId = getCurrentRoomId();
  if (roomId) {
    const apiPlayers = await getPlayersFromAPI(roomId);
    if (apiPlayers.length > 0) {
      return apiPlayers;
    }
  }
  
  // 方式3: 返回空数组 (兜底)
  return [];
}

/**
 * 获取当前玩家ID
 */
export function getCurrentPlayerId(): string | null {
  if (typeof window === 'undefined') return null;
  return sessionStorage.getItem('playerId');
}