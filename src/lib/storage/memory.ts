// 共享内存存储模块
// 这个文件提供全局唯一的Map实例，供所有API路由共享

import fs from 'fs';
import path from 'path';

export interface RoomData {
  id: string;
  game_name: string;
  thread_id: string;
  host_player_id: number;
  status: string;
  max_players: number;
  current_players: number;
  created_at: string;
}

export interface PlayerData {
  id: number;
  room_id: string;
  name: string;
  player_order: number;
  is_host: boolean;
  status: string;
  joined_at: string;
}

interface StorageData {
  rooms: Record<string, RoomData>;
  players: Record<string, PlayerData[]>;
  nextPlayerId: number;
}

// 全局单例存储
class MemoryStorage {
  private static instance: MemoryStorage;
  private readonly persistFile = path.join(process.cwd(), 'temp-rooms.json');
  
  public readonly rooms = new Map<string, RoomData>();
  public readonly players = new Map<string, PlayerData[]>();
  public nextPlayerId = 1;

  private constructor() {
    console.log('🏗️  MemoryStorage instance created at:', new Date().toISOString());
    this.loadFromFile();
  }

  private loadFromFile() {
    try {
      if (fs.existsSync(this.persistFile)) {
        const data = fs.readFileSync(this.persistFile, 'utf8');
        const parsed: StorageData = JSON.parse(data);
        
        // 恢复数据到Map
        Object.entries(parsed.rooms).forEach(([key, value]) => {
          this.rooms.set(key, value);
        });
        
        Object.entries(parsed.players).forEach(([key, value]) => {
          this.players.set(key, value);
        });
        
        this.nextPlayerId = parsed.nextPlayerId;
        
        console.log('📂 Loaded from file:', {
          rooms: this.rooms.size,
          players: this.players.size,
          nextPlayerId: this.nextPlayerId
        });
      }
    } catch (error) {
      console.error('❌ Failed to load from file:', error);
    }
  }

  private saveToFile() {
    try {
      const data: StorageData = {
        rooms: Object.fromEntries(this.rooms),
        players: Object.fromEntries(this.players),
        nextPlayerId: this.nextPlayerId
      };
      
      fs.writeFileSync(this.persistFile, JSON.stringify(data, null, 2));
    } catch (error) {
      console.error('❌ Failed to save to file:', error);
    }
  }

  // 包装原有方法以添加持久化
  public setRoom(key: string, value: RoomData) {
    this.rooms.set(key, value);
    this.saveToFile();
  }

  public setPlayers(key: string, value: PlayerData[]) {
    this.players.set(key, value);
    this.saveToFile();
  }

  public updateNextPlayerId() {
    this.nextPlayerId++;
    this.saveToFile();
    return this.nextPlayerId - 1;
  }

  public static getInstance(): MemoryStorage {
    if (!MemoryStorage.instance) {
      MemoryStorage.instance = new MemoryStorage();
    }
    return MemoryStorage.instance;
  }

  // 获取房间数据（确保从文件加载）
  public getRoom(roomId: string): { room: RoomData; players: PlayerData[] } | null {
    // 确保数据是最新的
    this.loadFromFile();
    
    const room = this.rooms.get(roomId);
    const players = this.players.get(roomId) || [];
    
    if (!room) {
      return null;
    }
    
    return { room, players };
  }

  // 调试方法
  public getDebugInfo() {
    return {
      timestamp: new Date().toISOString(),
      totalRooms: this.rooms.size,
      totalPlayerGroups: this.players.size,
      nextPlayerId: this.nextPlayerId,
      rooms: Array.from(this.rooms.entries()).map(([key, value]) => ({
        roomId: key,
        ...value
      })),
      players: Array.from(this.players.entries()).map(([key, value]) => ({
        roomId: key,
        players: value
      }))
    };
  }

  // 清理方法（可选）
  public clearAll() {
    this.rooms.clear();
    this.players.clear();
    this.nextPlayerId = 1;
    this.saveToFile();
  }
}

// 导出单例实例
export const memoryStorage = MemoryStorage.getInstance();