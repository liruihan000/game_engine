import { NextRequest, NextResponse } from 'next/server';
import { randomUUID } from 'crypto';
import { memoryStorage, type RoomData, type PlayerData } from '@/lib/storage/memory';

interface CreateRoomRequest {
  gameName: string;
  playerName: string;
}

// Helper function to create thread ID with LangGraph
async function createAgentThread(gameName: string): Promise<string> {
  try {
    // 🔑 调用真实的 LangGraph API 创建线程
    console.log('🧵 Creating real LangGraph thread for game:', gameName);
    
    const response = await fetch('http://localhost:8123/threads', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        assistant_id: "sample_agent",
        metadata: { 
          game_name: gameName,
          created_at: new Date().toISOString()
        }
      })
    });
    
    if (!response.ok) {
      throw new Error(`LangGraph API error: ${response.status}`);
    }
    
    const data = await response.json();
    const threadId = data.thread_id;
    
    console.log('✅ Real LangGraph thread created:', threadId);
    return threadId;
    
  } catch (error) {
    console.error('❌ Error creating LangGraph thread:', error);
    
    // 降级到模拟 threadId 以确保应用不崩溃
    const fallbackThreadId = `thread_${gameName}_${Date.now()}_${Math.random().toString(36).substring(2, 15)}`;
    console.log('🔄 Using fallback threadId:', fallbackThreadId);
    return fallbackThreadId;
  }
}

export async function POST(request: NextRequest) {
  console.log('🚀 CREATE ROOM API called at:', new Date().toISOString());
  try {
    const body: CreateRoomRequest = await request.json();
    console.log('📝 Request body:', body);
    
    if (!body.gameName || !body.playerName) {
      return NextResponse.json(
        { error: 'Game name and player name are required' },
        { status: 400 }
      );
    }

    // Generate unique IDs
    const roomId = randomUUID();
    // Host is always assigned ID=1 in a new room
    const playerId = 1;
    
    // Create agent thread
    const threadId = await createAgentThread(body.gameName);
    
    // Create room data
    const roomData: RoomData = {
      id: roomId,
      game_name: body.gameName,
      thread_id: threadId,
      host_player_id: playerId,
      status: 'waiting',
      max_players: 8,
      current_players: 1,
      created_at: new Date().toISOString()
    };
    
    // Create player data
    const playerData: PlayerData = {
      id: playerId,
      room_id: roomId,
      name: body.playerName,
      player_order: 1,
      is_host: true,
      status: 'active',
      joined_at: new Date().toISOString()
    };
    
    // Store in shared memory with persistence
    console.log('💾 Storing room in memory:', { roomId, roomData });
    memoryStorage.setRoom(roomId, roomData);
    memoryStorage.setPlayers(roomId, [playerData]);
    console.log('✅ Room stored successfully. Total rooms:', memoryStorage.rooms.size);
    console.log('🔗 Continuity check: room.thread_id (to be used as X-Thread-ID):', threadId);
    
    return NextResponse.json({
      roomId: roomId,
      threadId: threadId,
      playerId: playerId,
      playerOrder: 1,
      isHost: true,
      status: 'waiting'
    });
    
  } catch (error) {
    console.error('❌ Error creating room:', error);
    return NextResponse.json(
      { error: 'Failed to create room', details: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
}

// GET endpoint to retrieve room info (for future use)
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const roomId = searchParams.get('roomId');
    
    if (!roomId) {
      return NextResponse.json(
        { error: 'Room ID is required' },
        { status: 400 }
      );
    }
    
    const room = memoryStorage.rooms.get(roomId);
    const roomPlayers = memoryStorage.players.get(roomId);
    
    if (!room || !roomPlayers) {
      return NextResponse.json(
        { error: 'Room not found' },
        { status: 404 }
      );
    }
    
    return NextResponse.json({
      room,
      players: roomPlayers
    });
    
  } catch (error) {
    console.error('Error retrieving room:', error);
    return NextResponse.json(
      { error: 'Failed to retrieve room' },
      { status: 500 }
    );
  }
}