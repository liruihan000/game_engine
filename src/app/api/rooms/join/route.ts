import { NextRequest, NextResponse } from 'next/server';

interface JoinRoomRequest {
  roomId: string;
  playerName: string;
}

import { memoryStorage } from '@/lib/storage/memory';

export async function POST(request: NextRequest) {
  try {
    const body: JoinRoomRequest = await request.json();
    
    if (!body.roomId || !body.playerName) {
      return NextResponse.json(
        { error: 'Room ID and player name are required' },
        { status: 400 }
      );
    }

    // Check if room exists
    const room = memoryStorage.rooms.get(body.roomId);
    if (!room) {
      return NextResponse.json(
        { error: 'Room not found' },
        { status: 404 }
      );
    }

    // Check if room is full
    const roomPlayers = memoryStorage.players.get(body.roomId) || [];
    if (roomPlayers.length >= room.max_players) {
      return NextResponse.json(
        { error: 'Room is full' },
        { status: 400 }
      );
    }

    // Check if player name already exists in room
    const existingPlayer = roomPlayers.find(p => p.name === body.playerName);
    if (existingPlayer) {
      return NextResponse.json(
        { error: 'Player name already taken in this room' },
        { status: 400 }
      );
    }

    // Add player to room - 房间内连续ID分配
    const newPlayerId = roomPlayers.length > 0 ? 
      Math.max(...roomPlayers.map(p => p.id)) + 1 : 1;
    const playerData = {
      id: newPlayerId,
      room_id: body.roomId,
      name: body.playerName,
      player_order: roomPlayers.length + 1,
      is_host: false,
      status: 'active',
      joined_at: new Date().toISOString()
    };

    roomPlayers.push(playerData);
    memoryStorage.setPlayers(body.roomId, roomPlayers);

    // Update room player count
    room.current_players = roomPlayers.length;
    memoryStorage.setRoom(body.roomId, room);

    return NextResponse.json({
      roomId: body.roomId,
      threadId: room.thread_id, // 🔑 添加 threadId 到响应
      playerId: newPlayerId,
      playerOrder: roomPlayers.length,
      isHost: false,
      currentPlayers: roomPlayers.length,
      maxPlayers: room.max_players,
      players: roomPlayers.map(p => ({
        id: p.id,
        name: p.name,
        isHost: p.is_host
      }))
    });

  } catch (error) {
    console.error('Error joining room:', error);
    return NextResponse.json(
      { error: 'Failed to join room' },
      { status: 500 }
    );
  }
}