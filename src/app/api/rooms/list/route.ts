import { NextRequest, NextResponse } from 'next/server';
import { memoryStorage } from '@/lib/storage/memory';

export async function GET(request: NextRequest) {
  console.log('🔍 LIST ROOMS API called at:', new Date().toISOString());
  try {
    const { searchParams } = new URL(request.url);
    const gameName = searchParams.get('gameName');
    console.log('📝 Requested game name:', gameName);
    
    if (!gameName) {
      return NextResponse.json(
        { error: 'Game name is required' },
        { status: 400 }
      );
    }
    
    // Get all rooms for this game that are waiting for players
    console.log('🏠 Total rooms in memory:', memoryStorage.rooms.size);
    const allRooms = Array.from(memoryStorage.rooms.values());
    console.log('📋 All rooms:', allRooms.map(r => ({ name: r.game_name, status: r.status, players: r.current_players })));
    
    const availableRooms = allRooms
      .filter(room => {
        const matches = room.game_name === gameName && 
                       room.status === 'waiting' &&
                       room.current_players < room.max_players;
        console.log(`🔍 Room ${room.id}: game_name="${room.game_name}" === "${gameName}" = ${room.game_name === gameName}, status="${room.status}" === "waiting" = ${room.status === 'waiting'}, players=${room.current_players} < ${room.max_players} = ${room.current_players < room.max_players}, matches=${matches}`);
        return matches;
      })
      .map(room => {
        const roomPlayers = memoryStorage.players.get(room.id) || [];
        const hostPlayer = roomPlayers.find(p => p.is_host);
        
        return {
          roomId: room.id,
          hostName: hostPlayer?.name || 'Unknown',
          currentPlayers: room.current_players,
          maxPlayers: room.max_players,
          createdAt: room.created_at,
          players: roomPlayers.map(p => ({
            name: p.name,
            isHost: p.is_host
          }))
        };
      })
      .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()); // newest first
    
    console.log('✅ Available rooms found:', availableRooms.length);
    console.log('📤 Returning rooms:', availableRooms);
    
    return NextResponse.json(availableRooms);
    
  } catch (error) {
    console.error('Error listing rooms:', error);
    return NextResponse.json(
      { error: 'Failed to list rooms' },
      { status: 500 }
    );
  }
}