import { NextResponse } from 'next/server';
import { memoryStorage } from '@/lib/storage/memory';
import { randomUUID } from 'crypto';

export async function GET() {
  try {
    // Create a test room directly in memory
    const testRoomId = randomUUID();
    const testRoom = {
      id: testRoomId,
      game_name: 'test-game',
      thread_id: 'test-thread-123',
      host_player_id: 999,
      status: 'waiting',
      max_players: 8,
      current_players: 1,
      created_at: new Date().toISOString()
    };

    const testPlayer = {
      id: 999,
      room_id: testRoomId,
      name: 'Test Player',
      player_order: 1,
      is_host: true,
      status: 'active',
      joined_at: new Date().toISOString()
    };

    // Store test data
    memoryStorage.rooms.set(testRoomId, testRoom);
    memoryStorage.players.set(testRoomId, [testPlayer]);

    return NextResponse.json({
      message: 'Test room created',
      testRoomId,
      totalRooms: memoryStorage.rooms.size,
      totalPlayers: memoryStorage.players.size,
      nextPlayerId: memoryStorage.nextPlayerId,
      rooms: Array.from(memoryStorage.rooms.entries()),
      players: Array.from(memoryStorage.players.entries())
    });

  } catch (error) {
    console.error('Test API error:', error);
    return NextResponse.json(
      { error: 'Test failed', details: error },
      { status: 500 }
    );
  }
}

// Clear test data
export async function DELETE() {
  try {
    memoryStorage.clearAll();
    return NextResponse.json({
      message: 'Memory cleared',
      totalRooms: memoryStorage.rooms.size,
      totalPlayers: memoryStorage.players.size,
      nextPlayerId: memoryStorage.nextPlayerId
    });
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to clear memory', details: error },
      { status: 500 }
    );
  }
}