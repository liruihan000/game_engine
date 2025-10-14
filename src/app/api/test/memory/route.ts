import { NextResponse } from 'next/server';
import { memoryStorage } from '@/lib/storage/memory';
import { randomUUID } from 'crypto';

export async function GET() {
  try {
    // 直接在内存中创建一个测试房间
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

    // 存储测试数据
    memoryStorage.rooms.set(testRoomId, testRoom);
    memoryStorage.players.set(testRoomId, [testPlayer]);

    return NextResponse.json({
      message: '测试房间已创建',
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

// 清理测试数据
export async function DELETE() {
  try {
    memoryStorage.clearAll();
    return NextResponse.json({
      message: '内存已清理',
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