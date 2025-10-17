import { NextRequest, NextResponse } from 'next/server';
import { memoryStorage } from '@/lib/storage/memory';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const gameName = searchParams.get('gameName');
    
    // 获取内存中的所有数据
    const debugInfo = memoryStorage.getDebugInfo();
    const allRooms = Array.from(memoryStorage.rooms.values());
    
    return NextResponse.json({
      requestedGameName: gameName,
      totalRoomsInMemory: allRooms.length,
      allRoomsWithGameNames: allRooms.map(r => ({
        roomId: r.id,
        gameName: r.game_name,
        status: r.status,
        currentPlayers: r.current_players,
        maxPlayers: r.max_players
      })),
      matchingRooms: allRooms.filter(r => r.game_name === gameName),
      encodingTest: {
        original: gameName,
        decoded: decodeURIComponent(gameName || ''),
        encoded: encodeURIComponent(gameName || '')
      },
      memoryDebug: debugInfo
    });

  } catch (error) {
    return NextResponse.json({
      error: 'Test failed',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}