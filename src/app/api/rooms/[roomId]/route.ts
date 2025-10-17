import { NextRequest, NextResponse } from 'next/server';
import { memoryStorage } from '@/lib/storage/memory';

export async function GET(
  request: NextRequest,
  { params }: { params: { roomId: string } }
) {
  try {
    const { roomId } = params;
    
    if (!roomId) {
      return NextResponse.json(
        { error: 'Room ID is required' },
        { status: 400 }
      );
    }

    console.log(`🔍 Getting room data for: ${roomId}`);
    
    const roomData = memoryStorage.getRoom(roomId);
    
    if (!roomData) {
      console.log(`❌ Room not found: ${roomId}`);
      return NextResponse.json(
        { error: 'Room not found' },
        { status: 404 }
      );
    }

    console.log(`✅ Found room data for: ${roomId}`, roomData.room.game_name);
    
    return NextResponse.json(roomData);
    
  } catch (error) {
    console.error('❌ Error getting room data:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}