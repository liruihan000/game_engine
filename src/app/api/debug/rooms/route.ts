import { NextResponse } from 'next/server';
import { memoryStorage } from '@/lib/storage/memory';

export async function GET() {
  try {
    // Use debug info from the shared memory storage
    const debugInfo = memoryStorage.getDebugInfo();

    return NextResponse.json({
      ...debugInfo,
      message: "Using shared in-memory storage",
      solution: "All API routes share a single memory instance now"
    }, {
      headers: {
        'Content-Type': 'application/json',
      }
    });

  } catch (error) {
    console.error('Debug API error:', error);
    return NextResponse.json(
      { error: 'Failed to get debug info', details: error },
      { status: 500 }
    );
  }
}