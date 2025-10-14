import { NextResponse } from 'next/server';
import { memoryStorage } from '@/lib/storage/memory';

export async function GET() {
  try {
    // 使用共享存储的调试信息
    const debugInfo = memoryStorage.getDebugInfo();

    return NextResponse.json({
      ...debugInfo,
      message: "现在使用共享内存存储！",
      solution: "所有API路由现在共享同一个内存实例"
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