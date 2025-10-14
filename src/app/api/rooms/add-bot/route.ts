import { NextRequest, NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import * as path from 'path';
import * as yaml from 'js-yaml';
import { memoryStorage } from '@/lib/storage/memory';

interface GameData {
  declaration?: {
    min_players?: number;
    is_multiplayer?: boolean;
  };
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { roomId, gameName } = body;

    if (!roomId || !gameName) {
      return NextResponse.json(
        { error: 'Room ID and game name are required' },
        { status: 400 }
      );
    }

    // Get room data
    const roomData = memoryStorage.getRoom(roomId);
    if (!roomData) {
      return NextResponse.json(
        { error: 'Room not found' },
        { status: 404 }
      );
    }

    const { room, players: currentPlayers } = roomData;

    // Load DSL to get min_players
    const gamesDir = path.join(process.cwd(), 'games');
    const gameFiles = await fs.readdir(gamesDir);
    const targetFile = gameFiles.find(file => {
      const fileNameWithoutExt = file.toLowerCase().replace('.yaml', '');
      const normalizedFileName = fileNameWithoutExt.replace(/[^a-z0-9]/g, '-');
      const normalizedGameName = gameName.toLowerCase().replace(/[^a-z0-9]/g, '-');
      return normalizedFileName === normalizedGameName;
    });

    if (!targetFile) {
      return NextResponse.json(
        { error: `Game file not found for: ${gameName}` },
        { status: 404 }
      );
    }

    const filePath = path.join(gamesDir, targetFile);
    const fileContent = await fs.readFile(filePath, 'utf8');
    const gameData = yaml.load(fileContent) as GameData;

    const minPlayers = gameData?.declaration?.min_players || 4;
    const currentPlayerCount = currentPlayers.length;

    if (currentPlayerCount >= minPlayers) {
      return NextResponse.json(
        { error: `Room already has enough players (${currentPlayerCount}/${minPlayers})` },
        { status: 400 }
      );
    }

    // Calculate how many bots to add
    const botsNeeded = minPlayers - currentPlayerCount;
    const newPlayers = [...currentPlayers];

    // Find the next player number to use (starting from player2)
    let nextPlayerNumber = 2;
    const existingPlayerNames = currentPlayers.map(p => p.name.toLowerCase());
    
    // Find the next available player number
    while (existingPlayerNames.includes(`player${nextPlayerNumber}`)) {
      nextPlayerNumber++;
    }

    // Add bots with names like player2, player3, etc.
    for (let i = 0; i < botsNeeded; i++) {
      const playerId = memoryStorage.updateNextPlayerId();
      const botName = `player${nextPlayerNumber + i}`;
      
      const botPlayer = {
        id: playerId,
        room_id: roomId,
        name: botName,
        player_order: currentPlayerCount + i + 1,
        is_host: false,
        status: 'joined',
        joined_at: new Date().toISOString(),
      };
      newPlayers.push(botPlayer);
    }

    // Update room player count
    const updatedRoom = {
      ...room,
      current_players: newPlayers.length
    };

    // Save to memory storage
    memoryStorage.setRoom(roomId, updatedRoom);
    memoryStorage.setPlayers(roomId, newPlayers);

    console.log(`🤖 Added ${botsNeeded} bots to room ${roomId}:`, 
                newPlayers.slice(currentPlayerCount).map(p => p.name).join(', '));
    console.log(`📊 Total players: ${newPlayers.length}/${minPlayers}`);

    return NextResponse.json({
      success: true,
      message: `Added ${botsNeeded} bot(s) to reach minimum players`,
      playersAdded: botsNeeded,
      totalPlayers: newPlayers.length,
      minPlayers: minPlayers,
      addedBots: newPlayers.slice(currentPlayerCount).map(p => ({ id: p.id, name: p.name })),
      allPlayers: newPlayers
    });

  } catch (error) {
    console.error('❌ Error adding bots:', error);
    return NextResponse.json(
      { error: 'Failed to add bots' },
      { status: 500 }
    );
  }
}