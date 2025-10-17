import { NextRequest, NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import * as path from 'path';
import * as yaml from 'js-yaml';
import { memoryStorage } from '@/lib/storage/memory';

interface PlayerStatesTemplate {
  player_states: Record<string, Record<string, unknown>>;
}

interface GameDeclaration {
  player_states_template?: PlayerStatesTemplate;
  // Schema definition for per-player state fields (used for auto-generating defaults)
  player_states?: Record<string, { type?: string; example?: unknown } | unknown>;
  // Example players data structure
  players_example?: {
    players?: Record<string, Record<string, unknown>>;
  };
}

interface GameData {
  declaration?: GameDeclaration;
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { roomId, gameName, roomSession } = body;

    if (!roomId || !gameName) {
      return NextResponse.json(
        { error: 'Room ID and game name are required' },
        { status: 400 }
      );
    }

    // Prefer roomSession data if provided, fallback to storage
    let roomPlayers = [];
    if (roomSession?.players) {
      roomPlayers = roomSession.players;
      console.log('✅ Using roomSession players:', roomPlayers.length);
    } else {
      // Fallback to persistent storage
      const roomData = memoryStorage.getRoom(roomId);
      if (!roomData) {
        return NextResponse.json(
          { error: 'Room not found and no roomSession provided' },
          { status: 404 }
        );
      }
      roomPlayers = roomData.players || [];
    }
    
    if (roomPlayers.length === 0) {
      return NextResponse.json(
        { error: 'No players found in room' },
        { status: 404 }
      );
    }

    // Read game YAML file
    const gamesDir = path.join(process.cwd(), 'games');
    const gameFiles = await fs.readdir(gamesDir);
    const targetFile = gameFiles.find(file => {
      // Remove .yaml extension and compare
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

    // Defense mechanism 1: Check if player_states_template exists
    let template: Record<string, unknown> = {};
    
    if (gameData?.declaration?.player_states_template?.player_states) {
      // Try to get template with ID "1"
      template = gameData.declaration.player_states_template.player_states['1'];
      
      // Defense mechanism 2: If ID "1" doesn't exist, try first available ID
      if (!template) {
        const availableIds = Object.keys(gameData.declaration.player_states_template.player_states);
        if (availableIds.length > 0) {
          template = gameData.declaration.player_states_template.player_states[availableIds[0]];
        }
      }
    }
    
    // Debug logging
    console.log('🔍 Debug: gameData structure:', JSON.stringify({
      hasDeclaration: !!gameData?.declaration,
      hasPlayerStatesTemplate: !!gameData?.declaration?.player_states_template,
      hasPlayerStates: !!gameData?.declaration?.player_states,
      hasPlayersExample: !!gameData?.declaration?.players_example,
      gameDataKeys: Object.keys(gameData || {})
    }, null, 2));

    // Defense mechanism 3: If template still not found, auto-generate from player_states definition
    if (!template && gameData?.declaration?.player_states) {
      console.log('🛡️ No template found, auto-generating from player_states definition');
      template = {};
      const playerStatesSchema = gameData.declaration.player_states;
      
      // Generate default values based on field types
      for (const [fieldName, fieldDef] of Object.entries(playerStatesSchema)) {
        const fieldType = (fieldDef as any)?.type || 'string';
        const defaultValue = (fieldDef as any)?.example;
        
        switch (fieldType) {
          case 'string':
            template[fieldName] = defaultValue || '';
            break;
          case 'num':
          case 'number':
            template[fieldName] = defaultValue || 0;
            break;
          case 'boolean':
            template[fieldName] = defaultValue !== undefined ? defaultValue : true;
            break;
          case 'array':
          case 'list':
            template[fieldName] = defaultValue || [];
            break;
          case 'object':
          case 'dict':
            template[fieldName] = defaultValue || {};
            break;
          default:
            template[fieldName] = defaultValue || null;
        }
      }
    }
    
    // Final fallback: If we still have no template, return empty state for agent to handle
    if (!template || Object.keys(template).length === 0) {
      console.log('🛡️ No template available, returning empty state for agent to generate');
      return NextResponse.json({
        player_states: {},
        fallback_mode: true,
        message: 'No template found, agent will generate player_states'
      });
    }

    // Create initialized player_states with real players
    const initializedPlayers: Record<string, Record<string, unknown>> = {};
    
    roomPlayers.forEach((player) => {
      // Use gamePlayerId if available (from roomSession), otherwise fallback to index
      const playerId = player.gamePlayerId || (roomPlayers.indexOf(player) + 1).toString();
      initializedPlayers[playerId] = {
        ...template, // Copy all template fields
        name: player.name, // Replace with real player name
        id: player.id || playerId, // Store original player ID
        isHost: player.isHost || false
      };
    });

    const result = {
      player_states: initializedPlayers
    };

    return NextResponse.json(result);

  } catch (error) {
    console.error('Error initializing player states:', error);
    return NextResponse.json(
      { error: 'Failed to initialize player states' },
      { status: 500 }
    );
  }
}