import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import yaml from 'js-yaml';

interface GameInfo {
  name: string;
  description: string;
  isMultiplayer: boolean;
  filename: string;
}

export async function GET() {
  try {
    const gamesDir = path.join(process.cwd(), 'games');
    const files = fs.readdirSync(gamesDir).filter(file => file.endsWith('.yaml'));
    
    const games: GameInfo[] = [];
    
    for (const file of files) {
      try {
        const filePath = path.join(gamesDir, file);
        const fileContent = fs.readFileSync(filePath, 'utf8');
        const gameData = yaml.load(fileContent) as Record<string, unknown>;
        
        const declaration = gameData?.declaration as {
          description?: string;
          is_multiplayer?: boolean;
        } | undefined;
        
        if (declaration) {
          games.push({
            name: file.replace('.yaml', '').replace(/[-_]/g, ' '),
            description: declaration.description || 'No description available',
            isMultiplayer: declaration.is_multiplayer || false,
            filename: file.replace('.yaml', '')
          });
        }
      } catch (error) {
        console.error(`Error parsing ${file}:`, error);
        // Still add the game even if parsing fails
        games.push({
          name: file.replace('.yaml', '').replace(/[-_]/g, ' '),
          description: 'Game description unavailable',
          isMultiplayer: false,
          filename: file.replace('.yaml', '')
        });
      }
    }
    
    return NextResponse.json(games);
  } catch (error) {
    console.error('Error reading games directory:', error);
    return NextResponse.json({ error: 'Failed to load games' }, { status: 500 });
  }
}