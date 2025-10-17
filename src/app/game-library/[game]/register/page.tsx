"use client";

import { useState, useEffect } from "react";
import { useRouter, useParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft, User, Gamepad2, Users } from "lucide-react";
import Link from "next/link";

interface GameInfo {
  name: string;
  description: string;
  isMultiplayer: boolean;
  filename: string;
}

export default function GameRegister() {
  const router = useRouter();
  const params = useParams();
  const gameName = params.game as string;
  
  const [playerName, setPlayerName] = useState("");
  const [loading, setLoading] = useState(false);
  const [gameInfo, setGameInfo] = useState<GameInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchGameInfo();
  }, [gameName]); // eslint-disable-line react-hooks/exhaustive-deps

  const fetchGameInfo = async () => {
    try {
      const response = await fetch('/api/games');
      const games = await response.json();
      const game = games.find((g: GameInfo) => g.filename === gameName);
      if (game) {
        setGameInfo(game);
      } else {
        setError('Game not found');
      }
    } catch {
      setError('Failed to load game information');
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!playerName.trim()) return;
    
    setLoading(true);
    
    // Store player info in sessionStorage
    const playerSession = {
      playerName: playerName.trim(),
      gameName,
      gameDisplayName: gameInfo?.name || gameName
    };
    
    if (typeof window !== 'undefined') {
      sessionStorage.setItem('playerSession', JSON.stringify(playerSession));
    }
    
    // Navigate to room page
    router.push(`/game-library/${gameName}/room`);
  };

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-8 flex items-center justify-center">
        <Card className="w-full max-w-md">
          <CardHeader className="text-center">
            <CardTitle className="text-red-600">Error</CardTitle>
            <CardDescription>{error}</CardDescription>
          </CardHeader>
          <CardContent>
            <Link href="/game-library">
              <Button className="w-full">
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to Game Library
              </Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!gameInfo) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-8 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-8 flex items-center justify-center">
      <div className="w-full max-w-lg">
        {/* Back Button */}
        <Link href="/game-library" className="inline-flex items-center text-slate-600 hover:text-slate-900 mb-6 transition-colors">
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Game Library
        </Link>

        <Card className="shadow-xl bg-white/90 backdrop-blur border-0">
          <CardHeader className="text-center pb-6">
            <div className="flex items-center justify-center gap-3 mb-4">
              <Gamepad2 className="h-8 w-8 text-blue-600" />
              <CardTitle className="text-2xl font-bold text-slate-900 capitalize">
                {gameInfo.name}
              </CardTitle>
            </div>
            
            <div className="flex justify-center mb-4">
              <Badge 
                variant={gameInfo.isMultiplayer ? "default" : "secondary"}
                className="text-sm px-3 py-1"
              >
                {gameInfo.isMultiplayer ? (
                  <><Users className="h-4 w-4 mr-2" /> Multiplayer Game</>
                ) : (
                  <><User className="h-4 w-4 mr-2" /> Single Player Game</>
                )}
              </Badge>
            </div>
            
            <CardDescription className="text-base leading-relaxed">
              {gameInfo.description}
            </CardDescription>
          </CardHeader>

          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="playerName" className="text-base font-medium">
                  Your Name
                </Label>
                <Input
                  id="playerName"
                  type="text"
                  placeholder="Enter your player name"
                  value={playerName}
                  onChange={(e) => setPlayerName(e.target.value)}
                  className="h-12 text-base"
                  maxLength={20}
                  required
                />
                <p className="text-sm text-slate-600">
                  This name will be displayed to other players in the game.
                </p>
              </div>

              <Button
                type="submit"
                className="w-full h-12 text-base bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 shadow-lg hover:shadow-xl transition-all duration-300"
                disabled={!playerName.trim() || loading}
              >
                {loading ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Joining...
                  </>
                ) : (
                  <>
                    <User className="h-4 w-4 mr-2" />
                    Join Game
                  </>
                )}
              </Button>
            </form>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}