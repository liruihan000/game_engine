"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Gamepad2, Users, User } from "lucide-react";

interface GameInfo {
  name: string;
  description: string;
  isMultiplayer: boolean;
  filename: string;
}

export default function GameLibrary() {
  const [games, setGames] = useState<GameInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchGames();
  }, []);

  const fetchGames = async () => {
    try {
      const response = await fetch('/api/games');
      if (!response.ok) {
        throw new Error('Failed to fetch games');
      }
      const gameList = await response.json();
      setGames(gameList);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-slate-600">Loading games...</p>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center py-12">
            <p className="text-red-600 mb-4">Error: {error}</p>
            <Button onClick={fetchGames}>Try Again</Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Gamepad2 className="h-10 w-10 text-blue-600" />
            <h1 className="text-4xl font-bold text-slate-900">Game Library</h1>
          </div>
          <p className="text-xl text-slate-600 max-w-2xl mx-auto">
            Choose from our collection of interactive games. Each game supports multiple players and offers unique gameplay experiences.
          </p>
        </div>

        {/* Games Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {games.map((game) => (
            <Card key={game.filename} className="group hover:shadow-xl transition-all duration-300 border-0 shadow-lg bg-white/80 backdrop-blur">
              <CardHeader className="pb-4">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <CardTitle className="text-xl font-bold text-slate-900 group-hover:text-blue-600 transition-colors capitalize">
                      {game.name}
                    </CardTitle>
                    <div className="flex items-center gap-2 mt-2">
                      <Badge 
                        variant={game.isMultiplayer ? "default" : "secondary"}
                        className="text-xs"
                      >
                        {game.isMultiplayer ? (
                          <><Users className="h-3 w-3 mr-1" /> Multiplayer</>
                        ) : (
                          <><User className="h-3 w-3 mr-1" /> Single Player</>
                        )}
                      </Badge>
                    </div>
                  </div>
                </div>
              </CardHeader>
              
              <CardContent className="pt-0">
                <CardDescription className="text-slate-600 mb-6 line-clamp-3 leading-relaxed">
                  {game.description}
                </CardDescription>
                
                <Link href={`/game-library/${game.filename}/register`} className="block">
                  <Button 
                    className="w-full bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white shadow-lg hover:shadow-xl transition-all duration-300"
                  >
                    <Gamepad2 className="h-4 w-4 mr-2" />
                    Play Game
                  </Button>
                </Link>
              </CardContent>
            </Card>
          ))}
        </div>

        {games.length === 0 && !loading && (
          <div className="text-center py-12">
            <Gamepad2 className="h-16 w-16 text-slate-400 mx-auto mb-4" />
            <p className="text-slate-600 text-lg">No games available at the moment.</p>
          </div>
        )}
      </div>
    </div>
  );
}