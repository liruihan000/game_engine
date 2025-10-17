"use client";

import { useState, useEffect } from "react";
import { useRouter, useParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft, Play, Users, User, Crown, UserPlus } from "lucide-react";
import Link from "next/link";

interface PlayerSession {
  playerName: string;
  gameName: string;
  gameDisplayName: string;
}

interface RoomData {
  roomId: string;
  threadId: string;
  playerId: number;
  playerOrder?: number;
  isHost?: boolean;
  currentPlayers?: number;
  maxPlayers?: number;
  players?: Array<{
    id: number;
    name: string;
    isHost: boolean;
  }>;
}

interface AvailableRoom {
  roomId: string;
  hostName: string;
  currentPlayers: number;
  maxPlayers: number;
  createdAt: string;
  players: Array<{name: string; isHost: boolean}>;
}

export default function GameRoom() {
  const router = useRouter();
  const params = useParams();
  const gameName = params.game as string;
  
  const [playerSession, setPlayerSession] = useState<PlayerSession | null>(null);
  const [roomData, setRoomData] = useState<RoomData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<'main' | 'join'>('main');
  const [availableRooms, setAvailableRooms] = useState<AvailableRoom[]>([]);

  useEffect(() => {
    // Get player session from sessionStorage
    const sessionData = sessionStorage.getItem('playerSession');
    if (sessionData) {
      const session = JSON.parse(sessionData) as PlayerSession;
      if (session.gameName === gameName) {
        setPlayerSession(session);
      } else {
        // Wrong game, redirect to register
        router.push(`/game-library/${gameName}/register`);
      }
    } else {
      // No session, redirect to register
      router.push(`/game-library/${gameName}/register`);
    }
  }, [gameName, router]);

  const fetchAvailableRooms = async () => {
    if (!playerSession) return;
    
    console.log('🔍 Fetching available rooms for:', playerSession.gameName);
    setLoading(true);
    try {
      const url = `/api/rooms/list?gameName=${encodeURIComponent(playerSession.gameName)}`;
      console.log('📡 Request URL:', url);
      console.log('🏷️  Original game name:', playerSession.gameName);
      console.log('🔗 Encoded game name:', encodeURIComponent(playerSession.gameName));
      const response = await fetch(url);
      console.log('📨 List response status:', response.status);
      
      if (response.ok) {
        const rooms = await response.json();
        console.log('✅ Available rooms received:', rooms);
        setAvailableRooms(rooms);
      } else {
        const errorData = await response.json();
        console.error('❌ List API error:', errorData);
      }
    } catch (err) {
      console.error('❌ Failed to fetch available rooms:', err);
    } finally {
      setLoading(false);
    }
  };

  const joinSpecificRoom = async (roomId: string) => {
    if (!playerSession) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/rooms/join', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          roomId: roomId,
          playerName: playerSession.playerName,
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to join room');
      }
      
      const data = await response.json();
      setRoomData({
        roomId: data.roomId,
        threadId: data.threadId, // 🔑 使用 API 返回的 threadId
        playerId: data.playerId,
        playerOrder: data.playerOrder,
        isHost: data.isHost,
        currentPlayers: data.currentPlayers,
        maxPlayers: data.maxPlayers,
        players: data.players
      });
      
      // Store room data in sessionStorage
      const updatedSession = {
        ...playerSession,
        roomId: data.roomId,
        playerId: data.playerId
      };
      sessionStorage.setItem('playerSession', JSON.stringify(updatedSession));
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const showJoinRooms = () => {
    setMode('join');
    fetchAvailableRooms();
  };

  const createRoom = async () => {
    if (!playerSession) return;
    
    console.log('🎮 Creating room for:', playerSession);
    setLoading(true);
    setError(null);
    
    try {
      console.log('📡 Sending request to /api/rooms/create');
      const response = await fetch('/api/rooms/create', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          gameName: playerSession.gameName,
          playerName: playerSession.playerName,
        }),
      });
      
      console.log('📨 Response status:', response.status);
      
      if (!response.ok) {
        const errorData = await response.json();
        console.error('❌ API Error:', errorData);
        throw new Error(errorData.details || 'Failed to create room');
      }
      
      const data = await response.json();
      console.log('✅ Room created successfully:', data);
      setRoomData(data);
      
      // Store room data in sessionStorage
      const updatedSession = {
        ...playerSession,
        roomId: data.roomId,
        threadId: data.threadId,
        playerId: data.playerId
      };
      sessionStorage.setItem('playerSession', JSON.stringify(updatedSession));
      console.log('💾 Session updated:', updatedSession);
      
      // 🔑 触发房间切换事件（加入房间时）
      if (typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent('roomChanged', { 
          detail: { 
            roomId: data.roomId,
            threadId: data.threadId,
            action: 'joinRoom'
          }
        }));
        console.log('🏠 Dispatched roomChanged event for room join, threadId:', data.threadId);
      }
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const addBots = async () => {
    if (!roomData || !playerSession) return;
    
    setLoading(true);
    setError(null);
    
    try {
      console.log('🤖 Adding bots to reach minimum players...');
      const response = await fetch('/api/rooms/add-bot', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          roomId: roomData.roomId,
          gameName: playerSession.gameName,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to add bots');
      }

      const result = await response.json();
      console.log('🤖 Bots added successfully:', result);
      
      // Update room data with new player count
      setRoomData(prev => prev ? {
        ...prev,
        currentPlayers: result.totalPlayers,
        players: result.allPlayers.map((p: { id: number; name: string; is_host: boolean }) => ({
          id: p.id,
          name: p.name,
          isHost: p.is_host
        }))
      } : null);
      
    } catch (error) {
      console.error('❌ Error adding bots:', error);
      setError(error instanceof Error ? error.message : 'Failed to add bots');
    } finally {
      setLoading(false);
    }
  };

  const startGame = async () => {
    if (!roomData || !playerSession) return;
    
    setLoading(true);
    setError(null);
    
    try {
      // Prepare roomSession data
      const roomSessionData = {
        roomId: roomData.roomId,
        gameName: playerSession.gameName,
        totalPlayers: roomData.players?.length || 0,
        players: roomData.players?.map((player, index) => ({
          id: player.id,
          name: player.name,
          isHost: player.isHost,
          gamePlayerId: (index + 1).toString()
        })) || []
      };
      
      // First, initialize player states with real player data
      console.log('🎮 Initializing player states with roomSession...', roomSessionData);
      const initResponse = await fetch('/api/games/initialize-players', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          roomId: roomData.roomId,
          gameName: playerSession.gameName,
          roomSession: roomSessionData,
        }),
      });
      
      if (!initResponse.ok) {
        const errorData = await initResponse.json();
        throw new Error(errorData.error || 'Failed to initialize player states');
      }
      
      const response = await initResponse.json();
      console.log('✅ Player states API called, backend will handle initialization:', response);
      
      // Store game context
      const gameContext = {
        roomId: roomData.roomId,
        threadId: roomData.threadId,
        playerId: roomData.playerId,
        playerName: playerSession.playerName,
        gameName: playerSession.gameName
      };
      sessionStorage.setItem('gameContext', JSON.stringify(gameContext));
      
      // Store room session with all players info
      const roomSession = {
        roomId: roomData.roomId,
        gameName: playerSession.gameName,
        totalPlayers: roomData.players?.length || 0,
        players: roomData.players?.map((player, index) => ({
          id: player.id,
          name: player.name,
          isHost: player.isHost,
          gamePlayerId: (index + 1).toString() // For DSL mapping
        })) || [],
        timestamp: Date.now()
      };
      sessionStorage.setItem('roomSession', JSON.stringify(roomSession));
      console.log('💾 Room session stored:', roomSession);
      
      // 🔑 触发房间切换事件，通知 DynamicCopilotProvider 更新 threadId
      if (typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent('roomChanged', { 
          detail: { 
            roomId: roomData.roomId,
            threadId: roomData.threadId,
            action: 'startGame'
          }
        }));
        console.log('🏠 Dispatched roomChanged event for threadId:', roomData.threadId);
      }
      console.log('💾 Game context and player states stored');
      
      // Send start game message to Agent before navigating
      try {
        const response = await fetch('/api/copilotkit', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-Thread-ID': roomData.threadId
          },
          body: JSON.stringify({
            messages: [{
              role: 'user',
              content: 'start game'
            }],
            // 提前传入 gameName，服务端仍会以内存为准覆盖/兜底
            state: { gameName: playerSession.gameName }
          })
        });
        
        if (response.ok) {
          console.log('🎮 Start game message sent successfully');
        } else {
          console.log('⚠️ Failed to send start game message, continuing anyway');
        }
      } catch (error) {
        console.log('⚠️ Error sending start game message:', error, 'continuing anyway');
      }

      // Navigate to the main game engine
      router.push(`/?room=${roomData.roomId}&game=${playerSession.gameName}`);
      
    } catch (err) {
      console.error('❌ Failed to start game:', err);
      setError(err instanceof Error ? err.message : 'Failed to start game');
    } finally {
      setLoading(false);
    }
  };

  if (!playerSession) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-8 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-8 flex items-center justify-center">
      <div className="w-full max-w-2xl">
        {/* Back Button */}
        <Link href="/game-library" className="inline-flex items-center text-slate-600 hover:text-slate-900 mb-6 transition-colors">
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Game Library
        </Link>

        <Card className="shadow-xl bg-white/90 backdrop-blur border-0">
          <CardHeader className="text-center pb-6">
            <div className="flex items-center justify-center gap-3 mb-4">
              <Crown className="h-8 w-8 text-yellow-600" />
              <CardTitle className="text-3xl font-bold text-slate-900 capitalize">
                {playerSession.gameDisplayName}
              </CardTitle>
            </div>
            
            <CardDescription className="text-lg">
              Welcome, <span className="font-semibold text-slate-900">{playerSession.playerName}</span>!
              {mode === 'main' && " Choose whether to create a new room or join an existing one."}
              {mode === 'join' && " Select a room below to join the game."}
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-6">
            {!roomData ? (
              <>
                {mode === 'main' ? (
                  <div className="space-y-4">
                    <div className="text-center p-6 bg-slate-50 rounded-lg">
                      <Users className="h-12 w-12 text-slate-400 mx-auto mb-4" />
                      <h3 className="text-lg font-semibold text-slate-900 mb-4">Choose Your Option</h3>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <Button
                          onClick={createRoom}
                          disabled={loading}
                          className="h-14 text-base bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800 shadow-lg hover:shadow-xl transition-all duration-300"
                        >
                          <Crown className="h-5 w-5 mr-2" />
                          Create New Room
                        </Button>
                        
                        <Button
                          onClick={showJoinRooms}
                          variant="outline"
                          disabled={loading}
                          className="h-14 text-base border-2 border-blue-600 text-blue-600 hover:bg-blue-600 hover:text-white shadow-lg hover:shadow-xl transition-all duration-300"
                        >
                          <UserPlus className="h-5 w-5 mr-2" />
                          Join Existing Room
                        </Button>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <h3 className="text-lg font-semibold text-slate-900">Available Rooms</h3>
                      <Button
                        onClick={() => setMode('main')}
                        variant="ghost"
                        size="sm"
                      >
                        <ArrowLeft className="h-4 w-4 mr-2" />
                        Back
                      </Button>
                    </div>
                    
                    {loading ? (
                      <div className="text-center py-8">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
                        <p className="text-slate-600 mt-2">Loading rooms...</p>
                      </div>
                    ) : availableRooms.length === 0 ? (
                      <div className="text-center py-8 bg-slate-50 rounded-lg">
                        <Users className="h-12 w-12 text-slate-400 mx-auto mb-2" />
                        <p className="text-slate-600">No available rooms found.</p>
                        <p className="text-sm text-slate-500 mt-1">Be the first to create a room!</p>
                      </div>
                    ) : (
                      <div className="space-y-3">
                        {availableRooms.map((room) => (
                          <Card key={room.roomId} className="border border-slate-200 hover:border-blue-300 transition-colors">
                            <CardContent className="p-4">
                              <div className="flex items-center justify-between">
                                <div>
                                  <div className="flex items-center gap-2 mb-1">
                                    <Crown className="h-4 w-4 text-yellow-600" />
                                    <span className="font-semibold text-slate-900">{room.hostName}&apos;s Room</span>
                                  </div>
                                  <div className="flex items-center gap-4 text-sm text-slate-600">
                                    <span className="flex items-center gap-1">
                                      <Users className="h-3 w-3" />
                                      {room.currentPlayers}/{room.maxPlayers} players
                                    </span>
                                    <span className="text-xs text-slate-500">
                                      {new Date(room.createdAt).toLocaleTimeString()}
                                    </span>
                                  </div>
                                </div>
                                <Button
                                  onClick={() => joinSpecificRoom(room.roomId)}
                                  disabled={loading}
                                  size="sm"
                                  className="bg-blue-600 hover:bg-blue-700"
                                >
                                  Join Room
                                </Button>
                              </div>
                            </CardContent>
                          </Card>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </>
            ) : (
              <>
                {/* Room Created/Joined State */}
                <div className="space-y-4">
                  <div className="text-center p-6 bg-green-50 rounded-lg border border-green-200">
                    <div className="flex items-center justify-center gap-2 mb-2">
                      <Crown className="h-5 w-5 text-green-600" />
                      <Badge variant="outline" className="border-green-600 text-green-600">
                        {roomData.isHost ? 'Room Host' : 'Player'}
                      </Badge>
                    </div>
                    <h3 className="text-lg font-semibold text-green-900 mb-1">
                      {roomData.isHost ? 'Room Created Successfully!' : 'Joined Room Successfully!'}
                    </h3>
                    <p className="text-green-700">You can now start the game or wait for other players.</p>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <Card className="bg-slate-50">
                      <CardHeader className="pb-3">
                        <CardTitle className="text-sm font-medium text-slate-600">Room Details</CardTitle>
                      </CardHeader>
                      <CardContent className="pt-0 space-y-2">
                        <div className="flex justify-between">
                          <span className="text-sm text-slate-600">Room ID:</span>
                          <span className="text-sm font-mono bg-white px-2 py-1 rounded">{roomData.roomId.slice(0, 8)}...</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-slate-600">Player ID:</span>
                          <span className="text-sm font-semibold">#{roomData.playerId}</span>
                        </div>
                      </CardContent>
                    </Card>

                    <Card className="bg-slate-50">
                      <CardHeader className="pb-3">
                        <CardTitle className="text-sm font-medium text-slate-600">Current Players</CardTitle>
                      </CardHeader>
                      <CardContent className="pt-0">
                        {roomData.players && roomData.players.length > 0 ? (
                          <div className="space-y-2">
                            {roomData.players.map((player) => (
                              <div key={player.id} className="flex items-center gap-2">
                                <User className="h-4 w-4 text-slate-400" />
                                <span className="text-sm font-medium">{player.name}</span>
                                {player.isHost && <Badge variant="secondary" className="text-xs">Host</Badge>}
                              </div>
                            ))}
                          </div>
                        ) : (
                          <div className="flex items-center gap-2">
                            <User className="h-4 w-4 text-slate-400" />
                            <span className="text-sm font-medium">{playerSession.playerName}</span>
                            <Badge variant="secondary" className="text-xs">Host</Badge>
                          </div>
                        )}
                        
                        {roomData.isHost && (roomData.currentPlayers || 1) < 4 && (
                          <Button
                            onClick={addBots}
                            variant="outline"
                            size="sm"
                            disabled={loading}
                            className="w-full mt-3 text-xs border-dashed border-blue-300 text-blue-600 hover:bg-blue-50"
                          >
                            <UserPlus className="h-3 w-3 mr-1" />
                            Add Bots (Need {Math.max(0, 4 - (roomData.currentPlayers || 1))} more)
                          </Button>
                        )}
                        
                        <p className="text-xs text-slate-500 mt-2">
                          {roomData.currentPlayers || 1}/{roomData.maxPlayers || 8} players
                        </p>
                      </CardContent>
                    </Card>
                  </div>

                  <Button
                    onClick={startGame}
                    className="w-full h-14 text-lg bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 shadow-lg hover:shadow-xl transition-all duration-300"
                  >
                    <Play className="h-5 w-5 mr-2" />
                    Start Game
                  </Button>
                </div>
              </>
            )}

            {error && (
              <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-red-700 text-center">{error}</p>
                <Button
                  onClick={() => setError(null)}
                  variant="outline"
                  className="w-full mt-3 border-red-300 text-red-700 hover:bg-red-50"
                >
                  Dismiss
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
