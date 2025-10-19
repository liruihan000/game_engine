"use client";
/* eslint-disable @typescript-eslint/no-explicit-any */

import { useCoAgent, useCopilotAction, useCoAgentStateRender, useCopilotAdditionalInstructions, useLangGraphInterrupt, useCopilotChat } from "@copilotkit/react-core";
import { CopilotKitCSSProperties, CopilotChat, CopilotPopup } from "@copilotkit/react-ui";
import { TextMessage, MessageRole } from "@copilotkit/runtime-client-gql";
import { useCallback, useEffect, useRef, useState } from "react";
import type React from "react";
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import AppChatHeader, { PopupHeader } from "@/components/canvas/AppChatHeader";
import { X, Check, Loader2 } from "lucide-react"
import CardRenderer from "@/components/canvas/CardRenderer";
import ShikiHighlighter from "react-shiki/web";
import { motion, useScroll, useTransform, useMotionValueEvent } from "motion/react";
import { EmptyState } from "@/components/empty-state";
import { cn } from "@/lib/utils";
import type { AgentState, PlanStep, Item, ItemData, CardType, GamePosition, CharacterCardData, ActionButtonData, PhaseIndicatorData, TextDisplayData, VotingPanelData, AvatarSetData, BackgroundControlData, ResultDisplayData, TimerData, ComponentSize, ChatMessage, RoomPlayer, HandsCardData, ScoreBoardData, CoinDisplayData, StatementBoardData, ReactionTimerData, NightOverlayData, TurnIndicatorData, HealthDisplayData, InfluenceSetData, PlayerStatesDisplayData, PlayerActionsDisplayData } from "@/lib/canvas/types";

// === Unified Game Title and Phase Display System ===

// ✅ Optimized game title component - using unified state management without flicker
function GameTitle({ viewState, titleClasses }: { viewState: AgentState; titleClasses: string }) {
  const [mounted, setMounted] = useState(false);
  const { gameTitle } = useUnifiedGameState(viewState);  // 🎯 Get all logic in one line
  
  useEffect(() => {
    setMounted(true);
  }, []);
  
  // Use stable default during hydration to prevent mismatch
  const displayTitle = mounted ? gameTitle : 'Game Engine';
  
  return <div className={cn(titleClasses, "text-2xl font-semibold")}>{displayTitle}</div>;
}

// ✅ Optimized game phase component - using unified state management without flicker
function GamePhaseInfo({ viewState, titleClasses }: { viewState: AgentState; titleClasses: string }) {
  const [mounted, setMounted] = useState(false);
  const { phaseDisplay } = useUnifiedGameState(viewState);  // 🎯 Get all logic in one line
  
  useEffect(() => {
    setMounted(true);
  }, []);
  
  // Use stable default during hydration to prevent mismatch
  const displayPhase = mounted ? phaseDisplay : 'AI-Powered Game Engine';
  
  return <div className={cn(titleClasses, "mt-2 text-sm leading-6")}>{displayPhase}</div>;
}

// ✅ Unified game state management system - define once, reuse everywhere
function useUnifiedGameState(viewState: AgentState) {
  return {
    // 🎮 Game basic info - unified null handling and formatting
    gameName: viewState.gameName || null,
    gameTitle: (() => {
      const name = viewState.gameName;
      if (!name) return 'No Game Active';
      if (typeof name === 'string' && name.includes(':')) {
        return name.split(':')[0].trim();
      }
      return typeof name === 'string' ? name : 'Current Game';
    })(),
    
    // 📍 Phase info - unified formatting display
    phaseId: viewState.current_phase_id ?? null,  
    phaseName: viewState.current_phase_name || null,
    phaseDisplay: (() => {
      const id = viewState.current_phase_id;
      const name = viewState.current_phase_name;
      if (id !== undefined && name) return `Phase ${id}: ${name}`;
      if (id !== undefined) return `Phase ${id}`;
      return 'Not Started';
    })(),
    
    // 👥 Player states - unified processing and calculation
    playerStates: viewState.player_states ?? {},
    playerCount: Object.keys(viewState.player_states ?? {}).length,
    alivePlayers: Object.entries(viewState.player_states ?? {})
      .filter(([, data]: [string, any]) => data?.is_alive !== false)
      .map(([id]) => id),
    deadPlayers: viewState.deadPlayers ?? [],
    
    // 📝 Actions and votes - unified formatting
    playerActions: viewState.playerActions ?? {},
    votes: viewState.vote ?? [],
    votesDisplay: (viewState.vote ?? []).length > 0
      ? (viewState.vote ?? []).map((v: any) => `${v.playerid} → "${v.option}"`).join('; ')
      : 'No votes yet',
    
    // 📊 Computed properties - avoid repeated calculation
    recentActions: Object.entries(viewState.playerActions ?? {})
      .map(([id, data]: [string, any]) => ({ id, ...data }))
      .sort((a: any, b: any) => (b.timestamp || 0) - (a.timestamp || 0))
      .slice(0, 5),
  };
}

// ✅ Unified Agent instruction generation - directly process viewState, no Hook
function generateGameInstructions(viewState: AgentState): string {
  // Extract state directly, no React Hook calls
  const gameName = viewState.gameName;
  const gameTitle = gameName ? (gameName.includes(':') ? gameName.split(':')[0].trim() : gameName) : 'No Game Active';
  
  const phaseId = viewState.current_phase_id;
  const phaseName = viewState.current_phase_name;
  const phaseDisplay = (phaseId !== undefined && phaseName) ? `Phase ${phaseId}: ${phaseName}` 
    : (phaseId !== undefined ? `Phase ${phaseId}` : 'Not in any game');
  
  const playerStates = viewState.player_states ?? {};
  const playerCount = Object.keys(playerStates).length;
  const alivePlayers = Object.entries(playerStates).filter(([, data]: [string, any]) => data?.is_alive !== false);
  
  const votes = viewState.vote ?? [];
  const votesDisplay = votes.length > 0 
    ? votes.map((v: any) => `${v.playerid} → "${v.option}"`).join('; ')
    : 'No votes yet';
  
  return [
    "=== UNIFIED GAME STATE ===",
    `🎮 Game: ${gameTitle}`,
    `📍 Phase: ${phaseDisplay}`,
    `👥 Players: ${playerCount} total (${alivePlayers.length} alive)`,
    `🗳️  Votes: ${votesDisplay}`,
    "",
    "=== DETAILED DATA ===",
    `Player States: ${JSON.stringify(playerStates, null, 2)}`,
  ].join('\n');
}
import { GAME_GRID_STYLE, normalizePosition } from "@/lib/canvas/types";
import { initialState, isNonEmptyAgentState, defaultDataFor } from "@/lib/canvas/state";
// import { projectAddField4Item, projectSetField4ItemText, projectSetField4ItemDone, projectRemoveField4Item, chartAddField1Metric, chartSetField1Label, chartSetField1Value, chartRemoveField1Metric } from "@/lib/canvas/updates";
import useMediaQuery from "@/hooks/use-media-query";
import { GameChatArea } from "@/components/chat/GameChatArea";
import BroadcastInput from "@/components/tools/BroadcastInput";
import { getCurrentPlayerId } from "@/lib/player-utils";
import NewItemMenu from "@/components/canvas/NewItemMenu";

export default function CopilotKitPage() {
  // Use consistent agent name across all components - room isolation via threadId
  const { state, setState } = useCoAgent<AgentState>({
    name: "sample_agent", // 🔑 Fixed agent name, room isolation via threadId in DynamicCopilotProvider
    initialState: (() => {
      // One-time initialization logic - no useEffect needed
      if (typeof window === 'undefined') return initialState;
      
      try {
        // Get game context and room session from sessionStorage
        const gameContext = sessionStorage.getItem('gameContext');
        const roomSession = sessionStorage.getItem('roomSession');
        const urlGameName = new URLSearchParams(window.location.search).get('game');
        
        if (gameContext && roomSession) {
          const context = JSON.parse(gameContext);
          const room = JSON.parse(roomSession);
          // console.log('🎮 Game initialized from sessionStorage:', { gameName: context.gameName, players: room.players?.length });
          return {
            ...initialState,
            gameName: context.gameName,
            roomSession: room
          };
        }
        
        // Fallback to URL gameName only
        if (urlGameName) {
          console.log('🎮 Game initialized from URL:', urlGameName);
          return { ...initialState, gameName: urlGameName };
        }
      } catch (error) {
        console.error('Failed to initialize game state from sessionStorage:', error);
      }
      
      return initialState;
    })(),
  });

  const { appendMessage } = useCopilotChat();

  // Execution state to prevent concurrent operations
  const [isExecuting, setIsExecuting] = useState(false);

  // Unified user interaction handler - replaces scattered state logic
  const handleUserInteraction = useCallback(async (content: string, actionType?: string) => {
    // Prevent concurrent executions
    if (isExecuting) {
      console.log('⚠️ Operation already in progress, skipping...');
      return;
    }

    setIsExecuting(true);
    try {
      // Ensure state integrity
      const currentState = state ?? initialState;
      
      // Prefer filling missing state from sessionStorage first
      if (!currentState.gameName || !currentState.roomSession) {
        const gameContext = sessionStorage.getItem('gameContext');
        const roomSession = sessionStorage.getItem('roomSession');
        
        if (gameContext || roomSession) {
          setState((prev) => {
            const base = prev ?? initialState;
            const updates: Partial<AgentState> = {};
            
            // Fill gameName
            if (!base.gameName && gameContext) {
              const context = JSON.parse(gameContext);
              updates.gameName = context.gameName;
              console.log('🔄 Filled gameName from sessionStorage:', context.gameName);
            }
            
            // Fill roomSession
            if (!base.roomSession && roomSession) {
              updates.roomSession = JSON.parse(roomSession);
              console.log('🔄 Filled roomSession from sessionStorage');
            }
            
            // 保持现有的chatMessages，避免覆盖
            const result = { ...base, ...updates } as AgentState;
            if (base.chatMessages && base.chatMessages.length > 0) {
              result.chatMessages = base.chatMessages;
            }
            return result;
          });
        }
        // If sessionStorage is missing, fall back to URL
        else if (!currentState.gameName) {
          const urlGameName = new URLSearchParams(window.location.search).get('game');
          if (urlGameName) {
            setState(prev => ({ 
              ...prev, 
              gameName: urlGameName 
            } as AgentState));
            console.log('🔄 Filled gameName from URL:', urlGameName);
          }
        }
      }
      
      // Send message via unified handler
      await appendMessage(new TextMessage({
        role: MessageRole.User,
        content
      }));
      
      if (actionType && process.env.NODE_ENV === 'development') {
        console.log(`🎮 User interaction: ${actionType} - "${content}"`);
      }
    } catch (error) {
      console.error('User interaction failed:', error);
    } finally {
      setIsExecuting(false);
    }
  }, [state, setState, appendMessage, isExecuting]);

  // Handle action button clicks
  const handleButtonClick = useCallback(async (item: Item) => {
    // Prevent concurrent executions
    if (isExecuting) {
      console.log('⚠️ Operation already in progress, skipping button click...');
      return;
    }

    const buttonData = item.data as ActionButtonData;
    
    // Use unified interaction handling
    await handleUserInteraction(
      `Button "${item.name}" (ID: ${item.id}) has been clicked. Action: ${buttonData.action}`,
      'button_click'
    );
  }, [handleUserInteraction, isExecuting]);

  // Handle voting
  const handleVote = useCallback(async (votingId: string, playerId: string, option: string) => {
    // Update state first
    setState((prev) => {
      const base = prev ?? initialState;
      const currentVotes = base.vote ?? [];
      
      // Remove any existing vote from this player for this voting ID
      const filteredVotes = currentVotes.filter(v => !(v.voteid === votingId && v.playerid === playerId));
      
      // Add the new vote
      const newVote = {
        voteid: votingId,
        playerid: playerId,
        option: option
      };
      
      return {
        ...base,
        vote: [...filteredVotes, newVote]
      } as AgentState;
    });

    // Use unified interaction handling
    await handleUserInteraction(
      `Player ${playerId} voted "${option}" in voting ${votingId}`,
      'vote_cast'
    );
  }, [setState, handleUserInteraction]);

  // Simple cache for the last non-empty agent state
  const cachedStateRef = useRef<AgentState>(state ?? initialState);
  
  useEffect(() => {
    if (isNonEmptyAgentState(state)) {
      const newState = state as AgentState;
      cachedStateRef.current = newState;
    }
  }, [state]);
  // Use latest state when available, fall back to cached state to prevent flicker
  const viewState: AgentState = isNonEmptyAgentState(state) ? (state as AgentState) : cachedStateRef.current;

  // Handle chat messages with bot selection
  const handleSendChatMessage = useCallback(async (message: string, targetBotId?: string) => {
    const playerId = getCurrentPlayerId();
    const playerName = viewState.player_states?.[playerId || '']?.name as string || `Player ${playerId}`;
    
    if (!playerId) return;

    // 1. Immediately add user message to in-memory storage
    const userMessage: ChatMessage = {
      id: `msg-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`,
      playerId: playerId,
      playerName: playerName,
      message: message,
      timestamp: Date.now(),
      type: 'message'
    };

    setChatMessages(prev => [...prev, userMessage]);

    // 2. Send to Agent for bot reply handling
    if (targetBotId) {
      await handleUserInteraction(
        `Player ${playerName} to Bot ${targetBotId}: ${message}`,
        'direct_chat'
      );
    } else {
      await handleUserInteraction(
        `Player ${playerName} in game chat: ${message}`,
        'game_chat'
      );
    }
  }, [handleUserInteraction, viewState.player_states]);

  // State declarations - move before useCallback to avoid hoisting issues
  const [pendingTextPrompt, setPendingTextPrompt] = useState<{
    speakerId?: string;
    title?: string;
    placeholder?: string;
    toBotId?: string;
  } | null>(null);
  const [pendingTextValue, setPendingTextValue] = useState<string>("");

  // Broadcast input UI state
  const [pendingBroadcast, setPendingBroadcast] = useState<{
    title?: string;
    placeholder?: string;
    prefill?: string;
  } | null>(null);
  const [broadcastOpen, setBroadcastOpen] = useState<boolean>(false);

  // Agent-triggered UI tool to open floating broadcast input
  useCopilotAction({
    name: "createTextInputPanel",
    description: "Open a floating, pill-shaped text input panel for collecting user input and broadcasting.",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "title", type: "string", required: false, description: "Small caption above the input" },
      { name: "placeholder", type: "string", required: false, description: "Placeholder text in the input" },
      { name: "prefill", type: "string", required: false, description: "Prefill text for the input" },
    ],
    handler: ({ title, placeholder, prefill }: { title?: string; placeholder?: string; prefill?: string }) => {
      setPendingBroadcast({ title, placeholder, prefill });
      setBroadcastOpen(true);
      return "broadcast_input_opened";
    },
  });

  // Confirm handler for promptUserText dialog
  const handleConfirmPromptText = useCallback(async () => {
    if (!pendingTextPrompt) return;
    const speakerId = pendingTextPrompt.speakerId || getCurrentPlayerId();
    if (!speakerId) {
      setPendingTextPrompt(null);
      setPendingTextValue("");
      return;
    }
    const playerName = viewState.player_states?.[speakerId]?.name as string || `Player ${speakerId}`;
    const text = (pendingTextValue || "").trim();
    if (!text) return;
    if (pendingTextPrompt.toBotId) {
      await handleUserInteraction(`Player ${playerName} to Bot ${pendingTextPrompt.toBotId}: ${text}`, 'direct_chat');
    } else {
      await handleUserInteraction(`Player ${playerName} in game chat: ${text}`, 'user_input');
    }
    setPendingTextPrompt(null);
    setPendingTextValue("");
  }, [pendingTextPrompt, pendingTextValue, viewState.player_states, handleUserInteraction]);

  // Get available bots from roomSession in sessionStorage
  const getAvailableBots = useCallback(() => {
    const roomSession = viewState.roomSession;
    const deadPlayers = (viewState.deadPlayers || []).map(String);
    if (!roomSession?.players) return [];
    const players = roomSession.players as (RoomPlayer & { is_bot?: boolean; role?: string })[];
    return players
      .filter((player) => player.is_bot === true)
      .map((player) => ({
        id: String(player.id ?? ""),
        name: String(player.name ?? ""),
        role: player.role,
        isAlive: !deadPlayers.includes(String(player.id ?? ""))
      }));
  }, [viewState.roomSession, viewState.deadPlayers]);

  const isDesktop = useMediaQuery("(min-width: 768px)");
  const [showJsonView, setShowJsonView] = useState<boolean>(false);
  
  // Chat messages are stored locally, independent from Agent state
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  
  // Listen for room changes and clear chat history
  const currentRoomId = viewState.roomSession?.roomId;
  const previousRoomIdRef = useRef<string | undefined>(undefined);
  
  useEffect(() => {
    if (currentRoomId && previousRoomIdRef.current && currentRoomId !== previousRoomIdRef.current) {
      // Room changed, clear chat messages
      console.log('🏠 Room changed, clearing chat messages');
      setChatMessages([]);
    }
    previousRoomIdRef.current = currentRoomId;
  }, [currentRoomId]);
  const scrollAreaRef = useRef<HTMLDivElement | null>(null);
  const { scrollY } = useScroll({ container: scrollAreaRef });
  const headerScrollThreshold = 64;
  const headerOpacity = useTransform(scrollY, [0, headerScrollThreshold], [1, 0]);
  const titleInputRef = useRef<HTMLInputElement | null>(null);
  const descTextareaRef = useRef<HTMLInputElement | null>(null);
  // Strong idempotency during plan execution: allow only one creation per type while plan runs
  const createdByTypeRef = useRef<Partial<Record<CardType, string>>>({});
  const prevPlanStatusRef = useRef<string | null>(null);

  // Reset per-plan idempotency map on plan start/end or when plan definition changes
  useEffect(() => {
    const status = String(viewState?.planStatus ?? "");
    const prevStatus = prevPlanStatusRef.current;
    const started = status === "in_progress" && prevStatus !== "in_progress";
    const ended = prevStatus === "in_progress" && (status === "completed" || status === "failed" || status === "");
    if (started || ended) {
      createdByTypeRef.current = {};
    }
    prevPlanStatusRef.current = status;
  }, [viewState?.planStatus]);

  useMotionValueEvent(scrollY, "change", (y) => {
    const disable = y >= headerScrollThreshold;
    if (disable) {
      titleInputRef.current?.blur();
      descTextareaRef.current?.blur();
    }
  });

  // ✅ Optimized: Only log in development
  useEffect(() => {
    if (process.env.NODE_ENV === 'development') {
      console.log("[CoAgent state updated]", state);
    }
  }, [state]);

  // Reset JSON view when there are no items
  useEffect(() => {
    const itemsCount = (viewState?.items ?? []).length;
    if (itemsCount === 0 && showJsonView) {
      setShowJsonView(false);
    }
  }, [viewState?.items, showJsonView]);

  // ✅ Removed roomSession initialization useEffect - now handled in initialState

  // Use cached viewState to derive plan-related fields
  const planStepsMemo = (viewState?.planSteps ?? initialState.planSteps) as PlanStep[];
  const planStatusMemo = viewState?.planStatus ?? initialState.planStatus;
  const currentStepIndexMemo = typeof viewState?.currentStepIndex === "number" ? viewState.currentStepIndex : initialState.currentStepIndex;

  // One-time final summary renderer in chat when plan completes or fails
  useCoAgentStateRender<AgentState>({
    name: "sample_agent",
    nodeName: "plan-final-summary",
    render: ({ state }) => {
      const status = String(state?.planStatus ?? "");
      const steps = (state?.planSteps ?? []) as PlanStep[];
      if (!Array.isArray(steps) || steps.length === 0) return null;
      if (status !== "completed" && status !== "failed") return null;
      const count = steps.length;
      return (
        <div className="my-2 w-full">
          <Accordion type="single" collapsible defaultValue="done">
            <AccordionItem value="done">
              <AccordionTrigger className="text-xs">
                <span className="inline-flex items-center gap-2">
                  <Check className={cn("h-4 w-4", status === "completed" ? "text-green-600" : "text-red-600")} />
                  <span className="font-medium">{count} steps {status}</span>
                </span>
              </AccordionTrigger>
              <AccordionContent>
                <div className="rounded-2xl border shadow-sm bg-card p-4">
                  <div className="mb-2 text-xs font-semibold">Plan <span className={cn("ml-2 rounded-full px-2 py-0.5 text-[10px] font-medium border", status === "completed" ? "text-green-700 border-green-300 bg-green-50" : "text-red-700 border-red-300 bg-red-50")}>{status}</span></div>
                  <ol className="space-y-1">
                    {steps.map((s, i) => (
                      <li key={`${s.title ?? "step"}-${i}`} className="flex items-start gap-2">
                        <span className="mt-0.5 inline-flex h-4 w-4 items-center justify-center">
                          {String(s.status).toLowerCase() === "completed" ? (
                            <Check className="h-4 w-4 text-green-600" />
                          ) : String(s.status).toLowerCase() === "failed" ? (
                            <X className="h-4 w-4 text-red-600" />
                          ) : (
                            <span className="block h-2 w-2 rounded-full bg-gray-300" />
                          )}
                        </span>
                        <div className="flex-1 text-xs">
                          <div className={cn("leading-5", String(s.status).toLowerCase() === "completed" && "text-green-700", String(s.status).toLowerCase() === "failed" && "text-red-700")}>{s.title ?? `Step ${i + 1}`}</div>
                        </div>
                      </li>
                    ))}
                  </ol>
                </div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </div>
      );
    },
  });

  const getStatePreviewJSON = (s: AgentState | undefined): Record<string, unknown> => {
    const snapshot = (s ?? initialState) as AgentState;
    const { items, player_states } = snapshot;
    return {
      items: items ?? initialState.items,
      player_states: player_states,
    };
  };



  // Strengthen grounding: always prefer shared state over chat history
  useCopilotAdditionalInstructions({
    instructions: (() => {
      // ✅ Use unified game state and instruction generation - eliminate all duplicate logic!
      const items = viewState.items ?? initialState.items;
      const summary = items
        .slice(0, 5)
        .map((p: Item) => `id=${p.id} • name=${p.name} • type=${p.type}`)
        .join("\n");
      
      // 🎯 One line call to get all formatted state info
      const unifiedInstructions = generateGameInstructions(viewState);
      
      // ✅ Complete frontend tools directory - all tools available to Agent (59 total)
      const gameTools = {
        // 🎯 Basic operations
        setGlobalTitle: "Set the global page title",
        setGlobalDescription: "Set the global page subtitle/description", 
        setItemName: "Rename an existing item by id",
        setItemSubtitleOrDescription: "Set an item's subtitle/short description",
        setItemPosition: "Change an item's position on the grid",
        deleteItem: "Delete an item by id",
        clearCanvas: "Clear all canvas items except avatar sets",
        
        // 🎮 Core game components (create + update)
        createCharacterCard: "Create a character card (role, position, size, description)",
        updateCharacterCard: "Update existing character card properties",
        createActionButton: "Create an action button (label, action, enabled, position)",
        updateActionButton: "Update existing action button properties", 
        createPhaseIndicator: "Create a phase indicator (currentPhase, position, timer)",
        updatePhaseIndicator: "Update existing phase indicator properties",
        createTextDisplay: "Create a text panel (content, title, type, position)",
        updateTextDisplay: "Update existing text display content",
        createVotingPanel: "Create a voting panel (votingId, options, position)",
        updateVotingPanel: "Update existing voting panel options",
        
        // 👥 Player system
        createAvatarSet: "Create player avatars overlay (avatarType)",
        markPlayerDead: "Mark a player as dead (affects avatar display)",
        createPlayerStatesDisplay: "Display current player states in scrollable panel",
        createPlayerActionsDisplay: "Display player actions log in scrollable panel",
        
        // ⏰ Timer system
        createTimer: "Create countdown timer that expires and notifies agent",
        updateTimer: "Update existing timer duration or label",
        createReactionTimer: "Create quick reaction/challenge timer bar",
        startReactionTimer: "Start a reaction timer countdown",
        stopReactionTimer: "Stop a reaction timer", 
        resetReactionTimer: "Reset reaction timer to initial state",
        
        // 🎨 Visual effects
        changeBackgroundColor: "Create background control and set initial color",
        createBackgroundControl: "Create background color control component",
        createResultDisplay: "Create large gradient-styled result display",
        updateResultDisplay: "Update result display content",
        createNightOverlay: "Create night overlay with title and blur",
        setNightOverlay: "Update night overlay visibility and properties",
        
        // 🃏 Card games
        createHandsCard: "Create hand card for card games (cardType, cardName)",
        createHandsCardForPlayer: "Create hand card for specific player",
        updateHandsCard: "Update hand card properties",
        setHandsCardAudience: "Set hand card audience visibility",
        
        // 📊 Scoring system
        createScoreBoard: "Create score board with player entries",
        updateScoreBoard: "Update score board entries and properties",
        upsertScoreEntry: "Add or update a score entry",
        removeScoreEntry: "Remove score entry by player id",
        
        // 🏥 Health/status
        createHealthDisplay: "Create health/bullets display (value, max, style)",
        updateHealthDisplay: "Update health display values",
        createInfluenceSet: "Create influence cards set (Coup-style games)",
        updateInfluenceSet: "Update influence set properties",
        revealInfluenceCard: "Reveal influence card by index",
        
        // 📝 Special input
        createTextInputPanel: "Create text input panel for user input",
        createStatementBoard: "Create statement board for Two Truths and a Lie",
        updateStatementBoard: "Update statement board content",
        createTurnIndicator: "Create turn indicator showing current active player",
        updateTurnIndicator: "Update turn indicator properties",
        
        // 🤖 Interaction system
        addBotChatMessage: "Add bot message to chat",
        promptUserText: "Open dialog for user text input with speaker",
        submitVote: "Submit vote for voting panels"
      } as const;
      const gameToolsStr = JSON.stringify(gameTools, null, 2);
      
      const fieldSchema = "";
      const toolUsageHints = [
        "GAME TOOL USAGE HINTS:",
        "LOOP CONTROL: When asked to 'add a couple' items, add at most 2 and stop. Avoid repeated calls to the same mutating tool in one turn.",
        "VERIFICATION: After tools run, re-read the latest state and confirm what actually changed.",
      ].join("\n");
      return [
        "ALWAYS ANSWER FROM SHARED STATE (GROUND TRUTH).",
        "",
        unifiedInstructions,  // 🚀 Unified game state info - done in one line!
        "",
        "Items (sample):",
        summary || "(none)",
        "",
        "game_tool:",
        gameToolsStr,
        fieldSchema,
        toolUsageHints,
      ].join("\n");
    })(),
  });

  // HITL: dropdown selector for item choice using LangGraph interrupt
  useLangGraphInterrupt({
    enabled: ({ eventValue }) => {
      try {
        return typeof eventValue === "object" && eventValue?.type === "choose_item";
      } catch {
        return false;
      }
    },
    render: ({ event, resolve }) => {
      const items = viewState.items ?? initialState.items;
      if (!items.length) {
        return (
          <div className="rounded-md border bg-white p-4 text-sm shadow">
            <p>No items available.</p>
            <button
              className="mt-3 rounded border px-3 py-1"
              onClick={() => resolve("")}
            >
              Close
            </button>
          </div>
        );
      }
      let selectedId = items[0].id;
      return (
        <div className="rounded-md border bg-white p-4 text-sm shadow">
          <p className="mb-2 font-medium">Select an item</p>
          <p className="mb-3 text-xs text-gray-600">{(event?.value as { content?: string })?.content ?? "Which item should I use?"}</p>
          <select
            className="w-full rounded border px-2 py-1"
            defaultValue={selectedId}
            onChange={(e) => {
              selectedId = e.target.value;
            }}
          >
            {items.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name} ({p.id})
              </option>
            ))}
          </select>
          <div className="mt-3 flex justify-end gap-2">
            <button
              className="rounded border px-3 py-1"
              onClick={() => resolve("")}
            >
              Cancel
            </button>
            <button
              className="rounded border bg-blue-600 px-3 py-1 text-white"
              onClick={() => resolve(selectedId)}
            >
              Use item
            </button>
          </div>
        </div>
      );
    },
  });

  // HITL: choose a card type when not specified
  useLangGraphInterrupt({
    enabled: ({ eventValue }) => {
      try {
        return typeof eventValue === "object" && eventValue?.type === "choose_card_type";
      } catch {
        return false;
      }
    },
    render: ({ event, resolve }) => {
      const options: { id: CardType; label: string }[] = [
        { id: "character_card", label: "Character Card" },
        { id: "action_button", label: "Action Button" },
        { id: "phase_indicator", label: "Phase Indicator" },
        { id: "text_display", label: "Text Display" },
      ];
      let selected: CardType | "" = "";
      return (
        <div className="rounded-md border bg-white p-4 text-sm shadow">
          <p className="mb-2 font-medium">Select a card type</p>
          <p className="mb-3 text-xs text-gray-600">{(event?.value as { content?: string })?.content ?? "Which type of card should I create?"}</p>
          <select
            className="w-full rounded border px-2 py-1"
            defaultValue=""
            onChange={(e) => {
              selected = e.target.value as CardType;
            }}
          >
            <option value="" disabled>Select an item type…</option>
            {options.map((opt) => (
              <option key={opt.id} value={opt.id}>{opt.label}</option>
            ))}
          </select>
          <div className="mt-3 flex justify-end gap-2">
            <button className="rounded border px-3 py-1" onClick={() => resolve("")}>Cancel</button>
            <button
              className="rounded border bg-blue-600 px-3 py-1 text-white"
              onClick={() => selected && resolve(selected)}
              disabled={!selected}
            >
              Use type
            </button>
          </div>
        </div>
      );
    },
  });

  const updateItem = useCallback(
    (itemId: string, updates: Partial<Item>) => {
      setState((prev) => {
        const base = prev ?? initialState;
        const items: Item[] = base.items ?? [];
        const nextItems = items.map((p) => (p.id === itemId ? { ...p, ...updates } : p));
        return { ...base, items: nextItems } as AgentState;
      });
    },
    [setState]
  );

  const updateItemData = useCallback(
    (itemId: string, updater: (prev: ItemData) => ItemData) => {
      setState((prev) => {
        const base = prev ?? initialState;
        const items: Item[] = base.items ?? [];
        const nextItems = items.map((p) => (p.id === itemId ? { ...p, data: updater(p.data) } : p));
        return { ...base, items: nextItems } as AgentState;
      });
    },
    [setState]
  );

  const deleteItem = useCallback((itemId: string) => {
    setState((prev) => {
      const base = prev ?? initialState;
      const existed = (base.items ?? []).some((p) => p.id === itemId);
      const items: Item[] = (base.items ?? []).filter((p) => p.id !== itemId);
      return { ...base, items, lastAction: existed ? `deleted:${itemId}` : `not_found:${itemId}` } as AgentState;
    });
  }, [setState]);

  // Checklist item local helper removed; Copilot actions handle checklist CRUD



  // Remove checklist item local helper removed; use Copilot action instead

  // Dummy function for game engine (no tags needed)
  const toggleTag = useCallback(() => {
    // Game components don't use tags, so this is a no-op
  }, []);

  // Helper to generate default data by type

  const addItem = useCallback((type: CardType, name?: string, customData?: ItemData) => {
    const t: CardType = type;
    let createdId = "";
    setState((prev) => {
      const base = prev ?? initialState;
      const items: Item[] = base.items ?? [];
      // Derive next numeric id robustly from both itemsCreated counter and max existing id
      const maxExisting = items.reduce((max, it) => {
        const parsed = Number.parseInt(String(it.id ?? "0"), 10);
        return Number.isFinite(parsed) ? Math.max(max, parsed) : max;
      }, 0);
      const priorCount = Number.isFinite(base.itemsCreated) ? (base.itemsCreated as number) : 0;
      const nextNumber = Math.max(priorCount, maxExisting) + 1;
      createdId = String(nextNumber).padStart(4, "0");
      const item: Item = {
        id: createdId,
        type: t,
        name: name && name.trim() ? name.trim() : "",
        subtitle: "",
        data: customData || defaultDataFor(t),
      };
      const nextItems = [...items, item];
      // clamp to one per type when plan is active
      const planActive = String(base?.planStatus ?? "") === "in_progress";
      let deduped = nextItems;
      if (planActive) {
        const seen = new Set<string>();
        deduped = [];
        for (const it of nextItems) {
          const key = it.type;
          if (seen.has(key)) continue;
          seen.add(key);
          deduped.push(it);
        }
      }
      return { ...base, items: deduped, itemsCreated: nextNumber, lastAction: `created:${createdId}` } as AgentState;
    });
    return createdId;
  }, [setState]);



  // Frontend Actions (exposed as tools to the agent via CopilotKit)
  useCopilotAction({
    name: "setGlobalTitle",
    description: "Set the global title/name (outside of items).",
    available: "frontend", // ✅ Pure frontend operation, no Agent trigger
    followUp: false,
    parameters: [
      { name: "title", type: "string", required: true, description: "The new global title/name." },
    ],
    handler: ({ title }: { title: string }) => {
      setState((prev) => ({ ...(prev ?? initialState), globalTitle: title }));
    },
  });

  useCopilotAction({
    name: "setGlobalDescription",
    description: "Set the global description/subtitle (outside of items).",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "description", type: "string", required: true, description: "The new global description/subtitle." },
    ],
    handler: ({ description }: { description: string }) => {
      setState((prev) => ({ ...(prev ?? initialState), globalDescription: description }));
    },
  });

  // Frontend Actions (item-scoped)
  useCopilotAction({
    name: "setItemName",
    description: "Set an item's name/title.",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "name", type: "string", required: true, description: "The new item name/title." },
      { name: "itemId", type: "string", required: true, description: "Target item id." },
    ],
    handler: ({ name, itemId }: { name: string; itemId: string }) => {
      updateItem(itemId, { name });
    },
  });

  // Set item subtitle
  useCopilotAction({
    name: "setItemSubtitleOrDescription",
    description: "Set an item's description/subtitle (short description or subtitle).",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "subtitle", type: "string", required: true, description: "The new item description/subtitle." },
      { name: "itemId", type: "string", required: true, description: "Target item id." },
    ],
    handler: ({ subtitle, itemId }: { subtitle: string; itemId: string }) => {
      updateItem(itemId, { subtitle });
    },
  });



  useCopilotAction({
    name: "createCharacterCard",
    description: "Create a character card for the game.",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "name", type: "string", required: true, description: "Item name"},
      { name: "role", type: "string", required: true, description: "Character role (e.g., werewolf, seer, villager)" },
      { name: "position", type: "string", required: true, description: "Grid position (select: 'top-left' | 'top-center' | 'top-right' | 'middle-left' | 'center' | 'middle-right' | 'bottom-left' | 'bottom-center' | 'bottom-right')" },
      { name: "size", type: "string", required: false, description: "Card size (select: 'small' | 'medium' | 'large'; default: 'medium')" },
      { name: "description", type: "string", required: false, description: "Optional character description" },
      { name: "audience_type", type: "boolean", required: false, description: "Whether all players can see this (true) or only specific players (false). Default: true" },
      { name: "audience_ids", type: "string[]", required: false, description: "Array of player IDs who can see this component (only used when audience_type=false). Example: ['1', '2']" },
    ],
    handler: ({ name, role, position, size, description, audience_type, audience_ids }: { 
      name: string;
      role: string; 
      position: string; 
      size?: string; 
      description?: string;
      audience_type?: boolean;
      audience_ids?: string[];
    }) => {
      const normalized = (name ?? "").trim();
      
      // Name-based idempotency: if an item with same type+name exists, return it
      if (normalized) {
        const existing = (viewState.items ?? initialState.items).find((it) => 
          it.type === "character_card" && (it.name ?? "").trim() === normalized
        );
        if (existing) {
          return existing.id;
        }
      }
      
      const data: CharacterCardData = {
        role,
        position: normalizePosition(position || 'center'),
        size: size as ComponentSize,
        description,
        audience_type: audience_type ?? true,
        audience_ids: audience_ids ?? []
      };
      return addItem("character_card", name, data);
    },
  });

  useCopilotAction({
    name: "createActionButton",
    description: "Create an action button for player interactions.",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "name", type: "string", required: true, description: "Item name" },
      { name: "label", type: "string", required: true, description: "Button text" },
      { name: "action", type: "string", required: true, description: "Action identifier when clicked" },
      { name: "enabled", type: "boolean", required: true, description: "Whether button is clickable" },
      { name: "position", type: "string", required: true, description: "Grid position (select: 'top-left' | 'top-center' | 'top-right' | 'middle-left' | 'center' | 'middle-right' | 'bottom-left' | 'bottom-center' | 'bottom-right')" },
      { name: "size", type: "string", required: false, description: "Button size (select: 'small' | 'medium' | 'large')" },
      { name: "variant", type: "string", required: false, description: "Button style (select: 'primary' | 'secondary' | 'danger')" },
      { name: "audience_type", type: "boolean", required: false, description: "Whether all players can see this (true) or only specific players (false). Default: true" },
      { name: "audience_ids", type: "string[]", required: false, description: "Array of player IDs who can see this component (only used when audience_type=false). Example: ['1', '2']" },
    ],
    handler: ({ name, label, action, enabled, position, size, variant, audience_type, audience_ids }: {
      name: string;
      label: string;
      action: string;
      enabled: boolean;
      position: string;
      size?: string;
      variant?: string;
      audience_type?: boolean;
      audience_ids?: string[];
    }) => {
      const normalized = (name ?? "").trim();
      
      // Name-based idempotency
      if (normalized) {
        const existing = (viewState.items ?? initialState.items).find((it) => 
          it.type === "action_button" && (it.name ?? "").trim() === normalized
        );
        if (existing) {
          return existing.id;
        }
      }
      
      const data: ActionButtonData = {
        label,
        action,
        enabled,
        position: normalizePosition(position || 'center'),
        size: size as ComponentSize,
        variant: variant as "primary" | "secondary" | "danger",
        audience_type: audience_type ?? true,
        audience_ids: audience_ids ?? []
      };
      return addItem("action_button", name, data);
    },
  });

  useCopilotAction({
    name: "createPhaseIndicator",
    description: "Create a phase indicator to show current game phase.",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "name", type: "string", required: true, description: "Item name" },
      { name: "currentPhase", type: "string", required: true, description: "Current game phase" },
      { name: "position", type: "string", required: true, description: "Grid position (select: 'top-left' | 'top-center' | 'top-right' | 'middle-left' | 'center' | 'middle-right' | 'bottom-left' | 'bottom-center' | 'bottom-right')" },
      { name: "description", type: "string", required: false, description: "Optional phase description" },
      { name: "timeRemaining", type: "number", required: false, description: "Seconds remaining in phase" },
      { name: "audience_type", type: "boolean", required: false, description: "Whether all players can see this (true) or only specific players (false). Default: true" },
      { name: "audience_ids", type: "string[]", required: false, description: "Array of player IDs who can see this component (only used when audience_type=false). Example: ['1', '2']" },
    ],
    handler: ({ name, currentPhase, position, description, timeRemaining, audience_type, audience_ids }: {
      name: string;
      currentPhase: string;
      position: string;
      description?: string;
      timeRemaining?: number;
      audience_type?: boolean;
      audience_ids?: string[];
    }) => {
      const normalized = (name ?? "").trim();
      
      // Name-based idempotency
      if (normalized) {
        const existing = (viewState.items ?? initialState.items).find((it) => 
          it.type === "phase_indicator" && (it.name ?? "").trim() === normalized
        );
        if (existing) {
          return existing.id;
        }
      }
      
      const data: PhaseIndicatorData = {
        currentPhase,
        position: normalizePosition(position || 'center'),
        description,
        timeRemaining,
        audience_type: audience_type ?? true,
        audience_ids: audience_ids ?? []
      };
      return addItem("phase_indicator", name, data);
      
    },
  });

  useCopilotAction({
    name: "createTextDisplay",
    description: "Create a text display for game information.",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "name", type: "string", required: true, description: "Item name" },
      { name: "content", type: "string", required: true, description: "Main text content" },
      { name: "position", type: "string", required: true, description: "Grid position (select: 'center' | 'top-left' | 'top-center' | 'top-right' | 'middle-left'  | 'middle-right' | 'bottom-left' | 'bottom-center' | 'bottom-right')" },
      { name: "title", type: "string", required: false, description: "Optional title text" },
      { name: "type", type: "string", required: false, description: "Display type (select: 'info' | 'warning' | 'error' | 'success')" },
      { name: "audience_type", type: "boolean", required: false, description: "Whether all players can see this (true) or only specific players (false). Default: true" },
      { name: "audience_ids", type: "string[]", required: false, description: "Array of player IDs who can see this component (only used when audience_type=false). Example: ['1', '2']" },
    ],
    handler: ({ name, content, position, title, type, audience_type, audience_ids }: {
      name: string;
      content: string;
      position: string;
      title?: string;
      type?: string;
      audience_type?: boolean;
      audience_ids?: string[];
    }) => {
      const normalized = (name ?? "").trim();
      
      // Name-based idempotency
      if (normalized) {
        const existing = (viewState.items ?? initialState.items).find((it) => 
          it.type === "text_display" && (it.name ?? "").trim() === normalized
        );
        if (existing) {
          return existing.id;
        }
      }
      
      const data: TextDisplayData = {
        content,
        position: normalizePosition(position || 'center'),
        title,
        type: type as "info" | "warning" | "error" | "success",
        audience_type: audience_type ?? true,
        audience_ids: audience_ids ?? []
      };
      return addItem("text_display", name, data);
    },
  });

  useCopilotAction({
    name: "createVotingPanel",
    description: "Create a voting panel for player voting.",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "name", type: "string", required: true, description: "Item name" },
      { name: "votingId", type: "string", required: true, description: "Unique voting ID" },
      { name: "options", type: "string[]", required: true, description: "List of voting options" },
      { name: "position", type: "string", required: true, description: "Grid position (select: 'top-left' | 'top-center' | 'top-right' | 'middle-left' | 'center' | 'middle-right' | 'bottom-left' | 'bottom-center' | 'bottom-right')" },
      { name: "title", type: "string", required: false, description: "Optional voting title/question" },
      { name: "audience_type", type: "boolean", required: false, description: "Whether all players can see this (true) or only specific players (false). Default: true" },
      { name: "audience_ids", type: "string[]", required: false, description: "Array of player IDs who can see this component (only used when audience_type=false). Example: ['1', '2']" },
    ],
    handler: ({ name, votingId, options, position, title, audience_type, audience_ids }: {
      name: string;
      votingId: string;
      options: string[];
      position: string;
      title?: string;
      audience_type?: boolean;
      audience_ids?: string[];
    }) => {
      const normalized = (name ?? "").trim();
      
      // Check for duplicate votingId in existing vote records
      const existingVotes = (viewState.vote ?? []);
      const duplicateVoting = existingVotes.some(vote => vote.voteid === votingId);
      if (duplicateVoting) {
        throw new Error(`Voting ID ${votingId} already exists in vote records`);
      }
      
      // Name-based idempotency
      if (normalized) {
        const existing = (viewState.items ?? initialState.items).find((it) => 
          it.type === "voting_panel" && (it.name ?? "").trim() === normalized
        );
        if (existing) {
          return existing.id;
        }
      }
      
      const data: VotingPanelData = {
        votingId,
        options,
        position: normalizePosition(position || 'center'),
        title,
        audience_type: audience_type ?? true,
        audience_ids: audience_ids ?? []
      };
      return addItem("voting_panel", name, data);
    },
  });

  // Submit a vote without clicking the panel (chat-driven voting)
  useCopilotAction({
    name: "submitVote",
    description: "Submit a vote programmatically for the current player (no click needed)",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "votingId", type: "string", required: true, description: "Target voting ID" },
      { name: "option", type: "string", required: true, description: "Option to vote for (target name or id)" },
    ],
    handler: async ({ votingId, option }: { votingId: string; option: string }) => {
      const playerId = getCurrentPlayerId();
      if (!playerId) return "no_player";
      await handleVote(votingId, playerId, option);
      return `voted:${option}`;
    },
  });

  useCopilotAction({
    name: "createAvatarSet",
    description: "Create avatar set to display all players.",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "name", type: "string", required: true, description: "Item name" },
      { name: "avatarType", type: "string", required: true, description: "Avatar type (select: 'human' | 'wolf' | 'dog' | 'cat')" },
      { name: "audience_type", type: "boolean", required: false, description: "Whether all players can see this (true) or only specific players (false). Default: true" },
      { name: "audience_ids", type: "string[]", required: false, description: "Array of player IDs who can see this component (only used when audience_type=false). Example: ['1', '2']" },
    ],
    handler: ({ name, avatarType, audience_type, audience_ids }: {
      name: string;
      avatarType: string;
      audience_type?: boolean;
      audience_ids?: string[];
    }) => {
      const normalized = (name ?? "").trim();
      
      // Name-based idempotency
      if (normalized) {
        const existing = (viewState.items ?? initialState.items).find((it) => 
          it.type === "avatar_set" && (it.name ?? "").trim() === normalized
        );
        if (existing) {
          return existing.id;
        }
      }
      
      const data: AvatarSetData = {
        avatarType: avatarType || "human",
        audience_type: audience_type ?? true,
        audience_ids: audience_ids ?? []
      };
      return addItem("avatar_set", name, data);
    },
  });

  useCopilotAction({
    name: "markPlayerDead",
    description: "Mark a player as dead, making their avatar appear grayed out.",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "playerId", type: "string", required: true, description: "Player ID to mark as dead" },
      { name: "playerName", type: "string", required: true, description: "Player name for confirmation" },
    ],
    handler: ({ playerId, playerName }: {
      playerId: string;
      playerName: string;
    }) => {
      setState((prev) => {
        const base = prev ?? initialState;
        const currentDeadPlayers = base.deadPlayers ?? [];
        
        // Check if player is already dead
        if (currentDeadPlayers.includes(playerId)) {
          return base; // No change needed
        }
        
        // Add player to dead list
        return {
          ...base,
          deadPlayers: [...currentDeadPlayers, playerId],
          lastAction: `marked_dead:${playerId}:${playerName}`
        } as AgentState;
      });
      
      return `Player ${playerName} (ID: ${playerId}) has been marked as dead`;
    },
  });

  // Frontend action: create a timer component
  useCopilotAction({
    name: "createTimer",
    description: "Create a timer component fixed to top-left corner that counts down and automatically sends a message to Agent when time expires.",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "name", type: "string", required: true, description: "Timer name" },
      { name: "duration", type: "number", required: true, description: "Timer duration in seconds" },
      { name: "label", type: "string", required: false, description: "Optional label to display above timer" },
    ],
    handler: ({ name, duration, label }: {
      name: string;
      duration: number;
      label?: string;
    }) => {
      const normalized = (name ?? "").trim();
      
      // Name-based idempotency
      if (normalized) {
        const existing = (viewState.items ?? initialState.items).find((it) => 
          it.type === "timer" && (it.name ?? "").trim() === normalized
        );
        if (existing) {
          return existing.id;
        }
      }
      
      const data: TimerData = {
        duration: duration || 60,
        label: label || "",
        position: "top-left" as GamePosition,
        audience_type: true, // Force all timers to be public
        audience_ids: [] // Clear audience restrictions
      };
      
      const timerId = addItem("timer", name, data);
      
      // Start countdown and send message when expires
      setTimeout(async () => {
        await appendMessage(new TextMessage({
          role: MessageRole.User,
          content: `Timer "${name}" has expired after ${duration} seconds.`
        }));
        
        // Remove timer after expiring
        setState((prev) => {
          const base = prev ?? initialState;
          const items = base.items ?? [];
          const filteredItems = items.filter(item => item.id !== timerId);
          return { ...base, items: filteredItems } as AgentState;
        });
      }, duration * 1000);
      
      return timerId;
    },
  });

  useCopilotAction({
    name: "createBackgroundControl",
    description: "Create background color control panel.",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "name", type: "string", required: true, description: "Item name" },
      { name: "backgroundColor", type: "string", required: false, description: "Initial background color (select: 'white' | 'gray-900' | 'blue-50' | 'green-50' | 'purple-50')" },
      { name: "audience_type", type: "boolean", required: false },
      { name: "audience_ids", type: "string[]", required: false },
    ],
    handler: ({ name, backgroundColor, audience_type, audience_ids }: {
      name: string;
      backgroundColor?: string;
      audience_type?: boolean;
      audience_ids?: string[];
    }) => {
      const normalized = (name ?? "").trim();
      
      // Name-based idempotency
      if (normalized) {
        const existing = (viewState.items ?? initialState.items).find((it) => 
          it.type === "background_control" && (it.name ?? "").trim() === normalized
        );
        if (existing) {
          return existing.id;
        }
      }
      
      const data: BackgroundControlData = {
        backgroundColor: backgroundColor || "white",
        position: "center" as GamePosition,
        audience_type: audience_type ?? true,
        audience_ids: audience_ids ?? []
      };
      return addItem("background_control", name, data);
    },
  });

  // Backward-compatible color change tool (no-op create if exists)
  useCopilotAction({
    name: "changeBackgroundColor",
    description: "Change background color (creates control if missing)",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "name", type: "string", required: true },
      { name: "backgroundColor", type: "string", required: true },
    ],
    handler: ({ name, backgroundColor }: { name: string; backgroundColor: string }) => {
      // ensure exists
      let id = "";
      const normalized = (name ?? "").trim();
      const existing = (viewState.items ?? initialState.items).find((it) => it.type === "background_control" && (it.name ?? "").trim() === normalized);
      if (existing) {
        id = existing.id;
      } else {
        id = addItem("background_control", name, { backgroundColor: backgroundColor || "white", position: "center" as GamePosition, audience_type: true, audience_ids: [] } as BackgroundControlData);
      }
      updateItemData(id, (prev) => ({ ...(prev as BackgroundControlData), backgroundColor }));
      return id;
    },
  });

  useCopilotAction({
    name: "createResultDisplay",
    description: "Create a result display showing content as large artistic text with gradient colors.",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "name", type: "string", required: true, description: "Item name" },
      { name: "content", type: "string", required: true, description: "Result content to display" },
      { name: "position", type: "string", required: true, description: "Grid position (select: 'top-left' | 'top-center' | 'top-right' | 'middle-left' | 'center' | 'middle-right' | 'bottom-left' | 'bottom-center' | 'bottom-right')" },
    ],
    handler: ({ name, content, position }: {
      name: string;
      content: string;
      position: string;
    }) => {
      const normalized = (name ?? "").trim();
      
      // Name-based idempotency
      if (normalized) {
        const existing = (viewState.items ?? initialState.items).find((it) => 
          it.type === "result_display" && (it.name ?? "").trim() === normalized
        );
        if (existing) {
          return existing.id;
        }
      }
      
      const data: ResultDisplayData = {
        content: content || "RESULT",
        position: normalizePosition(position || 'center'),
        audience_type: true,
        audience_ids: []
      };
      return addItem("result_display", name, data);
    },
  });

  // Create a Hands Card (for card games)
  useCopilotAction({
    name: "createHandsCard",
    description: "Create a hand card item for card games (minimal, modern style)",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "name", type: "string", required: true, description: "Item name" },
      { name: "cardType", type: "string", required: true, description: "Card classification (e.g., attack, defense, spell)" },
      { name: "cardName", type: "string", required: true, description: "Display name of the card" },
      { name: "descriptions", type: "string", required: false, description: "Short description or effect" },
      { name: "color", type: "string", required: false, description: "Accent color (hex or token). Default neutral" },
      { name: "position", type: "string", required: false, description: "Grid position (default: bottom-center)" },
      { name: "audience_type", type: "boolean", required: false, description: "Whether all players can see this (true) or only specific players (false). Default: true" },
      { name: "audience_ids", type: "string[]", required: false, description: "Player IDs who can see this component when audience_type=false" },
    ],
    handler: ({ name, cardType, cardName, descriptions, color, position, audience_type, audience_ids }: {
      name: string;
      cardType: string;
      cardName: string;
      descriptions?: string;
      color?: string;
      position?: string;
      audience_type?: boolean;
      audience_ids?: string[];
    }) => {
      const normalized = (name ?? "").trim();
      // Name-based idempotency
      if (normalized) {
        const existing = (viewState.items ?? initialState.items).find((it) => 
          it.type === "hands_card" && (it.name ?? "").trim() === normalized
        );
        if (existing) return existing.id;
      }
      const data = {
        cardType,
        cardName,
        descriptions,
        color: color || "#2563eb",
        position: normalizePosition(position) || "bottom-center",
        audience_type: audience_type ?? true,
        audience_ids: audience_ids ?? [],
      } as HandsCardData;
      return addItem("hands_card", name, data);
    },
  });

  // Create a Score Board
  useCopilotAction({
    name: "createScoreBoard",
    description: "Create a scoreboard component with optional entries and styling",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "name", type: "string", required: true, description: "Item name" },
      { name: "title", type: "string", required: false, description: "Scoreboard title" },
      { name: "entries", type: "object[]", required: false, description: "Array of entries: [{id, name, score}]" },
      { name: "sort", type: "string", required: false, description: "Sort order (asc|desc)" },
      { name: "accentColor", type: "string", required: false, description: "Accent color" },
      { name: "position", type: "string", required: false, description: "Grid position (default: top-right)" },
      { name: "audience_type", type: "boolean", required: false, description: "true=public; false=private" },
      { name: "audience_ids", type: "string[]", required: false, description: "Visible player IDs if private" },
    ],
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    handler: ({ name, title, entries, sort, accentColor, position, audience_type, audience_ids }: any) => {
      const normalized = (name ?? "").trim();
      if (normalized) {
        const existing = (viewState.items ?? initialState.items).find((it) => it.type === "score_board" && (it.name ?? "").trim() === normalized);
        if (existing) return existing.id;
      }
      const data = {
        title,
        entries: Array.isArray(entries) ? entries : [],
        sort: (sort as "asc" | "desc") || "desc",
        accentColor: accentColor || "#2563eb",
        position: (position as GamePosition) || "top-right",
        audience_type: audience_type ?? true,
        audience_ids: audience_ids ?? [],
      } as ScoreBoardData;
      return addItem("score_board", name, data);
    },
  });

  // Update scoreboard metadata
  useCopilotAction({
    name: "updateScoreBoard",
    description: "Update scoreboard title/sort/accentColor/position",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "itemId", type: "string", required: true, description: "Target score_board item id" },
      { name: "title", type: "string", required: false },
      { name: "sort", type: "string", required: false },
      { name: "accentColor", type: "string", required: false },
      { name: "position", type: "string", required: false },
    ],
    handler: ({ itemId, title, sort, accentColor, position /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
}: any) => {
      updateItemData(String(itemId), (prev) => {
        const d = { ...(prev as ScoreBoardData) };
        if (typeof title === 'string') d.title = title;
        if (sort === 'asc' || sort === 'desc') d.sort = sort;
        if (typeof accentColor === 'string') d.accentColor = accentColor;
        if (typeof position === 'string') (d as any).position = position as GamePosition;
        return d;
      });
      return itemId;
    },
  });

  // Replace scoreboard entries
  useCopilotAction({
    name: "setScoreBoardEntries",
    description: "Replace all scoreboard entries",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "itemId", type: "string", required: true },
      { name: "entries", type: "object[]", required: true, description: "Array of {id, name, score}" },
    ],
    handler: ({ itemId, entries /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
}: any) => {
      const list = Array.isArray(entries) ? entries : [];
      updateItemData(itemId, (prev) => ({ ...(prev as ScoreBoardData), entries: list }));
      return itemId;
    },
  });

  // Upsert a single entry
  useCopilotAction({
    name: "upsertScoreEntry",
    description: "Add or update a scoreboard entry",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "itemId", type: "string", required: true },
      { name: "entryId", type: "string", required: true },
      { name: "name", type: "string", required: false },
      { name: "score", type: "number", required: false },
    ],
    handler: ({ itemId, entryId, name, score /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
}: any) => {
      updateItemData(String(itemId), (prev) => {
        const d = { ...(prev as ScoreBoardData) };
        const entries = Array.isArray(d.entries) ? [...d.entries] : [];
        const idx = entries.findIndex((e) => String(e.id) === String(entryId));
        if (idx >= 0) {
          entries[idx] = { ...entries[idx], name: name ?? entries[idx].name, score: typeof score === 'number' ? score : entries[idx].score };
        } else {
          entries.push({ id: String(entryId), name: name ?? String(entryId), score: typeof score === 'number' ? score : 0 });
        }
        d.entries = entries;
        return d;
      });
      return itemId;
    },
  });

  // Remove one score entry
  useCopilotAction({
    name: "removeScoreEntry",
    description: "Remove a single scoreboard entry by id",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "itemId", type: "string", required: true },
      { name: "entryId", type: "string", required: true },
    ],
    handler: ({ itemId, entryId /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
}: any) => {
      updateItemData(String(itemId), (prev) => {
        const d = { ...(prev as ScoreBoardData) };
        d.entries = (Array.isArray(d.entries) ? d.entries : []).filter((e) => String(e.id) !== String(entryId));
        return d;
      });
      return itemId;
    },
  });

  // Statement Board tools
  useCopilotAction({
    name: "createStatementBoard",
    description: "Create a statement board (up to 3 statements) with optional highlight",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "name", type: "string", required: true },
      { name: "statements", type: "string[]", required: false },
      { name: "highlightIndex", type: "number", required: false },
      { name: "locked", type: "boolean", required: false },
      { name: "accentColor", type: "string", required: false },
      { name: "position", type: "string", required: false },
      { name: "audience_type", type: "boolean", required: false },
      { name: "audience_ids", type: "string[]", required: false },
    ],
    handler: ({ name, statements, highlightIndex, locked, accentColor, position, audience_type, audience_ids /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
}: any) => {
      const normalized = (name ?? "").trim();
      if (normalized) {
        const existing = (viewState.items ?? initialState.items).find((it) => it.type === "statement_board" && (it.name ?? "").trim() === normalized);
        if (existing) return existing.id;
      }
      const data: StatementBoardData = {
        statements: Array.isArray(statements) ? statements.slice(0, 3) : ["", "", ""],
        highlightIndex: typeof highlightIndex === 'number' ? highlightIndex : -1,
        locked: typeof locked === 'boolean' ? locked : false,
        accentColor: accentColor || "#2563eb",
        position: normalizePosition(position) || "center",
        audience_type: audience_type ?? true,
        audience_ids: audience_ids ?? [],
      };
      return addItem("statement_board", name, data);
    },
  });

  useCopilotAction({
    name: "updateStatementBoard",
    description: "Update statement board fields",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "itemId", type: "string", required: true },
      { name: "statements", type: "string[]", required: false },
      { name: "highlightIndex", type: "number", required: false },
      { name: "locked", type: "boolean", required: false },
      { name: "accentColor", type: "string", required: false },
    ],
    handler: ({ itemId, statements, highlightIndex, locked, accentColor /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
}: any) => {
      updateItemData(String(itemId), (prev) => {
        const d = { ...(prev as StatementBoardData) };
        if (Array.isArray(statements)) d.statements = statements.slice(0, 3);
        if (typeof highlightIndex === 'number') d.highlightIndex = highlightIndex;
        if (typeof locked === 'boolean') d.locked = locked;
        if (typeof accentColor === 'string') d.accentColor = accentColor;
        return d;
      });
      return itemId;
    },
  });

  // Reaction Timer tools
  useCopilotAction({
    name: "createReactionTimer",
    description: "Create a reaction timer bar with duration and label",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "name", type: "string", required: true },
      { name: "duration", type: "number", required: false },
      { name: "label", type: "string", required: false },
      { name: "accentColor", type: "string", required: false },
      { name: "position", type: "string", required: false },
    ],
    handler: ({ name, duration, label, accentColor, position /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
}: any) => {
      const normalized = (name ?? "").trim();
      if (normalized) {
        const existing = (viewState.items ?? initialState.items).find((it) => it.type === "reaction_timer" && (it.name ?? "").trim() === normalized);
        if (existing) return existing.id;
      }
      const data: ReactionTimerData = {
        duration: typeof duration === 'number' ? Math.max(1, duration) : 10,
        startedAt: Date.now(), // Auto-start the timer
        running: true,         // Auto-start the timer
        label: label || "Reaction Window",
        accentColor: accentColor || "#22c55e",
        position: normalizePosition(position) || "top-center",
        audience_type: true, // Force all reaction timers to be public
        audience_ids: [], // Clear audience restrictions
      };
      return addItem("reaction_timer", name, data);
    },
  });

  useCopilotAction({
    name: "startReactionTimer",
    description: "Start a reaction timer (sets startedAt and running)",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "itemId", type: "string", required: true },
      { name: "duration", type: "number", required: false },
    ],
    handler: ({ itemId, duration /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
}: any) => {
      updateItemData(String(itemId), (prev) => {
        const d = { ...(prev as ReactionTimerData) };
        if (typeof duration === 'number') d.duration = Math.max(1, duration);
        d.startedAt = Date.now();
        d.running = true;
        return d;
      });
      return itemId;
    },
  });

  useCopilotAction({
    name: "stopReactionTimer",
    description: "Stop/pause a reaction timer",
    available: "remote",
    followUp: false,
    parameters: [ { name: "itemId", type: "string", required: true } ],
    handler: ({ itemId /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
}: any) => {
      updateItemData(itemId, (prev) => ({ ...(prev as ReactionTimerData), running: false }));
      return itemId;
    },
  });

  useCopilotAction({
    name: "resetReactionTimer",
    description: "Reset reaction timer to idle (clears startedAt, running=false)",
    available: "remote",
    followUp: false,
    parameters: [ { name: "itemId", type: "string", required: true } ],
    handler: ({ itemId /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
}: any) => {
      updateItemData(itemId, (prev) => ({ ...(prev as ReactionTimerData), startedAt: undefined, running: false }));
      return itemId;
    },
  });

  // Night Overlay tools
  useCopilotAction({
    name: "createNightOverlay",
    description: "Create a global night overlay (toggle visibility, optional text)",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "name", type: "string", required: true },
      { name: "visible", type: "boolean", required: false },
      { name: "title", type: "string", required: false },
      { name: "subtitle", type: "string", required: false },
      { name: "opacity", type: "number", required: false },
      { name: "blur", type: "boolean", required: false },
      { name: "audience_type", type: "boolean", required: false },
      { name: "audience_ids", type: "string[]", required: false },
    ],
    handler: ({ name, visible, title, subtitle, opacity, blur, audience_type, audience_ids /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
}: any) => {
      const normalized = (name ?? "").trim();
      if (normalized) {
        const existing = (viewState.items ?? initialState.items).find((it) => it.type === "night_overlay" && (it.name ?? "").trim() === normalized);
        if (existing) return existing.id;
      }
      const data: NightOverlayData = {
        visible: typeof visible === 'boolean' ? visible : true,
        title,
        subtitle,
        opacity: typeof opacity === 'number' ? Math.max(0, Math.min(1, opacity)) : 0.5,
        blur: typeof blur === 'boolean' ? blur : true,
        position: "center" as GamePosition,
        audience_type: audience_type ?? true,
        audience_ids: audience_ids ?? [],
      };
      return addItem("night_overlay", name, data as any);
    },
  });

  useCopilotAction({
    name: "setNightOverlay",
    description: "Toggle night overlay and optionally update text/opacity",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "itemId", type: "string", required: true },
      { name: "visible", type: "boolean", required: true },
      { name: "title", type: "string", required: false },
      { name: "subtitle", type: "string", required: false },
      { name: "opacity", type: "number", required: false },
      { name: "blur", type: "boolean", required: false },
    ],
    handler: ({ itemId, visible, title, subtitle, opacity, blur /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
}: any) => {
      updateItemData(String(itemId), (prev) => {
        const d = { ...(prev as NightOverlayData) };
        d.visible = !!visible;
        if (typeof title === 'string') d.title = title;
        if (typeof subtitle === 'string') d.subtitle = subtitle;
        if (typeof opacity === 'number') d.opacity = Math.max(0, Math.min(1, opacity));
        if (typeof blur === 'boolean') d.blur = blur;
        return d;
      });
      return itemId;
    },
  });

  // Turn Indicator tools
  useCopilotAction({
    name: "createTurnIndicator",
    description: "Create a pill turn indicator for the active player",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "name", type: "string", required: true },
      { name: "currentPlayerId", type: "string", required: true },
      { name: "playerName", type: "string", required: false },
      { name: "label", type: "string", required: false },
      { name: "accentColor", type: "string", required: false },
      { name: "position", type: "string", required: false },
      { name: "audience_type", type: "boolean", required: false },
      { name: "audience_ids", type: "string[]", required: false },
    ],
    handler: ({ name, currentPlayerId, playerName, label, accentColor, position, audience_type, audience_ids /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
}: any) => {
      const normalized = (name ?? "").trim();
      if (normalized) {
        const existing = (viewState.items ?? initialState.items).find((it) => it.type === "turn_indicator" && (it.name ?? "").trim() === normalized);
        if (existing) return existing.id;
      }
      const data: TurnIndicatorData = {
        currentPlayerId: String(currentPlayerId),
        playerName,
        label: label || "Speaker",
        accentColor: accentColor || "#2563eb",
        position: normalizePosition(position) || "top-center",
        audience_type: audience_type ?? true,
        audience_ids: audience_ids ?? [],
      };
      return addItem("turn_indicator", name, data);
    },
  });

  useCopilotAction({
    name: "updateTurnIndicator",
    description: "Update turn indicator fields",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "itemId", type: "string", required: true },
      { name: "currentPlayerId", type: "string", required: false },
      { name: "playerName", type: "string", required: false },
      { name: "label", type: "string", required: false },
      { name: "accentColor", type: "string", required: false },
      { name: "position", type: "string", required: false },
    ],
    handler: ({ itemId, currentPlayerId, playerName, label, accentColor, position /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
}: any) => {
      updateItemData(String(itemId), (prev) => {
        const d = { ...(prev as TurnIndicatorData) };
        if (typeof currentPlayerId === 'string') d.currentPlayerId = currentPlayerId;
        if (typeof playerName === 'string') d.playerName = playerName;
        if (typeof label === 'string') d.label = label;
        if (typeof accentColor === 'string') d.accentColor = accentColor;
        if (typeof position === 'string') d.position = position as GamePosition;
        return d;
      });
      return itemId;
    },
  });

  // Health Display (Bang!)
  useCopilotAction({
    name: "createHealthDisplay",
    description: "Create a health/bullets display",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "name", type: "string", required: true },
      { name: "value", type: "number", required: false },
      { name: "max", type: "number", required: false },
      { name: "style", type: "string", required: false },
      { name: "accentColor", type: "string", required: false },
      { name: "position", type: "string", required: false },
      { name: "audience_type", type: "boolean", required: false },
      { name: "audience_ids", type: "string[]", required: false },
    ],
    handler: ({ name, value, max, style, accentColor, position, audience_type, audience_ids /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
}: any) => {
      const normalized = (name ?? "").trim();
      if (normalized) {
        const existing = (viewState.items ?? initialState.items).find((it) => it.type === "health_display" && (it.name ?? "").trim() === normalized);
        if (existing) return existing.id;
      }
      const data: HealthDisplayData = {
        value: typeof value === 'number' ? Math.max(0, value) : 3,
        max: typeof max === 'number' ? Math.max(0, max) : 5,
        style: (style as any) || 'hearts',
        accentColor: accentColor || '#ef4444',
        position: (position as GamePosition) || 'top-right',
        audience_type: audience_type ?? true,
        audience_ids: audience_ids ?? [],
      };
      return addItem('health_display', name, data);
    },
  });

  useCopilotAction({
    name: "updateHealthDisplay",
    description: "Update health display (value/max/style/accentColor/position)",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "itemId", type: "string", required: true },
      { name: "value", type: "number", required: false },
      { name: "max", type: "number", required: false },
      { name: "style", type: "string", required: false },
      { name: "accentColor", type: "string", required: false },
      { name: "position", type: "string", required: false },
    ],
    handler: ({ itemId, value, max, style, accentColor, position /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
}: any) => {
      updateItemData(String(itemId), (prev) => {
        const d = { ...(prev as HealthDisplayData) };
        if (typeof value === 'number') d.value = Math.max(0, value);
        if (typeof max === 'number') d.max = Math.max(0, max);
        if (typeof style === 'string') d.style = style as any;
        if (typeof accentColor === 'string') d.accentColor = accentColor;
        if (typeof position === 'string') d.position = position as GamePosition;
        return d;
      });
      return itemId;
    },
  });

  // Influence Set (Coup)
  useCopilotAction({
    name: "createInfluenceSet",
    description: "Create a 2-card influence set for Coup",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "name", type: "string", required: true },
      { name: "ownerId", type: "string", required: true },
      { name: "cards", type: "object[]", required: false, description: "[{name, revealed}] length up to 2" },
      { name: "accentColor", type: "string", required: false },
      { name: "position", type: "string", required: false },
      { name: "audience_type", type: "boolean", required: false },
      { name: "audience_ids", type: "string[]", required: false },
    ],
    handler: ({ name, ownerId, cards, accentColor, position, audience_type, audience_ids /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
}: any) => {
      const normalized = (name ?? "").trim();
      if (normalized) {
        const existing = (viewState.items ?? initialState.items).find((it) => it.type === "influence_set" && (it.name ?? "").trim() === normalized);
        if (existing) return existing.id;
      }
      const data: InfluenceSetData = {
        ownerId: String(ownerId),
        cards: Array.isArray(cards) ? cards.slice(0, 2) : [ { name: "", revealed: false }, { name: "", revealed: false } ],
        accentColor: accentColor || '#a78bfa',
        position: (position as GamePosition) || 'bottom-center',
        audience_type: audience_type ?? true,
        audience_ids: audience_ids ?? [],
      };
      return addItem('influence_set', name, data);
    },
  });

  useCopilotAction({
    name: "updateInfluenceSet",
    description: "Update influence set fields (ownerId/cards/accentColor/position)",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "itemId", type: "string", required: true },
      { name: "ownerId", type: "string", required: false },
      { name: "cards", type: "object[]", required: false },
      { name: "accentColor", type: "string", required: false },
      { name: "position", type: "string", required: false },
    ],
    handler: ({ itemId, ownerId, cards, accentColor, position /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
}: any) => {
      updateItemData(String(itemId), (prev) => {
        const d = { ...(prev as InfluenceSetData) };
        if (typeof ownerId === 'string') d.ownerId = ownerId;
        if (Array.isArray(cards)) d.cards = cards.slice(0, 2);
        if (typeof accentColor === 'string') d.accentColor = accentColor;
        if (typeof position === 'string') d.position = position as GamePosition;
        return d;
      });
      return itemId;
    },
  });

  useCopilotAction({
    name: "revealInfluenceCard",
    description: "Reveal one influence card by index (0 or 1)",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "itemId", type: "string", required: true },
      { name: "index", type: "number", required: true },
      { name: "revealed", type: "boolean", required: false },
    ],
    handler: ({ itemId, index, revealed /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
}: any) => {
      updateItemData(String(itemId), (prev) => {
        const d = { ...(prev as InfluenceSetData) };
        const i = Math.max(0, Math.min(1, Number(index)));
        const cards = Array.isArray(d.cards) ? [...d.cards] : [];
        if (!cards[i]) cards[i] = { name: "", revealed: false } as any;
        cards[i] = { ...cards[i], revealed: typeof revealed === 'boolean' ? revealed : true };
        d.cards = cards.slice(0, 2);
        return d;
      });
      return itemId;
    },
  });

  // Update an existing Hands Card's core fields
  useCopilotAction({
    name: "updateHandsCard",
    description: "Update a hands card's basic fields (type, name, descriptions, color).",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "itemId", type: "string", required: true, description: "Target hands_card item id" },
      { name: "cardType", type: "string", required: false, description: "Card classification" },
      { name: "cardName", type: "string", required: false, description: "Display name of the card" },
      { name: "descriptions", type: "string", required: false, description: "Short description or effect" },
      { name: "color", type: "string", required: false, description: "Accent color (hex or token)" },
    ],
    handler: ({ itemId, cardType, cardName, descriptions, color }: {
      itemId: string;
      cardType?: string;
      cardName?: string;
      descriptions?: string;
      color?: string;
    }) => {
      updateItemData(String(itemId), (prev) => {
        const d = { ...(prev as HandsCardData) };
        if (typeof cardType === 'string') d.cardType = cardType;
        if (typeof cardName === 'string') d.cardName = cardName;
        if (typeof descriptions === 'string') d.descriptions = descriptions;
        if (typeof color === 'string') d.color = color;
        return d;
      });
      return itemId;
    },
  });

  // Common update tools: PhaseIndicator, TextDisplay, ActionButton, CharacterCard, VotingPanel, ResultDisplay, Timer, Item position
  useCopilotAction({
    name: "updatePhaseIndicator",
    description: "Update phase indicator fields",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "itemId", type: "string", required: true },
      { name: "currentPhase", type: "string", required: false },
      { name: "description", type: "string", required: false },
      { name: "timeRemaining", type: "number", required: false },
    ],
    handler: ({ itemId, currentPhase, description, timeRemaining /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
}: any) => {
      updateItemData(String(itemId), (prev) => {
        const d = { ...(prev as PhaseIndicatorData) };
        if (typeof currentPhase === 'string') d.currentPhase = currentPhase;
        if (typeof description === 'string') d.description = description;
        if (typeof timeRemaining === 'number') d.timeRemaining = timeRemaining;
        return d;
      });
      return itemId;
    },
  });

  useCopilotAction({
    name: "updateTextDisplay",
    description: "Update text display title/content/type",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "itemId", type: "string", required: true },
      { name: "title", type: "string", required: false },
      { name: "content", type: "string", required: false },
      { name: "type", type: "string", required: false },
    ],
    handler: ({ itemId, title, content, type /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
}: any) => {
      updateItemData(String(itemId), (prev) => {
        const d = { ...(prev as TextDisplayData) };
        if (typeof title === 'string') d.title = title;
        if (typeof content === 'string') d.content = content;
        if (typeof type === 'string') d.type = type as any;
        return d;
      });
      return itemId;
    },
  });

  useCopilotAction({
    name: "updateActionButton",
    description: "Update action button fields",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "itemId", type: "string", required: true },
      { name: "label", type: "string", required: false },
      { name: "action", type: "string", required: false },
      { name: "enabled", type: "boolean", required: false },
      { name: "variant", type: "string", required: false },
    ],
    handler: ({ itemId, label, action, enabled, variant /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
}: any) => {
      updateItemData(String(itemId), (prev) => {
        const d = { ...(prev as ActionButtonData) };
        if (typeof label === 'string') d.label = label;
        if (typeof action === 'string') d.action = action;
        if (typeof enabled === 'boolean') d.enabled = enabled;
        if (typeof variant === 'string') d.variant = variant as any;
        return d;
      });
      return itemId;
    },
  });

  useCopilotAction({
    name: "updateCharacterCard",
    description: "Update character card fields",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "itemId", type: "string", required: true },
      { name: "role", type: "string", required: false },
      { name: "description", type: "string", required: false },
      { name: "size", type: "string", required: false },
    ],
    handler: ({ itemId, role, description, size /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
}: any) => {
      updateItemData(String(itemId), (prev) => {
        const d = { ...(prev as CharacterCardData) };
        if (typeof role === 'string') d.role = role;
        if (typeof description === 'string') d.description = description;
        if (typeof size === 'string') (d as any).size = size as any;
        return d;
      });
      return itemId;
    },
  });

  useCopilotAction({
    name: "updateVotingPanel",
    description: "Update voting panel title/options",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "itemId", type: "string", required: true },
      { name: "title", type: "string", required: false },
      { name: "options", type: "string[]", required: false },
    ],
    handler: ({ itemId, title, options /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
}: any) => {
      updateItemData(String(itemId), (prev) => {
        const d = { ...(prev as VotingPanelData) };
        if (typeof title === 'string') d.title = title;
        if (Array.isArray(options)) d.options = options;
        return d;
      });
      return itemId;
    },
  });

  useCopilotAction({
    name: "updateResultDisplay",
    description: "Update result display content",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "itemId", type: "string", required: true },
      { name: "content", type: "string", required: false },
    ],
    handler: ({ itemId, content /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
}: any) => {
      updateItemData(String(itemId), (prev) => {
        const d = { ...(prev as ResultDisplayData) };
        if (typeof content === 'string') d.content = content;
        return d;
      });
      return itemId;
    },
  });

  useCopilotAction({
    name: "updateTimer",
    description: "Update timer duration/label",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "itemId", type: "string", required: true },
      { name: "duration", type: "number", required: false },
      { name: "label", type: "string", required: false },
    ],
    handler: ({ itemId, duration, label /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
}: any) => {
      updateItemData(String(itemId), (prev) => {
        const d = { ...(prev as TimerData) };
        if (typeof duration === 'number') d.duration = duration;
        if (typeof label === 'string') d.label = label;
        return d;
      });
      return itemId;
    },
  });

  useCopilotAction({
    name: "setItemPosition",
    description: "Update an item's grid position if supported",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "itemId", type: "string", required: true },
      { name: "position", type: "string", required: true },
    ],
    handler: ({ itemId, position /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
}: any) => {
      updateItemData(String(itemId), (prev) => {
        const data = { ...(prev as any) };
        if (typeof position === 'string') data.position = position as GamePosition;
        return data as ItemData;
      });
      return itemId;
    },
  });

  // Update Hands Card audience permissions
  useCopilotAction({
    name: "setHandsCardAudience",
    description: "Update audience for a hands card (public vs private players)",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "itemId", type: "string", required: true, description: "Target hands_card item id" },
      { name: "audience_type", type: "boolean", required: true, description: "true=public, false=private" },
      { name: "audience_ids", type: "string[]", required: false, description: "Player IDs when private" },
    ],
    handler: ({ itemId, audience_type, audience_ids }: { itemId: string; audience_type: boolean; audience_ids?: string[]; }) => {
      updateItemData(String(itemId), (prev) => {
        const d = { ...(prev as HandsCardData) };
        d.audience_type = audience_type;
        d.audience_ids = audience_ids ?? [];
        return d;
      });
      return itemId;
    },
  });

  // Convenience: create a private Hands Card for a specific player
  useCopilotAction({
    name: "createHandsCardForPlayer",
    description: "Create a private hands card visible only to the specified player",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "name", type: "string", required: true, description: "Item name" },
      { name: "playerId", type: "string", required: true, description: "Target player id (gamePlayerId)" },
      { name: "cardType", type: "string", required: true, description: "Card classification" },
      { name: "cardName", type: "string", required: true, description: "Display name of the card" },
      { name: "descriptions", type: "string", required: false, description: "Short description or effect" },
      { name: "color", type: "string", required: false, description: "Accent color (hex or token)" },
      { name: "position", type: "string", required: false, description: "Grid position (default: bottom-center)" },
    ],
    handler: ({ name, playerId, cardType, cardName, descriptions, color, position }: {
      name: string;
      playerId: string;
      cardType: string;
      cardName: string;
      descriptions?: string;
      color?: string;
      position?: string;
    }) => {
      const data: HandsCardData = {
        cardType,
        cardName,
        descriptions,
        color: color || "#2563eb",
        position: normalizePosition(position) || "bottom-center",
        audience_type: false,
        audience_ids: [String(playerId)],
      };
      return addItem("hands_card", name, data);
    },
  });

  // Frontend action: delete an item by id
  // Create Player States Display
  useCopilotAction({
    name: "createPlayerStatesDisplay",
    description: "Create a display panel showing current player states with real-time updates",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "name", type: "string", required: true, description: "Item name" },
      { name: "title", type: "string", required: false, description: "Display title (default: 'Player States')" },
      { name: "position", type: "string", required: false, description: "Grid position (default: 'middle-left')" },
      { name: "maxHeight", type: "string", required: false, description: "Max height for scrolling (default: '400px')" },
      { name: "audience_type", type: "boolean", required: false, description: "true=public; false=private" },
      { name: "audience_ids", type: "string[]", required: false, description: "Visible player IDs if private" },
    ],
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    handler: ({ name, title, position, maxHeight, audience_type, audience_ids }: any) => {
      const normalized = (name ?? "").trim();
      if (normalized) {
        const existing = (viewState.items ?? initialState.items).find((it) => it.type === "player_states_display" && (it.name ?? "").trim() === normalized);
        if (existing) return existing.id;
      }
      const data = {
        title: title || "Player States",
        position: (position as GamePosition) || "middle-left",
        maxHeight: maxHeight || "400px",
        audience_type: audience_type ?? true,
        audience_ids: audience_ids ?? [],
      } as PlayerStatesDisplayData;
      return addItem("player_states_display", name, data);
    },
  });

  // Create Player Actions Display
  useCopilotAction({
    name: "createPlayerActionsDisplay", 
    description: "Create a scrollable display panel showing player actions log with latest actions at the top",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "name", type: "string", required: true, description: "Item name" },
      { name: "title", type: "string", required: false, description: "Display title (default: 'Player Actions')" },
      { name: "position", type: "string", required: false, description: "Grid position (default: 'middle-right')" },
      { name: "maxHeight", type: "string", required: false, description: "Max height for scrolling (default: '400px')" },
      { name: "maxItems", type: "number", required: false, description: "Maximum number of actions to display (default: 50)" },
      { name: "audience_type", type: "boolean", required: false, description: "true=public; false=private" },
      { name: "audience_ids", type: "string[]", required: false, description: "Visible player IDs if private" },
    ],
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    handler: ({ name, title, position, maxHeight, maxItems, audience_type, audience_ids }: any) => {
      const normalized = (name ?? "").trim();
      if (normalized) {
        const existing = (viewState.items ?? initialState.items).find((it) => it.type === "player_actions_display" && (it.name ?? "").trim() === normalized);
        if (existing) return existing.id;
      }
      const data = {
        title: title || "Player Actions", 
        position: (position as GamePosition) || "middle-right",
        maxHeight: maxHeight || "400px",
        maxItems: maxItems || 50,
        audience_type: audience_type ?? true,
        audience_ids: audience_ids ?? [],
      } as PlayerActionsDisplayData;
      return addItem("player_actions_display", name, data);
    },
  });

  useCopilotAction({
    name: "deleteItem",
    description: "Delete an item by id.",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "itemId", type: "string", required: true, description: "Target item id." },
    ],
    handler: ({ itemId }: { itemId: string }) => {
      const existed = (viewState.items ?? initialState.items).some((p) => p.id === itemId);
      deleteItem(itemId);
      return existed ? `deleted:${itemId}` : `not_found:${itemId}`;
    },
  });

  useCopilotAction({
    name: "clearCanvas",
    description: "Clear all items from the canvas except avatar sets. This is useful when transitioning between game phases or starting fresh.",
    available: "remote",
    followUp: false,
    parameters: [],
    handler: () => {
      console.log("🔧 clearCanvas handler called - starting execution");
      setState((prev) => {
        const base = prev ?? initialState;
        const items = base.items ?? [];
        // Keep only avatar_set items
        const avatarItems = items.filter(item => item.type === "avatar_set");
        const removedCount = items.length - avatarItems.length;
        console.log(`🧹 Clearing canvas: removed ${removedCount} items, kept ${avatarItems.length} avatars`);
        return { 
          ...base, 
          items: avatarItems,
          lastAction: `cleared_canvas:${removedCount}_removed`
        } as AgentState;
      });
      
      const originalCount = (viewState.items ?? []).length;
      const avatarCount = (viewState.items ?? []).filter(item => item.type === "avatar_set").length;
      const removedCount = originalCount - avatarCount;
      const result = `Canvas cleared. Removed ${removedCount} items, kept ${avatarCount} avatar sets.`;
      console.log("✅ clearCanvas handler completed, returning:", result);
      return result;
    },
  });

  // Chat-related Agent tools
  useCopilotAction({
    name: "addBotChatMessage",
    description: "Add a bot message to the game chat area",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "botId", type: "string", required: true, description: "ID of the bot sending the message" },
      { name: "botName", type: "string", required: true, description: "Name of the bot sending the message" },
      { name: "message", type: "string", required: true, description: "Bot's chat message" },
      { name: "messageType", type: "string", required: false, description: "Message type: 'message', 'system', or 'action'" },
    ],
    handler: ({ botId, botName, message, messageType }: { 
      botId: string; 
      botName: string; 
      message: string; 
      messageType?: string;
    }) => {
      const botMessage: ChatMessage = {
        id: `bot-msg-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`,
        playerId: botId,
        playerName: botName,
        message: message,
        timestamp: Date.now(),
        type: (messageType as 'message' | 'system' | 'action') || 'message'
      };


      setChatMessages(prev => [...prev, botMessage]);

      return `Bot ${botName} sent message: ${message}`;
    },
  });

  // A UI tool for the Agent to prompt user for a text paragraph, then send it with a specified speaker
  useCopilotAction({
    name: "promptUserText",
    description: "Open a dialog for the user to input a paragraph and confirm; reported as spoken by a specific player.",
    available: "remote",
    followUp: false,
    parameters: [
      { name: "speakerId", type: "string", required: false, description: "Player ID who is speaking (default: current player)" },
      { name: "title", type: "string", required: false, description: "Dialog title" },
      { name: "placeholder", type: "string", required: false, description: "Textarea placeholder" },
      { name: "toBotId", type: "string", required: false, description: "Optional target bot id for direct message" },
    ],
    handler: ({ speakerId, title, placeholder, toBotId }: { speakerId?: string; title?: string; placeholder?: string; toBotId?: string }) => {
      setPendingTextPrompt({ speakerId, title, placeholder, toBotId });
      return "prompt_opened";
    },
  });

  const titleClasses = cn(
    /* base styles */
    "w-full outline-none rounded-md px-2 py-1",
    "bg-transparent placeholder:text-gray-400",
    "ring-1 ring-transparent transition-all ease-out",
    /* hover styles */
    "hover:ring-border",
    /* focus styles */
    "focus:ring-2 focus:ring-accent/50 focus:shadow-sm focus:bg-accent/10",
    "focus:shadow-accent focus:placeholder:text-accent/65 focus:text-accent",
  );

  return (
    <div
      style={{ "--copilot-kit-primary-color": "#2563eb" } as CopilotKitCSSProperties}
      className="h-[calc(100vh-3.5rem)] flex flex-col min-h-0 overflow-hidden"
    >
      {/* Main Layout */}
      <div className="flex flex-1 min-h-0 overflow-hidden">
        {/* Chat Sidebar */}
        <aside className="-order-1 max-md:hidden flex flex-col min-w-80 w-[30vw] max-w-120 p-4 pr-0">
          <div className="h-full flex flex-col align-start w-full shadow-lg rounded-2xl border border-sidebar-border overflow-hidden">
            {/* Chat Header */}
            <AppChatHeader />
            {/* Sidebar Plan Tracker or Completed Summary */}
            {(() => {
              const steps = planStepsMemo;
              const count = steps.length;
              const status = String(planStatusMemo ?? "");
              if (!Array.isArray(steps) || count === 0 || status === "completed" || status === "failed" || status === "") return null;
              if (status === "completed") {
                return (
                  <div className="px-4 pt-3 border-b">
                    <Accordion type="single" collapsible>
                      <AccordionItem value="done">
                        <AccordionTrigger className="text-xs pt-0 pb-3">
                          <span className="inline-flex items-center gap-2">
                            <Check className="h-4 w-4 text-green-600" />
                            <span className="font-medium">{count} steps completed</span>
                          </span>
                        </AccordionTrigger>
                        <AccordionContent>
                          <div className="rounded-xl border bg-card p-3">
                            <div className="mb-1 text-xs font-semibold">Plan <span className="ml-2 rounded-full px-2 py-0.5 text-[10px] font-medium border text-green-700 border-green-300 bg-green-50">completed</span></div>
                            <ol className="space-y-1">
                              {steps.map((s, i) => (
                                <li key={`${s.title ?? "step"}-${i}`} className="flex items-start gap-2">
                                  <span className="mt-0.5 inline-flex h-4 w-4 items-center justify-center">
                                    <Check className="h-4 w-4 text-green-600" />
                                  </span>
                                  <div className="flex-1 text-xs">
                                    <div className="leading-5 text-green-700">{s.title ?? `Step ${i + 1}`}</div>
                                  </div>
                                </li>
                              ))}
                            </ol>
                          </div>
                        </AccordionContent>
                      </AccordionItem>
                    </Accordion>
                  </div>
                );
              }
              return (
                <div className="p-4 py-3 border-b">
                  <div className="rounded-xl border bg-card p-3">
                    <div className="mb-1 text-xs font-semibold">Plan <span className="ml-2 rounded-full px-2 py-0.5 text-[10px] font-medium border text-blue-700 border-blue-300 bg-blue-50">in_progress</span></div>
                    <ol className="space-y-1">
                      {steps.map((s, i) => {
                        const st = String(s?.status ?? "pending").toLowerCase();
                        const isActive = typeof currentStepIndexMemo === "number" && currentStepIndexMemo === i && st === "in_progress";
                        const isDone = st === "completed";
                        const isFailed = st === "failed";
                        return (
                          <li key={`${s.title ?? "step"}-${i}`} className="flex items-start gap-2">
                            <span className="mt-0.5 inline-flex h-4 w-4 items-center justify-center">
                              {isDone ? (
                                <Check className="h-4 w-4 text-green-600" />
                              ) : isActive ? (
                                <Loader2 className="h-4 w-4 animate-spin text-blue-600" />
                              ) : isFailed ? (
                                <X className="h-4 w-4 text-red-600" />
                              ) : (
                                <span className="block h-2 w-2 rounded-full bg-gray-300" />
                              )}
                            </span>
                            <div className="flex-1 text-xs">
                              <div className={cn("leading-5", isDone && "text-green-700", isActive && "text-blue-700", isFailed && "text-red-700")}>{s.title ?? `Step ${i + 1}`}</div>
                            </div>
                          </li>
                        );
                      })}
                    </ol>
                  </div>
                </div>
              );
            })()}
            
            {/* Player States and Actions Display - Outside Canvas */}
            <div className="p-4 space-y-3">
              {/* Player States Display */}
              <div className="bg-card border border-border rounded-lg p-3 shadow-sm">
                <div className="text-sm font-medium text-foreground mb-2">
                  Player States
                </div>
                <div className="space-y-2 overflow-y-auto max-h-[180px]">
                  {Object.keys(viewState.player_states || {}).length === 0 ? (
                    <div className="text-xs text-muted-foreground text-center py-4">
                      No player states available
                    </div>
                  ) : (
                    Object.entries(viewState.player_states || {}).map(([playerId, state]) => (
                      <div key={playerId} className="p-2 bg-muted/50 rounded border text-xs">
                        <div className="font-medium mb-1">Player {playerId}</div>
                        <pre className="whitespace-pre-wrap text-xs font-mono overflow-hidden">
                          {JSON.stringify(state, null, 2)}
                        </pre>
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* Player Actions Display */}
              <div className="bg-card border border-border rounded-lg p-3 shadow-sm">
                <div className="text-sm font-medium text-foreground mb-2">
                  Player Actions
                </div>
                <div className="space-y-2 overflow-y-auto max-h-[180px]">
                  {(() => {
                    const playerActions = viewState.playerActions || {};
                    const playerEntries = Object.entries(playerActions);
                    
                    return playerEntries.length === 0 ? (
                      <div className="text-xs text-muted-foreground text-center py-4">
                        No player actions recorded
                      </div>
                    ) : (
                      playerEntries.map(([playerId, playerData]) => {
                        if (!playerData || typeof playerData !== 'object' || !playerData.actions) {
                          return null;
                        }
                        
                        const actions = Object.entries(playerData.actions)
                          .map(([actionId, actionData]: [string, any]) => ({
                            actionId,
                            phase: actionData?.phase || '',
                            timestamp: actionData?.timestamp || 0,
                            action: actionData?.action || ''
                          }))
                          .sort((a, b) => b.timestamp - a.timestamp);
                        
                        return (
                          <div key={playerId} className="p-3 bg-card border border-border rounded-lg">
                            <div className="text-sm font-medium text-foreground mb-2">
                              {playerData.name}
                            </div>
                            <div className="space-y-1">
                              {actions.length === 0 ? (
                                <div className="text-xs text-muted-foreground">No actions</div>
                              ) : (
                                actions.map((actionData) => (
                                  <div key={actionData.actionId} className="text-xs p-2 bg-muted/30 rounded">
                                    <div className="flex items-center justify-between mb-1">
                                      <span className="font-medium">{actionData.phase}</span>
                                      <span className="text-muted-foreground">
                                        {new Date(actionData.timestamp).toLocaleTimeString()}
                                      </span>
                                    </div>
                                    <div>{actionData.action}</div>
                                  </div>
                                ))
                              )}
                            </div>
                          </div>
                        );
                      })
                    );
                  })()}
                </div>
              </div>
            </div>
            
            {/* Chat Content - conditionally rendered to avoid duplicate rendering */}
            {isDesktop && (
              <CopilotChat
                className="flex-1 overflow-auto w-full scroll-smooth"
                labels={{
                  title: "Game Master",
                  initial:
                    "🎲 Welcome to the AI Game Engine! Click 'Start' to begin a game, or I'll broadcast updates during gameplay.",
                }}
                suggestions={[
                  {
                    title: "Start",
                    message: "Start game.",
                  },
                ]}
              />
            )}
          </div>
        </aside>
        {/* Main Content */}
        <main className="relative flex flex-1 h-full overflow-visible">
          <div ref={scrollAreaRef} className="relative overflow-visible size-full px-4 sm:px-8 md:px-10 py-4">
            <div className={cn(
              "relative mx-auto max-w-7xl h-full min-h-8",
              (showJsonView || (viewState.items ?? []).length === 0) && "flex flex-col",
            )}>
              {/* Global Title & Description (hidden in JSON view) */}
              {!showJsonView && (
                <motion.div style={{ opacity: headerOpacity }} className="sticky top-0 mb-6">
                  <GameTitle viewState={viewState} titleClasses={titleClasses} />
                  <GamePhaseInfo viewState={viewState} titleClasses={titleClasses} />
                </motion.div>
              )}
              
              {(viewState.items ?? []).length === 0 ? (
                <EmptyState className="flex-1">
                  <div className="mx-auto max-w-lg text-center">
                    <h2 className="text-lg font-semibold text-foreground">Ready to Play!</h2>
                    <p className="mt-2 text-sm text-muted-foreground">
                      Click Start to begin your game.
                    </p>
                    <div className="mt-6 flex justify-center">
                      <Button
                        type="button"
                        variant="outline"
                        className={cn(
                          "gap-2 text-base font-semibold md:h-10",
                          "bg-green-50 hover:bg-green-100 border-green-200 text-green-700",
                        )}
                        onClick={async () => {
                          // Use unified interaction handling
                          await handleUserInteraction("Start game.", "start_game");
                        }}
                      >
                        🎮 Start Game
                      </Button>
                    </div>
                  </div>
                </EmptyState>
              ) : (
                <div className="flex-1 py-0 overflow-visible">
                  {showJsonView ? (
                    <div className="pb-16 size-full">
                      <div className="rounded-2xl border shadow-sm bg-card size-full overflow-auto max-md:text-sm">
                        <ShikiHighlighter language="json" theme="github-light">
                          {JSON.stringify(getStatePreviewJSON(viewState), null, 2)}
                        </ShikiHighlighter>
                      </div>
                    </div>
                  ) : (
                    <div className="relative overflow-visible">
                      {/* Avatar sets render as overlay */}
                      <div className="absolute inset-0 z-10 pointer-events-none" style={{ left: '-150px', right: '-150px', top: '0', bottom: '0' }}>
                        {(viewState.items ?? [])
                          .filter(item => item.type === "avatar_set")
                          .map(item => (
                            <CardRenderer 
                              key={item.id} 
                              item={item} 
                              onUpdateData={(updater) => updateItemData(item.id, updater)} 
                              onToggleTag={() => toggleTag()} 
                              onButtonClick={handleButtonClick} 
                              onVote={handleVote} 
                              playerStates={viewState.player_states} 
                              deadPlayers={viewState.deadPlayers}
                            />
                          ))}
                      </div>
                      
                      {/* Card table wrapper (wood rim) */}
                      <div className="rounded-[28px] p-4 bg-[linear-gradient(135deg,#7b4a2e,#9a6b3f,#7a4e2b)] shadow-[0_12px_30px_rgba(0,0,0,0.4)] [box-shadow:inset_0_0_0_2px_rgba(255,255,255,0.12),inset_0_0_0_1px_rgba(0,0,0,0.25)]">
                        {/* Felt surface canvas */}
                        <div
                          style={GAME_GRID_STYLE}
                          className="relative pb-20 rounded-[18px] border border-[#2a3f2f]/40 ring-1 ring-[#1a2d20]/40 bg-[radial-gradient(80%_80%_at_30%_20%,#1b5e2a_0%,#155c2b_55%,#0e4a22_100%)] overflow-hidden"
                          data-canvas-container
                        >
                      {/* Render all 9 region containers */}
                      {(["top-left", "top-center", "top-right", "middle-left", "center", "middle-right", "bottom-left", "bottom-center", "bottom-right"] as GamePosition[]).map(position => {
                        const itemsInRegion = (viewState.items ?? []).filter(item => {
                          // Exclude avatar_set items as they render as overlay
                          if (item.type === "avatar_set") return false;
                          // Exclude night_overlay items as they render as global overlays
                          if (item.type === "night_overlay") return false;
                          const itemData = item.data as ItemData;
                          const itemPosition = (itemData as { position?: string })?.position;
                          const normalizedPosition = itemPosition ? normalizePosition(itemPosition) : 'center';
                          return normalizedPosition === position;
                        });

                        return (
                          <div
                            key={position}
                            style={{ gridArea: position }}
                            className="flex flex-col items-center justify-center gap-4 p-2 rounded-lg min-h-[100px]"
                          >
                            {itemsInRegion.map((item) => {
                              // Check audience permissions for delete button visibility
                              const currentPlayerId = getCurrentPlayerId();
                              const itemData = item.data as ItemData & { audience_type: boolean; audience_ids: string[] };
                              const hasPermission = !currentPlayerId || 
                                itemData.audience_type === true || 
                                itemData.audience_ids?.includes(currentPlayerId);
                              
                              return (
                                <div key={item.id} className="relative group">
                                  {hasPermission && (
                                    <button
                                      type="button"
                                      aria-label="Delete card"
                                      className="absolute -right-2 -top-2 z-10 inline-flex h-6 w-6 items-center justify-center rounded-full bg-red-500 text-white opacity-0 group-hover:opacity-100 hover:bg-red-600 transition-opacity"
                                      onClick={() => deleteItem(item.id)}
                                    >
                                      <X className="h-3 w-3" />
                                    </button>
                                  )}
                                  
                                  <CardRenderer item={item} onUpdateData={(updater) => updateItemData(item.id, updater)} onToggleTag={() => toggleTag()} onButtonClick={handleButtonClick} onVote={handleVote} playerStates={viewState.player_states} deadPlayers={viewState.deadPlayers} playerActions={viewState.playerActions} />
                                </div>
                              );
                            })}
                          </div>
                        );
                      })}
                        </div>
                      </div>

                      {/* Render night overlay components as global overlays */}
                      {(viewState.items ?? []).filter(item => item.type === "night_overlay").map((item) => {
                        // Check audience permissions for delete button visibility
                        const currentPlayerId = getCurrentPlayerId();
                        const itemData = item.data as ItemData & { audience_type: boolean; audience_ids: string[] };
                        const hasPermission = !currentPlayerId || 
                          itemData.audience_type === true || 
                          itemData.audience_ids?.includes(currentPlayerId);
                          
                        return (
                          <div key={item.id}>
                            <CardRenderer 
                              item={item} 
                              onUpdateData={(updater) => updateItemData(item.id, updater)} 
                              onToggleTag={() => toggleTag()} 
                              onButtonClick={handleButtonClick} 
                              onVote={handleVote} 
                              playerStates={viewState.player_states} 
                              deadPlayers={viewState.deadPlayers} 
                            />
                            {hasPermission && (
                              <button
                                type="button"
                                aria-label="Delete night overlay"
                                className="fixed top-4 right-4 z-[70] inline-flex h-8 w-8 items-center justify-center rounded-full bg-red-500 text-white hover:bg-red-600 transition-colors pointer-events-auto"
                                onClick={() => deleteItem(item.id)}
                              >
                                <X className="h-4 w-4" />
                              </button>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
          
          {/* Continue button - shown when game has items */}
          {(viewState.items ?? []).length > 0 && (
            <div className="flex justify-center py-4">
              <Button
                variant="default" 
                size="lg"
                className="bg-blue-600 hover:bg-blue-700 text-white font-semibold px-8 py-3"
                disabled={isExecuting}
                onClick={async () => {
                  // Use unified interaction handling
                  await handleUserInteraction("Continue", "continue_game");
                }}
              >
                {isExecuting ? "Processing..." : "Continue"}
              </Button>
            </div>
          )}
          
          {(viewState.items ?? []).length > 0 ? (
            <div className="absolute left-1/2 -translate-x-1/2 bottom-4 flex gap-2">
              <NewItemMenu 
                onSelect={(type) => addItem(type)}
                className="bg-green-50 hover:bg-green-100 border-green-200 text-green-700"
              />
              <Button
                type="button"
                variant="outline"
                className="gap-1.25 text-base font-semibold"
                onClick={() => setShowJsonView((v) => !v)}
              >
                {showJsonView
                  ? "Canvas"
                  : <>JSON</>
                }
              </Button>
            </div>
          ) : null}
        </main>
        
        {/* Right Chat Area */}
        <aside className="max-md:hidden flex flex-col min-w-80 w-[30vw] max-w-120 p-4 pl-0">
          <div className="h-full flex flex-col w-full shadow-lg rounded-2xl border border-sidebar-border overflow-hidden">
            <GameChatArea
              messages={chatMessages}
              currentPlayerId={typeof window !== 'undefined' ? getCurrentPlayerId() : null}
              currentPlayerName={viewState.player_states?.[getCurrentPlayerId() || '']?.name as string || ''}
              onSendMessage={handleSendChatMessage}
              playerCount={(viewState.roomSession?.players?.length ?? viewState.roomSession?.totalPlayers ?? Object.keys(viewState.player_states || {}).length)}
              availableBots={getAvailableBots()}
            />
          </div>
        </aside>
      </div>
      <div className="md:hidden">
        {/* Mobile Chat Popup - conditionally rendered to avoid duplicate rendering */}
        {!isDesktop && (
          <CopilotPopup
            className="scroll-smooth"
            Header={PopupHeader}
            labels={{
              title: "Game Master", 
              initial:
                "🎲 Welcome to the AI Game Engine! Click 'Start' to begin a game, or I'll broadcast updates during gameplay.",
            }}
            suggestions={[
              {
                title: "Start",
                message: "Start game.",
              },
            ]}
          />
        )}
      </div>

      {/* Dialog for promptUserText */}
      <Dialog open={!!pendingTextPrompt} onOpenChange={(open) => { if (!open) { setPendingTextPrompt(null); setPendingTextValue(""); } }}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{pendingTextPrompt?.title || 'Enter the text you want to send'}</DialogTitle>
          </DialogHeader>
          <div className="space-y-2">
            <Textarea
              value={pendingTextValue}
              onChange={(e) => setPendingTextValue(e.target.value)}
              placeholder={pendingTextPrompt?.placeholder || 'Please enter...'}
              rows={6}
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => { setPendingTextPrompt(null); setPendingTextValue(""); }}>Cancel</Button>
            <Button onClick={handleConfirmPromptText}>Send</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Floating Broadcast Input */}
      <BroadcastInput
        open={broadcastOpen}
        title={pendingBroadcast?.title || "Broadcast"}
        placeholder={pendingBroadcast?.placeholder || "Type a broadcast message..."}
        initialValue={pendingBroadcast?.prefill || ""}
        onConfirm={async (text: string) => {
          // Add a system-style message to chat (local memory store)
          const sysMessage: ChatMessage = {
            id: `bc-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
            playerId: "broadcast",
            playerName: "Broadcast",
            message: text,
            timestamp: Date.now(),
            type: 'system',
          };
          setChatMessages(prev => [...prev, sysMessage]);
          // Sync minimal info to shared state for agent context
          setState(prev => ({ ...(prev ?? initialState), lastBroadcast: text } as AgentState));
          // Notify agent
          await handleUserInteraction(`Broadcast: ${text}`, 'broadcast');
          // Close UI
          setBroadcastOpen(false);
          setPendingBroadcast(null);
        }}
        onClose={() => {
          setBroadcastOpen(false);
          setPendingBroadcast(null);
        }}
      />
    </div>
  );
}
