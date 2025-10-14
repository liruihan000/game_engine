"use client";

import { useCoAgent, useCopilotAction, useCoAgentStateRender, useCopilotAdditionalInstructions, useLangGraphInterrupt, useCopilotChat } from "@copilotkit/react-core";
import { CopilotKitCSSProperties, CopilotChat, CopilotPopup } from "@copilotkit/react-ui";
import { TextMessage, MessageRole } from "@copilotkit/runtime-client-gql";
import { useCallback, useEffect, useRef, useState } from "react";
import type React from "react";
import { Button } from "@/components/ui/button"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import AppChatHeader, { PopupHeader } from "@/components/canvas/AppChatHeader";
import { X, Check, Loader2 } from "lucide-react"
import CardRenderer from "@/components/canvas/CardRenderer";
import ShikiHighlighter from "react-shiki/web";
import { motion, useScroll, useTransform, useMotionValueEvent } from "motion/react";
import { EmptyState } from "@/components/empty-state";
import { cn } from "@/lib/utils";
import type { AgentState, PlanStep, Item, ItemData, CardType, GamePosition, CharacterCardData, ActionButtonData, PhaseIndicatorData, TextDisplayData, VotingPanelData, AvatarSetData, BackgroundControlData, ResultDisplayData, TimerData, ComponentSize } from "@/lib/canvas/types";
import { GAME_GRID_STYLE } from "@/lib/canvas/types";
import { initialState, isNonEmptyAgentState, defaultDataFor } from "@/lib/canvas/state";
// import { projectAddField4Item, projectSetField4ItemText, projectSetField4ItemDone, projectRemoveField4Item, chartAddField1Metric, chartSetField1Label, chartSetField1Value, chartRemoveField1Metric } from "@/lib/canvas/updates";
import useMediaQuery from "@/hooks/use-media-query";

export default function CopilotKitPage() {
  // Use consistent agent name across all components - room isolation via threadId
  const { state, setState } = useCoAgent<AgentState>({
    name: "sample_agent", // 🔑 Fixed agent name, room isolation via threadId in DynamicCopilotProvider
    initialState,
  });

  const { appendMessage } = useCopilotChat();

  // Helper function to ensure gameName is set from URL
  const ensureGameName = useCallback(() => {
    const params = new URLSearchParams(window.location.search);
    const gameName = params.get('game');
    
    if (gameName) {
      const currentGameName = state?.gameName;
      if (currentGameName !== gameName) {
        console.log('🎮 Setting gameName from URL:', gameName);
        setState((prev) => {
          const base = prev ?? initialState;
          return { ...base, gameName } as AgentState;
        });
        return true; // Indicate that gameName was updated
      }
    }
    return false; // No update needed
  }, [setState, state]);

  // Handle action button clicks
  const handleButtonClick = useCallback(async (item: Item) => {
    const buttonData = item.data as ActionButtonData;
    
    // Ensure gameName is set before sending message
    ensureGameName();
    
    // Send message to CopilotChat
    await appendMessage(
      new TextMessage({
        role: MessageRole.User,
        content: `Button "${item.name}" (ID: ${item.id}) has been clicked. Action: ${buttonData.action}`
      })
    );
  }, [appendMessage, ensureGameName]);

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

    // Send message to CopilotChat
    await appendMessage(
      new TextMessage({
        role: MessageRole.User,
        content: `Player ${playerId} voted "${option}" in voting ${votingId}`
      })
    );
  }, [setState, appendMessage]);

  // Global cache for the last non-empty agent state
  const cachedStateRef = useRef<AgentState>(state ?? initialState);
  useEffect(() => {
    if (isNonEmptyAgentState(state)) {
      cachedStateRef.current = state as AgentState;
    }
  }, [state]);
  // we use viewState to avoid transient flicker; TODO: troubleshoot and remove this workaround
  const viewState: AgentState = isNonEmptyAgentState(state) ? (state as AgentState) : cachedStateRef.current;

  const isDesktop = useMediaQuery("(min-width: 768px)");
  const [showJsonView, setShowJsonView] = useState<boolean>(false);
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

  useEffect(() => {
    console.log("[CoAgent state updated]", state);
  }, [state]);

  // Reset JSON view when there are no items
  useEffect(() => {
    const itemsCount = (viewState?.items ?? []).length;
    if (itemsCount === 0 && showJsonView) {
      setShowJsonView(false);
    }
  }, [viewState?.items, showJsonView]);

  // Initialize player_states from sessionStorage when game starts
  useEffect(() => {
    console.log('🔍 Checking for player states...');
    // Get current room context
    const gameContext = sessionStorage.getItem('gameContext');
    if (!gameContext) {
      console.log('No game context found');
      return;
    }
    
    const context = JSON.parse(gameContext);
    const playerStatesKey = `playerStates_${context.roomId}`;
    const playerStatesData = sessionStorage.getItem(playerStatesKey);

    if (playerStatesData) {
      try {
        const parsedData = JSON.parse(playerStatesData);
        console.log('📥 Found player states for room:', context.roomId, parsedData);
        
        // Initialize player_states in AgentState
        setState(prevState => {
          if (prevState?.player_states) {
            console.log('⏭️ Player states already set, skipping...');
            return prevState;
          }
          
          console.log('✅ Initializing player states...');
          return {
            ...(prevState || initialState),
            player_states: parsedData
          };
        });
        
        // Clear room-specific sessionStorage to prevent re-initialization
        sessionStorage.removeItem(playerStatesKey);
        console.log('✅ Player states initialized and sessionStorage cleared for room:', context.roomId);
        
      } catch (error) {
        console.error('❌ Failed to parse player states from sessionStorage:', error);
        sessionStorage.removeItem(playerStatesKey); // Clear invalid data
      }
    }
  }, [setState]); // Only depend on setState

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
      const items = viewState.items ?? initialState.items;
      const playerStates = viewState.player_states ?? {};
      const votes = viewState.vote ?? [];
      const deadPlayers = viewState.deadPlayers ?? [];
      const gTitle = "Game Engine";
      const gDesc = "AI-Powered Game Engine";
      const summary = items
        .slice(0, 5)
        .map((p: Item) => `id=${p.id} • name=${p.name} • type=${p.type}`)
        .join("\n");
      
      const playerStatesStr = Object.keys(playerStates).length > 0 
        ? JSON.stringify(playerStates, null, 2)
        : "(none)";
      
      const votesStr = votes.length > 0
        ? votes.map(v => `${v.playerid} voted "${v.option}" in ${v.voteid}`).join("; ")
        : "(none)";
      
      const deadPlayersStr = deadPlayers.length > 0
        ? deadPlayers.join(", ")
        : "(none)";
      
      const currentGameName = viewState.gameName || "(none)";
      
      const fieldSchema = "";
      const toolUsageHints = [
        "GAME TOOL USAGE HINTS:",
        "LOOP CONTROL: When asked to 'add a couple' items, add at most 2 and stop. Avoid repeated calls to the same mutating tool in one turn.",
        "VERIFICATION: After tools run, re-read the latest state and confirm what actually changed.",
      ].join("\n");
      return [
        "ALWAYS ANSWER FROM SHARED STATE (GROUND TRUTH).",
        `Global Title: ${gTitle || "(none)"}`,
        `Global Description: ${gDesc || "(none)"}`,
        `Current Game: ${currentGameName}`,
        "Items (sample):",
        summary || "(none)",
        "Player States:",
        playerStatesStr,
        "Current Votes:",
        votesStr,
        "Dead Players:",
        deadPlayersStr,
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
    available: "remote",
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
    parameters: [
      { name: "name", type: "string", required: true, description: "Item name"},
      { name: "role", type: "string", required: true, description: "Character role (e.g., werewolf, seer, villager)" },
      { name: "position", type: "string", required: true, description: "Grid position (select: 'top-left' | 'top-center' | 'top-right' | 'middle-left' | 'center' | 'middle-right' | 'bottom-left' | 'bottom-center' | 'bottom-right')" },
      { name: "size", type: "string", required: false, description: "Card size (select: 'small' | 'medium' | 'large'; default: 'medium')" },
      { name: "description", type: "string", required: false, description: "Optional character description" },
    ],
    handler: ({ name, role, position, size, description }: { 
      name: string;
      role: string; 
      position: string; 
      size?: string; 
      description?: string; 
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
        position: position as GamePosition,
        size: size as ComponentSize,
        description
      };
      return addItem("character_card", name, data);
    },
  });

  useCopilotAction({
    name: "createActionButton",
    description: "Create an action button for player interactions.",
    available: "remote",
    parameters: [
      { name: "name", type: "string", required: true, description: "Item name" },
      { name: "label", type: "string", required: true, description: "Button text" },
      { name: "action", type: "string", required: true, description: "Action identifier when clicked" },
      { name: "enabled", type: "boolean", required: true, description: "Whether button is clickable" },
      { name: "position", type: "string", required: true, description: "Grid position (select: 'top-left' | 'top-center' | 'top-right' | 'middle-left' | 'center' | 'middle-right' | 'bottom-left' | 'bottom-center' | 'bottom-right')" },
      { name: "size", type: "string", required: false, description: "Button size (select: 'small' | 'medium' | 'large')" },
      { name: "variant", type: "string", required: false, description: "Button style (select: 'primary' | 'secondary' | 'danger')" },
    ],
    handler: ({ name, label, action, enabled, position, size, variant }: {
      name: string;
      label: string;
      action: string;
      enabled: boolean;
      position: string;
      size?: string;
      variant?: string;
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
        position: position as GamePosition,
        size: size as ComponentSize,
        variant: variant as "primary" | "secondary" | "danger"
      };
      return addItem("action_button", name, data);
    },
  });

  useCopilotAction({
    name: "createPhaseIndicator",
    description: "Create a phase indicator to show current game phase.",
    available: "remote",
    parameters: [
      { name: "name", type: "string", required: true, description: "Item name" },
      { name: "currentPhase", type: "string", required: true, description: "Current game phase" },
      { name: "position", type: "string", required: true, description: "Grid position (select: 'top-left' | 'top-center' | 'top-right' | 'middle-left' | 'center' | 'middle-right' | 'bottom-left' | 'bottom-center' | 'bottom-right')" },
      { name: "description", type: "string", required: false, description: "Optional phase description" },
      { name: "timeRemaining", type: "number", required: false, description: "Seconds remaining in phase" },
    ],
    handler: ({ name, currentPhase, position, description, timeRemaining }: {
      name: string;
      currentPhase: string;
      position: string;
      description?: string;
      timeRemaining?: number;
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
        position: position as GamePosition,
        description,
        timeRemaining
      };
      return addItem("phase_indicator", name, data);
    },
  });

  useCopilotAction({
    name: "createTextDisplay",
    description: "Create a text display for game information.",
    available: "remote",
    parameters: [
      { name: "name", type: "string", required: true, description: "Item name" },
      { name: "content", type: "string", required: true, description: "Main text content" },
      { name: "position", type: "string", required: true, description: "Grid position (select: 'top-left' | 'top-center' | 'top-right' | 'middle-left' | 'center' | 'middle-right' | 'bottom-left' | 'bottom-center' | 'bottom-right')" },
      { name: "title", type: "string", required: false, description: "Optional title text" },
      { name: "type", type: "string", required: false, description: "Display type (select: 'info' | 'warning' | 'error' | 'success')" },
    ],
    handler: ({ name, content, position, title, type }: {
      name: string;
      content: string;
      position: string;
      title?: string;
      type?: string;
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
        position: position as GamePosition,
        title,
        type: type as "info" | "warning" | "error" | "success"
      };
      return addItem("text_display", name, data);
    },
  });

  useCopilotAction({
    name: "createVotingPanel",
    description: "Create a voting panel for player voting.",
    available: "remote",
    parameters: [
      { name: "name", type: "string", required: true, description: "Item name" },
      { name: "votingId", type: "string", required: true, description: "Unique voting ID" },
      { name: "options", type: "string[]", required: true, description: "List of voting options" },
      { name: "position", type: "string", required: true, description: "Grid position (select: 'top-left' | 'top-center' | 'top-right' | 'middle-left' | 'center' | 'middle-right' | 'bottom-left' | 'bottom-center' | 'bottom-right')" },
      { name: "title", type: "string", required: false, description: "Optional voting title/question" },
    ],
    handler: ({ name, votingId, options, position, title }: {
      name: string;
      votingId: string;
      options: string[];
      position: string;
      title?: string;
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
        position: position as GamePosition,
        title
      };
      return addItem("voting_panel", name, data);
    },
  });

  useCopilotAction({
    name: "createAvatarSet",
    description: "Create avatar set to display all players.",
    available: "remote",
    parameters: [
      { name: "name", type: "string", required: true, description: "Item name" },
      { name: "avatarType", type: "string", required: true, description: "Avatar type (select: 'human' | 'wolf' | 'dog' | 'cat')" },
    ],
    handler: ({ name, avatarType }: {
      name: string;
      avatarType: string;
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
        avatarType: avatarType || "human"
      };
      return addItem("avatar_set", name, data);
    },
  });

  useCopilotAction({
    name: "markPlayerDead",
    description: "Mark a player as dead, making their avatar appear grayed out.",
    available: "remote",
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
    name: "changeBackgroundColor",
    description: "Create background color control panel with multiple color options including white and dark gray.",
    available: "remote",
    parameters: [
      { name: "name", type: "string", required: true, description: "Item name" },
      { name: "backgroundColor", type: "string", required: false, description: "Initial background color (select: 'white' | 'gray-900' | 'blue-50' | 'green-50' | 'purple-50')" },
    ],
    handler: ({ name, backgroundColor }: {
      name: string;
      backgroundColor?: string;
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
        backgroundColor: backgroundColor || "white"
      };
      return addItem("background_control", name, data);
    },
  });

  useCopilotAction({
    name: "createResultDisplay",
    description: "Create a result display showing content as large artistic text with gradient colors.",
    available: "remote",
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
        position: position as GamePosition,
      };
      return addItem("result_display", name, data);
    },
  });

  // Frontend action: delete an item by id
  useCopilotAction({
    name: "deleteItem",
    description: "Delete an item by id.",
    available: "remote",
    parameters: [
      { name: "itemId", type: "string", required: true, description: "Target item id." },
    ],
    handler: ({ itemId }: { itemId: string }) => {
      const existed = (viewState.items ?? initialState.items).some((p) => p.id === itemId);
      deleteItem(itemId);
      return existed ? `deleted:${itemId}` : `not_found:${itemId}`;
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
      className="h-[calc(100vh-3.5rem)] flex flex-col"
    >
      {/* Main Layout */}
      <div className="flex flex-1 overflow-visible">
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
            {/* Chat Content - conditionally rendered to avoid duplicate rendering */}
            {isDesktop && (
              <CopilotChat
                className="flex-1 overflow-auto w-full"
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
                  <div className={cn(titleClasses, "text-2xl font-semibold")}>
                    Game Engine
                  </div>
                  <div className={cn(titleClasses, "mt-2 text-sm leading-6")}>
                    AI-Powered Game Engine
                  </div>
                </motion.div>
              )}
              
              {(viewState.items ?? []).length === 0 ? (
                <EmptyState className="flex-1">
                  <div className="mx-auto max-w-lg text-center">
                    <h2 className="text-lg font-semibold text-foreground">Ready to Play!</h2>
                    <p className="mt-2 text-sm text-muted-foreground">
                      Click Start to begin your {(() => {
                        if (typeof window === 'undefined') return 'game';
                        const params = new URLSearchParams(window.location.search);
                        const gameName = params.get('game') || 'game';
                        return gameName;
                      })()} game.
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
                          // Set gameName from URL and wait for state update
                          const gameNameUpdated = ensureGameName();
                          
                          // Wait a moment for state to propagate if needed
                          if (gameNameUpdated) {
                            await new Promise(resolve => setTimeout(resolve, 100));
                          }
                          
                          console.log('🎮 Current state gameName:', state?.gameName);
                          
                          // Send start game message
                          await appendMessage(new TextMessage({
                            role: MessageRole.User,
                            content: "Start game."
                          }));
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
                      
                      <div style={GAME_GRID_STYLE} className="pb-20 bg-white" data-canvas-container>
                      {/* Render all 9 region containers */}
                      {(["top-left", "top-center", "top-right", "middle-left", "center", "middle-right", "bottom-left", "bottom-center", "bottom-right"] as GamePosition[]).map(position => {
                        const itemsInRegion = (viewState.items ?? []).filter(item => {
                          // Exclude avatar_set items as they render as overlay
                          if (item.type === "avatar_set") return false;
                          const itemData = item.data as ItemData;
                          return (itemData as { position?: GamePosition })?.position === position;
                        });

                        return (
                          <div
                            key={position}
                            style={{ gridArea: position }}
                            className="flex flex-col items-center justify-center gap-4 p-2 rounded-lg min-h-[100px]"
                          >
                            {itemsInRegion.map((item) => (
                              <div key={item.id} className="relative group">
                                <button
                                  type="button"
                                  aria-label="Delete card"
                                  className="absolute -right-2 -top-2 z-10 inline-flex h-6 w-6 items-center justify-center rounded-full bg-red-500 text-white opacity-0 group-hover:opacity-100 hover:bg-red-600 transition-opacity"
                                  onClick={() => deleteItem(item.id)}
                                >
                                  <X className="h-3 w-3" />
                                </button>
                                
                                <CardRenderer item={item} onUpdateData={(updater) => updateItemData(item.id, updater)} onToggleTag={() => toggleTag()} onButtonClick={handleButtonClick} onVote={handleVote} playerStates={viewState.player_states} deadPlayers={viewState.deadPlayers} />
                              </div>
                            ))}
                          </div>
                        );
                      })}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
          {(viewState.items ?? []).length > 0 ? (
            <div className={cn(
              "absolute left-1/2 -translate-x-1/2 bottom-4",
              "inline-flex rounded-lg shadow-lg bg-card",
              "[&_button]:bg-card [&_button]:w-22 md:[&_button]:h-10",
              "[&_button]:shadow-none! [&_button]:hover:bg-accent",
              "[&_button]:hover:border-accent [&_button]:hover:text-accent",
              "[&_button]:hover:bg-accent/10!",
            )}>
              <Button
                type="button"
                variant="outline"
                className={cn(
                  "gap-1.25 text-base font-semibold rounded-r-none border-r-0",
                  "bg-green-50 hover:bg-green-100 border-green-200 text-green-700",
                )}
                onClick={async () => {
                  // Set gameName from URL and wait for state update
                  const gameNameUpdated = ensureGameName();
                  
                  // Wait a moment for state to propagate if needed
                  if (gameNameUpdated) {
                    await new Promise(resolve => setTimeout(resolve, 100));
                  }
                  
                  console.log('🎮 Current state gameName:', state?.gameName);
                  
                  // Send start game message
                  await appendMessage(new TextMessage({
                    role: MessageRole.User,
                    content: "Start game."
                  }));
                }}
              >
                🎮 Start
              </Button>
              <Button
                type="button"
                variant="outline"
                className={cn(
                  "gap-1.25 text-base font-semibold rounded-l-none",
                )}
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
      </div>
      <div className="md:hidden">
        {/* Mobile Chat Popup - conditionally rendered to avoid duplicate rendering */}
        {!isDesktop && (
          <CopilotPopup
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
    </div>
  );
}
