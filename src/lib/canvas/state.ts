import { AgentState, CardType, ItemData, VotingPanelData, TextDisplayData, AvatarSetData, BackgroundControlData, ResultDisplayData, TimerData, HandsCardData, ScoreBoardData, CoinDisplayData, StatementBoardData, ReactionTimerData, NightOverlayData, TurnIndicatorData, HealthDisplayData, InfluenceSetData, BroadcastInputData, PlayerStatesDisplayData, PlayerActionsDisplayData } from "@/lib/canvas/types";

export const initialState: AgentState = {
  items: [],
  lastAction: "",
  itemsCreated: 0,
  player_states: {}, // Initialize as empty object instead of undefined
  vote: [], // Initialize empty vote records array
  deadPlayers: [], // Initialize empty dead players array
  planSteps: [],
  currentStepIndex: -1,
  planStatus: "",
  gameName: undefined, // No game selected initially
  chatMessages: [], // Initialize empty chat messages array
  lastBroadcast: "",
};

export function isNonEmptyAgentState(value: unknown): value is AgentState {
  if (value == null || typeof value !== "object") return false;
  const keys = Object.keys(value as Record<string, unknown>);
  // Always consider a state with basic AgentState fields as valid
  // This prevents player_states from being lost during state transitions
  return keys.length > 0;
}

export function defaultDataFor(type: CardType): ItemData {
  switch (type) {
    case "character_card":
      return {
        role: "",
        position: "center",
        audience_type: true,
        audience_ids: [],
      };
    case "action_button":
      return {
        label: "Action",
        action: "",
        enabled: true,
        position: "center",
        audience_type: true,
        audience_ids: [],
      };
    case "phase_indicator":
      return {
        currentPhase: "",
        position: "top-center",
        audience_type: true,
        audience_ids: [],
      };
    case "text_display":
      return {
        content: "",
        position: "center",
        audience_type: true,
        audience_ids: [],
      } as TextDisplayData;
    case "voting_panel":
      return {
        votingId: "",
        title: "",
        options: [],
        position: "center",
        audience_type: true,
        audience_ids: [],
      } as VotingPanelData;
    case "avatar_set":
      return {
        avatarType: "human",
        audience_type: true,
        audience_ids: [],
      } as AvatarSetData;
    case "background_control":
      return {
        backgroundColor: "white",
        position: "center",
        audience_type: true,
        audience_ids: [],
      } as BackgroundControlData;
    case "result_display":
      return {
        content: "",
        position: "center",
        audience_type: true,
        audience_ids: [],
      } as ResultDisplayData;
    case "timer":
      return {
        duration: 5, // default 5 seconds
        label: "",
        position: "top-left",
        audience_type: true,
        audience_ids: [],
      } as TimerData;
    case "hands_card":
      return {
        cardType: "card",
        cardName: "",
        descriptions: "",
        color: "#2563eb", // default primary blue
        position: "bottom-center",
        audience_type: true,
        audience_ids: [],
      } as HandsCardData;
    case "score_board":
      return {
        title: "Scoreboard",
        entries: [],
        sort: "desc",
        accentColor: "#2563eb",
        position: "top-right",
        audience_type: true,
        audience_ids: [],
      } as ScoreBoardData;
    case "coin_display":
      return {
        currency: "gold",
        count: 1,
        accentColor: "#f59e0b", // amber-500 gold
        showLabel: false,
        position: "top-right",
        audience_type: true,
        audience_ids: [],
      } as CoinDisplayData;
    case "statement_board":
      return {
        statements: ["", "", ""],
        highlightIndex: -1,
        locked: false,
        accentColor: "#2563eb",
        position: "center",
        audience_type: true,
        audience_ids: [],
      } as StatementBoardData;
    case "reaction_timer":
      return {
        duration: 10,
        startedAt: undefined,
        running: false,
        label: "Reaction Window",
        accentColor: "#22c55e", // green-500
        position: "top-center",
        audience_type: true,
        audience_ids: [],
      } as ReactionTimerData;
    case "night_overlay":
      return {
        visible: true,
        title: "Night Phase",
        subtitle: "Secret actions in progress",
        opacity: 0.3,
        blur: false,
        position: "center",
        audience_type: true,
        audience_ids: [],
      } as NightOverlayData;
    case "turn_indicator":
      return {
        currentPlayerId: "",
        playerName: "",
        label: "Speaker",
        accentColor: "#2563eb",
        position: "top-center",
        audience_type: true,
        audience_ids: [],
      } as TurnIndicatorData;
    case "health_display":
      return {
        value: 3,
        max: 5,
        style: "hearts",
        accentColor: "#ef4444", // red-500
        position: "top-right",
        audience_type: true,
        audience_ids: [],
      } as HealthDisplayData;
    case "influence_set":
      return {
        ownerId: "",
        cards: [
          { name: "", revealed: false },
          { name: "", revealed: false },
        ],
        accentColor: "#a78bfa", // violet-300
        position: "bottom-center",
        audience_type: true,
        audience_ids: [],
      } as InfluenceSetData;
    case "broadcast_input":
      return {
        title: "Broadcast",
        placeholder: "Type a broadcast message...",
        confirmLabel: "Send",
        position: "center",
        audience_type: true,
        audience_ids: [],
      } as BroadcastInputData;
    case "player_states_display":
      return {
        title: "Player States",
        position: "middle-left",
        maxHeight: "400px",
        audience_type: true,
        audience_ids: [],
      } as PlayerStatesDisplayData;
    case "player_actions_display":
      return {
        title: "Player Actions",
        position: "middle-right", 
        maxHeight: "400px",
        maxItems: 50,
        audience_type: true,
        audience_ids: [],
      } as PlayerActionsDisplayData;
    default:
      return {
        content: "",
        position: "center",
        audience_type: true,
        audience_ids: [],
      };
  }
}
