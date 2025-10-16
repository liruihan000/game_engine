import { AgentState, CardType, ItemData, VotingPanelData, TextDisplayData, AvatarSetData, BackgroundControlData, ResultDisplayData, TimerData } from "@/lib/canvas/types";

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
      };
    case "action_button":
      return {
        label: "Action",
        action: "",
        enabled: true,
        position: "center",
      };
    case "phase_indicator":
      return {
        currentPhase: "",
        position: "top-center",
      };
    case "text_display":
      return {
        content: "",
        position: "center",
      } as TextDisplayData;
    case "voting_panel":
      return {
        votingId: "",
        title: "",
        options: [],
        position: "center",
      } as VotingPanelData;
    case "avatar_set":
      return {
        avatarType: "human",
      } as AvatarSetData;
    case "background_control":
      return {
        backgroundColor: "white",
      } as BackgroundControlData;
    case "result_display":
      return {
        content: "",
        position: "center",
      } as ResultDisplayData;
    case "timer":
      return {
        duration: 5, // default 60 seconds
        label: "",
      } as TimerData;
    default:
      return {
        content: "",
        position: "center",
      };
  }
}


