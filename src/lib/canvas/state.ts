import { AgentState, CardType, ItemData } from "@/lib/canvas/types";

export const initialState: AgentState = {
  items: [],
  lastAction: "",
  itemsCreated: 0,
  planSteps: [],
  currentStepIndex: -1,
  planStatus: "",
  
};

export function isNonEmptyAgentState(value: unknown): value is AgentState {
  if (value == null || typeof value !== "object") return false;
  const keys = Object.keys(value as Record<string, unknown>);
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
      };
    default:
      return {
        content: "",
        position: "center",
      };
  }
}


