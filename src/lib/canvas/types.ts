export interface ChecklistItem {
  id: string;
  text: string;
  done: boolean;
  proposed: boolean;
}

export interface LinkItem {
  title: string;
  url: string;
}

export type CardType = 
  // MVP Game Components Only
  | "character_card"    // 角色卡
  | "action_button"     // 按钮
  | "phase_indicator"   // 阶段指示器
  | "text_display"      // 文本展示板
  // Future components (commented for MVP)
  // | "voting_panel" | "timer" | "player_list" | "score_board" 
  // | "challenge_modal" | "elimination_display" | "game_log";

// Preset position types for game components
export type GamePosition = 
  | "top-left" | "top-center" | "top-right"
  | "middle-left" | "center" | "middle-right" 
  | "bottom-left" | "bottom-center" | "bottom-right";

// Universal size levels for all components (Agent-friendly)
export type ComponentSize = "small" | "medium" | "large";

// CSS Grid configuration for game canvas - center area is larger
export const GAME_GRID_STYLE = {
  display: 'grid',
  gridTemplateAreas: `
    "top-left top-center top-right"
    "middle-left center middle-right"
    "bottom-left bottom-center bottom-right"
  `,
  gridTemplateColumns: '1fr 2fr 1fr',  // center column is 2x larger
  gridTemplateRows: '1fr 2fr 1fr',     // center row is 2x larger
  gap: '1.5rem',
  minHeight: '100vh',
  padding: '1.5rem'
};


// MVP Game Component Data Structures
export interface CharacterCardData {
  role: string;        // werewolf, seer, villager, etc.
  description?: string;
  position: GamePosition;
  size?: ComponentSize; // optional preset size
}

export interface ActionButtonData {
  label: string;       // button text
  action: string;      // action type when clicked
  enabled: boolean;    // whether button is clickable
  variant?: "primary" | "secondary" | "danger";
  position: GamePosition;
  size?: ComponentSize; // optional preset size
}

export interface PhaseIndicatorData {
  currentPhase: string;  // "night", "day", "voting", etc.
  description?: string;  // phase description
  timeRemaining?: number; // seconds left in phase
  position: GamePosition;
  size?: ComponentSize; // optional preset size
}

export interface TextDisplayData {
  title?: string;      // optional title
  content: string;     // main text content
  type?: "info" | "warning" | "error" | "success";
  position: GamePosition;
  size?: ComponentSize; // optional preset size
}

export type ItemData = CharacterCardData | ActionButtonData | PhaseIndicatorData | TextDisplayData;

export interface Item {
  id: string;
  type: CardType;
  name: string; // editable title
  subtitle: string; // subtitle shown under the title
  data: ItemData;
}

export interface PlanStep {
  title: string;
  status: "pending" | "in_progress" | "completed" | "blocked" | "failed";
  note?: string;
}

export interface AgentState {
  items: Item[];
  lastAction?: string;
  itemsCreated: number;
}


