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
  | "voting_panel"      // 投票面板
  | "avatar_set"        // 头像套装
  | "background_control" // 背景颜色控制
  | "result_display"    // 结果展示纯艺术字
  | "timer"             // 定时器
  // Future components (commented for MVP)
  // | "player_list" | "score_board" 
  // | "challenge_modal" | "elimination_display" | "game_log";

// Preset position types for game components
export type GamePosition = 
  | "top-left" | "top-center" | "top-right"
  | "middle-left" | "center" | "middle-right" 
  | "bottom-left" | "bottom-center" | "bottom-right";

// Universal size levels for all components (Agent-friendly)
export type ComponentSize = "small" | "medium" | "large";

// CSS Grid configuration for game canvas - equal proportions
export const GAME_GRID_STYLE = {
  display: 'grid',
  gridTemplateAreas: `
    "top-left top-center top-right"
    "middle-left center middle-right"
    "bottom-left bottom-center bottom-right"
  `,
  gridTemplateColumns: '1fr 1fr 1fr',  // equal proportions
  gridTemplateRows: '1fr 1fr 1fr',     // equal proportions
  gap: '1.5rem',
  minHeight: '80vh',  // Reduced to prevent overflow
  maxHeight: '80vh',  // Add max height constraint
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
}

export interface TextDisplayData {
  title?: string;      // optional title
  content: string;     // main text content
  type?: "info" | "warning" | "error" | "success";
  position: GamePosition;
}

export interface VotingPanelData {
  votingId: string;    // unique voting ID from backend
  title?: string;      // voting title/question
  options: string[];   // list of voting options
  position: GamePosition;
}

export interface AvatarSetData {
  avatarType: string; // avatar type: "human" | "wolf" | "dog" | "cat" 
}

export interface BackgroundControlData {
  backgroundColor: string; // background color options: "white" | "gray-900" | "blue-50" | "green-50" | "purple-50"
}

export interface ResultDisplayData {
  content: string; // result content to display as artistic text
  position: GamePosition;
}

export interface TimerData {
  duration: number; // timer duration in seconds
  label?: string; // optional timer label displayed above countdown (e.g., "Night Phase Timer", "Discussion Time")
  // position is fixed to top-left corner, no position parameter needed
}

export type ItemData = CharacterCardData | ActionButtonData | PhaseIndicatorData | TextDisplayData | VotingPanelData | AvatarSetData | BackgroundControlData | ResultDisplayData | TimerData;

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

export interface VoteRecord {
  voteid: string;
  playerid: string;
  option: string;
}

export interface AgentState {
  items: Item[];
  lastAction?: string;
  itemsCreated: number;
  player_states?: Record<string, Record<string, unknown>>; // Dictionary of all game players: {"1": {name: "Alice", ...}, "2": {name: "Bob", ...}}
  vote: VoteRecord[]; // Array of vote records
  deadPlayers?: string[]; // Array of dead player IDs
  planSteps?: PlanStep[];
  currentStepIndex?: number;
  planStatus?: string;
  gameName?: string; // Current game DSL name (e.g., "werewolf", "coup")
  roomSession?: Record<string, any>; // Room session data from frontend
}


