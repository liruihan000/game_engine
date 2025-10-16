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

// Audience permissions for UI components
export interface AudiencePermissions {
  audience_type: boolean; // true = 所有人能看, false = 仅指定玩家能看
  audience_ids: string[]; // 当audience_type=false时，指定哪些玩家ID能看到
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
  | "hands_card"        // 手牌卡片（Hand Card for card games）
  | "score_board"       // 计分板（Score Board）
  | "coin_display"      // 硬币/金币显示（单个或多个）
  | "statement_board"   // 三句真话一假話的陈述板
  | "reaction_timer"    // 反应/挑战计时条
  | "night_overlay"     // 夜晚遮罩层（全局覆蓋）
  | "turn_indicator"    // 回合指示器（当前发言/行動者）
  | "health_display"    // 生命/子彈顯示（Bang! 等）
  | "influence_set"     // 影響力卡組（Coup 兩張）
  | "broadcast_input"   // 廣播輸入框
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

// Position normalization to handle common Agent mistakes
export function normalizePosition(position: string): GamePosition {
  const positionMap: Record<string, GamePosition> = {
    // Common mistakes -> correct positions
    'middle-center': 'center',
    'center-center': 'center',
    'middle-middle': 'center',
    'mid-center': 'center',
    'central': 'center',
    // Already correct positions (pass through)
    'top-left': 'top-left',
    'top-center': 'top-center', 
    'top-right': 'top-right',
    'middle-left': 'middle-left',
    'center': 'center',
    'middle-right': 'middle-right',
    'bottom-left': 'bottom-left',
    'bottom-center': 'bottom-center',
    'bottom-right': 'bottom-right'
  };
  
  return positionMap[position] || 'center'; // Default to center if unknown
}


// MVP Game Component Data Structures
export interface CharacterCardData extends AudiencePermissions {
  role: string;        // werewolf, seer, villager, etc.
  description?: string;
  position: GamePosition;
  size?: ComponentSize; // optional preset size
}

export interface ActionButtonData extends AudiencePermissions {
  label: string;       // button text
  action: string;      // action type when clicked
  enabled: boolean;    // whether button is clickable
  variant?: "primary" | "secondary" | "danger";
  position: GamePosition;
  size?: ComponentSize; // optional preset size
}

export interface PhaseIndicatorData extends AudiencePermissions {
  currentPhase: string;  // "night", "day", "voting", etc.
  description?: string;  // phase description
  timeRemaining?: number; // seconds left in phase
  position: GamePosition;
}

export interface TextDisplayData extends AudiencePermissions {
  title?: string;      // optional title
  content: string;     // main text content
  type?: "info" | "warning" | "error" | "success";
  position: GamePosition;
}

export interface VotingPanelData extends AudiencePermissions {
  votingId: string;    // unique voting ID from backend
  title?: string;      // voting title/question
  options: string[];   // list of voting options
  position: GamePosition;
}

export interface AvatarSetData extends AudiencePermissions {
  avatarType: string; // avatar type: "human" | "wolf" | "dog" | "cat" 
}

export interface BackgroundControlData extends AudiencePermissions {
  backgroundColor: string; // background color options: "white" | "gray-900" | "blue-50" | "green-50" | "purple-50"
  position: GamePosition; // placement in game grid
}

export interface ResultDisplayData extends AudiencePermissions {
  content: string; // result content to display as artistic text
  position: GamePosition;
}

export interface TimerData extends AudiencePermissions {
  duration: number; // timer duration in seconds
  label?: string; // optional timer label displayed above countdown (e.g., "Night Phase Timer", "Discussion Time")
  position: GamePosition; // placement in game grid
}

// Hands card data for card games
export interface HandsCardData extends AudiencePermissions {
  cardType: string;        // card classification e.g., "attack", "defense", "spell"
  cardName: string;        // display name of the card
  descriptions?: string;   // brief description or effect text
  color?: string;          // accent color (hex or tailwind token)
  position: GamePosition;  // canvas grid position
}

// Score board types
export interface ScoreBoardEntry {
  id: string;      // player id or participant id
  name: string;    // display name
  score: number;   // current score
}

export interface ScoreBoardData extends AudiencePermissions {
  title?: string;             // e.g., "Scoreboard"
  entries: ScoreBoardEntry[]; // list of scores
  sort?: "asc" | "desc";      // optional sort order
  accentColor?: string;       // accent color for header/labels
  position: GamePosition;     // placement on grid
}

// Coin (gold) display data - supports single or multiple via count
export interface CoinDisplayData extends AudiencePermissions {
  currency?: string;         // e.g., "gold" | "coin"
  count: number;             // number of coins (1 for single)
  accentColor?: string;      // coin color, default golden amber
  showLabel?: boolean;       // whether to show label text
  position: GamePosition;    // grid position
}

// Statement board for "Two Truths and a Lie"
export interface StatementBoardData extends AudiencePermissions {
  statements: string[];      // up to 3 statements
  highlightIndex?: number;   // 0..2 highlights the suspected lie or current focus
  locked?: boolean;          // when true, disables editing or indicates finalized
  accentColor?: string;      // accent for numbers/highlight
  position: GamePosition;    // grid position
}

// Reaction/challenge timer bar for quick windows
export interface ReactionTimerData extends AudiencePermissions {
  duration: number;          // seconds
  startedAt?: number;        // epoch ms when started
  running?: boolean;         // whether counting down
  label?: string;            // optional label
  accentColor?: string;      // bar color
  position: GamePosition;    // grid position
}

// Night overlay (global dimmer)
export interface NightOverlayData extends AudiencePermissions {
  visible: boolean;          // show/hide overlay
  title?: string;            // optional headline (e.g., "Night Phase")
  subtitle?: string;         // optional subline
  opacity?: number;          // 0..1 overlay darkness (default 0.5)
  blur?: boolean;            // apply backdrop blur to underlying content
  position: GamePosition;    // placement in game grid
}

// Turn indicator (current player)
export interface TurnIndicatorData extends AudiencePermissions {
  currentPlayerId: string;   // player id of active
  playerName?: string;       // optional display name
  label?: string;            // e.g., "Your Turn" / "Speaker"
  accentColor?: string;      // pill accent
  position: GamePosition;    // placement on grid
}

// Health / bullets display (Bang! etc.)
export interface HealthDisplayData extends AudiencePermissions {
  value: number;             // current value
  max?: number;              // maximum (optional)
  style?: "hearts" | "bullets"; // visual style
  accentColor?: string;      // accent color
  position: GamePosition;    // placement
}

// Coup influence set (2 cards)
export interface InfluenceCard {
  name: string;              // e.g., Duke, Assassin, Ambassador, Captain, Contessa
  revealed: boolean;         // face-up if revealed/lost
}

export interface InfluenceSetData extends AudiencePermissions {
  ownerId: string;           // player id owning the influences
  cards: InfluenceCard[];    // typically length=2
  accentColor?: string;      // border/accent
  position: GamePosition;    // placement
}

// Broadcast input component
export interface BroadcastInputData extends AudiencePermissions {
  title?: string;            // header text (e.g., "Broadcast", "Announce")
  placeholder?: string;      // input placeholder
  confirmLabel?: string;     // button text (e.g., "Send", "Broadcast")
  position: GamePosition;    // placement
}

// Room session types
export interface RoomPlayer {
  id: number;
  name: string;
  isHost: boolean;
  gamePlayerId: string;
}

export interface RoomSession {
  roomId: string;
  gameName: string;
  totalPlayers: number;
  players: RoomPlayer[];
  timestamp: number;
}

export type ItemData = CharacterCardData | ActionButtonData | PhaseIndicatorData | TextDisplayData | VotingPanelData | AvatarSetData | BackgroundControlData | ResultDisplayData | TimerData | HandsCardData | ScoreBoardData | CoinDisplayData | StatementBoardData | ReactionTimerData | NightOverlayData | TurnIndicatorData | HealthDisplayData | InfluenceSetData | BroadcastInputData;

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

export interface GameAction {
  type: 'vote' | 'night_action' | 'protect' | 'investigate' | 'eliminate';
  target?: string;    // Target player ID
  value?: string;     // Additional action data
}

export interface ChatMessage {
  id: string;
  playerId: string;
  playerName: string;
  message: string;
  timestamp: number;
  type?: 'message' | 'system' | 'action' | 'broadcast';
  // 🔒 Bot message visibility controls
  visibility?: 'public' | 'private' | 'hidden';  // Message visibility level (hidden = placeholder)
  target_audience?: string[];                     // Target audience for private messages (player IDs)
  // 🎬 Game action integration
  game_action?: GameAction;                       // Optional game action to execute
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
  roomSession?: RoomSession; // Room session data from frontend
  playerActions?: Record<string, {
    name: string;
    actions: string;
    timestamp: number;
    phase: string;
  }>; // Player action tracking by ID: {"1": {name: "Alice", actions: "voted for Bob", timestamp: 1634567890, phase: "day_voting"}}
  chatMessages?: ChatMessage[]; // Chat messages array
  lastBroadcast?: string; // Last broadcast text submitted via BroadcastInput
}
