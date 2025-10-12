# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Reusable LLM Game Engine** built on CopilotKit + LangGraph. Games are described via DSL, not coded. Only DSL rules change - everything else stays reusable.

## Development Commands

- `pnpm dev` - Runs UI (:3000) and agent (:8123) concurrently  
- `pnpm dev:agent` - Agent server only
- `pnpm dev:ui` - UI server only
- `pnpm lint` - ESLint

Setup: `echo 'OPENAI_API_KEY=your-key' > agent/.env`

## Architecture

### 1. Backend Orchestrator (`agent/agent.py`)
- **DSL-Driven**: Game rules from external files, no hardcoded game logic
- **Atomic Operations**: Generic tools that work across all games
- **Generic Reasoning**: DM agent uses only DSL + current state

### 2. Frontend UI Composer (`src/app/page.tsx`)
- **useCopilotAction Pattern**: Each UI tool exposed via individual action
```typescript
useCopilotAction({
  name: "displayVotePanel",
  description: "Show voting interface to players",
  available: "remote", 
  parameters: [
    { name: "options", type: "array", required: true },
    { name: "targetPlayers", type: "array", required: true }
  ],
  handler: ({ options, targetPlayers }) => {
    // Generic voting UI logic
  }
});
```

### 3. Game State (Generic)
```typescript
interface GameEngineState {
  phase: string                    // Current game phase
  players: Player[]               // Active players  
  gameState: Record<string, any>  // Dynamic game data
  dsl: GameDSL                   // Loaded game rules
  events: GameEvent[]            // Game events
}
```

## Key Principles

1. **Zero Game-Specific Code**: No `if game === "werewolf"` conditions
2. **DSL-First**: All game behavior defined in external DSL files  
3. **Atomic Tools**: Both backend operations and frontend actions are atomic
4. **Generic Prompts**: All system prompts work across diverse game types

## File Structure

```
/agent/agent.py          # Generic orchestrator (reusable)
/src/app/page.tsx        # UI with useCopilotAction tools (reusable)
/games/                  # Game DSL files (only part that changes)
  werewolf.yaml
  coup.yaml
```

## Frontend-Backend State Synchronization Architecture

### Core Files and Their Roles

#### 1. `/src/app/page.tsx` - State Synchronization Hub
**Key Functions:**

```typescript
// 核心状态同步 - 双向连接前后端
const { state, setState } = useCoAgent<AgentState>({
  name: "sample_agent", 
  initialState,
});

// 关键函数: updateItem - 状态更新的心脏
const updateItem = useCallback((itemId: string, updates: Partial<Item>) => {
  setState((prev) => {
    const base = prev ?? initialState;
    const items: Item[] = base.items ?? [];
    const nextItems = items.map((p) => (p.id === itemId ? { ...p, ...updates } : p));
    return { ...base, items: nextItems } as AgentState;
  });
}, [setState]);

// 工具注册 - Agent可调用的前端能力
useCopilotAction({
  name: "putCharacterCard",
  description: "Display character card on canvas",
  available: "remote",
  parameters: [
    { name: "character", type: "object", required: true }
  ],
  handler: ({ character }) => {
    updateItem(character.id, { 
      type: "character",
      data: character 
    });
  }
});

// 实时状态注入Agent
useCopilotAdditionalInstructions({
  instructions: (() => {
    const gameState = viewState;
    return `GAME STATE: phase=${gameState.phase}, players=${gameState.players?.length}`;
  })()
});

// 人机交互中断支持
useLangGraphInterrupt({
  enabled: ({ eventValue }) => eventValue?.type === "player_choice",
  render: ({ event, resolve }) => <ChoiceUI onChoice={resolve} />
});
```

#### 2. `/src/app/api/copilotkit/route.ts` - Runtime Bridge
```typescript
const runtime = new CopilotRuntime({
  agents: {
    "sample_agent": new LangGraphAgent({
      deploymentUrl: "http://localhost:8123",
      graphId: "sample_agent",
    }),
  }
});
```

#### 3. `/src/components/canvas/` - Reusable Game UI Components
- `CardRenderer.tsx` - 游戏卡片渲染
- `ItemHeader.tsx` - 元素头部
- `AppChatHeader.tsx` - 聊天头部  
- `NewItemMenu.tsx` - 创建菜单

### State Synchronization Flow

```
用户操作 → updateItem() → setState() → useCoAgent → WebSocket → Python Agent
    ↓                                                             ↓
UI更新 ← 前端重渲染 ← useCoAgent ← WebSocket ← Command(update={...})
```

**详细机制**:
1. 用户交互触发 `updateItem()`
2. `setState()` 更新本地状态
3. `useCoAgent` 自动同步到Python Agent
4. Agent处理逻辑，返回 `Command(update={...})`
5. 前端接收更新，UI立即响应

### Why updateItem is Critical

`updateItem` 是整个状态同步架构的核心：

1. **双向同步桥梁**: 唯一能同时更新前端UI和后端Agent状态
2. **工具执行基础**: 所有Agent工具最终都通过它执行状态变更
3. **一致性保障**: 确保前后端看到同一状态，避免数据竞争
4. **实时响应**: Agent决策 → updateItem → UI立即更新

```typescript
// 没有updateItem，就没有:
// - 实时游戏状态更新
// - Agent工具的UI反馈  
// - 玩家操作的状态同步
// - 多步游戏流程的连续执行
```

## Architecture Components

### 1. Backend Game Orchestrator (`agent/agent.py`)
- **DSL-Driven**: Game rules from external files
- **Generic Tools**: Atomic operations work across games
- **State Sync**: Uses CopilotKitState for bidirectional sync

### 2. Frontend UI Composer (`src/app/page.tsx`)
- **State Hub**: useCoAgent manages shared state
- **Tool Registry**: useCopilotAction exposes capabilities
- **Component Library**: Atomic UI components for games

### 3. Game State (Synchronized)
```typescript
interface GameEngineState {
  phase: string                    // Current game phase
  players: Player[]               // Active players  
  gameState: Record<string, any>  // Dynamic game data
  characters: Character[]         // Game characters
  events: GameEvent[]            // Game event history
  dsl: GameDSL                   // Loaded game rules
}
```

## Key Integration Functions

```typescript
// 状态同步核心
useCoAgent<GameState>()           // 双向状态同步
updateItem(id, updates)           // 状态更新心脏  
setState(updater)                 // 本地状态管理

// Agent工具系统
useCopilotAction()               // 注册游戏工具
useLangGraphInterrupt()          // 人机交互支持
useCopilotAdditionalInstructions() // 实时上下文注入

// 组件渲染
CardRenderer                     // 游戏元素渲染
updateItemData()                 // 数据层更新
```

## useCopilotAction 工作机制

**所有的 useCopilotAction 都通过相同机制工作**：

### 1. 核心执行流程
```typescript
// Agent调用工具
Agent.call("toolName", { params })

// useCopilotAction handler被触发  
useCopilotAction({
  name: "toolName",
  handler: ({ params }) => {
    setState((prev) => ({ ...prev, fieldName: newValue }));
  }
});

// 状态更新链路
setState → state更新 → viewState自动更新 → React重新渲染
```

### 2. viewState渲染机制
```typescript
// viewState是state的实时映射
const viewState: AgentState = isNonEmptyAgentState(state) ? 
  (state as AgentState) : cachedStateRef.current;

// 渲染中使用viewState
<input value={viewState?.globalTitle ?? initialState.globalTitle} />
{(viewState.items ?? []).map(item => <CardRenderer item={item} />)}
```

### 3. 简单字段 vs 复杂结构
**简单字段** (如 globalTitle):
- `setState` → 直接更新单个字段
- 渲染: 直接读取 `viewState.fieldName`

**复杂结构** (如 items数组):
- `setState` → 数组操作 (添加、删除、更新)
- 渲染: 遍历、过滤、映射等复杂操作

### 4. 双向同步保证
- **前端→后端**: setState触发useCoAgent同步到Python Agent
- **后端→前端**: Agent状态变更自动更新到前端viewState  
- **一致性**: 确保前后端始终看到相同的状态

## Development Protocol

**⚠️ IMPORTANT: All code changes must be discussed and approved before implementation**

Before making any modifications:
1. Explain the design logic and reasoning
2. Get approval from the user  
3. Then proceed with implementation

This ensures architectural consistency and prevents breaking changes to the reusable game engine.

## Success Criteria

- Add new game with only DSL file changes
- >90% game completion rate across game types  
- Real-time bidirectional state synchronization
- Seamless Agent tool execution and UI feedback