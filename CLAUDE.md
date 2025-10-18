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

1. **All comments should be English**
2. **Zero Game-Specific Code**: No `if game === "werewolf"` conditions
3. **DSL-First**: All game behavior defined in external DSL files  
4. **Atomic Tools**: Both backend operations and frontend actions are atomic
5. **Generic Prompts**: All system prompts work across diverse game types
6. **Simplest Logic**: Always use the most direct approach - prefer `setState` over helper functions when possible

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



## 添加新UI组件需要更新的位置

基于代码分析，要添加一个新的UI组件需要更新以下位置：

1. 类型定义 (src/lib/canvas/types.ts)

// 添加新的数据类型
export interface NewComponentData {
property1: string;
property2?: boolean;
position: GamePosition;
size?: ComponentSize;
}

// 更新 ItemData 联合类型
export type ItemData =
| CharacterCardData
| ActionButtonData
| PhaseIndicatorData
| TextDisplayData
| NewComponentData;  // ← 添加新类型

// 更新 CardType 联合类型
export type CardType =
| "character_card"
| "action_button"
| "phase_indicator"
| "text_display"
| "new_component";  // ← 添加新类型

1. 默认数据 (src/lib/canvas/state.ts)

export const defaultDataFor = (type: CardType): ItemData => {
switch (type) {
case "character_card": return { /* ... */ };
case "action_button": return { /* ... */ };
case "phase_indicator": return { /* ... */ };
case "text_display": return { /* ... */ };
case "new_component": return {  // ← 添加新组件默认数据
property1: "",
property2: false,
position: "center",
size: "medium"
};
default: return { position: "center" };
}
};

1. 渲染组件 (src/components/canvas/CardRenderer.tsx)

// 在 CardRenderer 函数中添加新的渲染逻辑
if (item.type === "new_component") {
const d = item.data as NewComponentData;
const getSizeClasses = (size: string = 'medium') => {
const sizeMap: Record<string, string> = {
small: "w-30 h-20",
medium: "w-50 h-30",
large: "w-70 h-40"
};
return sizeMap[size] || sizeMap.medium;
};

```
return (
  <div className={`${getSizeClasses(d.size)} bg-card border rounded-lg p-3`}>
    {/* 新组件的UI实现 */}
    <div>{d.property1}</div>
  </div>
);

```

}

1. Agent工具注册 (src/app/page.tsx)

useCopilotAction({
name: "createNewComponent",
description: "Create a new component for the game.",
available: "remote",
parameters: [
{ name: "name", type: "string", required: true, description: "Component name"
},
{ name: "property1", type: "string", required: true, description: "Property
description" },
{ name: "position", type: "string", required: true, description: "Grid
position" },
// ... 其他参数
],
handler: ({ name, property1, position, /* 其他参数 */ }: {
name: string;
property1: string;
position: string;
// ... 其他类型
}) => {
const normalized = (name ?? "").trim();

```
  // 重复性检查
  if (normalized) {
    const existing = (viewState.items ?? initialState.items).find((it) =>
      it.type === "new_component" && (it.name ?? "").trim() === normalized
    );
    if (existing) {
      return existing.id;
    }
  }

  const data: NewComponentData = {
    property1,
    position: position as GamePosition,
    // ... 其他属性
  };
  return addItem("new_component", name, data);
},

```

});

1. NewItemMenu (src/components/canvas/NewItemMenu.tsx)

// 如果需要手动创建菜单，添加新选项
const options: { id: CardType; label: string }[] = [
{ id: "character_card", label: "Character Card" },
{ id: "action_button", label: "Action Button" },
{ id: "phase_indicator", label: "Phase Indicator" },
{ id: "text_display", label: "Text Display" },
{ id: "new_component", label: "New Component" },  // ← 添加
];

🎯 总结

必须更新的核心位置 (4个):

1. 类型定义 - types.ts
2. 默认数据 - state.ts
3. 渲染组件 - CardRenderer.tsx
4. Agent工具 - page.tsx (useCopilotAction)

可选更新的位置 (2个):
5. 创建菜单 - NewItemMenu.tsx

1. Agent文件 (agent/agent_no_loop.py)

FRONTEND_TOOL_ALLOWLIST = set([
# Game component creation tools
"createCharacterCard",
"createActionButton",
"createPhaseIndicator",
"createTextDisplay",
"createNewComponent",  # ← 添加新工具
# Component management tools
"deleteItem"
])