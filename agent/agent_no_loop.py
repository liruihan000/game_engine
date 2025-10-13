"""
This is the main entry point for the LLM Game Engine DM Agent.
It defines the workflow graph, game state, tools, nodes and edges for 
running diverse game types from DSL descriptions.
"""

# Apply patch for CopilotKit import issue before any other imports
# This fixes the incorrect import path in copilotkit.langgraph_agent (bug in v0.1.63)
import sys
import yaml
import os
import logging
from dotenv import load_dotenv


# Load environment variables with absolute path
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(env_path)

# Monitoring configuration
VERBOSE_LOGGING = True  # Set to False to disable detailed logging

# 直接配置 logger，不依赖 basicConfig
logger = logging.getLogger('GameAgent')
logger.handlers.clear()  # 清除现有 handlers

if VERBOSE_LOGGING:
    logger.setLevel(logging.INFO)
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    
    # 文件处理器 - 使用时间戳创建新的日志文件
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'/home/lee/canvas-with-langgraph-python/logs/agent_{timestamp}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.propagate = False  # 防止传播到root logger
    logger.info(f"Logging to: {log_file}")
else:
    logger.setLevel(logging.CRITICAL)

# Only apply the patch if the module doesn't already exist
if 'langgraph.graph.graph' not in sys.modules:
    # Create a mock module for the incorrect import path that CopilotKit expects
    class _MockModule:
        pass

    # Import the necessary modules first
    import langgraph
    import langgraph.graph
    import langgraph.graph.state

    # Import CompiledStateGraph from the correct location
    from langgraph.graph.state import CompiledStateGraph

    # Create the fake module path that CopilotKit incorrectly expects
    _mock_graph_module = _MockModule()
    _mock_graph_module.CompiledGraph = CompiledStateGraph

    # Add it to sys.modules so CopilotKit's incorrect import will work
    sys.modules['langgraph.graph.graph'] = _mock_graph_module

# Now we can safely import everything else
from typing import Any, List, Optional, Dict
from typing_extensions import Literal
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from copilotkit import CopilotKitState
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
try:
    # Prefer absolute import when package context is available
    from agent.state_ops import (
        add_item,
        update_item,
        update_item_data_merge,
        delete_item as delete_item_state,
    )
except Exception:
    # Fallback: executed as a loose module (no package); import from same dir
    import os as _os
    import sys as _sys
    _sys.path.append(_os.path.dirname(_os.path.abspath(__file__)))
    from state_ops import (
        add_item,
        update_item,
        update_item_data_merge,
        delete_item as delete_item_state,
    )
from pydantic import BaseModel, Field

class AgentState(CopilotKitState):
    """
    Game Engine DM Agent State
    
    Inherits from CopilotKitState for bidirectional frontend-backend sync.
    Contains game-specific fields for phase management, character tracking,
    event handling, and DSL-driven game logic.
    """
    tools: List[Any] = []
    # Shared state fields synchronized with the frontend
    items: List[Dict[str, Any]] = []  # Game elements (character cards, UI components, game objects)
    # Game DM state (for interactive game engine)
    current_phase_id: int = 0 
    dsl: dict = {}  # Rules DSL (loaded once)
    players: List[Dict[str, Any]] = []  # e.g., [{id, name}]
    actions: List[Dict[str, Any]] = []  # e.g., [{id, name}]

async def load_game_dsl() -> dict:
    """Load the game DSL from YAML file"""
    import asyncio
    import aiofiles
    
    try:
        dsl_path = '/home/lee/canvas-with-langgraph-python/games/simple_choice_game.yaml'
        async with aiofiles.open(dsl_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            dsl_content = yaml.safe_load(content)
        return dsl_content
    except Exception as e:
        print(f"Failed to load DSL: {e}")
        return {}





def summarize_items_for_prompt(state: AgentState) -> str:
    """Simplified for game engine - items not used for complex project management"""
    try:
        items = state.get("items", []) or []
        if not items:
            return "(no items)"
        return f"{len(items)} item(s) present"
    except Exception:
        return "(unable to summarize items)"



class CreateActionButtonArgs(BaseModel):
    name: str = Field(..., description="Item name")
    label: str = Field(..., description="Button text")
    action: str = Field(..., description="Action identifier when clicked")
    enabled: bool = Field(..., description="Whether button is clickable")
    position: str = Field(
        ..., description="Grid position (select: 'top-left' | 'top-center' | 'top-right' | 'middle-left' | 'center' | 'middle-right' | 'bottom-left' | 'bottom-center' | 'bottom-right')"
    )
    size: Optional[str] = Field(None, description="Button size (select: 'small' | 'medium' | 'large')")
    variant: Optional[str] = Field(None, description="Button style (select: 'primary' | 'secondary' | 'danger')")


@tool("createActionButton", args_schema=CreateActionButtonArgs)
def createActionButton(
    name: str,
    label: str,
    action: str,
    enabled: bool,
    position: str,
    size: Optional[str] = None,
    variant: Optional[str] = None,
) -> str:
    """Create an action button for player interactions."""
    return "queued"


class CreateCharacterCardArgs(BaseModel):
    name: str = Field(..., description="Item name")
    role: str = Field(..., description="Character role (e.g., werewolf, seer, villager)")
    position: str = Field(
        ..., description="Grid position (select: 'top-left' | 'top-center' | 'top-right' | 'middle-left' | 'center' | 'middle-right' | 'bottom-left' | 'bottom-center' | 'bottom-right')"
    )
    size: Optional[str] = Field(None, description="Card size (select: 'small' | 'medium' | 'large'; default: 'medium')")
    description: Optional[str] = Field(None, description="Optional character description")


@tool("createCharacterCard", args_schema=CreateCharacterCardArgs)
def createCharacterCard(
    name: str,
    role: str,
    position: str,
    size: Optional[str] = None,
    description: Optional[str] = None,
) -> str:
    """Create a character card for the game."""
    return "queued"

class CreatePhaseIndicatorArgs(BaseModel):
    name: str = Field(..., description="Item name")
    currentPhase: str = Field(..., description="Current game phase")
    position: str = Field(
        ..., description="Grid position (select: 'top-left' | 'top-center' | 'top-right' | 'middle-left' | 'center' | 'middle-right' | 'bottom-left' | 'bottom-center' | 'bottom-right')"
    )
    size: Optional[str] = Field(None, description="Indicator size (select: 'small' | 'medium' | 'large')")
    description: Optional[str] = Field(None, description="Optional phase description")
    timeRemaining: Optional[int] = Field(None, description="Seconds remaining in phase")


@tool("createPhaseIndicator", args_schema=CreatePhaseIndicatorArgs)
def createPhaseIndicator(
    name: str,
    currentPhase: str,
    position: str,
    size: Optional[str] = None,
    description: Optional[str] = None,
    timeRemaining: Optional[int] = None,
) -> str:
    """Create a phase indicator to show current game phase."""
    return "queued"


class CreateTextDisplayArgs(BaseModel):
    name: str = Field(..., description="Item name")
    content: str = Field(..., description="Main text content")
    position: str = Field(
        ..., description="Grid position (select: 'top-left' | 'top-center' | 'top-right' | 'middle-left' | 'center' | 'middle-right' | 'bottom-left' | 'bottom-center' | 'bottom-right')"
    )
    size: Optional[str] = Field(None, description="Display size (select: 'small' | 'medium' | 'large')")
    title: Optional[str] = Field(None, description="Optional title text")
    type: Optional[str] = Field(None, description="Display type (select: 'info' | 'warning' | 'error' | 'success')")


@tool("createTextDisplay", args_schema=CreateTextDisplayArgs)
def createTextDisplay(
    name: str,
    content: str,
    position: str,
    size: Optional[str] = None,
    title: Optional[str] = None,
    type: Optional[str] = None,
) -> str:
    """Create a text display for game information."""
    return "queued"


class DeleteItemArgs(BaseModel):
    itemId: str = Field(..., description="Target item id.")


@tool("deleteItem", args_schema=DeleteItemArgs)
def deleteItem(itemId: str) -> str:
    """Delete an item by id."""
    return "queued"


class SetItemNameArgs(BaseModel):
    name: str = Field(..., description="The new item name/title.")
    itemId: str = Field(..., description="Target item id.")


@tool("setItemName", args_schema=SetItemNameArgs)
def setItemName(itemId: str, name: str) -> str:
    """Set an item's name/title."""
    return "queued"


class SetItemSubtitleArgs(BaseModel):
    subtitle: str = Field(..., description="The new item description/subtitle.")
    itemId: str = Field(..., description="Target item id.")


@tool("setItemSubtitleOrDescription", args_schema=SetItemSubtitleArgs)
def setItemSubtitleOrDescription(itemId: str, subtitle: str) -> str:
    """Set an item's description/subtitle (short description or subtitle)."""
    return "queued"


backend_tools = [
    createActionButton,
    createCharacterCard,
    createPhaseIndicator,
    createTextDisplay,
    deleteItem,
    setItemName,
    setItemSubtitleOrDescription,
]

# Backend tool dispatcher registry (reduce per-tool boilerplate)
def _build_button_data(args: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "label": args.get("label"),
        "action": args.get("action"),
        "enabled": bool(args.get("enabled", True)),
        "position": args.get("position"),
        "size": args.get("size"),
        "variant": args.get("variant"),
    }


def _build_phase_indicator_data(args: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "currentPhase": args.get("currentPhase"),
        "position": args.get("position"),
        "size": args.get("size"),
        "description": args.get("description"),
        "timeRemaining": args.get("timeRemaining"),
    }


# text_display data builder
def _build_text_display_data(args: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "content": args.get("content"),
        "position": args.get("position"),
        "size": args.get("size"),
        "title": args.get("title"),
        "type": args.get("type"),
    }


# character_card data builder
def _build_character_card_data(args: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "role": args.get("role"),
        "position": args.get("position"),
        "size": args.get("size"),
        "description": args.get("description"),
    }


# Map tool name -> operation spec
BACKEND_TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # addItem ops（与前端一致：统一按 name 幂等）
    "createActionButton": {
        "op": "addItem",
        "item_type": "action_button",
        "build_data": _build_button_data,
    },
    "createPhaseIndicator": {
        "op": "addItem",
        "item_type": "phase_indicator",
        "build_data": _build_phase_indicator_data,
    },
    "createCharacterCard": {
        "op": "addItem",
        "item_type": "character_card",
        "build_data": _build_character_card_data,
    },
    # Optionally support createTextDisplay alias
    "createTextDisplay": {
        "op": "addItem",
        "item_type": "text_display",
        "build_data": _build_text_display_data,
    },
    # deleteItem op
    "deleteItem": {
        "op": "deleteItem",
    },
    # updateItem ops (top-level fields)
    "setItemName": {
        "op": "updateItem",
        "build_updates": lambda a: {"name": str(a.get("name", "")).strip()},
    },
    "setItemSubtitleOrDescription": {
        "op": "updateItem",
        "build_updates": lambda a: {"subtitle": a.get("subtitle")},
    },
}

# Extract tool names from backend_tools for comparison
backend_tool_names = [tool.name for tool in backend_tools]

# Frontend tool allowlist for game engine (DM tools)
FRONTEND_TOOL_ALLOWLIST = set([
    # Game component creation tools
    "createCharacterCard",
    "createActionButton", 
    "createPhaseIndicator",
    "createTextDisplay",
    # Component management tools
    "deleteItem"
])


async def ActionExecutor(state: AgentState, config: RunnableConfig) -> Command[Literal["__end__"]]:
    """
    Execute actions from state.actions by calling frontend tools.
    No loop logic - just execute and end.
    """
    logger.info(f"[ActionExecutor][start] ==== start ActionExecutor ====")

    # 1. Define the model
    model = init_chat_model("openai:gpt-4o")

    # 2. Prepare and bind frontend tools to the model
    def _extract_tool_name(tool: Any) -> Optional[str]:
        """Extract a tool name from either a LangChain tool or an OpenAI function spec dict."""
        try:
            # OpenAI tool spec dict: { "type": "function", "function": { "name": "..." } }
            if isinstance(tool, dict):
                fn = tool.get("function", {}) if isinstance(tool.get("function", {}), dict) else {}
                name = fn.get("name") or tool.get("name")
                if isinstance(name, str) and name.strip():
                    return name
                return None
            # LangChain tool object or @tool-decorated function
            name = getattr(tool, "name", None)
            if isinstance(name, str) and name.strip():
                return name
            return None
        except Exception:
            return None

    # Frontend tools may arrive either under state["tools"] or within the CopilotKit envelope
    raw_tools = (state.get("tools", []) or [])
    try:
        ck = state.get("copilotkit", {}) or {}
        raw_actions = ck.get("actions", []) or []
        if isinstance(raw_actions, list) and raw_actions:
            raw_tools = [*raw_tools, *raw_actions]
    except Exception:
        pass

    deduped_frontend_tools: List[Any] = []
    seen: set[str] = set()
    for t in raw_tools:
        name = _extract_tool_name(t)
        if not name:
            continue
        if name not in FRONTEND_TOOL_ALLOWLIST:
            continue
        if name in seen:
            continue
        seen.add(name)
        deduped_frontend_tools.append(t)

    # cap to well under 128 (OpenAI tools limit)
    MAX_FRONTEND_TOOLS = 110
    if len(deduped_frontend_tools) > MAX_FRONTEND_TOOLS:
        deduped_frontend_tools = deduped_frontend_tools[:MAX_FRONTEND_TOOLS]

    combined_tools = [ *deduped_frontend_tools]
    model_with_tools = model.bind_tools(
        combined_tools,
        parallel_tool_calls=True,  # Allow multiple tool calls in single response
    )

    # 3. Prepare system message with actions to execute
    items_summary = summarize_items_for_prompt(state)
    last_action = state.get("lastAction", "")
    actions_to_execute = state.get("actions", []) or []
    
    dsl_content = await load_game_dsl()
    current_phase_id = state.get("current_phase_id", "")
    dsl_info = f"LOADED GAME DSL:\n{dsl_content}\n" if dsl_content else "No DSL loaded.\n"
    
    game_schema = ""

    system_message = SystemMessage(
        content=(
            f"itemsState (ground truth):\n{items_summary}\n"
            f"lastAction (ground truth): {last_action}\n"
            f"Current phase: {current_phase_id}\n\n"
            "GAME DSL REFERENCE (only for understanding game flow):\n"
            f"{dsl_info}\n"
            f"{game_schema}\n"
            "ACTION EXECUTOR INSTRUCTION:\n"
            f"- You have the following actions to execute: {actions_to_execute}\n"
            "- **CRITICAL**: You MUST call ALL required tools in THIS SINGLE response. Do NOT return until you have made ALL tool calls.\n"
            "- If a referenced tool name is unavailable, SUBSTITUTE it with the closest available backend tool that achieves the same effect (use your best judgment).\n"
            "- Make MULTIPLE tool calls in this single response to complete all actions at once.\n"
            "- Use the game schema above to ensure proper data structure for each tool call.\n"
            "- After making ALL tool calls, include a brief, friendly message about what phase/stage is ready.\n"
            "- Keep the message concise and game-narrative style (e.g., 'The game scene is ready. Please choose your character.').\n"
        )
    )

    # 4. Trim messages and filter out orphaned ToolMessages
    full_messages = state.get("messages", []) or []
    
    # Debug: Log message history
    # logger.info(f"[ActionExecutor][DEBUG] Total messages: {len(full_messages)}")
    # for i, msg in enumerate(full_messages[-5:]):
    #     if isinstance(msg, AIMessage):
    #         tool_calls = getattr(msg, "tool_calls", []) or []
    #         logger.info(f"[ActionExecutor][DEBUG][{i}] AIMessage with {len(tool_calls)} tool_calls")
    #         for tc in tool_calls:
    #             tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
    #             tc_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
    #             logger.info(f"  - tool: {tc_name} (id: {tc_id})")
    #     elif isinstance(msg, ToolMessage):
    #         tc_id = getattr(msg, "tool_call_id", None)
    #         content_preview = str(getattr(msg, "content", ""))[:50]
    #         logger.info(f"[ActionExecutor][DEBUG][{i}] ToolMessage for tool_call_id={tc_id}, content={content_preview}")
    #     else:
    #         msg_type = type(msg).__name__
    #         content_preview = str(getattr(msg, "content", ""))[:50]
    #         logger.info(f"[ActionExecutor][DEBUG][{i}] {msg_type}: {content_preview}")
    
    trimmed_messages = full_messages[-30:]  # Increased to accommodate multiple tool calls
    
    # Filter out incomplete AIMessage + ToolMessage sequences
    # AIMessage with tool_calls must be followed by ALL corresponding ToolMessages
    filtered_messages = []
    i = 0
    while i < len(trimmed_messages):
        msg = trimmed_messages[i]
        
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            # Collect expected tool_call_ids
            tool_calls = getattr(msg, "tool_calls", [])
            expected_ids = {tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None) for tc in tool_calls}
            expected_ids.discard(None)
            
            # Collect following ToolMessages
            tool_messages = []
            j = i + 1
            while j < len(trimmed_messages) and (isinstance(trimmed_messages[j], ToolMessage) or getattr(trimmed_messages[j], "type", None) == "tool"):
                tool_messages.append(trimmed_messages[j])
                j += 1
            
            # Check if all expected tool_call_ids have responses
            received_ids = {getattr(tm, "tool_call_id", None) for tm in tool_messages}
            received_ids.discard(None)
            
            # Only keep if ALL tool_calls have responses
            if expected_ids and expected_ids == received_ids:
                filtered_messages.append(msg)
                filtered_messages.extend(tool_messages)
            
            i = j
        elif isinstance(msg, ToolMessage) or getattr(msg, "type", None) == "tool":
            # Orphaned ToolMessage, skip
            i += 1
        else:
            # Keep other messages (HumanMessage, SystemMessage)
            filtered_messages.append(msg)
            i += 1
    
    trimmed_messages = filtered_messages
    
    latest_state_system = SystemMessage(
        content=(
            "LATEST GROUND TRUTH (authoritative):\n"
            f"- items:\n{items_summary}\n"
            f"- lastAction: {last_action}\n"
            f"- current_phase_id: {current_phase_id}\n"
            f"- actions to execute: {actions_to_execute}\n"
        )
    )

    response = await model_with_tools.ainvoke([
        system_message,
        *trimmed_messages,
        latest_state_system,
    ], config)

    # Log output
    try:
        content_preview = getattr(response, "content", None)
        if isinstance(content_preview, str):
            logger.info(f"[ActionExecutor][OUTPUT] content: {content_preview[:400]}")
        else:
            logger.info(f"[ActionExecutor][OUTPUT] content: (non-text)")
        tool_calls = getattr(response, "tool_calls", []) or []
        if tool_calls:
            for tc in tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                logger.info(f"[ActionExecutor][TOOL_CALL] tool_call name={name} args={args}")
        else:
            logger.info("[ActionExecutor][TOOL_CALL] tool_calls: none")
    except Exception:
        pass

    # Check tool calls and handle BACKEND tools by mutating shared state directly
    try:
        tool_calls = getattr(response, "tool_calls", []) or []
    except Exception:
        tool_calls = []

    backend_calls = []
    for tc in tool_calls:
        tc_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
        if tc_name and tc_name in backend_tool_names:
            backend_calls.append(tc)

    if backend_calls:
        # Apply backend tool effects server-side so UI renders without frontend actions
        items = (state.get("items", []) or []).copy()
        try:
            # Derive counter consistently using helper
            counter = state.get("itemsCreated", 0) or 0

            last_created_id = None
            last_deleted_id = None
            last_not_found_id = None
            for tc in backend_calls:
                tc_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                if not isinstance(args, dict):
                    continue

                spec = BACKEND_TOOL_REGISTRY.get(tc_name or "")
                if not spec:
                    continue

                op = spec.get("op")
                if op == "addItem":
                    item_type = spec.get("item_type")
                    build_data = spec.get("build_data")
                    normalized = str(args.get("name", "")).strip()  # 与前端保持 name 幂等
                    data = build_data(args) if callable(build_data) else {}
                    items, counter, created_id, existing_id = add_item(
                        items=items,
                        items_created=counter,
                        item_type=str(item_type),
                        name=normalized,
                        data=data,
                    )
                    last_created_id = created_id or existing_id
                elif op == "deleteItem":
                    item_id = str(args.get("itemId", ""))
                    items, existed = delete_item_state(items, item_id)
                    if existed:
                        last_deleted_id = item_id
                    else:
                        last_not_found_id = item_id
                elif op == "updateItem":
                    item_id = str(args.get("itemId", ""))
                    build_updates = spec.get("build_updates")
                    updates = build_updates(args) if callable(build_updates) else {}
                    items, changed = update_item(items, item_id, updates)
                    # 前端 update 不写 lastAction，这里保持一致：不更新 lastAction
                elif op == "updateItemData":
                    item_id = str(args.get("itemId", ""))
                    build_patch = spec.get("build_data_patch")
                    patch = build_patch(args) if callable(build_patch) else {}
                    items, changed = update_item_data_merge(items, item_id, patch)
                    # 同上，保持与前端一致：不更新 lastAction

            return Command(
                goto=END,
                update={
                    "messages": [],
                    "items": items,
                    "itemsCreated": counter,
                    "lastAction": (
                        f"deleted:{last_deleted_id}" if last_deleted_id else (
                            f"not_found:{last_not_found_id}" if last_not_found_id else (
                                f"created:{last_created_id}" if last_created_id else state.get("lastAction", "")
                            )
                        )
                    ),
                    "current_phase_id": state.get("current_phase_id", ""),
                    "actions": [],
                    "__last_tool_guidance": None,
                },
            )
        except Exception as e:
            logger.error(f"[ActionExecutor][backend_tools] failed to apply backend tool calls: {e}")

    # No frontend tool calls, actions completed
    logger.info(f"[ActionExecutor][end] === ENDING (NO TOOLS) ===")
    return Command(
        goto=END,
        update={
            "messages": [],
            "items": state.get("items", []),
            "itemsCreated": state.get("itemsCreated", 0),
            "lastAction": state.get("lastAction", ""),
            "current_phase_id": state.get("current_phase_id", ""),
            "actions": [],  # Clear actions after execution
            "__last_tool_guidance": None,
        }
    )


async def PhaseEvaluator(state: AgentState, config: RunnableConfig) -> Command[Literal["PhaseEvaluator", "ActionExecutor", "__end__"]]:

    logger.info(f"[PhaseEvaluator][start] ==== start chatnode ====")


    # 1. Define the model
    model = init_chat_model("openai:gpt-4o")



    # 3. Define the system message by which the chat model will be run
    items_summary = summarize_items_for_prompt(state)
    post_tool_guidance = state.get("__last_tool_guidance", None)
    last_action = state.get("lastAction", "")

    
    # # Load DSL if not already loaded
    # dsl_content = state.get("dsl", {})
    # if not dsl_content:
    dsl_content = await load_game_dsl()

    current_phase_id = state.get("current_phase_id", 0)
    
    # Format DSL in readable YAML format for prompt
    if dsl_content:
        import json as _json
        dsl_info = f"LOADED GAME DSL (all phases):\n{_json.dumps(dsl_content, indent=2, ensure_ascii=False)}\n"
    else:
        dsl_info = "No DSL loaded.\n"
    
    # Get current phase details
    current_phase = dsl_content.get(current_phase_id, {}) if dsl_content else {}
    if current_phase:
        import json as _json
        current_phase_str = f"Current phase (ID {current_phase_id}):\n{_json.dumps(current_phase, indent=2, ensure_ascii=False)}\n"
    else:
        current_phase_str = f"Current phase ID: {current_phase_id} (not found in DSL)\n"

    system_message = SystemMessage(
        content=(
            # Ground truth snapshot
            f"itemsState (ground truth):\n{items_summary}\n"
            f"lastAction (ground truth): {last_action}\n"
            f"All phases: {dsl_info}\n"
            f"{current_phase_str}"

            # PhaseEvaluator-style instruction (JSON-only)
            "PHASE EVALUATION INSTRUCTION:\n"
            "- Decide if the current phase is complete according to the current phase's completion criteria and the historical operation records in trimmed_messages.\n"
            "- For player_action completion criteria: Consider BOTH button clicks AND text messages expressing the same intent as valid player actions.\n"
            "  Example: If waiting for 'select_ronin', accept messages like 'select ronin', 'choose ronin', 'I choose Ronin', etc.\n"
            "- If complete, OUTPUT JSON ONLY: {\"transition\": true, \"next_phase_id\": <number>, \"note\": <short>}\n"
            "- If not complete, OUTPUT JSON ONLY: {\"transition\": false, \"note\": <short why not>, \"actions\": [...]}\n"
            "- Actions format: [{\"description\": \"what to do\", \"tools\": [\"tool_name1\", \"tool_name2\"]}, ...]\n"
            "  Example: [{\"description\": \"Display phase indicator at top\", \"tools\": [\"phase_indicator\"]}, {\"description\": \"Show welcome text\", \"tools\": [\"text_display\"]}]\n"
        )
    )
    
    # 4. Trim long histories and filter out orphaned ToolMessages
    full_messages = state.get("messages", []) or []
    
    # Debug: Log message history
    # logger.info(f"[PhaseEvaluator][DEBUG] Total messages: {len(full_messages)}")
    # for i, msg in enumerate(full_messages[-5:]):
    #     if isinstance(msg, AIMessage):
    #         tool_calls = getattr(msg, "tool_calls", []) or []
    #         logger.info(f"[PhaseEvaluator][DEBUG][{i}] AIMessage with {len(tool_calls)} tool_calls")
    #         for tc in tool_calls:
    #             tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
    #             tc_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
    #             logger.info(f"  - tool: {tc_name} (id: {tc_id})")
    #     elif isinstance(msg, ToolMessage):
    #         tc_id = getattr(msg, "tool_call_id", None)
    #         content_preview = str(getattr(msg, "content", ""))[:50]
    #         logger.info(f"[PhaseEvaluator][DEBUG][{i}] ToolMessage for tool_call_id={tc_id}, content={content_preview}")
    #     else:
    #         msg_type = type(msg).__name__
    #         content_preview = str(getattr(msg, "content", ""))[:50]
    #         logger.info(f"[PhaseEvaluator][DEBUG][{i}] {msg_type}: {content_preview}")
    
    trimmed_messages = full_messages[-15:]  # Increased to accommodate multiple tool calls
    
    # Filter out incomplete AIMessage + ToolMessage sequences (same logic as ActionExecutor)
    filtered_messages = []
    i = 0
    while i < len(trimmed_messages):
        msg = trimmed_messages[i]
        
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            tool_calls = getattr(msg, "tool_calls", [])
            expected_ids = {tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None) for tc in tool_calls}
            expected_ids.discard(None)
            
            tool_messages = []
            j = i + 1
            while j < len(trimmed_messages) and (isinstance(trimmed_messages[j], ToolMessage) or getattr(trimmed_messages[j], "type", None) == "tool"):
                tool_messages.append(trimmed_messages[j])
                j += 1
            
            received_ids = {getattr(tm, "tool_call_id", None) for tm in tool_messages}
            received_ids.discard(None)
            
            if expected_ids and expected_ids == received_ids:
                filtered_messages.append(msg)
                filtered_messages.extend(tool_messages)
            
            i = j
        elif isinstance(msg, ToolMessage) or getattr(msg, "type", None) == "tool":
            i += 1
        else:
            filtered_messages.append(msg)
            i += 1
    
    trimmed_messages = filtered_messages

    # 5. Append a final, authoritative state snapshot after chat history
    #
    # Ensure the latest shared state takes priority over chat history and
    # stale tool results. This enforces state-first grounding, reduces drift, and makes
    # precedence explicit. Optional post-tool guidance confirms successful actions
    # (e.g., deletion) instead of re-stating absence.
    latest_state_system = SystemMessage(
        content=(
            "LATEST GROUND TRUTH (authoritative):\n"
            f"- items:\n{items_summary}\n"
            f"- lastAction: {last_action}\n"
            f"{current_phase_str}"
            "Resolution policy: If ANY prior message mentions values that conflict with the above,\n"
            "those earlier mentions are obsolete and MUST be ignored.\n"
            "When asked 'what is it now', ALWAYS read from this LATEST GROUND TRUTH.\n"
            + ("\nIf the last tool result indicated success (e.g., 'deleted:ID'), confirm the action rather than re-stating absence." if post_tool_guidance else "")
        )
    )

   

    response = await model.ainvoke([
        system_message,
        *trimmed_messages,
        latest_state_system,
    ], config)

    # Log LLM output content and planned tool calls (not just the last turn)
    try:
        content_preview = getattr(response, "content", None)
        if isinstance(content_preview, str):
            logger.info(f"[LLM][OUTPUT] content: {content_preview[:400]}")
        else:
            logger.info(f"[LLM][OUTPUT] content: (non-text)")
        tool_calls = getattr(response, "tool_calls", []) or []
        if tool_calls:
            for tc in tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                logger.info(f"[LLM][TOOL_CALL] tool_call name={name} args={args}")
        else:
            logger.info("[LLM][TOOL_CALL] tool_calls: none")
    except Exception:
        pass
    

    
    
    # Try to extract transition, next_phase_id, and actions from the model JSON content
    actions_out = []
    transition = False
    next_phase_id = state.get("current_phase_id", "")
    try:
        raw_out = str(getattr(response, "content", "")).strip()
        import json as _json
        parsed = _json.loads(raw_out) if raw_out else {}
        if isinstance(parsed, dict):
            transition = parsed.get("transition", False)
            if transition and "next_phase_id" in parsed:
                next_phase_id = parsed.get("next_phase_id")
            if not transition and isinstance(parsed.get("actions"), list):
                actions_out = parsed.get("actions")
    except Exception:
        actions_out = []
        transition = False

    # Decide routing based on transition
    if transition:
        logger.info(f"[PhaseEvaluator][end] === PHASE TRANSITION: {state.get('current_phase_id', '')} -> {next_phase_id} ===")
        return Command(
            goto="PhaseEvaluator",
            update={
                "messages": [],  # Don't render to frontend during internal phase transition
                # persist shared state keys so UI edits survive across runs
                "items": state.get("items", []),
                "itemsCreated": state.get("itemsCreated", 0),
                "lastAction": state.get("lastAction", ""),
                # persist game dm fields
                "current_phase_id": next_phase_id,
                "actions": actions_out,
                "__last_tool_guidance": None,
            }
        )
    else:
        logger.info(f"[PhaseEvaluator][end] === NO TRANSITION, GOTO ActionExecutor ===")
        return Command(
            goto="ActionExecutor",
            update={
                "messages": [],  # Don't render to frontend during internal hop
                # persist shared state keys so UI edits survive across runs
                "items": state.get("items", []),
                "itemsCreated": state.get("itemsCreated", 0),
                "lastAction": state.get("lastAction", ""),
                # persist game dm fields
                "current_phase_id": state.get("current_phase_id", ""),
                "actions": actions_out,
                "__last_tool_guidance": None,
            }
        )




def route_to_tool_node(response: BaseMessage):
    """
    Route to tool node if any tool call in the response matches a backend tool name.
    """
    tool_calls = getattr(response, "tool_calls", None)
    if not tool_calls:
        return False

    for tool_call in tool_calls:
        name = tool_call.get("name")
        if name in backend_tool_names:
            return True
    return False

# Define the workflow graph
workflow = StateGraph(AgentState)
workflow.add_node("PhaseEvaluator", PhaseEvaluator)
workflow.add_node("ActionExecutor", ActionExecutor)
# Note: tool_node is for backend tools if needed in the future
# workflow.add_node("tool_node", ToolNode(tools=backend_tools))
# workflow.add_edge("tool_node", "PhaseEvaluator")

# Graph flow (dynamic routing via Command(goto=...)):
# PhaseEvaluator -> PhaseEvaluator (if phase complete, continue to next phase)
# PhaseEvaluator -> ActionExecutor (if phase incomplete, execute actions)
# ActionExecutor -> END (after executing actions)
workflow.set_entry_point("PhaseEvaluator")

graph = workflow.compile()


# if __name__ == "__main__":
#     import asyncio
#     from langchain_core.messages import HumanMessage

#     async def main():
#         out = await graph.ainvoke({"messages": [HumanMessage(content="start game")]})
#         print(out)

#     asyncio.run(main())