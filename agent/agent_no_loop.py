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
import json as _json


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
    player_states: Dict[str, Dict[str, Any]] = {}  # Player states keyed by player ID (e.g., {"1": {name: "Alice", ...}, "2": {name: "Bob", ...}})
    # Game DM state (for interactive game engine)
    current_phase_id: int = 0 
    dsl: dict = {}  # Rules DSL (loaded once)
    actions: List[Dict[str, Any]] = []  # e.g., [{id, name}]
    gameName: str = ""  # Current game DSL name (e.g., "werewolf", "coup")
    deadPlayers: List[str] = []  # List of dead player IDs
    vote: List[Dict[str, Any]] = []  # List of vote records

async def load_game_dsl() -> dict:
    """Deprecated default DSL loader.

    This agent now uses gameName-based DSL resolution and should not fall back
    to a hardcoded default game. Returning empty dict to indicate 'no DSL'.
    """
    logger.info("[DSL] load_game_dsl() called but default loading is disabled; returning empty DSL")
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


async def initialize_player_states_from_dsl(dsl_content: dict, room_players: list) -> dict:
    """
    Initialize player_states from DSL template, similar to initialize-players API.
    
    Args:
        dsl_content: The loaded DSL content
        room_players: List of players from room (e.g., [{"name": "Alice"}, {"name": "Bob"}])
        
    Returns:
        Dictionary of initialized player_states
    """
    try:
        # Defense mechanism 1: Check if player_states_template exists
        template = {}
        
        if dsl_content and 'declaration' in dsl_content and 'player_states_template' in dsl_content['declaration']:
            player_states_template = dsl_content['declaration']['player_states_template']
            if 'player_states' in player_states_template:
                # Try to get template with ID "1"
                template = player_states_template['player_states'].get('1', {})
                
                # Defense mechanism 2: If ID "1" doesn't exist, try first available ID
                if not template:
                    available_ids = list(player_states_template['player_states'].keys())
                    if available_ids:
                        template = player_states_template['player_states'][available_ids[0]]
        
        # Defense mechanism 3: If template still not found, auto-generate from player_states definition
        if not template and dsl_content and 'declaration' in dsl_content and 'player_states' in dsl_content['declaration']:
            logger.info('🛡️ No template found, auto-generating from player_states definition')
            player_states_schema = dsl_content['declaration']['player_states']
            
            # Generate default values based on field types
            for field_name, field_def in player_states_schema.items():
                field_type = field_def.get('type', 'string')
                default_value = field_def.get('example')
                
                if field_type == 'string':
                    template[field_name] = default_value or ''
                elif field_type in ['num', 'number']:
                    template[field_name] = default_value or 0
                elif field_type == 'boolean':
                    template[field_name] = default_value if default_value is not None else True
                elif field_type in ['array', 'list']:
                    template[field_name] = default_value or []
                elif field_type in ['object', 'dict']:
                    template[field_name] = default_value or {}
                else:
                    template[field_name] = default_value or None
        
        # Final fallback: If we still have no template, return empty state
        if not template or len(template) == 0:
            logger.info('🛡️ No template available, returning empty player_states')
            return {}
        
        # Create initialized player_states with real players
        initialized_players = {}
        
        for index, player in enumerate(room_players):
            player_id = str(index + 1)
            initialized_players[player_id] = {
                **template,  # Copy all template fields
                'name': player.get('name', f'Player {player_id}'),  # Replace with real player name
            }
        
        return initialized_players
        
    except Exception as e:
        logger.error(f"Error initializing player_states from DSL: {e}")
        return {}

backend_tools = []

# Extract tool names from backend_tools for comparison
backend_tool_names = [tool.name for tool in backend_tools]

# Frontend tool allowlist for game engine (DM tools)
FRONTEND_TOOL_ALLOWLIST = set([
    # Game component creation tools
    "createCharacterCard",
    "createActionButton", 
    "createPhaseIndicator",
    "createTextDisplay",
    "createVotingPanel",
    "createAvatarSet",
    "changeBackgroundColor",
    "createResultDisplay",
    "createTimer",
    # Component management tools
    "deleteItem",
    "clearCanvas",
    # Player state management
    "markPlayerDead"
])


async def ActionExecutor(state: AgentState, config: RunnableConfig) -> Command[Literal["__end__"]]:
    """
    Execute actions from state.actions by calling frontend tools.
    No loop logic - just execute and end.
    NOTE: This node does NOT evaluate or perform phase transitions.
    All transition decisions must be made in PhaseEvaluator to avoid
    duplicate or invalid hops. next_phase_id is intentionally ignored here.
    """
    logger.info(f"[ActionExecutor][start] ==== start ActionExecutor ====")
    try:
        logger.info(f"[ActionExecutor][game] gameName={state.get('gameName', None)}")
    except Exception:
        pass
    
    # Debug: Print entire received state
    logger.info(f"[ActionExecutor][DEBUG] Full received state keys: {list(state.keys())}")
    if "player_states" in state:
        logger.info(f"[ActionExecutor][DEBUG] Received player_states: {state.get('player_states', {})}")
    else:
        logger.info(f"[ActionExecutor][DEBUG] NO player_states in received state")

    # 1. Define the model
    model = init_chat_model("openai:gpt-4.1")

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

    model_with_tools = model.bind_tools(
        deduped_frontend_tools,
        parallel_tool_calls=True,  # Allow multiple tool calls in single response
    )

    # 3. Prepare system message with actions to execute
    items_summary = summarize_items_for_prompt(state)
    last_action = state.get("lastAction", "")
    actions_to_execute = state.get("actions", []) or []
    
    # Load DSL based on gameName in state (same as PhaseEvaluator)
    current_game_name = state.get('gameName')
    logger.info(f"[ActionExecutor][DSL] Current game name: {current_game_name}")
    
    dsl_content = {}
    if current_game_name:
        try:
            import aiofiles
            dsl_file_path = os.path.join(os.path.dirname(__file__), '..', 'games', f"{current_game_name}.yaml")
            if os.path.exists(dsl_file_path):
                async with aiofiles.open(dsl_file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    dsl_content = yaml.safe_load(content)
                logger.info(f"[ActionExecutor][DSL] Successfully loaded DSL for game: {current_game_name}")
            else:
                logger.warning(f"[ActionExecutor][DSL] DSL file not found: {dsl_file_path}")
        except Exception as e:
            logger.error(f"[ActionExecutor][DSL] Failed to load DSL for {current_game_name}: {e}")
    
    # No fallback: require gameName-based DSL; keep empty if unavailable
    if not dsl_content:
        logger.warning("[ActionExecutor][DSL] No DSL loaded (missing or invalid gameName). Proceeding without DSL context.")
    
    current_phase_id = state.get("current_phase_id", 0)
    dsl_info = f"LOADED GAME DSL:\n{dsl_content}\n" if dsl_content else "No DSL loaded.\n"
    
    # Get current phase details from phases (same as PhaseEvaluator)
    phases = dsl_content.get('phases', {}) if dsl_content else {}
    current_phase = phases.get(current_phase_id, {}) if phases else {}
    
    # Print current phase details
    if current_phase:
        logger.info(f"[ActionExecutor][DSL] Current phase ID: {current_phase_id}")
        logger.info(f"[ActionExecutor][DSL] Current phase: {current_phase}")
    else:
        logger.info(f"[ActionExecutor][DSL] Current phase ID: {current_phase_id} (not found in DSL phases)")
    if current_phase:
        current_phase_str = f"Current phase (ID {current_phase_id}):\n{current_phase}\n"
    else:
        current_phase_str = f"Current phase ID: {current_phase_id} (not found in DSL phases)\n"
    
    game_schema = ""

    system_message = SystemMessage(
        content=(
            f"itemsState (ground truth):\n{items_summary}\n"
            f"lastAction (ground truth): {last_action}\n"
            f"{current_phase_str}\n"
            "GAME DSL REFERENCE (only for understanding game flow):\n"
            f"{dsl_info}\n"
            f"{game_schema}\n"
            "ACTION EXECUTOR INSTRUCTION:\n"
            f"- You have the following actions to execute: {actions_to_execute}\n"
            "- **CRITICAL**: You MUST call ALL required tools in THIS SINGLE response. Do NOT return until you have made ALL tool calls.\n"
            "- For each action in the list, call its specified tools. If an action has tools: [A, B], you must call both A and B.\n"
            "- Make MULTIPLE tool calls in this single response to complete all actions at once.\n"
            "- Use the game schema above to ensure proper data structure for each tool call.\n"
            "- After making ALL tool calls, include a brief, friendly message about what phase/stage is ready.\n"
            "- Keep the message concise and game-narrative style (e.g., 'The game scene is ready. Please choose your character.').\n"
            "TOOL USAGE RULES:\n"
            "- Tool names must match exactly (no 'functions.' prefix).\n"
            "- When you create items (e.g., via `createTextDisplay`), capture the returned id (e.g., '0002') and reuse it in later calls that require 'itemId' (e.g., 'deleteItem'). Do not pass the item name; always pass the exact id.\n"
            "- To delete multiple items, call 'deleteItem' once per 'itemId'.\n"
            "LAYOUT PLACEMENT RULES:\n"
            "- Prefer placing text tools (createTextDisplay, createResultDisplay) at grid position 'center' by default.\n"
            "- If there are already 4 or more items on canvas, distribute subsequent items across: 'top-center', 'bottom-center', 'middle-left', 'middle-right', then 'top-left', 'top-right', 'bottom-left', 'bottom-right'.\n"
            "- Keep related elements sensible (e.g., phase indicator near top-center; voting panels center/bottom-center).\n"
            f"- Total tools to call this turn: {sum(len(action.get('tools', [])) for action in actions_to_execute if isinstance(action, dict))}\n"
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

    # Check if this response contains frontend tool calls that need to be delivered to the client
    try:
        tool_calls = getattr(response, "tool_calls", []) or []
    except Exception:
        tool_calls = []
    
    # has_frontend_tool_calls = False
    # for tc in tool_calls:
    #     name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
    #     if name and name not in backend_tool_names:
    #         has_frontend_tool_calls = True
    #         break

    # # If the model produced FRONTEND tool calls, deliver them to the client and stop the turn.
    # # The client will execute and post ToolMessage(s), after which the next run can resume.
    # if has_frontend_tool_calls:
    #     logger.info(f"[ActionExecutor][end] === ENDING WITH FRONTEND TOOLS ===")
    #     return Command(
    #         goto=END,
    #         update={
    #             "messages": [response],
    #             "items": state.get("items", []),
    #             "itemsCreated": state.get("itemsCreated", 0),
    #             "lastAction": state.get("lastAction", ""),
    #             "current_phase_id": state.get("current_phase_id", ""),
    #             "actions": state.get("actions", []),  # Keep actions until tools are executed
    #             "__last_tool_guidance": (
    #                 "Frontend tool calls issued. Waiting for client tool results before continuing."
    #             ),
    #         }
    #     )

    # No frontend tool calls, actions completed
    logger.info(f"[ActionExecutor][end] === ENDING (NO TOOLS) ===")
    return Command(
        goto=END,
        update={
            "messages": [response],
            "items": state.get("items", []),
            "itemsCreated": state.get("itemsCreated", 0),
            "lastAction": state.get("lastAction", ""),
            "player_states": state.get("player_states", {}),  # Persist player_states
            "current_phase_id": state.get("current_phase_id", 0),
            "actions": [],  # Clear actions after execution
            "__last_tool_guidance": None,
        }
    )


async def PhaseEvaluator(state: AgentState, config: RunnableConfig) -> Command[Literal["PhaseEvaluator", "ActionExecutor", "__end__"]]:

    logger.info(f"[PhaseEvaluator][start] ==== start chatnode ====")
    try:
        logger.info(f"[PhaseEvaluator][game] gameName={state.get('gameName', None)}")
    except Exception:
        pass
    
    # Debug: Print entire received state
    logger.info(f"[PhaseEvaluator][DEBUG] Full received state keys: {list(state.keys())}")
    if "player_states" in state:
        logger.info(f"[PhaseEvaluator][DEBUG] Received player_states: {state.get('player_states', {})}")
    else:
        logger.info(f"[PhaseEvaluator][DEBUG] NO player_states in received state")
    
    if "vote" in state:
        logger.info(f"[PhaseEvaluator][DEBUG] Received votes: {state.get('vote', [])}")
    else:
        logger.info(f"[PhaseEvaluator][DEBUG] NO votes in received state")
    
    if "items" in state:
        logger.info(f"[PhaseEvaluator][DEBUG] Received items: {state.get('items', [])}")
    else:
        logger.info(f"[PhaseEvaluator][DEBUG] NO items in received state")

    # 1. Define the model
    model = init_chat_model("openai:gpt-4.1")



    # 3. Define the system message by which the chat model will be run
    items_summary = summarize_items_for_prompt(state)
    post_tool_guidance = state.get("__last_tool_guidance", None)
    last_action = state.get("lastAction", "")

    
    # Load DSL based on gameName in state
    current_game_name = state.get('gameName')
    logger.info(f"[PhaseEvaluator][DSL] Current game name: {current_game_name}")
    
    dsl_content = {}
    if current_game_name:
        try:
            import aiofiles
            dsl_file_path = os.path.join(os.path.dirname(__file__), '..', 'games', f"{current_game_name}.yaml")
            if os.path.exists(dsl_file_path):
                async with aiofiles.open(dsl_file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    dsl_content = yaml.safe_load(content)
                logger.info(f"[PhaseEvaluator][DSL] Successfully loaded DSL for game: {current_game_name}")
            else:
                logger.warning(f"[PhaseEvaluator][DSL] DSL file not found: {dsl_file_path}")
        except Exception as e:
            logger.error(f"[PhaseEvaluator][DSL] Failed to load DSL for {current_game_name}: {e}")
    
    # No fallback: require gameName-based DSL; keep empty if unavailable
    if not dsl_content:
        logger.warning("[PhaseEvaluator][DSL] No DSL loaded (missing or invalid gameName). Proceeding without DSL context.")
    # 当前阶段 ID：默认从 0 开始（DSL phases 使用 0 起始的递增整数键）
    current_phase_id = state.get("current_phase_id", 0)
    # 归一化：若为数字字符串，转换为 int，防止后续 dict 查找键类型不匹配
    if isinstance(current_phase_id, str) and current_phase_id.isdigit():
        try:
            current_phase_id = int(current_phase_id)
        except Exception:
            pass
    
    # Initialize player_states if empty and we have DSL
    current_player_states = state.get("player_states", {})
    if not current_player_states and dsl_content:
        logger.warning("[PhaseEvaluator] player_states is empty but DSL is loaded. This should not happen if initialize-players API was called correctly.")
        logger.info("[PhaseEvaluator] Skipping player_states initialization - expecting frontend to provide initialized player_states")
    else:
        logger.info(f"[PhaseEvaluator] Using existing player_states: {len(current_player_states)} players")
    
    # Format DSL in readable YAML format for prompt
    if dsl_content:
        dsl_info = f"LOADED GAME DSL (all phases):{dsl_content}"
    else:
        dsl_info = "No DSL loaded.\n"
    
    # Get current phase details from phases (not declaration.phases)
    phases = dsl_content.get('phases', {}) if dsl_content else {}
    current_phase = phases.get(current_phase_id, {}) if phases else {}
    # Frontend-available tools (allowlisted) with brief descriptions for the model (static JSON text)
    game_tool_str = (
        '{\n'
        '  "createCharacterCard": "Create a character card (role, position, optional size/desc).",\n'
        '  "createActionButton": "Create an action button (label, action id, enabled, position).",\n'
        '  "createPhaseIndicator": "Show current phase indicator (phase, position, optional desc/timer).",\n'
        '  "createTextDisplay": "Text panel for info (content, optional title/type, position).",\n'
        '  "createVotingPanel": "Voting panel (votingId, options, position, optional title).",\n'
        '  "createAvatarSet": "Overlay of player avatars around the canvas (avatarType).",\n'
        '  "changeBackgroundColor": "Create background control and set initial color/theme.",\n'
        '  "createResultDisplay": "Large gradient-styled result banner (content, position).",\n'
        '  "createTimer": "Countdown timer that expires and notifies the agent.",\n'
        '  "deleteItem": "Delete a canvas item by id.",\n'
        '  "clearCanvas": "Clear all canvas items except avatar sets (phase reset).",\n'
        '  "markPlayerDead": "Mark a player as dead (grays out their avatar)."\n'
        '}\n'
    )
    
    # Print current phase details
    if current_phase:
        logger.info(f"[PhaseEvaluator][DSL] Current phase ID: {current_phase_id}")
        logger.info(f"[PhaseEvaluator][DSL] Current phase: {current_phase}")
        current_phase_str = f"Current phase (ID {current_phase_id}):\n{current_phase}\n"
    else:
        logger.info(f"[PhaseEvaluator][DSL] Current phase ID: {current_phase_id} (not found in DSL phases)")
        current_phase_str = f"Current phase ID: {current_phase_id} (not found in DSL)\n"
    logger.info(f"[PhaseEvaluator][DSL] Current phase str: {current_phase_str}")

    system_message = SystemMessage(
        content=(
            # Ground truth snapshot
            f"itemsState (ground truth):\n{items_summary}\n"
            f"lastAction (ground truth): {last_action}\n"
            f"All phases: {dsl_info}\n"
            f"{current_phase_str}"
            f"game_tool:\n{game_tool_str}\n"

            # PhaseEvaluator-style instruction (JSON-only)
            "PHASE EVALUATION INSTRUCTION:\n"
            "STEP 1: PLAYER STATES - First, check if any player states need updating based on game events and messages.\n"
            "BOT AUTOMATION RULES:\n"
            "- Player \"1\" is the ONLY HUMAN player - update only based on their actual messages/actions\n" 
            "- ALL other players (\"2\", \"3\", \"4\", etc.) are BOTS - automatically simulate their behavior\n"
            "- For EVERY bot player in EVERY phase, you MUST simulate appropriate decisions and update their player_states accordingly\n"
            "- Bot behavior examples: vote for random players, use night abilities, make game decisions\n"
            "- To update player states, include \"player_states_updates\" in your JSON response with complete player state objects.\n"
            "- IMPORTANT: Return the COMPLETE player state object for each player, but only update the specific field values that have changed based on game events. Keep all other existing field values unchanged.\n"
            "- Format: \"player_states_updates\": {\"1\": {\"name\": \"Alpha\", \"role\": \"Werewolf\", \"is_alive\": true, \"vote\": \"Beta\"}, \"2\": {\"name\": \"Beta\", \"role\": \"Detective\", \"is_alive\": true, \"vote\": \"Alpha\"}, \"3\": {\"name\": \"Gamma\", \"role\": \"Villager\", \"is_alive\": false, \"vote\": \"\"}}\n"
            "- Only existing keys can be updated, no new keys will be added. Each player object must include all existing fields for that player.\n"
            "- Use exact player numeric IDs as keys (\"1\", \"2\", \"3\", etc.) and be very strict about only updating values that have actually changed due to game events.\n"
            "STEP 2: PHASE EVALUATION - Decide if the current phase is complete according to the completion criteria.\n"
            "- For player_action completion criteria: Consider BOTH button clicks AND text messages expressing the same intent as valid player actions.\n"
            "- Example: If waiting for voting phase completion with \"wait_for: all_players_action\" and \"condition: player.is_alive == true\":\n"
            "  - Accept button clicks on voting UI\n"
            "  - Accept text messages like 'I vote for Alpha', 'vote Beta', 'eliminate Gamma', 'my vote goes to Alpha'\n"
            "  - Check that ALL surviving players have cast their votes and relevant player_states.vote fields have been updated\n"
            "- IMPORTANT: You must ONLY choose actions from the current phase's actions list. Do NOT execute next-phase actions early.\n"
            "- IMPORTANT: Only when the current phase's completion criteria are satisfied should you set \"transition\": true with a VALID \"next_phase_id\" (existing phase id).\n"
            
            "STEP 3: OUTPUT FORMAT - Provide JSON response:\n"
            "- If complete, OUTPUT JSON ONLY: {\"transition\": true, \"next_phase_id\": <number>, \"note\": <short>, \"player_states_updates\": {...}}\n"
            "- If not complete, OUTPUT JSON ONLY: {\"transition\": false, \"note\": <short why not>, \"actions\": [...], \"player_states_updates\": {...}}\n"
            "- Actions format: [{\"description\": \"what to do\", \"tools\": [\"tool_name1\", \"tool_name2\"]}, ...]\n"
            "  Example: [{\"description\": \"Create and display voting UI for all surviving players\", \"tools\": [\"createActionButton\"]}, {\"description\": \"Show current phase indicator\", \"tools\": [\"createPhaseIndicator\"]}]\n"
            "- If next_phase is null, set transition=false and DO NOT include next_phase_id.\n"
            "- IMPORTANT: If there is NO valid next phase, set transition=false and DO NOT include next_phase_id.\n"
            "- You MUST execute ALL tools listed in the current phase's actions, in this turn. Also consider any extra actions that improve UX (e.g., timers, UI cleanup) and include them.\n"
            "- In most scenes, clear the canvas before laying out new UI. Include a 'clearCanvas' step as the first action when appropriate.\n"
            "EXAMPLES1 \n"
            "1) Phase 0 — Game Introduction (transition=false; execute all actions)\n"
            "{\n"
            "  \"transition\": false,\n"
            "  \"note\": \"Display introduction, rules, win conditions; prepare next\",\n"
            "  \"actions\": [\n"
            "    { \"description\": \"Reset scene for a clean intro\", \"tools\": [\"clearCanvas\"] },\n"
            "    { \"description\": \"Global title & description\", \"tools\": [\"setGlobalTitle\", \"setGlobalDescription\"] },\n"
            "    { \"description\": \"Background control and theme\", \"tools\": [\"createBackgroundControl\", \"changeBackgroundColor\"] },\n"
            "    { \"description\": \"Show player avatars\", \"tools\": [\"createAvatarSet\"] },\n"
            "    { \"description\": \"Show current phase indicator\", \"tools\": [\"createPhaseIndicator\"] },\n"
            "    { \"description\": \"Show game rules overview\", \"tools\": [\"createTextDisplay\"] },\n"
            "    { \"description\": \"Show win conditions\", \"tools\": [\"createTextDisplay\"] }\n"
            "  ],\n"
            "  \"player_states_updates\": {}\n"
            "}\n"
            "EXAMPLES2 \n"
            "2) Phase 0 complete → transition to Phase 1 (Role Assignment)\n"
            "{\n"
            "  \"transition\": true,\n"
            "  \"next_phase_id\": 1,\n"
            "  \"note\": \"Introduction displayed; moving to Role Assignment\",\n"
            "  \"player_states_updates\": {}\n"
            "}\n"
            "EXAMPLES3 \n"
            "3) Phase 2 — Night — Werewolves Choose Target (transition=false; prepare night UI)\n"
            "{\n"
            "  \"transition\": false,\n"
            "  \"note\": \"Waiting for werewolves to choose a target\",\n"
            "  \"actions\": [\n"
            "    { \"description\": \"Reset scene for night\", \"tools\": [\"clearCanvas\"] },\n"
            "    { \"description\": \"Night ambiance\", \"tools\": [\"changeBackgroundColor\"] },\n"
            "    { \"description\": \"Phase indicator\", \"tools\": [\"createPhaseIndicator\"] },\n"
            "    { \"description\": \"Night instructions for werewolves\", \"tools\": [\"createTextDisplay\"] },\n"
            "    { \"description\": \"Optional: short timer to limit the night\", \"tools\": [\"createTimer\"] }\n"
            "  ],\n"
            "  \"player_states_updates\": {}\n"
            "}\n"

            "EXAMPLES4 \n"
            "4) Phase 5 — Dawn — Night Resolution and Reveal (transition=false; resolve and reveal)\n"
            "{\n"
            "  \"transition\": false,\n"
            "  \"note\": \"Reveal night results to all players\",\n"
            "  \"actions\": [\n"
            "    { \"description\": \"Reset scene for dawn\", \"tools\": [\"clearCanvas\"] },\n"
            "    { \"description\": \"Dawn ambiance\", \"tools\": [\"changeBackgroundColor\"] },\n"
            "    { \"description\": \"Phase indicator\", \"tools\": [\"createPhaseIndicator\"] },\n"
            "    { \"description\": \"Mark dead player if any and show result\", \"tools\": [\"markPlayerDead\", \"createResultDisplay\", \"createTextDisplay\"] }\n"
            "  ],\n"
            "  \"player_states_updates\": {}\n"
            "}\n"
            "EXAMPLES5 \n"
            "5) Phase 99 — Game Over (next_phase=null → transition=false)\n"
            "{\n"
            "  \"transition\": false,\n"
            "  \"note\": \"Game Over — display final results\",\n"
            "  \"actions\": [\n"
            "    { \"description\": \"Reset scene for finale\", \"tools\": [\"clearCanvas\"] },\n"
            "    { \"description\": \"Final background\", \"tools\": [\"changeBackgroundColor\"] },\n"
            "    { \"description\": \"Show final results\", \"tools\": [\"createResultDisplay\", \"createTextDisplay\"] }\n"
            "  ],\n"
            "  \"player_states_updates\": {}\n"
            "}\n"
            "TOOL USAGE RULES:\n"
            "- Tool names must match exactly (no 'functions.' prefix).\n"
            "- When you create items (e.g., via `createTextDisplay`), capture the returned id (e.g., '0002') and reuse it in later calls that require 'itemId' (e.g., 'deleteItem'). Do not pass the item name; always pass the exact id.\n"
            "- To delete multiple items, call 'deleteItem' once per 'itemId'.\n"
            "LAYOUT PLACEMENT RULES:\n"
            "- Prefer placing text tools (createTextDisplay, createResultDisplay) at grid position 'center'.\n"
            "- If 'center' is crowded or you place multiple texts, use 'bottom-center' then 'top-center' as fallbacks.\n"
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

    # Print player states AFTER calling LLM  
    logger.info(f"[PhaseEvaluator][AFTER LLM CALL] Player States:")
    state_player_states_after = state.get("player_states", {})
    logger.info(f"[PhaseEvaluator][AFTER LLM CALL] Player States: {state_player_states_after}")
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
    

    
    
    # 尝试从 LLM JSON 输出中抽取：transition、next_phase_id、actions、player_states 更新
    actions_out = []
    transition = False
    next_phase_id = state.get("current_phase_id", 0)
    player_states_updates = {}
    
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
            # Extract player_states updates if provided
            if isinstance(parsed.get("player_states_updates"), dict):
                player_states_updates = parsed.get("player_states_updates")
    except Exception:
        actions_out = []
        transition = False
        player_states_updates = {}
    
    # 处理 player_states 更新：严格的值更新，不新增字段
    updated_player_states = current_player_states.copy()
    if player_states_updates:
        logger.info(f"[PhaseEvaluator] Processing player_states updates: {player_states_updates}")
        
        for player_id, updates in player_states_updates.items():
            if player_id in updated_player_states and isinstance(updates, dict):
                # Only update existing keys, strict value updates
                for key, value in updates.items():
                    if key in updated_player_states[player_id]:
                        updated_player_states[player_id][key] = value
                        logger.info(f"[PhaseEvaluator] Updated player {player_id}.{key} = {value}")
                    else:
                        logger.warning(f"[PhaseEvaluator] Ignored new key {key} for player {player_id}")
            else:
                logger.warning(f"[PhaseEvaluator] Player {player_id} not found or invalid updates format")
    else:
        logger.info(f"[PhaseEvaluator] No player_states updates requested")

    # Normalize next_phase_id to integer if it's a numeric string to prevent phase lookup failures
    try:
        if isinstance(next_phase_id, str) and next_phase_id.isdigit():
            next_phase_id = int(next_phase_id)
    except Exception:
        pass

    # 校验 next_phase_id 是否有效：仅当 transition=True 且 next_phase_id 存在于 DSL phases 时才允许跳转
    # def _is_valid_next_phase(pid, phases_dict):
    #     """Validate next_phase_id against DSL phases.
    #     - None -> invalid
    #     - integer key in phases -> valid
    #     - numeric string (e.g., "3") -> cast to int and check -> valid/invalid
    #     - others -> invalid
    #     """
    #     if pid is None:
    #         return False
    #     if pid in phases_dict:
    #         return True
    #     if isinstance(pid, str) and pid.isdigit():
    #         try:
    #             return int(pid) in phases_dict
    #         except Exception:
    #             return False
    #     return False

    # if transition:
    #     phases = dsl_content.get('phases', {}) if dsl_content else {}
    #     if not _is_valid_next_phase(next_phase_id, phases):
    #         logger.warning(f"[PhaseEvaluator] Ignoring invalid next_phase_id={next_phase_id}; staying in current phase")
    #         transition = False
    #     else:
    #         # 统一 current_phase_id 的类型为 int，以匹配 DSL phases 的整数键
    #         if isinstance(next_phase_id, str) and next_phase_id.isdigit():
    #             try:
    #                 next_phase_id = int(next_phase_id)
    #             except Exception:
    #                 # 理论上不会走到这里，前面已校验 isdigit
    #                 pass

    # 根据最终 transition 决策路由
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
                "player_states": updated_player_states,  # Complete updated player_states
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
                "player_states": updated_player_states,  # Complete updated player_states
                # persist game dm fields
                "current_phase_id": state.get("current_phase_id", 0),
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
