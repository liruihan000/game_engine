"""
DM Agent with Bot - Simple routing node implementation
This creates an initial routing node that directs flow based on current phase.
"""

import logging
import os  
import yaml
from dotenv import load_dotenv
from typing import Literal, List, Dict, Any, Optional
from prompt.prompt_loader import _load_prompt_async
# Load environment variables with absolute path
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(env_path)
from typing_extensions import TypedDict
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from copilotkit import CopilotKitState
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain.tools import tool
from tools.backend_tools import (
    set_next_phase,
    update_player_state,
    add_game_note,
    update_player_actions,
    update_player_name,
    update_complete_player_states,
    set_feedback_decision,
    _execute_update_player_state,
    _execute_update_player_actions,
    _execute_add_game_note,
    _execute_update_player_name,
    _execute_update_complete_player_states,
    _execute_set_feedback_decision,
)
from tools.utils import (
    get_phase_info_from_dsl,
    clean_llm_json_response,
    format_declaration_for_prompt,
    format_dict_for_prompt,
    format_current_phase_for_prompt,
    summarize_items_for_prompt,
    process_human_action_if_needed,
    filter_incomplete_message_sequences,
    _limit_actions_per_player,
    load_dsl_by_gamename,
    initialize_player_states_from_dsl,
    filter_backend_tools_from_messages,
    filter_backend_tools_from_response,
)
import json
import uuid
import time

# Global state version counter for monotonic versioning
_state_version_counter = 0

def get_next_state_version() -> int:
    """Get next monotonic state version number."""
    global _state_version_counter
    _state_version_counter += 1
    return _state_version_counter

# Monitoring configuration
VERBOSE_LOGGING = True  # Set to False to disable detailed logging

# Configure logger directly, not depending on basicConfig
logger = logging.getLogger('DMAgentWithBot')
logger.handlers.clear()  # Clear existing handlers

if VERBOSE_LOGGING:
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    
    # File handler - merge daily logs in dev mode (avoid multiple files from hot reload)
    from datetime import datetime
    date_str = datetime.now().strftime('%Y%m%d')
    log_file = f'/home/lee/game_engine/logs/dm_agent_bot_{date_str}.log'
    
    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.propagate = False  # Prevent propagation to root logger
    logger.info(f"Logging to: {log_file}")
else:
    logger.setLevel(logging.CRITICAL)



class AgentState(CopilotKitState):
    """Agent state for bot workflow with all required fields"""
    tools: List[Any] = []
    # Shared state fields synchronized with the frontend
    items: List[Dict[str, Any]] = []  # Game elements (character cards, UI components, game objects)
    # Game DM state (for interactive game engine)
    current_phase_id: int = 0
    current_phase_name: str = ""  # Current phase name from DSL (e.g., "Role Assignment")
    player_states: Dict[str, Any] = {}
    gameName: str = ""  # Current game DSL name (e.g., "werewolf", "coup")
    dsl: dict = {}
    # need_feed_back_dict: dict = {}
    roomSession: Dict[str, Any] = {}  # Room session data from frontend
    # Chat-specific fields for chatbot synchronization
    playerActions: Dict[str, Any] = {}  # Player actions
    phase_history: List[Dict[str, Any]] = []  # Phase transition history
    game_notes: List[str] = []  # Game events, state changes, and decision reminders
    # Version control fields for state synchronization
    stateVersion: int = 0  # Monotonic version number
    stateTimestamp: float = 0.0  # Unix timestamp
    updatedBy: str = ""  # Which node updated the state



# async def ActionValidatorNode(state: AgentState, config: RunnableConfig) -> Command[Literal["PhaseNode", "ActionExecutor"]]:
#     """
#     ActionValidatorNode - Currently in BYPASS mode (validation disabled).
#     Simply passes through to allow execution to continue without validation.
#     """
#     # Log raw_messages at node start
#     raw_messages = state.get("messages", [])
#     logger.info(f"[ActionValidatorNode] raw_messages: {raw_messages}")
    
#     logger.info("[ActionValidatorNode] ⚡ BYPASS MODE - Skipping validation, allowing execution to continue")
    
#     # Reset retry count and continue to PhaseNode for phase progression
#     return Command(goto="PhaseNode", update={"retry_count": 0})


# Centralized backend tools (shared across nodes)
backend_tools = [
    update_player_state,
    update_complete_player_states,
    set_next_phase,
    update_player_actions,
]

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
    "createDeathMarker",
    "promptUserText",
    # Card game UI
    "createHandsCard",
    "setHandsCardAudience",
    "createHandsCardForPlayer",
    # Text input panel tool - for user input collection and broadcast
    "createTextInputPanel",
    # Scoreboard tools
    "createScoreBoard",
    "setScoreBoardEntries",
    "upsertScoreEntry",
    "removeScoreEntry",
    # Chat-driven vote
    "submitVote",
    # Coins UI tools
    "createCoinDisplay",
    "incrementCoinCount",
    "setCoinAudience",
    # Statement board & Reaction timer
    "createStatementBoard",
    "createReactionTimer",
    "startReactionTimer",
    "stopReactionTimer",
    "resetReactionTimer",
    # Night overlay & Turn indicator
    "createTurnIndicator",
    # Health & Influence
    "createHealthDisplay",
    "createInfluenceSet",
    "revealInfluenceCard",
    # Component management tools
    "clearCanvas",
    # Player state management
    "markPlayerDead",
    # Chat tools
    "addBotChatMessage"
])


BACKEND_TOOL_NAMES = {t.name for t in backend_tools}


async def InitialRouterNode(state: AgentState, config: RunnableConfig) -> Command[Literal["ChatBotNode", "BotBehaviorNode"]]:
    """
    Initial routing node that loads DSL, processes human actions, and routes to appropriate nodes.
    
    Routes to: 
    - ChatBotNode for chat messages
    - BotBehaviorNode for game progression (after processing human actions)
    """

    # === DETAILED INPUT LOGGING ===
    current_phase_id = state.get('current_phase_id', 0)
    player_states = state.get("player_states", {})
    current_phase_id = state.get('current_phase_id', 0)
    
    # Define backend tools that don't require frontend interaction
    backend_tool_names = BACKEND_TOOL_NAMES
    
    full_messages = state.get("messages", []) or []
    try:
        if full_messages:
            last_msg = full_messages[-1]
            if isinstance(last_msg, AIMessage):
                pending_frontend_call = False
                for tc in getattr(last_msg, "tool_calls", []) or []:
                    name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                    if name and name not in backend_tool_names:
                        pending_frontend_call = True
                        break
            if pending_frontend_call:
                try:
                    # print("[TRACE] Pending frontend tool calls detected; skipping LLM this turn and waiting for ToolMessage(s).")
                    logger.info("[chatnode][end] Pending frontend tool calls detected; skipping LLM this turn and waiting for ToolMessage(s).")
                except Exception:
                    pass
                return Command(
                    goto=END,
                    update={
                        "messages": ([]),
                    }
                )
    except Exception:
        pass
    
    # Load DSL if not already present
    dsl_content = state.get("dsl", {})
    if not dsl_content and state.get("gameName"):
        current_game_name = state.get("gameName")
        logger.info(f"[InitialRouter] Loading DSL for game: {current_game_name}")
        dsl_content = await load_dsl_by_gamename(current_game_name)
        logger.info(f"[InitialRouter] dsl_content: {dsl_content}")
        if not dsl_content:
            logger.warning(f"[InitialRouter] Failed to load DSL for game: {current_game_name}")
    
    current_phase_id = state.get('current_phase_id', 0)
    player_states = state.get("player_states", {})
    
    # Initialize updates with DSL
    updates = {}
    if dsl_content:
        updates["dsl"] = dsl_content
    
    # Initialize player_states if empty and we have roomSession data
    if not player_states and dsl_content:
        room_session = state.get("roomSession")
        if room_session and room_session.get("players"):
            try:
                room_players = room_session["players"]
                initialized_states = await initialize_player_states_from_dsl(dsl_content, room_players)
                if initialized_states:
                    updates["player_states"] = initialized_states
                else:
                    logger.warning("[InitialRouter] Failed to initialize player_states from DSL")
            except Exception as e:
                logger.error(f"[InitialRouter] Error initializing player_states: {e}")
        else:
            logger.warning("[InitialRouter] No roomSession data available for player_states initialization")
    else:
        logger.info(f"[InitialRouter] Using existing player_states: {len(player_states)} players")
        # Still pass existing player_states to ensure they're available to next node
        if player_states:
            updates["player_states"] = player_states
    
    # if current_phase_id == 0:
    #     # If phase 0 UI hasn't been rendered yet, go to ActionExecutor once; otherwise, advance via PhaseNode
    #     if not bool(state.get("phase0_ui_done", False)):
    #         logger.info("[InitialRouter] phase0_ui_done=false: Routing to ActionExecutor (phase 0)")
    #         return Command(goto="ActionExecutor", update={**updates, "dsl": state.get("dsl", {} )})
    #     else:
    #         logger.info("[InitialRouter] phase0_ui_done=true: Routing to PhaseNode for transition")
    #         return Command(goto="PhaseNode", update=updates)
    # else:
    # Check latest message for game chat and route to ChatBotNode if needed
    messages = state.get("messages", [])
    try:
        if messages:
            last_msg = messages[-1]
            
            # Check if it's a human message using isinstance
            if isinstance(last_msg, HumanMessage):
                content = last_msg.content
                
                # Check for game chat patterns
                if 'in game chat:' in content or 'to Bot' in content:
                    final_dsl = dsl_content if dsl_content else state.get("dsl", {})
                    updates["dsl"] = final_dsl
                    return Command(goto="ChatBotNode", update=updates)
    except Exception as e:
        logger.error(f"[InitialRouter] Error checking chat message: {e}")

    # Process human actions before routing to BotBehaviorNode
    updated_player_actions = process_human_action_if_needed(
        messages,
        state.get("playerActions", {}),
        state.get("playerStates", {}), 
        state.get("currentPhaseId", 0),
        state.get("roomSession", {}),
        dsl_content
    )
    updates["playerActions"] = updated_player_actions

    # Ensure DSL is properly passed - use dsl_content if loaded, otherwise fallback to state
    final_dsl = dsl_content if dsl_content else state.get("dsl", {})
    updates["dsl"] = final_dsl
    
    # Generate monotonic state version for InitialRouter updates
    state_version = get_next_state_version()
    timestamp = time.time()
    
    logger.info(f"[State Version] {state_version} - Phase {current_phase_id} - Updated by InitialRouter at {timestamp}")
    
    # Add version control to updates (avoid underscore prefix for CopilotKit compatibility)
    updates["stateVersion"] = state_version
    updates["stateTimestamp"] = timestamp
    updates["updatedBy"] = "InitialRouter"
    
    return Command(goto="BotBehaviorNode", update=updates)

async def ChatBotNode(state: AgentState, config: RunnableConfig) -> Command[Literal["__end__"]]:
    """LLM-driven chat bot node"""
    logger.info("[ChatBotNode] Processing chat message")
    
    # # Log raw_messages at node start
    # raw_messages = state.get("messages", [])
    # logger.info(f"[ChatBotNode] raw_messages: {raw_messages}")
    
    # Get basic state information
    messages = state.get("messages", [])
    player_states = state.get("player_states", {})
    dsl = state.get("dsl", {})
    current_phase_id = state.get("current_phase_id", 0)
    playerActions = state.get("playerActions", {})
    roomSession = state.get("roomSession", {})
    player_states= state.get("player_states", {})
    current_phase_id = state.get("current_phase_id", 0)
    dsl_content = state.get("dsl", {})
    phases = dsl_content.get('phases', {}) if dsl_content else {}
    # Try both int and string keys to handle YAML parsing variations
    current_phase = phases.get(current_phase_id, {}) or phases.get(str(current_phase_id), {})
    declaration = dsl_content.get('declaration', {}) if dsl_content else {}
    if not messages:
        return Command(goto=END, update={})
    
    # Check if last message is game chat - hardcoded logic
    last_msg = messages[-1]
    if not isinstance(last_msg, HumanMessage):
        return Command(goto=END, update={})
    
    content = last_msg.content.lower()
    if 'in game chat:' not in content and 'to bot' not in content:
        return Command(goto=END, update={})
    
    model = init_chat_model("openai:gpt-4.1-mini")
    
    # Get available tools to call addBotChatMessage
    raw_tools = state.get("tools", []) or []
    try:
        ck = state.get("copilotkit", {}) or {}
        raw_actions = ck.get("actions", []) or []
        if isinstance(raw_actions, list) and raw_actions:
            raw_tools = [*raw_tools, *raw_actions]
    except Exception:
        pass
    
    # Filter to addBotChatMessage tool
    chat_tools = []
    for tool in raw_tools:
        name = None
        if isinstance(tool, dict):
            fn = tool.get("function", {})
            name = fn.get("name") if isinstance(fn, dict) else None
        else:
            name = getattr(tool, "name", None)
        
        if name == "addBotChatMessage":
            chat_tools.append(tool)
            break
    
    
    # Bind tools to model
    model_with_tools = model.bind_tools(chat_tools)
    game_notes = state.get('game_notes', [])

    chat_system_prompt = await _load_prompt_async("chat_system_prompt")


    # Enhanced LLM system for intelligent bot chat responses
    system_prompt = f"""
    🤖 **INTELLIGENT BOT CHAT SYSTEM**

    📊 **GAME CONTEXT**:
    - Current Phase: {current_phase_id}
    - Player States:
      {format_dict_for_prompt(player_states)}
    - Player Actions:
      {format_dict_for_prompt(_limit_actions_per_player(playerActions, 3) if playerActions else {})}


    🚫 **MANDATORY LIFE STATUS CHECK**:
    - Dead players: {[f"Player {pid} ({data.get('name', f'Bot {pid}')})" for pid, data in player_states.items() if not data.get('is_alive', True) and pid != "1"]}
    - Living bots: {[f"Player {pid} ({data.get('name', f'Bot {pid}')}): {data.get('role', 'Unknown role')}" for pid, data in player_states.items() if data.get('is_alive', True) and pid != "1"]}
    - **CRITICAL**: Dead players CANNOT speak or respond to chat - exclude them entirely!

    💬 **USER MESSAGE**: {last_msg.content}
    {chat_system_prompt}
    """
    
    try:
        response = await model_with_tools.ainvoke([SystemMessage(content=system_prompt)])

        
        # Check if tool was called
        tool_calls = getattr(response, "tool_calls", []) or []
        if tool_calls:
            logger.info(f"[ChatBotNode][TOOL_CALLS] Tool calls details: {tool_calls}")
            return Command(goto=END, update={"messages": [response]})
        else:
            logger.info("[ChatBotNode][TOOL_CALLS] No tool calls made - likely not a chat message")
            return Command(goto=END, update={})
        
    except Exception as e:
        logger.error(f"[ChatBotNode] LLM call failed: {e}")
        return Command(goto=END, update={})

async def BotBehaviorNode(state: AgentState, config: RunnableConfig) -> Command[Literal["ActionExecutor"]]:
    """
    BotBehaviorNode - Simple bot behavior simulation for non-human players.
    Only player_id=1 is human, all others are bots that need simulated actions.
    """

    # Get current game state
    current_phase_id = state.get("current_phase_id", 0)
    dsl_content = state.get("dsl", {})
    phases = dsl_content.get('phases', {}) if dsl_content else {}
    current_phase = phases.get(current_phase_id, {}) or phases.get(str(current_phase_id), {})
    player_states = state.get("player_states", {})
    declaration = dsl_content.get('declaration', {}) if dsl_content else {}
    # Initialize LLM
    model = init_chat_model("anthropic:claude-haiku-4-5-20251001")
    model_with_tools = model.bind_tools([update_player_actions])
    
    # Simple bot behavior prompt
    system_message = SystemMessage(
        content=(
            "🤖 **BOT BEHAVIOR SIMULATOR**\n"
            "You are controlling ALL bot players (any player_id ≠ 1).\n"
            f"Declaration:\n{format_declaration_for_prompt(declaration)}\n\n"
            f"{format_current_phase_for_prompt(current_phase, current_phase_id)}\n"
            f"Player States:\n{format_dict_for_prompt(player_states)}\n\n"
            f"Items Summary:\n{summarize_items_for_prompt(state)}\n\n"
            
            "**Your Job**: For each bot player, decide what they should do in this phase.\n"
            "- Look at the current phase completion criteria\n"
            "- Check each bot's role and current state\n"
            "- Generate realistic bot actions using update_player_actions(player_id, actions, phase)\n"
            "- The items summary is the current UI of the game, you can use it to decide what to do\n"
            "- You must do the actions that the completion criteria includes for different roles."
            "- Bots should act strategically based on their roles and game state.\n\n"
            
            "**Examples**: voting, accusations, night actions, defenses, etc.\n"
            "Keep it simple but realistic for each bot's role and situation."
        )
    )

    # Process messages using utility function to filter backend tools
    full_messages = state.get("messages", []) or []
    processed_messages = filter_backend_tools_from_messages(full_messages)
    processed_messages = filter_incomplete_message_sequences(processed_messages)

    # Call LLM with backend tool bound
    response = await model_with_tools.ainvoke([
        system_message,
        *processed_messages,
    ], config)
    
    # Apply backend tool effects inline (no ToolMessage)
    tool_calls = getattr(response, "tool_calls", []) or []
    current_player_states = dict(state.get("player_states", {}))
    current_player_actions = dict(state.get("playerActions", {}))

    for tc in tool_calls:
        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
        args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
        if not isinstance(args, dict):
            try:
                import json as _json
                args = _json.loads(args)
            except Exception:
                args = {}
        if name == "update_player_actions":
            pid = args.get("player_id")
            actions = args.get("actions")
            phase = args.get("phase")
            if pid and actions and phase:
                current_player_actions = _execute_update_player_actions(
                    current_player_actions, pid, actions, phase, state.get("roomSession", {}), current_player_states
                )
    
    # Clean up tool calls from response to avoid tool_use/tool_result mismatch
    if hasattr(response, 'tool_calls'):
        response.tool_calls = []
    
    # Route to ActionExecutor
    logger.info("[BotBehaviorNode] Routing to ActionExecutor")
    return Command(
        goto="ActionExecutor",
        update={
            "player_states": current_player_states,
            "playerActions": current_player_actions,
            "roomSession": state.get("roomSession", {}),
            "dsl": state.get("dsl", {})
        }
    )



async def ActionExecutor(state: AgentState, config: RunnableConfig) -> Command[Literal["UIUpdateNode"]]:
    """
    Execute actions from DSL and current phase by calling frontend tools.
    Audience-aware rendering: always choose explicit audience permissions per component
    and render public, group, and individual UIs according to the DSL phase design.
    Can make announcements based on RefereeNode conclusions.
    """
    
    # 1. Define the model
    model = init_chat_model("anthropic:claude-haiku-4-5-20251001")
    # model=init_chat_model("openai:gpt-4.1-mini")

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
    
    backend_tools = [update_complete_player_states, set_next_phase]

    # Add all backend tools to ActionExecutor
    all_tools = [*deduped_frontend_tools, *backend_tools]
    
    model_with_tools = model.bind_tools(
        all_tools,
        parallel_tool_calls=True,  # Allow multiple tool calls in single response
    )

    # 3. Prepare system states with current state and actions to execute
    items_summary = summarize_items_for_prompt(state)
    current_phase_id = state.get("current_phase_id", 0)
    dsl_content = state.get("dsl", {})
    
    # Special check for phase 0: Must ensure ActionExecutor has run at least once before allowing transition
    if current_phase_id == 0:
        phase_history = state.get("phase_history", [])
        phase0_executed = any(entry.get("phase_id") == 0 for entry in phase_history)
        if not phase0_executed:
            phases = dsl_content.get('phases', {}) if dsl_content else {}
            phase_name = phases.get(0, {}).get('name', 'Phase 0') or phases.get('0', {}).get('name', 'Phase 0')
            phase_entry = {
                "phase_id": 0,
                "phase_name": phase_name,
                "timestamp": __import__('datetime').datetime.now().isoformat()
            }
            phase_history.append(phase_entry)
            return Command(
                goto="UIUpdateNode",
                update={
                    "current_phase_id": 0,
                    "phase_history": phase_history
                }
            )
    declaration = dsl_content.get('declaration', {}) if dsl_content else {}
    player_states = state.get("player_states", {})
    playerActions = state.get("playerActions", {})
    phases = dsl_content.get('phases', {}) if dsl_content else {}
    current_phase = phases.get(current_phase_id, {}) or phases.get(str(current_phase_id), {})
    if current_phase:
        current_phase_str = f"Current phase (ID {current_phase_id}):\n{current_phase}\n"
    else:
        logger.info(f"[ActionExecutor][DSL] Current phase ID: {current_phase_id} (not found in DSL phases)")
        current_phase_str = f"Current phase ID: {current_phase_id} (not found in DSL phases)\n"


    system_message = SystemMessage(
        content=(
            "🎮 **YOU ARE THE DM (DUNGEON MASTER) - BACKEND ANALYSIS NODE**\n"
            "Focus on player operation analysis and state management only. UI updates are handled by the next node.\n\n"

            "📊 **CURRENT GAME STATE**:\n"
            f"- Declaration:\n    {format_declaration_for_prompt(declaration)}\n\n"
            f"- Phase History: {state.get('phase_history', [])}\n\n"
            f"- Current Phase:\n{format_current_phase_for_prompt(current_phase, current_phase_id)}\n\n"
            f"- Player States:\n{format_dict_for_prompt(player_states)}\n\n"
            f"- Player Actions:\n{format_dict_for_prompt(playerActions)}\n\n"
            f"- Screen Components: {items_summary}\n\n"

            "🔄 **MANDATORY 3-STEP DM WORKFLOW** (MUST execute ALL steps in order):\n\n"
            "⚠️ **CRITICAL EXECUTION RULE**: You MUST complete ALL 3 STEPS in EVERY response. Skipping any step is forbidden!\n\n"

            "**STEP 1: Analyze Current Player Operations**\n"
            "• **IMPORTANT**: Analyze player operations from playerActions data structure ONLY\n"
            "• Do NOT read or analyze human messages - only use playerActions state\n"
            "• playerActions contains all recorded player behaviors from previous rounds\n"
            "• Find each player's specific behaviors in current phase: vote choices, input content, target selections\n"
            "• Use ONLY actual existing data from playerActions - NEVER fabricate or guess\n"
            "• Use update_player_actions tool to record/update player operations if needed\n"
            "• Output format: 'Player X: [specific operation content from playerActions]'\n"
            "• If no operation record exists in playerActions, output 'Player X: No operations recorded'\n\n"

            

            "**STEP 2: Phase Management & Transition Decision**\n"
            "• **MANDATORY Phase Decision**: ALWAYS check current phase completion criteria\n"
            f"• Current phase completion criteria: {current_phase.get('completion_criteria', {})}\n"
            "• **MANDATORY Phase Update**: You MUST call set_next_phase in EVERY response\n"
            "  - **SPECIAL CASE**: If current_phase_id=0 AND items are empty, stay in phase 0 (UI not initialized yet)\n"
            "  - If phase is complete: advance to next phase\n"
            "  - If phase is not complete: stay in current phase (call set_next_phase with same phase_id)\n"
            "• Output format: 'Phase Decision: [Complete/Incomplete] - Advancing to Phase X / Staying in Phase X'\n\n"

            "**STEP 3: Update Player States**\n"
            "• Based on STEP 1's actual operations, update corresponding player_states\n"
            "• **CRITICAL ROLE ASSIGNMENT**: If players don't have roles assigned, you need to assign roles and update state\n"
            "  - **MANDATORY**: EVERY player must receive a role - check all player IDs in roomSession\n"
            "  - **VERIFICATION**: Before proceeding, verify each player has a role assigned\n"
            "  - **NO EXCEPTIONS**: Never leave any player without a role assignment\n"
            "• **BATCH UPDATE PREFERRED**: For role assignment or complete state initialization, use update_complete_player_states\n"
            "  - First call initialize_player_states_from_dsl to get DSL template with all required fields\n"
            "  - Fill the template with actual player data (names, roles, etc.)\n"
            "  - Use update_complete_player_states(player_states_dict) to batch update all players\n"
            "  - Example: update_complete_player_states({\"1\": {\"name\": \"Alice\", \"role\": \"werewolf\", \"is_alive\": true}, \"2\": {\"name\": \"Bob\", \"role\": \"villager\", \"is_alive\": true}})\n"
            "• Use update_player_state only for single field updates after initial setup\n"
            "• Update logic must comply with DSL-defined state field meanings\n"
            "• Calculate scores, life/death status, game progress, etc.\n"
            "• Output format: 'Updated Player X: field_name = new_value (reason)'\n\n"
            
            "🔧 **Available Backend Tools**:\n"
            "• update_player_actions(player_id, actions, phase) - Record player operations\n"
            "• update_player_state(player_id, state_name, state_value) - Update single player state field\n"
            "• update_complete_player_states(player_states_dict) - Batch update complete player states (preferred for role assignment)\n"
            "• set_next_phase(next_phase_id, transition_reason) - Enter next phase\n\n"

            "📋 **Output Format Requirements**:\n"
            "In your response message, clearly display in sections:\n"
            "1. **Current Player Operations**: [List each player's specific operations from playerActions]\n"
            "2. **State Updates**: [List all state changes and reasons]\n"
            "3. **Phase Management**: [Phase completion analysis and transition decision]\n\n"

            "🚨 **Critical Constraints**:\n"
            "- NEVER fabricate player operation data - only read from playerActions state\n"
            "- NEVER analyze human messages for player operations - ignore message content\n"
            "- ONLY source for player operations is playerActions data structure\n"
            "- MANDATORY: You MUST call set_next_phase in EVERY response - no exceptions!\n"
            "- SPECIAL: If phase_id=0 AND items=[], stay in phase 0 (UI initialization needed first)\n"
            "- Only use backend tools (update_player_actions, update_player_state, set_next_phase)\n"
            "- Dead players cannot participate in any game mechanics\n"
            "- Strictly follow DSL definitions for state updates\n"
            "- Do NOT attempt to create UI components - that's handled by the next node\n"
        )
    )



    # 4. Process messages: remove ToolMessages and tool_calls from AIMessages, no quantity limit
    full_messages = state.get("messages", []) or []
    
    
    latest_state_system = SystemMessage(
        content=(
            "LATEST GROUND TRUTH (authoritative):\n"
            f"- items: {items_summary}\n"
            f"- current_phase_id: {current_phase_id}\n"
            f"- player_states:\n{format_dict_for_prompt(player_states)}\n"
        )
    )

    # Process messages using utility function to filter backend tools
    processed_messages = filter_backend_tools_from_messages(full_messages)
    processed_messages = filter_incomplete_message_sequences(processed_messages)
    response = await model_with_tools.ainvoke([
        system_message,
        latest_state_system,
        *processed_messages,
    ], config)


    



    # Process backend tool calls in ActionExecutor
    tool_calls = getattr(response, "tool_calls", []) or []
    current_player_states = dict(state.get("player_states", {}))
    current_player_actions = dict(state.get("playerActions", {}))
    current_game_notes = list(state.get("game_notes", []))
    updated_phase_id = state.get("current_phase_id", 0)

    for tc in tool_calls:
        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
        args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
        if not isinstance(args, dict):
            try:
                import json as _json
                args = _json.loads(args)
            except Exception:
                args = {}
        
        # Handle backend tool calls
        if name == "update_player_state":
            pid = args.get("player_id")
            state_name = args.get("state_name")
            state_value = args.get("state_value")
            if pid and state_name is not None:
                current_player_states = _execute_update_player_state(
                    current_player_states, pid, state_name, state_value
                )
                logger.info(f"[ActionExecutor] Updated player state: {pid}.{state_name} = {state_value}")
        elif name == "update_player_actions":
            pid = args.get("player_id")
            actions = args.get("actions")
            phase = args.get("phase")
            if pid and actions and phase:
                current_player_actions = _execute_update_player_actions(
                    current_player_actions, pid, actions, phase, state.get("roomSession", {}), current_player_states
                )
                logger.info(f"[ActionExecutor] Updated player actions: {pid} in phase {phase}")
        elif name == "update_complete_player_states":
            player_states_dict = args.get("player_states_dict")
            if player_states_dict:
                current_player_states = _execute_update_complete_player_states(
                    current_player_states, player_states_dict
                )
                logger.info(f"[ActionExecutor] Batch updated complete player states: {len(player_states_dict)} players")
        elif name == "add_game_note":
            note_type = args.get("note_type")
            content = args.get("content")
            if note_type and content:
                current_game_notes = _execute_add_game_note(
                    current_game_notes, note_type, content
                )
                logger.info(f"[ActionExecutor] Added game note: {note_type} - {content}")
        elif name == "set_next_phase":
            next_phase_id = args.get("next_phase_id")
            transition_reason = args.get("transition_reason", "")
            if next_phase_id is not None:
                # Validate phase ID using normalize function
                phases = dsl_content.get('phases', {})
                
                def _normalize_and_validate_phase_id(pid: Any, phases_dict: dict) -> tuple[Any, bool]:
                    """Return (normalized_pid, is_valid)"""
                    try:
                        if pid is None:
                            return pid, False
                        # Check direct match
                        if pid in phases_dict:
                            return pid, True
                        # Check string version of pid
                        if str(pid) in phases_dict:
                            return pid, True
                        # Check integer version if pid is a numeric string
                        if isinstance(pid, str) and pid.isdigit():
                            int_pid = int(pid)
                            if int_pid in phases_dict:
                                return int_pid, True
                        return pid, False
                    except Exception:
                        return pid, False
                
                normalized_pid, is_valid = _normalize_and_validate_phase_id(next_phase_id, phases)
                if is_valid:
                    updated_phase_id = normalized_pid
                    logger.info(f"[ActionExecutor] Set next phase: {normalized_pid}, reason: {transition_reason}")
                else:
                    logger.warning(f"[ActionExecutor] Invalid phase ID {next_phase_id}, maintaining current phase")
    
    # Update phase history when phase changes
    current_phase_history = list(state.get("phase_history", []))
    if updated_phase_id != state.get("current_phase_id", 0):
        phases = dsl_content.get('phases', {})
        phase_name = phases.get(updated_phase_id, {}).get('name', f'Phase {updated_phase_id}') or phases.get(str(updated_phase_id), {}).get('name', f'Phase {updated_phase_id}')
        
        phase_entry = {
            "phase_id": updated_phase_id,
            "phase_name": phase_name
        }
        current_phase_history.append(phase_entry)
        logger.info(f"[ActionExecutor] Updated phase history: added phase {updated_phase_id} ({phase_name})")
    
    final_player_states = current_player_states

    # Remove backend tool calls from response using utility function
    response = filter_backend_tools_from_response(response, BACKEND_TOOL_NAMES)
    

    # Generate monotonic state version
    state_version = get_next_state_version()
    timestamp = time.time()
    
    logger.info(f"[State Version] {state_version} - Phase {updated_phase_id} - Updated by ActionExecutor at {timestamp}")

    return Command(
        goto="UIUpdateNode",
        update={
            # Use final_messages like agent.py
            "messages": response,
            "player_states": final_player_states,  # Updated via backend tool calls
            "playerActions": current_player_actions,  # Updated via backend tool calls
            "phase_history": current_phase_history,  # Updated when phase changes
            "current_phase_id": updated_phase_id,
            "current_phase_name": get_phase_info_from_dsl(updated_phase_id, dsl_content)[1],
        }
    )




async def UIUpdateNode(state: AgentState, config: RunnableConfig) -> Command[Literal["__end__"]]:
    """
    UI Update Node - Handle all frontend UI creation and updates
    """
    
    # 1. Define the model
    model = init_chat_model("anthropic:claude-haiku-4-5-20251001")
    # model=init_chat_model("openai:gpt-4.1-mini")
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

    # Only bind frontend tools to UIUpdateNode
    model_with_tools = model.bind_tools(
        deduped_frontend_tools,
        parallel_tool_calls=True,  # Allow multiple tool calls in single response
    )

    # 3. Prepare system states for UI updates
    items_summary = summarize_items_for_prompt(state)
    current_phase_id = state.get("current_phase_id", 0)
    dsl_content = state.get("dsl", {})
    player_states = state.get("player_states", {})
    phases = dsl_content.get('phases', {}) if dsl_content else {}
    current_phase = phases.get(current_phase_id, {}) or phases.get(str(current_phase_id), {})
    declaration = dsl_content.get('declaration', {}) if dsl_content else {}

    system_message = SystemMessage(
        content=(
            "🎨 **YOU ARE THE UI UPDATE SPECIALIST**\n"
            "Your job is to create and update UI components based on current game state.\n\n"

            "📊 **CURRENT GAME STATE**:\n"
            f"- Declaration:\n{format_declaration_for_prompt(declaration)}\n"
            f"- Current Phase:\n{format_current_phase_for_prompt(current_phase, current_phase_id)}\n"
            f"- Player States:\n{format_dict_for_prompt(player_states)}\n"
            f"- Current Screen: {items_summary}\n\n"

            "🎯 **UI UPDATE WORKFLOW**:\n\n"

            "**STEP 1: Check Current Phase Requirements**\n"
            f"• Current phase completion criteria: {current_phase.get('completion_criteria', {})}\n"
            f"• Required UI components: {current_phase.get('actions', [])}\n\n"

            "**STEP 2: Create Required UI Components**\n"
            "• **UI Operation Order**: clearCanvas() → create tools series\n"
            "• **CRITICAL AUDIENCE PERMISSIONS**: Control who can see what - ESSENTIAL FOR GAME BALANCE!\n"
            "  - audience_type=true: ALL players can see (public info)\n"
            "  - audience_type=false + audience_ids=['1','2']: ONLY specified players can see (private info)\n"
            "• **EXAMPLES**: Role cards (private), voting results (public), night actions (private to wolves)\n"
            "• **CHARACTER CARDS MUST HAVE AUDIENCE**: Every createCharacterCard MUST specify audience_ids=['player_id'] for the specific player who owns that character/role\n"
            "• **MANDATORY FOR ALL PLAYERS**: Create character cards for EVERY player - check roomSession for complete player list\n"
            "• **VERIFICATION REQUIRED**: Ensure no player is missing their character card\n"
            "• Create all components needed for current phase\n"
            "• Ensure proper positioning to avoid overlaps\n"
            "• Include proper audience targeting\n"
            "• **REMEMBER**: Always update/create score displays when scores change\n"
            "• Use createScoreBoard or updateScoreBoard to show current player scores\n\n"
            "• **CRITICAL**: If phase requires player statements/input, you MUST create createTextInputPanel\n"
            "• **WITHOUT TEXT INPUT PANEL, PLAYERS CANNOT COMMUNICATE OR PARTICIPATE**\n"
            "• Examples of phases requiring input: defense phase, accusation phase, testimony phase\n"
            "• Format: createTextInputPanel(title='Enter your defense', placeholder='Type your response...')\n\n"
            "• **CRITICAL**: If phase requires player choices/voting, you MUST create createVotingPanel\n"
            "• **WITHOUT VOTING PANEL, PLAYERS CANNOT MAKE CHOICES OR VOTE**\n"
            "• Examples of phases requiring choices: voting phase, elimination phase, target selection\n"
            "• Format: createVotingPanel(votingId='vote1', options=['Player A', 'Player B'], title='Vote to eliminate')\n"
            "• **IMPORTANT**: Use audience_type/audience_ids for voting panels too!\n"
            "  - Public voting: audience_type=true (everyone sees)\n"
            "  - Secret voting: audience_type=false + audience_ids=['specific_players']\n"
            "• Players use submitVote automatically when they click voting options\n"
           

            "**STEP 3: Optional Broadcasting (when needed)**\n"
            "• Use addBotChatMessage to broadcast important announcements\n"
            "• Examples: phase changes, game events, player eliminations, role reveals\n"
            "• Format: addBotChatMessage(message='Your announcement text')\n"
            "• Only use when you need to inform all players about game state changes\n\n"

            "🚨 **Critical Requirements**:\n"
            "- Every clearCanvas MUST be followed by create tools\n"
            "- **CRITICAL**: ALWAYS set audience_type/audience_ids - NEVER leave blank!\n"
            "- **Game will break if players see private info they shouldn't see**\n"
            "- **REMEMBER**: Update score displays every time scores change in the game\n"
            "- **MANDATORY**: Create createTextInputPanel if phase requires player input - NO EXCEPTIONS!\n"
            "- **MANDATORY**: Create createVotingPanel if phase requires player choices - NO EXCEPTIONS!\n"
            "- Players cannot participate without proper input/voting panels in interactive phases\n"
            "- Use addBotChatMessage for important game announcements\n"
        )
    )

    # Multiple safety: filter incomplete message sequences
    full_messages = state.get("messages", []) or []
    
    
    # Apply safety filtering
    safe_messages = filter_incomplete_message_sequences(full_messages)

    latest_state_system = SystemMessage(
        content=(
            "LATEST GROUND TRUTH (authoritative):\n"
            f"- items: {items_summary}\n"
            f"- current_phase_id: {current_phase_id}\n"
            f"- player_states:\n{format_dict_for_prompt(player_states)}\n"
        )
    )
    
    response = await model_with_tools.ainvoke([
        system_message,
        *safe_messages,
    ], config)

    logger.info(f"[UIUpdateNode] Created UI components")



    # Guard: if the model only returned deletions, force a follow-up to produce creation calls and merge
    try:
        orig_tool_calls = getattr(response, "tool_calls", []) or []
        def _get_tool_name(tc):
            return tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
        deletion_names = {"clearCanvas"}
        only_deletions = bool(orig_tool_calls) and all((_get_tool_name(tc) in deletion_names) for tc in orig_tool_calls)
        if only_deletions:
            logger.warning("[ActionExecutor][GUARD] Only clearCanvas tool calls detected; issuing follow-up request for creation tools.")
            strict_creation_system = SystemMessage(
                content=(
                    "You returned ONLY clearCanvas tools. Now you MUST produce the required creation tools for the current phase in this follow-up.\n"
                    "Rules:\n"
                    "- Do NOT call clearCanvas again.\n"
                    "- Call only creation tools to render the phase UI (e.g., createPhaseIndicator, createTimer, createVotingPanel, createTextDisplay, createDeathMarker, etc.).\n"
                    f"- Current phase context: ID {current_phase_id}. Follow its 'actions' strictly.\n"
                    "- Include proper audience permissions on each component (audience_type=true for public; or audience_type=false with audience_ids list).\n"
                )
            )
            # Claude requires at least one HumanMessage
            followup_human_msg = HumanMessage(content="Please create the missing UI components based on the system instructions.")
            followup_response = await model_with_tools.ainvoke([
                strict_creation_system,
                latest_state_system,
                followup_human_msg,
            ], config)
            try:
                logger.info(f"[ActionExecutor][GUARD][FOLLOWUP] Raw response content: {followup_response.content}")
                logger.info(f"[ActionExecutor][GUARD][FOLLOWUP] Response type: {type(followup_response)}")
            except Exception:
                pass
            followup_calls = getattr(followup_response, "tool_calls", []) or []
            creation_calls_only = [tc for tc in followup_calls if (_get_tool_name(tc) not in deletion_names)]
            if creation_calls_only:
                merged_calls = [*orig_tool_calls, *creation_calls_only]
                response = AIMessage(content="", tool_calls=merged_calls)
                logger.info(f"[ActionExecutor][GUARD] Merged deletion + creation tool calls: {len(merged_calls)} total")
            else:
                logger.warning("[ActionExecutor][GUARD] Follow-up produced no creation calls; using original deletions only.")
    except Exception:
        logger.exception("[ActionExecutor][GUARD] Creation follow-up failed")
    
    return Command(
        goto=END,
        update={
            "messages": response,
            "items": state.get("items", []),
            "player_states": state.get("player_states", {}),
            "playerActions": state.get("playerActions", {}),
            "game_notes": state.get("game_notes", []),
            "phase_history": state.get("phase_history", []),
            "current_phase_id": state.get("current_phase_id", 0),
            "current_phase_name": state.get("current_phase_name", ""),
            "actions": [],
            "dsl": state.get("dsl", {}),
            "roomSession": state.get("roomSession", {}),
            "phase0_ui_done": True,
        }
    )


# Define the workflow graph
workflow = StateGraph(AgentState)

# Add all nodes
workflow.add_node("InitialRouterNode", InitialRouterNode)
workflow.add_node("ChatBotNode", ChatBotNode)
workflow.add_node("BotBehaviorNode", BotBehaviorNode)
# workflow.add_node("RoleAssignmentNode", RoleAssignmentNode)
workflow.add_node("ActionExecutor", ActionExecutor)
workflow.add_node("UIUpdateNode", UIUpdateNode)
# workflow.add_node("ActionValidatorNode", ActionValidatorNode)

# Set entry point
workflow.set_entry_point("InitialRouterNode")

# Compile the graph (LangGraph API handles persistence itself in local_dev/cloud)
graph = workflow.compile()



