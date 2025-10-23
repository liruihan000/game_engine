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
    set_feedback_decision,
    _execute_update_player_state,
    _execute_update_player_actions,
    _execute_add_game_note,
    _execute_update_player_name,
    _execute_set_feedback_decision,
)
from tools.utils import (
    get_phase_info_from_dsl,
    clean_llm_json_response,
    summarize_items_for_prompt,
    process_human_action_if_needed,
    filter_incomplete_message_sequences,
    _limit_actions_per_player,
    load_dsl_by_gamename,
    initialize_player_states_from_dsl,
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
    update_player_actions,
    set_next_phase,
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
    # Print game name from state
    game_name = state.get("gameName", "")
    logger.info(f"[InitialRouterNode] Game name from state: {game_name}")
    
    
    # === DETAILED INPUT LOGGING ===
    current_phase_id = state.get('current_phase_id', 0)
    player_states = state.get("player_states", {})
    # Keep original for updates, limit for internal use
    playerActions = state.get("playerActions", {})
    logger.info(f"[InitialRouter][INPUT] current_phase_id: {current_phase_id}")
    logger.info(f"[InitialRouter][INPUT] player_states: {player_states}")
    logger.info(f"[InitialRouter][INPUT] playerActions: {playerActions}")
    logger.info(f"[InitialRouter][INPUT] state keys: {list(state.keys())}")

    
    current_phase_id = state.get('current_phase_id', 0)
    logger.info(f"[InitialRouter] Starting with phase_id: {current_phase_id}")
    
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
        logger.info(f"[InitialRouter] player_states empty ({len(player_states)} players), checking roomSession...")
        room_session = state.get("roomSession")
        if room_session and room_session.get("players"):
            logger.info("[InitialRouter] Initializing player_states from roomSession")
            try:
                room_players = room_session["players"]
                initialized_states = await initialize_player_states_from_dsl(dsl_content, room_players)
                if initialized_states:
                    updates["player_states"] = initialized_states
                    logger.info(f"[InitialRouter] ✅ Initialized player_states: {len(initialized_states)} players")
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
    
    if not chat_tools:
        logger.warning("[ChatBotNode] addBotChatMessage tool not available")
        return Command(goto=END, update={})
    
    # Bind tools to model
    model_with_tools = model.bind_tools(chat_tools)
    
    # Log game_notes for debugging
    game_notes = state.get('game_notes', [])
    logger.info(f"[ChatBotNode] Game Notes Count: {len(game_notes)}")
    if game_notes:
        logger.info(f"[ChatBotNode] All Game Notes: {game_notes}")
    else:
        logger.info(f"[ChatBotNode] No Game Notes Available")

    chatbot_system_prompt = await _load_prompt_async("chatbot_system_prompt")

    # Enhanced LLM system for intelligent bot chat responses
    system_prompt = f"""
    🤖 **INTELLIGENT BOT CHAT SYSTEM**

    📊 **GAME CONTEXT**:
    - Current Phase: {current_phase_id}
    - Player States: {player_states}
    - Player Actions: {_limit_actions_per_player(playerActions, 3) if playerActions else {}}
    - Game Notes: {game_notes[-5:] if game_notes else 'None'}

    🚫 **MANDATORY LIFE STATUS CHECK**:
    - Dead players: {[f"Player {pid} ({data.get('name', f'Bot {pid}')})" for pid, data in player_states.items() if not data.get('is_alive', True) and pid != "1"]}
    - Living bots: {[f"Player {pid} ({data.get('name', f'Bot {pid}')}): {data.get('role', 'Unknown role')}" for pid, data in player_states.items() if data.get('is_alive', True) and pid != "1"]}
    - **CRITICAL**: Dead players CANNOT speak or respond to chat - exclude them entirely!

    💬 **USER MESSAGE**: {last_msg.content}
    {chatbot_system_prompt}
    """
    
    try:
        response = await model_with_tools.ainvoke([SystemMessage(content=system_prompt)])
        
        # === DETAILED LLM RESPONSE LOGGING ===
        logger.info(f"[ChatBotNode][LLM_OUTPUT] Raw response content: {response.content}")
        logger.info(f"[ChatBotNode][LLM_OUTPUT] Response type: {type(response)}")
        
        # Check if tool was called
        tool_calls = getattr(response, "tool_calls", []) or []
        logger.info(f"[ChatBotNode][TOOL_CALLS] Total tool calls: {len(tool_calls)}")
        if tool_calls:
            logger.info(f"[ChatBotNode][TOOL_CALLS] Tool calls details: {tool_calls}")
            return Command(goto=END, update={"messages": [response]})
        else:
            logger.info("[ChatBotNode][TOOL_CALLS] No tool calls made - likely not a chat message")
            return Command(goto=END, update={})
        
    except Exception as e:
        logger.error(f"[ChatBotNode] LLM call failed: {e}")
        return Command(goto=END, update={})

async def BotBehaviorNode(state: AgentState, config: RunnableConfig) -> Command[Literal["PhaseNode"]]:
    """
    BotBehaviorNode analyzes bot behavior and generates responses for non-human players.
    
    Input:
    - trimmed_messages: Recent message history
    - player_states: Current player states
    - current_phase and declaration: Phase configuration
    - need_feed_back_dict: Required feedback info
    
    Output:
    """
    # Print game name from state
    game_name = state.get("gameName", "")
    logger.info(f"[BotBehaviorNode] Game name from state: {game_name}")
    
    # Log raw_messages at node start
    # raw_messages = state.get("messages", [])
    # logger.info(f"[BotBehaviorNode] raw_messages: {raw_messages}")
    
    logger.info("[BotBehaviorNode] Starting bot behavior analysis")

    
    # Extract inputs - simplified with BaseMessage
    # messages = state.get("messages", [])  # Not used in BotBehaviorNode
    player_states = state.get("player_states", {})
    current_phase_id = state.get("current_phase_id", 0)
    # Remove need_feed_back_dict dependency - use autonomous analysis only
    dsl_content = state.get("dsl", {})
    
    # Get current phase details and NEXT phase for pre-analysis
    phases = dsl_content.get('phases', {}) if dsl_content else {}
    # Try both int and string keys to handle YAML parsing variations
    current_phase = phases.get(current_phase_id, {}) or phases.get(str(current_phase_id), {})
    
    # BotBehaviorNode should only focus on current phase - no next phase prediction
    # This prevents bots from gaming the system by knowing future phases
    
    declaration = dsl_content.get('declaration', {}) if dsl_content else {}
    playerActions = state.get("playerActions", {})
    
    # Log phase info  
    logger.info(f"[BotBehaviorNode] current_phase_id: {current_phase_id}")
    logger.info(f"[BotBehaviorNode] current_phase: {current_phase}")
    logger.info(f"[BotBehaviorNode] player_states: {player_states}")
    
    # Log game_notes for debugging
    game_notes = state.get('game_notes', [])
    logger.info(f"[BotBehaviorNode] Game Notes Count: {len(game_notes)}")
    if game_notes:
        logger.info(f"[BotBehaviorNode] Full Game Notes: {game_notes}")
    else:
        logger.info(f"[BotBehaviorNode] No Game Notes Available")

    # Initialize LLM
    model = init_chat_model("openai:gpt-4.1-mini")
    model_with_tools = model.bind_tools([update_player_actions])
    items_summary = summarize_items_for_prompt(state)
    bot_behavior_system_prompt = await _load_prompt_async("bot_behavior_system_prompt")
    
    # System message with precise analysis based on FeedbackDecisionNode logic
    system_message = SystemMessage(
        content=(
            "🤖 **BOT BEHAVIOR GENERATION - CURRENT PHASE FOCUS**\n\n"
            f"📊 **CURRENT GAME STATE**:\n\n"
            f"- **Current Phase ({current_phase_id})**: {current_phase}\n\n"
            f"- **Player States**: {player_states}\n"
            f"- **Player Actions**: {_limit_actions_per_player(playerActions, 3) if playerActions else {}}\n\n"
            f"- **Game Notes**: {game_notes[-5:] if game_notes else 'None'}\n\n"
            f"- **Items State**: {items_summary}\n\n"
            f"{bot_behavior_system_prompt}"
            
        )
    )
    
    # Only treat update_player_actions as backend here
    backend_tool_names = {"update_player_actions"}
    
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
                    goto=END
                )
    except Exception:
        pass


    # Call LLM with backend tool bound
    response = await model_with_tools.ainvoke([system_message], config)
    
    # === DETAILED LLM RESPONSE LOGGING ===
    logger.info(f"[BotBehaviorNode][LLM_OUTPUT] Raw response content: {response.content}")
    logger.info(f"[BotBehaviorNode][LLM_OUTPUT] Response type: {type(response)}")
    
    # No JSON parsing needed - we only expect tool calls
    # The bot behavior is recorded via update_player_actions tool calls
    
    # Apply backend tool effects inline (no ToolMessage)
    tool_calls = getattr(response, "tool_calls", []) or []
    logger.info(f"[BotBehaviorNode][TOOL_CALLS] Total tool calls: {len(tool_calls)}")
    logger.info(f"[BotBehaviorNode][TOOL_CALLS] Tool calls details: {tool_calls}")
    current_player_states = dict(state.get("player_states", {}))
    current_player_actions = dict(state.get("playerActions", {}))
    logger.info(f"[BotBehaviorNode] current_player_actions: {current_player_actions}")
    # No longer using need_feed_back_dict - using autonomous analysis
    logger.info(f"[BotBehaviorNode][DEBUG] Processing all backend tool calls without filtering")
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
    
    # Route to RefereeNode
    logger.info("[BotBehaviorNode] Routing to RefereeNode")
    return Command(
        goto="PhaseNode",
        update={
            "player_states": current_player_states,
            "playerActions": current_player_actions,
            "roomSession": state.get("roomSession", {}),
            "dsl": state.get("dsl", {})
        }
    )

async def RefereeNode(state: AgentState, config: RunnableConfig) -> Command[Literal["ActionExecutor"]]:
    """
    RefereeNode processes player behaviors and updates game state according to rules.
    
    Input:
    - trimmed_messages: Recent message history
    - last human message: Most recent human player input
    - player_states: Current player states
    - current_phase and declaration: Phase configuration  
    
    Output:
    - Updated player_states based on game rules and player actions
    """
    # Print game name from state
    game_name = state.get("gameName", "")
    logger.info(f"[RefereeNode] Game name from state: {game_name}")
    
    # Log raw_messages at node start
    # raw_messages = state.get("messages", [])
    # logger.info(f"[RefereeNode] raw_messages: {raw_messages}")
    
    logger.info("[RefereeNode] Starting referee analysis and state updates")
    
    # Extract inputs - simplified with BaseMessage
    messages = state.get("messages", [])  # Safe dictionary access to BaseMessage list
    trimmed_messages = messages[-10:] if messages else []
    trimmed_messages = filter_incomplete_message_sequences(trimmed_messages)
    player_states = state.get("player_states", {})
    current_phase_id = state.get("current_phase_id", 0)
    dsl_content = state.get("dsl", {})
    
    
    # Get current phase details
    phases = dsl_content.get('phases', {}) if dsl_content else {}
    # Try both int and string keys to handle YAML parsing variations
    current_phase = phases.get(current_phase_id, {}) or phases.get(str(current_phase_id), {})
    next_phase = current_phase
    declaration = dsl_content.get('declaration', {}) if dsl_content else {}
    
    # Log phase info
    logger.info(f"[RefereeNode] current_phase_id: {current_phase_id}")
    logger.info(f"[RefereeNode] current_phase: {current_phase}")
    logger.info(f"[RefereeNode] player_states (input): {player_states}")

    playerActions = state.get("playerActions", {})
    
    # Log game_notes for debugging
    game_notes = state.get('game_notes', [])
    logger.info(f"[RefereeNode] Game Notes Count: {len(game_notes)}")
    if game_notes:
        logger.info(f"[RefereeNode] Current Game Notes: {game_notes}")
    else:
        logger.info(f"[RefereeNode] No Game Notes Available")

    phase_history = state.get('phase_history', [])
    last_phase_id = phase_history[-2]['phase_id'] if len(phase_history) >= 2 else None
    
    # Get last phase details from DSL using last_phase_id
    last_phase = None
    if last_phase_id is not None:
        phases = dsl_content.get('phases', {}) if dsl_content else {}
        # Try both int and string keys to handle YAML parsing variations
        last_phase = phases.get(last_phase_id, {}) or phases.get(str(last_phase_id), {})
    
    # Initialize LLM
    model = init_chat_model("openai:gpt-4.1-mini")
    # Bind state management tools
    model_with_tools = model.bind_tools([update_player_state, add_game_note], parallel_tool_calls=True)
    referee_system_prompt_1 = await _load_prompt_async("referee_system_prompt_1")
    referee_system_prompt_2 = await _load_prompt_async("referee_system_prompt_2")
    # Create system message with all inputs
    system_message = SystemMessage(
        content=(
            " **REFEREE NODE: STATE MANAGER & GAME NOTES KEEPER**\n\n"

            f" **CURRENT GAME ANALYSIS**:\n"
            f"- Phase ID: {current_phase_id} | Phase: {current_phase.get('name', 'Unknown')}\n"
            f"- Next Phase: {current_phase}\n"
            f"- Current Phase: {last_phase.get('name', f'Phase {last_phase_id}') if last_phase else 'None'}\n"
            f"- Phase History: {state.get('phase_history', [])[-5:] if state.get('phase_history') else 'None'}\n"
            f"- Player States: {player_states}\n"
            f"- Player Actions: {_limit_actions_per_player(playerActions, 3) if playerActions else {}}\n"
            f"- Game Notes: {game_notes[-5:] if game_notes else 'None'}\n"
            f"- Declaration Rules: {declaration}\n\n"
            f"{referee_system_prompt_1}"
            
            "🔮 **PHASE-AWARE DECISION MAKING**:\n"
            f"**Current Phase Analysis** ({current_phase.get('name', 'Unknown')}):\n"
            f"• Completion criteria: {current_phase.get('completion_criteria', {}).get('type', 'Unknown')}\n"
            f"• Expected actions: {current_phase.get('completion_criteria', {}).get('description', 'None')}\n"
            f"**Next Phase Preview** ({next_phase.get('name', 'Unknown') if next_phase else 'Game End'}):\n"
            f"• What's coming: {next_phase.get('description', 'Unknown') if next_phase else 'Final phase'}\n"
            f"• Preparation needed: Use this context to create better game notes and state updates\n\n"
            f"{referee_system_prompt_2}"
        )
    )
    

    # Only treat update_player_state as backend here
    backend_tool_names = {"update_player_state"}
    
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
                    goto=END
                )
    except Exception:
        pass
    
    # Call LLM with tool bound
    response = await model_with_tools.ainvoke([system_message], config)
    
    # === DETAILED LLM RESPONSE LOGGING ===
    logger.info(f"[RefereeNode][LLM_OUTPUT] Raw response content: {response.content}")
    logger.info(f"[RefereeNode][LLM_OUTPUT] Response type: {type(response)}")
    
    # No JSON expected; start with current states
    updated_player_states = player_states
    conclusions = []
    
    # Apply tool calls inline (no ToolMessage) 
    tool_calls = getattr(response, "tool_calls", []) or []
    logger.info(f"[RefereeNode][TOOL_CALLS] Total tool calls: {len(tool_calls)}")
    logger.info(f"[RefereeNode][TOOL_CALLS] Tool calls details: {tool_calls}")
    
    current_player_states = dict(updated_player_states)
    current_game_notes = list(state.get("game_notes", []))
    
    for tc in tool_calls:
        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
        args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
        if not isinstance(args, dict):
            try:
                import json as _json
                args = _json.loads(args)
            except Exception:
                args = {}
        
        if name == "update_player_state":
            pid = args.get("player_id")
            state_name = args.get("state_name")
            state_value = args.get("state_value")
            if pid and state_name is not None:
                current_player_states = _execute_update_player_state(
                    current_player_states, pid, state_name, state_value
                )
        elif name == "add_game_note":
            note_type = args.get("note_type")
            content = args.get("content")
            if note_type and content:
                current_game_notes = _execute_add_game_note(
                    current_game_notes, note_type, content
                )

    # Notes are now created via tool calls (add_game_note), no need for direct creation
    # current_game_notes already contains all tool-created notes
    
    # Route to RoleAssignmentNode with updated player states, conclusions, and game notes
    notes_created = len(current_game_notes) - len(state.get("game_notes", []))
    logger.info(f"[RefereeNode] Created {notes_created} new game notes via tools, routing to RoleAssignmentNode")
    return Command(
        goto="ActionExecutor",
        update={
            "player_states": current_player_states,
            "game_notes": current_game_notes,
            "roomSession": state.get("roomSession", {}),
            "dsl": state.get("dsl", {}),
            "phase_history": state.get("phase_history", [])
        }
    )

# async def RoleAssignmentNode(state: AgentState, config: RunnableConfig) -> Command[Literal["ActionExecutor"]]:
#     """
#     RoleAssignmentNode: Pre-assignment of roles using LLM before ActionExecutor.
#     - Uses LLM to intelligently assign roles based on DSL requirements
#     - ActionExecutor continues normally with DSL-defined role assignment actions
#     - Ensures balanced game setup and proper role distribution
#     """
    
#     logger.info("[RoleAssignmentNode] Starting intelligent role assignment")
    
#     # Extract inputs
#     dsl_content = state.get("dsl", {})
#     player_states = state.get("player_states", {})
#     current_phase_id = state.get("current_phase_id", 0)
    
#     logger.info(f"[RoleAssignmentNode][INPUT] current_phase_id: {current_phase_id}")
#     logger.info(f"[RoleAssignmentNode][INPUT] player_states: {player_states}")
    
#     # Check if game defines roles for assignment
#     declaration = dsl_content.get('declaration', {}) if dsl_content else {}
#     dsl_roles = declaration.get('roles', [])
    
#     # Check if game uses role system: both DSL roles and player_states have 'role' key
#     has_dsl_roles = bool(dsl_roles)
#     has_role_key_in_player_states = False
#     if player_states:
#         # Check if any player has 'role' key in their state structure
#         sample_player = next(iter(player_states.values()), {})
#         has_role_key_in_player_states = 'role' in sample_player
    
#     role_assignment_detected = has_dsl_roles and has_role_key_in_player_states
#     if has_dsl_roles and not has_role_key_in_player_states:
#         logger.info(f"[RoleAssignmentNode] Game defines {len(dsl_roles)} roles in DSL, but player_states don't have 'role' key - skipping role assignment")
#     elif role_assignment_detected:
#         logger.info(f"[RoleAssignmentNode] Game defines {len(dsl_roles)} roles and player_states have 'role' key, assignment needed")
#     else:
#         logger.info("[RoleAssignmentNode] No roles defined in DSL or no 'role' key in player_states")
    
#     # Check if roles are already assigned
#     all_roles_assigned = True
#     unassigned_players = []
#     if player_states and role_assignment_detected:
#         for player_id, player_data in player_states.items():
#             player_role = player_data.get('role', '')
#             if not player_role:  # Empty or missing role
#                 all_roles_assigned = False
#                 unassigned_players.append(player_id)
    
#     # Skip if no role assignment needed or all roles already assigned
#     if not role_assignment_detected or all_roles_assigned:
#         logger.info("[RoleAssignmentNode] No role assignment needed, passing through to ActionExecutor")
#         return Command(
#             goto="ActionExecutor",
#             update={
#                 "current_phase_id": current_phase_id,
#                 "player_states": player_states,
#                 "roomSession": state.get("roomSession", {}),
#                 "dsl": dsl_content,
#                 "playerActions": state.get("playerActions", {}),
#             }
#         )
    
#     # Use LLM for intelligent role assignment
#     logger.info(f"[RoleAssignmentNode] Using LLM to assign roles to {len(unassigned_players)} players")
    
#     model = init_chat_model("openai:gpt-4.1-mini-mini")
#     model_with_tools = model.bind_tools([update_player_name], parallel_tool_calls=True)
    
#     # Create intelligent role assignment prompt
#     system_message = SystemMessage(
#         content=(
#             "INTELLIGENT ROLE ASSIGNMENT TASK\n"
#             f"Game: {declaration.get('description', 'Unknown Game')}\n"
#             f"Available Roles: {dsl_roles}\n"
#             f"Total Players: {len(player_states)}\n"
#             f"Unassigned Players: {unassigned_players}\n"
#             f"Current Player States: {player_states}\n"
#             f"Min Players: {declaration.get('min_players', 'Unknown')}\n\n"
            
#             "TASK: Assign roles to unassigned players using game balance and strategy.\n"
            
#             "ASSIGNMENT RULES:\n"
#             "- NEVER overwrite existing roles (skip players who already have roles)\n"
#             "- Use update_player_name tool for each assignment\n"
#             "- Ensure game balance based on player count and game mechanics\n"
#             "- Consider role interactions and win conditions\n"
#             "- Player 1 is the human - give them an engaging role when possible\n"
#             "- Distribute special roles fairly among all players\n\n"
            
#             "GAME BALANCE STRATEGY:\n"
#             "1. Calculate optimal role distribution for current player count\n"
#             "2. Assign evil/mafia/werewolf roles appropriately (usually 20-30% of players)\n"
#             "3. Assign power roles (Detective, Doctor, etc.) for game depth\n"
#             "4. Fill remaining slots with basic roles (Villager, etc.)\n"
#             "5. Ensure no team has overwhelming advantage\n\n"
            
#             "SPECIFIC CONSIDERATIONS:\n"
#             "- For Werewolf games: 1-2 werewolves for 5-7 players, 2-3 for 8+ players\n"
#             "- Ensure at least one investigative role and one protective role\n"
#             "- Balance information roles vs action roles\n"
#             "- Consider faction balance for multiplayer games\n\n"
            
#             "Execute role assignments using update_player_name tools now."
#         )
#     )
    
#     try:
#         # Add 10 second timeout to prevent hanging
#         import asyncio
#         response = await asyncio.wait_for(
#             model_with_tools.ainvoke([system_message], config),
#             timeout=10.0
#         )
        
#         # === LLM RESPONSE LOGGING ===
#         logger.info(f"[RoleAssignmentNode][LLM_OUTPUT] Response content: {response.content}")
        
#         # Process role assignment tool calls
#         tool_calls = getattr(response, "tool_calls", []) or []
#         logger.info(f"[RoleAssignmentNode][TOOL_CALLS] Total: {len(tool_calls)}")
        
#         updated_player_states = dict(player_states)
#         if tool_calls:
#             for tc in tool_calls:
#                 name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
#                 if name == "update_player_name":
#                     args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
#                     if not isinstance(args, dict):
#                         try:
#                             import json as _json
#                             args = _json.loads(args)
#                         except Exception:
#                             args = {}
#                     pid = args.get("player_id")
#                     player_name = args.get("name")
#                     role = args.get("role")
#                     if pid and role:
#                         updated_player_states = _execute_update_player_name(
#                             updated_player_states, pid, player_name, role
#                         )
#                         logger.info(f"[RoleAssignmentNode] LLM assigned: Player {pid} ({player_name}) -> role={role}")
        
#         logger.info("[RoleAssignmentNode] Role assignment completed, routing to ActionExecutor")
        
#         return Command(
#             goto="ActionExecutor",
#             update={
#                 "current_phase_id": current_phase_id,
#                 "player_states": updated_player_states,
#                 "roomSession": state.get("roomSession", {}),
#                 "dsl": dsl_content,
#                 "playerActions": state.get("playerActions", {}),
#             }
#         )
        
    # except asyncio.TimeoutError:
    #     logger.warning("[RoleAssignmentNode] LLM call timed out after 10 seconds, skipping role assignment")
    #     # Skip role assignment and continue with ActionExecutor
    #     return Command(
    #         goto="ActionExecutor",
    #         update={
    #             "current_phase_id": current_phase_id,
    #             "player_states": player_states,
    #             "roomSession": state.get("roomSession", {}),
    #             "dsl": dsl_content,
    #             "playerActions": state.get("playerActions", {}),
    #         }
    #     )
    # except Exception as e:
    #     logger.error(f"[RoleAssignmentNode] LLM call failed: {e}, skipping role assignment")
    #     # Skip role assignment and continue with ActionExecutor
    #     return Command(
    #         goto="ActionExecutor",
    #         update={
    #             "current_phase_id": current_phase_id,
    #             "player_states": player_states,
    #             "roomSession": state.get("roomSession", {}),
    #             "dsl": dsl_content,
    #             "playerActions": state.get("playerActions", {}),
    #         }
    #     )

async def PhaseNode(state: AgentState, config: RunnableConfig) -> Command[Literal["RefereeNode", "ActionExecutor"]]:
    """
    PhaseNode determines the next phase based on DSL and current game state.
    
    Input:
    - dsl: Game DSL rules
    - current_phase_id: Current phase identifier
    - current_phase and declaration: Phase configuration
    
    Output:
    - next_phase_id: Determined next phase
    """
    # Print game name from state
    game_name = state.get("gameName", "")
    logger.info(f"[PhaseNode] Game name from state: {game_name}")
    
    # Log raw_messages at node start
    # raw_messages = state.get("messages", [])
    # logger.info(f"[PhaseNode] raw_messages: {raw_messages}")
    
    logger.info("[PhaseNode] Starting phase transition analysis")
    
    # Extract inputs
    dsl_content = state.get("dsl", {})
    current_phase_id = state.get("current_phase_id", 0)
    player_states = state.get("player_states", {})
    playerActions = state.get("playerActions", {})
    # === DETAILED INPUT LOGGING ===
    logger.info(f"[PhaseNode][INPUT] current_phase_id: {current_phase_id}")
    logger.info(f"[PhaseNode][INPUT] player_states: {player_states}")
    logger.info(f"[PhaseNode][INPUT] playerActions: {playerActions}")
    logger.info(f"[PhaseNode][INPUT] state keys: {list(state.keys())}")
   
    
    # Get current phase details (needed for phase 0 check)
    phases = dsl_content.get('phases', {}) if dsl_content else {}
    
    # Special check for phase 0: Must ensure ActionExecutor has run at least once before allowing transition
    if current_phase_id == 0:
        phase_history = state.get("phase_history", [])
        logger.info(f"[PhaseNode] [phase_history] : {phase_history}")
        # Check if phase 0 exists in history
        phase0_executed = any(entry.get("phase_id") == 0 for entry in phase_history)
        
        if not phase0_executed:
            logger.info("[PhaseNode] Phase 0 hasn't been executed yet by ActionExecutor; staying at phase 0")
            
            # Record phase 0 in history before executing
            phase_name = phases.get(0, {}).get('name', 'Phase 0') or phases.get('0', {}).get('name', 'Phase 0')
            phase_entry = {
                "phase_id": 0,
                "phase_name": phase_name,
                "timestamp": __import__('datetime').datetime.now().isoformat()
            }
            phase_history.append(phase_entry)
            
            return Command(
                goto="ActionExecutor",
                update={
                    "current_phase_id": 0,
                    "player_states": player_states,
                    "roomSession": state.get("roomSession", {}),
                    "dsl": dsl_content,
                    "phase_history": phase_history
                }
            )
        else:
            logger.info("[PhaseNode] Phase 0 has been executed, proceeding with transition analysis")
    
    # Try both int and string keys to handle YAML parsing variations
    current_phase = phases.get(current_phase_id, {}) or phases.get(str(current_phase_id), {})
    declaration = dsl_content.get('declaration', {}) if dsl_content else {}
    items_summary = summarize_items_for_prompt(state)
    logger.info(f"[PhaseNode][output] items_summary: {items_summary}")
    # Log phase info
    logger.info(f"[PhaseNode] current_phase_id: {current_phase_id}")
    logger.info(f"[PhaseNode] current_phase: {current_phase}")
    logger.info(f"[PhaseNode] player_states: {player_states}")
    
    # Log game_notes for debugging
    game_notes = state.get('game_notes', [])
    logger.info(f"[PhaseNode] Game Notes Count: {len(game_notes)}")
    if game_notes:
        logger.info(f"[PhaseNode] All Game Notes: {game_notes}")
    else:
        logger.info(f"[PhaseNode] No Game Notes Available")

    # Initialize LLM with set_next_phase tool
    model = init_chat_model("openai:gpt-4.1-mini")
    model_with_tools = model.bind_tools([set_next_phase])
    logger.info(f"[PhaseNode] Phase {current_phase_id}: Phase transition analysis with set_next_phase tool")
    
    messages = state.get("messages", []) or []
    trimmed_messages = messages[-10:]  # Get more messages before filtering
    filtered_messages = filter_incomplete_message_sequences(trimmed_messages)
    trimmed_messages = filtered_messages[-3:]  # Keep only 3 after filtering

    # PhaseNode focuses purely on phase transition - no role assignment
    PhaseNode_system_prompt=await _load_prompt_async("PhaseNode_system_prompt")
    
    system_message = SystemMessage(
        content=(
            "PHASE TRANSITION ANALYSIS WITH ROLE MANAGEMENT\n"
            f"itemsState (current frontend layout):\n{items_summary}\n"
            f"Current Phase ID: {current_phase_id}\n"
            f"Current Phase Details: {current_phase}\n"
            f"Game Declaration: {declaration}\n"
            f"Player States: {player_states}\n"
            f"Game Notes: {game_notes[-5:] if game_notes else 'None'}\n"
            f"🚫 Living players: {[pid for pid, data in player_states.items() if data.get('is_alive', True)]}\n"
            f"🚫 Dead players: {[pid for pid, data in player_states.items() if not data.get('is_alive', True)]}\n"
            f"Phase History (last 5): {state.get('phase_history', [])[-5:]}\n" 
            f"Player Actions: {_limit_actions_per_player(playerActions, 3) if playerActions else {}}\n\n"
            f"{PhaseNode_system_prompt}"
            
        )
    )
    
    # Call LLM with tools for all phases (needed for set_next_phase tool)
    logger.info("[PhaseNode] About to call LLM with set_next_phase tool")
    try:
        response = await model_with_tools.ainvoke([system_message], config)
        logger.info("[PhaseNode] LLM call completed successfully")
    except Exception as e:
        logger.error(f"[PhaseNode] LLM call failed: {e}")
        raise
    
    # === DETAILED LLM RESPONSE LOGGING ===
    logger.info(f"[PhaseNode][LLM_OUTPUT] Raw response content: {response.content}")
    logger.info(f"[PhaseNode][LLM_OUTPUT] Response type: {type(response)}")
    
    # PhaseNode no longer handles role assignment - check for phase transition tool calls
    tool_calls = getattr(response, "tool_calls", []) or []
    logger.info(f"[PhaseNode][TOOL_CALLS] Total tool calls: {len(tool_calls)}")
    if tool_calls:
        logger.info(f"[PhaseNode][TOOL_CALLS] Tool calls details: {tool_calls}")
    
    # Extract phase decision from tool calls
    transition_from_tool = None
    next_phase_id_from_tool = None
    transition_reason = ""
    for tc in tool_calls:
        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
        if name == "set_next_phase":
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
            if not isinstance(args, dict):
                try:
                    import json as _json
                    args = _json.loads(args)
                except Exception:
                    args = {}
            transition_from_tool = args.get("transition")
            next_phase_id_from_tool = args.get("next_phase_id")
            transition_reason = args.get("transition_reason", "")
            logger.info(f"[PhaseNode] Tool call: transition={transition_from_tool}, next_phase_id={next_phase_id_from_tool}, reason: {transition_reason}")
            break
    
    # Use tool call result or fallback logic
    if transition_from_tool is not None and next_phase_id_from_tool is not None:
        parsed_transition = bool(transition_from_tool)
        proposed_next_phase_id = next_phase_id_from_tool
        logger.info(f"[PhaseNode] Using tool call result: transition={parsed_transition}, next_phase_id={proposed_next_phase_id}")
        logger.info(f"[PhaseNode] Transition reason: {transition_reason}")
    else:
        # Fallback: try to parse text-based tool call
        import re
        raw_content = str(response.content)
        logger.info(f"[PhaseNode] Attempting to parse text-based tool call from: {raw_content}")
        
        # Look for set_next_phase pattern in text
        pattern = r'set_next_phase\s*\(\s*transition\s*=\s*(true|false)\s*,\s*next_phase_id\s*=\s*(\d+)\s*,\s*transition_reason\s*=\s*[\'"]([^\'"]*)[\'"]'
        match = re.search(pattern, raw_content, re.IGNORECASE)
        
        if match:
            parsed_transition = match.group(1).lower() == 'true'
            proposed_next_phase_id = int(match.group(2))
            transition_reason = match.group(3)
            logger.info(f"[PhaseNode] Parsed text-based tool call: transition={parsed_transition}, next_phase_id={proposed_next_phase_id}, reason: {transition_reason}")
        else:
            # Final fallback: no transition
            parsed_transition = False
            proposed_next_phase_id = current_phase_id
            transition_reason = "No valid tool call found"
            logger.warning(f"[PhaseNode] No valid tool call made, staying at phase_id {proposed_next_phase_id}")

    # Validate and normalize phase id
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

    # Determine target phase
    target_phase_id = current_phase_id  # Default: no transition

    if parsed_transition:
        normalized_pid, is_valid = _normalize_and_validate_phase_id(proposed_next_phase_id, phases)
        if is_valid:
            target_phase_id = normalized_pid
            logger.info(f"[PhaseNode] Transition approved → {current_phase_id} -> {target_phase_id}")
        else:
            logger.warning(f"[PhaseNode] Invalid next_phase_id={proposed_next_phase_id}; staying at {current_phase_id}")
    else:
        logger.info(f"[PhaseNode] No transition; staying at phase {current_phase_id}")

    # Record current phase in history
    current_phase_history = state.get("phase_history", [])
    phase_name = phases.get(target_phase_id, {}).get('name', f'Phase {target_phase_id}') or phases.get(str(target_phase_id), {}).get('name', f'Phase {target_phase_id}')
    
    phase_entry = {
        "phase_id": target_phase_id,
        "phase_name": phase_name,
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }
    current_phase_history.append(phase_entry)

    logger.info("[PhaseNode] Routing to ActionExecutor")
    
    # Get phase info using helper function
    target_phase, target_phase_name = get_phase_info_from_dsl(target_phase_id, dsl_content)
    
    phasenode_outputs = {
        "current_phase_id": target_phase_id,
        "current_phase_name": target_phase_name,
        "player_states": state.get("player_states", {}),
        "roomSession": state.get("roomSession", {}),
        "dsl": state.get("dsl", {}),
        "phase_history": current_phase_history
    }
    
    # === DETAILED OUTPUT LOGGING ===
    logger.info(f"[PhaseNode][OUTPUT] Command goto: ActionExecutor")
    logger.info(f"[PhaseNode][OUTPUT] Updates keys: {list(phasenode_outputs.keys())}")
    logger.info(f"[PhaseNode][OUTPUT] Updates current_phase_id: {phasenode_outputs.get('current_phase_id')}")
    logger.info(f"[PhaseNode][OUTPUT] Updates player_states: {phasenode_outputs.get('player_states', 'NOT_SET')}")
    logger.info(f"[PhaseNode][OUTPUT] Updates playerActions: NOT_INCLUDED")
    
    return Command(
        goto="RefereeNode",
        update=phasenode_outputs
    )

async def ActionExecutor(state: AgentState, config: RunnableConfig) -> Command[Literal["__end__"]]:
    """
    Execute actions from DSL and current phase by calling frontend tools.
    Audience-aware rendering: always choose explicit audience permissions per component
    and render public, group, and individual UIs according to the DSL phase design.
    Can make announcements based on RefereeNode conclusions.
    """
    # Print game name from state
    game_name = state.get("gameName", "")
    logger.info(f"[ActionExecutor] Game name from state: {game_name}")
    
    # Log raw_messages at node start
    # raw_messages = state.get("messages", [])
    # logger.info(f"[ActionExecutor] raw_messages: {raw_messages}")
    
    logger.info(f"[ActionExecutor][start] ==== start ActionExecutor ====")
    
    # Extract phase info for logging
    current_phase_id = state.get("current_phase_id", 0)
    dsl_content = state.get("dsl", {})
    
    # === DETAILED INPUT LOGGING ===
    player_states = state.get("player_states", {})
    playerActions = state.get("playerActions", {})
    logger.info(f"[ActionExecutor][INPUT] current_phase_id: {current_phase_id}")
    logger.info(f"[ActionExecutor][INPUT] player_states: {player_states}")
    logger.info(f"[ActionExecutor][INPUT] playerActions: {playerActions}")
    logger.info(f"[ActionExecutor][INPUT] state keys: {list(state.keys())}")
    phases = dsl_content.get('phases', {})
    current_phase = phases.get(current_phase_id, {}) or phases.get(str(current_phase_id), {})
    
    # Log phase info
    logger.info(f"[ActionExecutor] current_phase_id: {current_phase_id}")
    logger.info(f"[ActionExecutor] current_phase: {current_phase}")
    logger.info(f"[ActionExecutor] player_states: {state.get('player_states', {})}")
    
    # Debug: Print entire received state
    logger.info(f"[ActionExecutor][DEBUG] Full received state keys: {list(state.keys())}")
    if "player_states" in state:
        logger.info(f"[ActionExecutor][DEBUG] Received player_states: {state.get('player_states', {})}")
    
    # 1. Define the model
    model = init_chat_model("openai:gpt-4.1-mini")

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

    # 3. Prepare system message with current state and actions to execute
    items_summary = summarize_items_for_prompt(state)
    logger.info(f"[ActionExecutor][output] items_summary: {items_summary}")
    current_phase_id = state.get("current_phase_id", 0)
    dsl_content = state.get("dsl", {})
    declaration = dsl_content.get('declaration', {}) if dsl_content else {}
    player_states = state.get("player_states", {})
    playerActions = state.get("playerActions", {})
    
    # Get current phase details
    phases = dsl_content.get('phases', {}) if dsl_content else {}
    # Try both int and string keys to handle YAML parsing variations
    current_phase = phases.get(current_phase_id, {}) or phases.get(str(current_phase_id), {})
    
    # Log game_notes for debugging
    game_notes = state.get('game_notes', [])
    logger.info(f"[ActionExecutor] Game Notes Count: {len(game_notes)}")
    if game_notes:
        logger.info(f"[ActionExecutor] Current Game Notes: {game_notes}")
    else:
        logger.info(f"[ActionExecutor] No Game Notes Available")

    # Generate actions from DSL phase if not explicitly provided
    actions_to_execute = state.get("actions", []) or []
    if not actions_to_execute and current_phase:
        # Extract actions from current phase
        phase_actions = current_phase.get("actions", [])
        if phase_actions:
            actions_to_execute = [{"description": f"Execute phase {current_phase_id} actions", "tools": phase_actions}]
    
    # Print current phase details
    if current_phase:
        logger.info(f"[ActionExecutor][DSL] Current phase ID: {current_phase_id}")
        logger.info(f"[ActionExecutor][DSL] Current phase: {current_phase}")
        current_phase_str = f"Current phase (ID {current_phase_id}):\n{current_phase}\n"
    else:
        logger.info(f"[ActionExecutor][DSL] Current phase ID: {current_phase_id} (not found in DSL phases)")
        current_phase_str = f"Current phase ID: {current_phase_id} (not found in DSL phases)\n"

    dsl_info = f"LOADED GAME DSL:\n{dsl_content}\n" if dsl_content else "No DSL loaded.\n"

    # Role assignment is handled exclusively by RefereeNode; ActionExecutor does not assign roles

    # (no-op here)
    ActionExecutor_system_prompt = await _load_prompt_async("ActionExecutor_system_prompt")

    system_message = SystemMessage(
        content=(
            "🎯 **YOU ARE THE DM (DUNGEON MASTER / GAME MASTER)**\n"
            "As the DM, you have complete responsibility for running this game. You must:\n\n"
             "📊 **CURRENT GAME STATE** (Analyze these carefully):\n"
            f"itemsState (current frontend layout): {items_summary}\n"
            f"{current_phase_str}\n"
            f"player_states: {player_states}\n"
            f"phase history: {state.get('phase_history', [])}\n" 
            f"game_notes: {game_notes[-5:] if game_notes else 'None'}\n"
            f"Game Description: {declaration.get('description', 'No description available')}\n"
            "GAME DSL REFERENCE (for understanding game flow):\n"
            "🎯 ACTION EXECUTOR:\n"
            f"Actions to execute: {actions_to_execute}\n\n"
            f"{ActionExecutor_system_prompt}"
            
            f"🚨 **PHASE 1 ENFORCEMENT** - Current phase_id is {current_phase_id}:\n"
            "IF current_phase_id == 1 (Role Assignment):\n"
            "  • You MUST create createCharacterCard for each player showing their specific role\n"
            f"  • Current player_states show these roles: {[(pid, pdata.get('role', 'NO_ROLE')) for pid, pdata in player_states.items()]}\n"
            "  • Create one createCharacterCard for each player with their actual role from player_states\n"
            "  • Use audience_type=false and audience_ids=[player_id] for each card\n"
            "  • DO NOT create generic role assignment messages - CREATE SPECIFIC ROLE CARDS!\n\n"
            
            "🔧 TOOL USAGE:\n"
            "- Exact tool names (no prefixes), capture returned IDs for reuse\n"
        )
    )


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
                    goto=END
                )
    except Exception:
        pass



    # 4. Trim messages and filter out orphaned ToolMessages
    full_messages = state.get("messages", []) or []
    trimmed_messages = full_messages[-20:]  # Increased to accommodate multiple tool calls
    
    # Filter out incomplete AIMessage + ToolMessage sequences using global function
    trimmed_messages = filter_incomplete_message_sequences(trimmed_messages)
    
    trimmed_messages = [msg for msg in trimmed_messages if not isinstance(msg, HumanMessage)]
    
    latest_state_system = SystemMessage(
        content=(
            "LATEST GROUND TRUTH (authoritative):\n"
            f"- items: {items_summary}\n"
            f"- current_phase_id: {current_phase_id}\n"
            f"- player_states: {player_states}\n"
            f"- actions to execute: {actions_to_execute}\n"
        )
    )

    response = await model_with_tools.ainvoke([
        system_message,
        latest_state_system,
    ], config)

    # === DETAILED LLM RESPONSE LOGGING ===
    logger.info(f"[ActionExecutor][LLM_OUTPUT] Raw response content: {response.content}")
    logger.info(f"[ActionExecutor][LLM_OUTPUT] Response type: {type(response)}")
    
    # Log output
    try:
        content_preview = getattr(response, "content", None)
        if isinstance(content_preview, str):
            logger.info(f"[ActionExecutor][LLM_OUTPUT] Content preview: {content_preview[:400]}")
        else:
            logger.info(f"[ActionExecutor][LLM_OUTPUT] Content: (non-text)")
        tool_calls = getattr(response, "tool_calls", []) or []
        logger.info(f"[ActionExecutor][TOOL_CALLS] Total tool calls: {len(tool_calls)}")
        if tool_calls:
            logger.info(f"[ActionExecutor][TOOL_CALLS] Tool calls details: {tool_calls}")
            for tc in tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                logger.info(f"[ActionExecutor][TOOL_CALLS] Individual tool_call name={name} args={args}")
        else:
            logger.info("[ActionExecutor][TOOL_CALL] tool_calls: none")
    except Exception:
        pass

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
            followup_response = await model_with_tools.ainvoke([
                strict_creation_system,
                latest_state_system,
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

    # Do not change phase here; PhaseNode is authoritative for transitions
    current_phase_id = state.get("current_phase_id", 0)
    updated_phase_id = current_phase_id
    logger.info(f"[ActionExecutor] Maintaining current phase_id: {updated_phase_id}")
    
    # Do not modify player_states here; RefereeNode owns role assignment
    final_player_states = state.get("player_states", {})

    # Actions completed, end execution; mark phase 0 UI as done so InitialRouter won't loop back
    logger.info(f"[ActionExecutor][end] === ENDING ===")
    # === DETAILED OUTPUT LOGGING ===
    logger.info(f"[ActionExecutor][OUTPUT] Command goto: END")
    logger.info(f"[ActionExecutor][OUTPUT] final_player_states: {final_player_states}")
    logger.info(f"[ActionExecutor][OUTPUT] updated_phase_id: {updated_phase_id}")

    # Generate monotonic state version
    state_version = get_next_state_version()
    timestamp = time.time()
    
    logger.info(f"[State Version] {state_version} - Phase {updated_phase_id} - Updated by ActionExecutor at {timestamp}")

    return Command(
        goto="__end__",
        update={
            # Use final_messages like agent.py
            "messages": response,
            "items": state.get("items", []),
            "player_states": final_player_states,  # Updated with role assignments
            "current_phase_id": updated_phase_id,
            "current_phase_name": get_phase_info_from_dsl(updated_phase_id, dsl_content)[1],
            "actions": [],  # Clear actions after execution
            "dsl": state.get("dsl", {}),  # Persist DSL
            "roomSession": state.get("roomSession", {}),  # Persist roomSession
            "phase0_ui_done": True if updated_phase_id == 0 else state.get("phase0_ui_done", True),
            # 🔑 Monotonic version control (avoid underscore prefix for CopilotKit compatibility)
            "stateVersion": state_version,
            "stateTimestamp": timestamp,
            "updatedBy": "ActionExecutor",
        }
    )

# Define the workflow graph
workflow = StateGraph(AgentState)

# Add all nodes
workflow.add_node("InitialRouterNode", InitialRouterNode)
workflow.add_node("ChatBotNode", ChatBotNode)
workflow.add_node("BotBehaviorNode", BotBehaviorNode)
workflow.add_node("RefereeNode", RefereeNode)
workflow.add_node("PhaseNode", PhaseNode)
# workflow.add_node("RoleAssignmentNode", RoleAssignmentNode)
workflow.add_node("ActionExecutor", ActionExecutor)
# workflow.add_node("ActionValidatorNode", ActionValidatorNode)

# Set entry point
workflow.set_entry_point("InitialRouterNode")

# Compile the graph (LangGraph API handles persistence itself in local_dev/cloud)
graph = workflow.compile()
