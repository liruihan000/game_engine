"""
DM Agent with Bot - Simple routing node implementation
This creates an initial routing node that directs flow based on current phase.
"""

import logging
import os  
import yaml
from dotenv import load_dotenv
from typing import Literal, List, Dict, Any, Optional
from tools.prompts import _load_prompt_async
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
    
    
    # Bind tools to model
    model_with_tools = model.bind_tools(chat_tools)
    game_notes = state.get('game_notes', [])

    chat_system_prompt = await _load_prompt_async("chat_system_prompt")


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
    {chat_system_prompt}
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

async def BotBehaviorNode(state: AgentState, config: RunnableConfig) -> Command[Literal["RefereeNode"]]:
    """
    BotBehaviorNode analyzes bot behavior and generates responses for non-human players.
    
    Input:
    - trimmed_messages: Recent message history
    - player_states: Current player states
    - current_phase and declaration: Phase configuration
    - need_feed_back_dict: Required feedback info
    
    Output:
    """

    player_states = state.get("player_states", {})
    current_phase_id = state.get("current_phase_id", 0)
    # Remove need_feed_back_dict dependency - use autonomous analysis only
    dsl_content = state.get("dsl", {})
    
    # Get current phase details and NEXT phase for pre-analysis
    phases = dsl_content.get('phases', {}) if dsl_content else {}
    # Try both int and string keys to handle YAML parsing variations
    current_phase = phases.get(current_phase_id, {}) or phases.get(str(current_phase_id), {})
    declaration = dsl_content.get('declaration', {}) if dsl_content else {}
    playerActions = state.get("playerActions", {})
    game_notes = state.get('game_notes', [])

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
            f"- **Player Actions**: {_limit_actions_per_player(playerActions, 1) if playerActions else {}}\n\n"
            f"- **Game Notes**: {game_notes[-5:] if game_notes else 'None'}\n\n"
            f"- **Items State**: {items_summary}\n\n"
            f"- **Declaration**: {declaration}\n\n"
            f"- **Bot Behavior System Prompt**: {bot_behavior_system_prompt}\n\n"
          
        )
    )

    # Call LLM with backend tool bound
    response = await model_with_tools.ainvoke([system_message], config)
    
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
    
    # Route to RefereeNode
    logger.info("[BotBehaviorNode] Routing to RefereeNode")
    return Command(
        goto="RefereeNode",
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


    playerActions = state.get("playerActions", {})
    game_notes = state.get('game_notes', [])
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
    model_with_tools = model.bind_tools([update_player_state, add_game_note, set_next_phase], parallel_tool_calls=True)
    
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
            
            "🎯 **TRIPLE MISSION**:\n"
            "1. **State Updates**: Process actions → update player_states (highest priority)\n"
            "2. **Game Notes**: Record events, decisions, and reminders for all nodes\n"
            "3. **Phase-Aware Analysis**: Use current + next phase info for smarter decisions\n\n"
            
            "📋 **CRITICAL STATE UPDATE RULES**:\n"
            "🚨 **UNDERSTAND PLAYER STATES FIRST**: Carefully read declaration.player_states definitions!\n"
            "• Each field has specific meaning and update conditions\n"
            "• speaker_rounds_completed: Only increment AFTER player actually completed their speaker turn to 1\n"
            "• is_speaker: Only update based on actual game flow, not speculation\n"
            "• statements_published: Only set to true AFTER statements are actually shared\n"
            "• DO NOT fabricate or assume state changes - base on actual evidence\n"
            "• Example: speaker_rounds_completed += 1 ONLY after player finished speaking phase\n"
            "• Example: is_speaker = False ONLY when speaker role actually transitions\n"
            "• TIMING MATTERS: Update states when events actually happen, not when anticipated\n\n"
            
            "🗳️ **STRICT PLAYER ACTIONS ANALYSIS RULE**:\n"
            "🚨 **CRITICAL**: Only use ACTUAL data from playerActions - NEVER use example values!\n"
            "• Find each player's LATEST action: highest timestamp AND matching current phase name\n"
            "• Extract EXACT vote choices, targets, statements from actual playerActions content\n"
            "• Example process: If playerActions shows 'Player 2 voted for statement 1' → vote_choice=1\n"
            "• If playerActions shows 'shared statements: I love dogs, I hate cats, I own 5 birds' → use THESE exact statements\n"
            "• FORBIDDEN: Using example values like 'I've been to Japan' when playerActions says different\n"
            "• FORBIDDEN: Inventing vote results not present in playerActions\n"
            "• MANDATORY: Cross-reference action timestamp and phase name before processing\n"
            "• UPDATE game_notes with ACTUAL OUTCOMES from playerActions data only\n"
            "• UPDATE player_states based on REAL actions, not hypothetical examples\n"
            "🔍 **COMPREHENSIVE STATE UPDATE CHECK**:\n"
            "• **SCAN ALL PLAYER_STATES**: Check every field in every player's state for needed updates\n"
            "• **SCORE CALCULATIONS**: For reveal/results phases, calculate and update scores based on actual data\n"
            "  - Example: Two Truths - compare vote_choice vs lie_index, update score accordingly\n"
            "  - Example: Werewolf - update elimination counts, survival streaks, etc.\n"
            "• **GAME PROGRESSION**: Update round counters, phase completions, win conditions\n"
            "• **ACHIEVEMENT TRACKING**: Update any achievement or milestone fields\n"
            "• **MANDATORY**: Every reveal/results phase MUST include score/progress updates\n\n"
            
            "📝 **GAME NOTES WRITING STANDARDS**:\n"
            "✅ CORRECT: 'Player 1 voted statement 2 (correct +1 point), Player 3 voted statement 1 (wrong +0 points)'\n"
            "❌ WRONG: 'Players voted', 'differing votes', 'All players have voted'\n"
            "✅ CORRECT: 'Player 2 (speaker) chose statement 2 as lie, earned +0 points this round'\n"
            "❌ WRONG: 'Voting completed', 'votes received and recorded'\n"
            "✅ CORRECT: 'Round totals: Player 1: 5 points, Player 2: 3 points, Player 3: 2 points'\n"
            "RULE: Always specify WHO did WHAT with exact POINTS EARNED and TOTAL SCORES\n\n"
            
            "🚫 **VOTING VALIDATION & ERROR HANDLING**:\n"
            "1. Check voting eligibility BEFORE updating player_states:\n"
            "   • can_vote=true AND is_speaker=false (for statement voting)\n"
            "2. If invalid vote detected:\n"
            "   • DO NOT call update_player_state for vote_choice\n"
            "   • Record in game_notes: 'Player X vote invalid - is current speaker'\n"
            "3. Only process and record VALID votes from eligible players\n\n"
            
            "🏆 **SCORING & RESULTS RECORDING**:\n"
            "After voting phase, MUST do:\n"
            "1. Get correct answer from player_states: speaker's 'chosen_lie' field\n"
            "2. Compare votes: 'Player 1 voted X (correct/wrong), Player 3 voted Y (correct/wrong)'\n"
            "3. UPDATE player_states scores: Call update_player_state for each player's new total score\n"
            "4. Record detailed results in game_notes:\n"
            "   • Round outcome: 'Player 1 voted 2 (correct), Player 3 voted 1 (wrong)'\n"
            "   • Score changes: 'Player 1: +1 point, Player 2: +0 points, Player 3: +0 points'\n"
            "   • Current totals: 'Total scores - Player 1: 3 points, Player 2: 1 point, Player 3: 2 points'\n"
            "MANDATORY: Update both player_states scores AND record complete results in game_notes\n\n"
            
            "💀 **DEATH STATUS & RESULTS ANNOUNCEMENT CHECK**:\n"
            f"• **CURRENT STATUS**: Living: {[pid for pid, data in player_states.items() if data.get('is_alive', True)]}, Dead: {[pid for pid, data in player_states.items() if not data.get('is_alive', True)]}\n"
            "🚨 **CRITICAL**: If current phase is a results/announcement phase, check player survival:\n"
            "• **VALIDATE**: All state updates must respect death/elimination status\n"
            "• **CRITICAL RULE**: Dead players (is_alive=false) CANNOT perform ANY actions or participate\n"
            "• Examine player_states for is_alive=false players\n"
            "• If any player died (is_alive changed from true to false):\n"
            "  - **RECORD**: Add 🔴 CRITICAL game note when setting is_alive=false for any player\n"
            "  - Record death in game_notes: 'Player X (RoleName) has been eliminated/died this phase'\n"
            "  - Include elimination reason if available in playerActions or game context\n"
        "  - Calculate impact on game state (team balance, role distribution)\n"
            "• If multiple deaths occurred, record each separately with specific details\n"
            "📝 **MANDATORY PHASE SUMMARY REQUIREMENT**:\n"
            "• **ALWAYS write a narrative summary** of what happened in this phase to game_notes\n"
            "• **FORMAT EXAMPLES**:\n"
            "  - Night phases: 'Last night, the Werewolves chose to eliminate Player 1 (Detective). However, Player 1 was protected by the Doctor and survived. There were no deaths last night.'\n"
            "  - Day phases: 'During day voting, Player 2 (Werewolf) was eliminated by majority vote. The village successfully identified a werewolf.'\n"
            "  - Reveal phases: 'Dawn revealed the night outcomes: Player 3 (Villager) was eliminated by werewolves. The Doctor's protection saved Player 1. Current survivors: Players 1, 2, 4.'\n"
            "  - Alternative outcomes: 'Last night, the Werewolves eliminated Pla～yer 3 (Villager). The Doctor protected Player 1, but Player 3 was not protected and died.'\n"
            "• **INCLUDE**: Actions taken, protection attempts, actual outcomes, survival/death results, revelations\n"
            "• **WRITE**: Clear, narrative-style summaries that explain cause and effect\n"
            "• **REVEAL PHASES SPECIAL**: Include what was reveale～d, who survived/died, current game state\n"
            "• **CONCLUSION REQUIREMENT**: Write comprehensive game state conclusion to game_notes\n"
            "  - Living players summary: 'Remaining alive: Player 2 (Doctor), Player 4 (Villager)'\n"
            "  - Team/role analysis: 'Team balance: 2 Villagers vs 1 Werewolf remaining'\n"
            "  - Game progression: 'Phase X completed with Y eliminations'\n\n"
            
            "🔮 **PHASE-AWARE DECISION MAKING**:\n"
            f"**Current Phase Analysis** ({current_phase.get('name', 'Unknown')}):\n"
            f"• Completion criteria: {current_phase.get('completion_criteria', {}).get('type', 'Unknown')}\n"
            f"• Expected actions: {current_phase.get('completion_criteria', {}).get('description', 'None')}\n"
            f"**Next Phase Preview** ({next_phase.get('name', 'Unknown') if next_phase else 'Game End'}):\n"
            f"• What's coming: {next_phase.get('description', 'Unknown') if next_phase else 'Final phase'}\n"
            f"• Preparation needed: Use this context to create better game notes and state updates\n\n"
            
            "🎭 **CRITICAL NEXT PHASE ROLE ASSIGNMENT RULE**:\n"
            "⚠️ **IF NEXT PHASE REQUIRES ROLE ASSIGNMENT** (Phase names like 'Role Assignment', 'Identity Assignment', 'Speaker Selection'):\n"
            "• **MANDATORY PREPARATION**: You MUST assign roles NOW in current phase and store in player_states\n"
            "• **STORE IN STATES**: Use update_player_state to set role field for each player\n"
            "• **RECORD IN GAME NOTES**: Add NEXT_PHASE type note documenting role assignments for future reference\n"
            "• **EXAMPLE**: If next phase is 'Role Assignment', assign roles like role='Werewolf', role='Villager' NOW\n"
            "• **TIMING**: Do this BEFORE phase transitions to ensure roles are ready when needed\n"
            "• **GAME NOTES FORMAT**: add_game_note('NEXT_PHASE', 'Roles assigned: Player1=Werewolf, Player2=Villager for upcoming Role Assignment phase')\n\n"
            "• **If next phase is speaker rotation, do Speaker Rotation Analysis**: For speaker rotation phases, count completed vs remaining turns\n"
            "  - add_game_note('SPEAKER_STATUS', 'Progress: 2 players completed speaking, 2 players remaining')\n\n"
            
            "• Correct who is the current speaker, who is the last speaker.\n"
            "• Correct do you need to select ann one for next round to do something?.\n"
            
            " **Then analyze Game Declaration rules** for elimination/death conditions, role abilities, and win conditions\n"
            " **Then examine Player Actions** to see what each player actually did:\n"
            "\n"
            "**Action-to-State Updates:**\n"
            "- **Voting actions** ('voted to eliminate X'): Track votes and apply elimination rules from Declaration\n"
            "  * If rules say 'most voted dies' and X got most votes → set X's is_alive=false\n"
            "  * Update vote tracking fields in player states\n"
            "  * **CRITICAL**: Add 🔴 CRITICAL game note when player dies\n"
            "- **Role ability actions** ('protected/investigated/targeted player X'):\n"
            "  * Set action completion: night_action_submitted=True or day_action_submitted=True\n"
            "  * Set last_night_action='[action_type]' or last_day_action='[action_type]'\n"
            "  * Set targets: last_night_target=X or last_day_target=X\n"
            "  * Apply ability effects based on Declaration rules (protection saves, investigation reveals, etc.)\n"
            "- **Game rule applications**:\n"
            "  * Cross-reference actions with Declaration rules to determine state changes\n"
            "  * Update known_alignments for investigators based on their targets\n"
            "  * Handle elimination/death according to game-specific rules\n"
            "\n"
            "**EXECUTION APPROACH**:\n"
            "1. **Read Game Declaration rules** to understand what causes state changes\n"
            "2. **Analyze each player's actions** from Player Actions data\n"
            "3. **Make referee judgments** - determine results, winners/losers, rule violations\n"
            "4. **Apply rules to actions** to determine what states should change\n"
            "5. **Use update_player_state tool** to make only the necessary state updates\n"
            "6. **Use add_game_note tool** to record all judgment results and reasoning\n"
            "\n"
            "**Examples of Rule-Based Updates**:\n"
            "• If Declaration says 'werewolves win when equal/outnumber villagers' + current states show this → game_over updates\n"
            "• If Declaration describes voting elimination + Player Actions show vote tallies → update is_alive for eliminated player\n"
            "• If Declaration defines role abilities + Player Actions show ability usage → update target/action tracking fields\n\n"

            "TASK 2: EMERGENCY Role Assignment (ONLY if ALL players have completely empty roles).\n"
            "IMPORTANT: Primary role assignment is handled by PhaseNode for phases 0-2.\n"
            "RefereeNode should ONLY assign roles as emergency fallback.\n\n"
            
            "Role Assignment Emergency Criteria (ALL must be true):\n"
            "- Phase name/description explicitly mentions 'assign' or 'role' (case-insensitive)\n"  
            "- declaration.roles exists\n"
            "- ALL players have completely empty/missing role field (not just some)\n"
            "- No player has any assigned role whatsoever\n\n"
            
            "CRITICAL PROTECTION - When assigning roles:\n"
            "- NEVER overwrite any existing role (even empty string '' counts as 'assigned')\n"
            "- If ANY player already has a role, SKIP role assignment entirely\n"
            "- Only proceed if player_states shows ALL players with role=null or role missing\n"
            "- This ensures PhaseNode has priority for role assignment"
           
           "- The ouput is updated_player_states, which is a updated version of player_states, keep the same keys and values for each player, don't change the keys and values for each player, only update the values that have changed.\n"
            
            
            "  Use update_player_state tool to update player state, you can call the tool for multiple in one time.\n"
            "- Maintain consistency with game rules and phase requirements\n"
            "- Handle eliminations, votes, role abilities, and status changes\n"
            "- Use player IDs as string keys (\"1\", \"2\", \"3\", etc.)\n"
            "- Include conclusions array with key events that happened:\n"
            "  * Player eliminations/deaths (who died and how)\n"
            "  * Game outcomes (who won/lost)\n"
            "  * Important discoveries or revelations\n"
            "  * No events (if nothing significant happened)\n\n"
            
            "💀 **CRITICAL ELIMINATION RULE**:\n"
            "- If ANY player is eliminated/dies/is voted out during the game:\n"
            "  * IMMEDIATELY set their is_alive=false using update_player_state tool\n"
            "  * This marks them as dead and removes them from active gameplay\n"
            "  * Dead players cannot vote, act, or participate in any game mechanics\n"
            "  * Example: update_player_state(player_id='3', state_name='is_alive', state_value=False)\n"
            "- ALWAYS update player status when processing elimination events\n"
            "- Remember: is_alive=false means the player is considered dead in the game\n\n"
            
            "📝 **GAME NOTES REQUIREMENTS**:\n"
            "After processing actions, create comprehensive notes for other nodes:\n"
            "• **Critical Events**: '🔴 CRITICAL: Player 3 (Doctor) eliminated - marked is_alive=false'\n"
            "• Important decisions**: '🎯 DECISION: Selected Player 2 as next speaker (turn_order)'\n"
            "• **Bot Reminders**: '🤖 BOT ACTION: Player 4 needs to complete werewolf vote'\n"
            "• **Voting Status**: '⚠️ REMINDER: Player 1 has NOT voted in current phase'\n"
            "* **Select the next player to do something**."
            "These notes guide ActionExecutor, BotBehaviorNode, and PhaseNode decisions.\n\n"
            
            "🎯 **REFEREE JUDGMENT & RESULT ANALYSIS**:\n"
            "As the game referee, you must analyze player actions and make official judgments:\n"
            "• **Voting Results**: Count votes, determine eliminations, record outcomes\n"
            "  - add_game_note('DECISION', 'Vote tally complete: Player 3 eliminated with 3 votes vs Player 2 with 1 vote')\n"
            "• **Night Action Results**: Process werewolf attacks, doctor protections, detective investigations\n"
            "  - add_game_note('DECISION', 'Werewolves targeted Player 4, Doctor protected Player 4 - no elimination')\n"
            "• **Win Condition Checks**: Evaluate if game end conditions are met\n"
            "  - add_game_note('GAME_STATUS', 'Win condition check: 2 werewolves vs 3 villagers - game continues')\n"
            "• **Rule Violations**: Identify invalid actions or rule violations\n"
            "  - add_game_note('CRITICAL', 'Player 3 attempted to vote while dead - action ignored')\n"
            "• **Phase Branch & End Condition Analysis**: If next phase has branches, analyze conditions and suggest path\n"
            "  - add_game_note('PHASE_SUGGESTION', 'End condition analysis: 3/4 players completed speaking, suggest continue current phase')\n"
            "  - add_game_note('BRANCH_RECOMMENDATION', 'Branch condition met: all players finished, recommend transition to results phase')\n"
           
            "⚠️ **IMPORTANT**: When calling add_game_note, provide CLEAN content without emoji prefixes:\n"
            "✅ CORRECT: add_game_note('CRITICAL', 'Player 3 eliminated - marked is_alive=false')\n"
            "❌ WRONG: add_game_note('CRITICAL', '🔴 CRITICAL: Player 3 eliminated')\n"
            "The function will automatically add the appropriate emoji and formatting.\n\n"
            
            "🚨 **TOOL EXECUTION ORDER**:\n"
            "1. **FIRST**: Analyze player actions and make referee judgments\n"
            "2. **SECOND**: Use update_player_state tools for all state changes based on judgments\n"
            "3. **THIRD**: Use add_game_note tools to record judgment results and outcomes\n"
            "4. **MULTIPLE CALLS**: You can call both tools multiple times as needed\n\n"
        )
    )
    
    

    response = await model_with_tools.ainvoke([system_message], config)



    tool_calls = getattr(response, "tool_calls", []) or []
    current_game_notes = list(state.get("game_notes", []))
    transition_from_tool = None
    next_phase_id_from_tool = None
    transition_reason = ""
    
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
                    dict(player_states), pid, state_name, state_value
                )
        elif name == "add_game_note":
            note_type = args.get("note_type")
            content = args.get("content")
            if note_type and content:
                current_game_notes = _execute_add_game_note(
                    current_game_notes, note_type, content
                )
        elif name == "set_next_phase":
            transition_from_tool = args.get("transition")
            next_phase_id_from_tool = args.get("next_phase_id")
            transition_reason = args.get("transition_reason", "")
            logger.info(f"[RefereeNode] Tool call: transition={transition_from_tool}, next_phase_id={next_phase_id_from_tool}, reason: {transition_reason}")

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



    normalized_pid, is_valid = _normalize_and_validate_phase_id(next_phase_id_from_tool, phases)
    if is_valid:
        target_phase_id = normalized_pid
    else:
        target_phase_id = current_phase_id


    # Record current phase in history
    current_phase_history = state.get("phase_history", [])
    phase_name = phases.get(target_phase_id, {}).get('name', f'Phase {target_phase_id}') or phases.get(str(target_phase_id), {}).get('name', f'Phase {target_phase_id}')
    
    phase_entry = {
        "phase_id": target_phase_id,
        "phase_name": phase_name
    }
    current_phase_history.append(phase_entry)



    return Command(
        goto="ActionExecutor",
        update={
            "player_states": current_player_states,
            "game_notes": current_game_notes,
            "roomSession": state.get("roomSession", {}),
            "dsl": state.get("dsl", {}),
            "phase_history": state.get("phase_history", []),
            "current_phase_id": target_phase_id,
            "phase_history": current_phase_history
        }
    )

async def ActionExecutor(state: AgentState, config: RunnableConfig) -> Command[Literal["__end__"]]:
    """
    Execute actions from DSL and current phase by calling frontend tools.
    Audience-aware rendering: always choose explicit audience permissions per component
    and render public, group, and individual UIs according to the DSL phase design.
    Can make announcements based on RefereeNode conclusions.
    """
    
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

    # 3. Prepare system states with current state and actions to execute
    items_summary = summarize_items_for_prompt(state)
    current_phase_id = state.get("current_phase_id", 0)
    dsl_content = state.get("dsl", {})
    declaration = dsl_content.get('declaration', {}) if dsl_content else {}
    player_states = state.get("player_states", {})
    playerActions = state.get("playerActions", {})
    phases = dsl_content.get('phases', {}) if dsl_content else {}
    current_phase = phases.get(current_phase_id, {}) or phases.get(str(current_phase_id), {})
    game_notes = state.get('game_notes', [])
    if current_phase:
        current_phase_str = f"Current phase (ID {current_phase_id}):\n{current_phase}\n"
    else:
        logger.info(f"[ActionExecutor][DSL] Current phase ID: {current_phase_id} (not found in DSL phases)")
        current_phase_str = f"Current phase ID: {current_phase_id} (not found in DSL phases)\n"


    system_message = SystemMessage(
        content=(
            "🎯 **YOU ARE THE DM (DUNGEON MASTER / GAME MASTER)**\n"
            "As the DM, you have complete responsibility for running this game. You must:\n\n"
             "📊 **CURRENT GAME STATE** (Analyze these carefully):\n"
            f"itemsState (current frontend layout): {items_summary}\n"
            f"**Current Phase**: {current_phase_str}\n"
            f"player_states: {player_states}\n"
            f"phase history: {state.get('phase_history', [])}\n" 
            f"game_notes: {game_notes[-5:] if game_notes else 'None'}\n"
            f"Game Description: {declaration.get('description', 'No description available')}\n"
            f"**Current Actions to Execute**: {current_phase.get('actions', [])}\n"
            "GAME DSL REFERENCE (for understanding game flow):\n"
 
            "📋 **DM CORE RESPONSIBILITIES** (Master these completely):\n"
            "1. **GAME NOTES AWARENESS**: Read game_notes for critical state changes and UI guidance\n"
            "2. **DEAD PLAYER FILTERING**: NEVER create voting options for players with is_alive=false\n"
            "3. **SPEAKER MANAGEMENT**: Always identify who is the current speaker this round\n"
            "4. **ROUND CONCLUSIONS**: Understand what happened last round and what was concluded\n"
            "5. **PERSISTENT DISPLAYS**: Know what information must stay visible on screen always\n"
            "6. **RULE MASTERY**: Deeply understand the game rules and DSL inside-out\n"
            "7. **SCREEN STATE AWARENESS**: Use itemsState to know what players currently see\n"
            "8. **COMPONENT LIFECYCLE**: Determine what UI components to keep vs delete vs create\n"
            "9. **DELETE BEFORE CREATE**: You MUST delete outdated components before creating new ones\n"
            "10. **ROUND OBJECTIVES**: Clearly understand what this round is trying to achieve\n"
            "11. **PROGRESSION CONDITIONS**: Know what conditions move the game to the next round\n"
            "💀 **DEATH MARKER MANDATORY REQUIREMENTS**:\n"
            "• **DEATH STATUS CHECK**: EVERY round, check player_states for is_alive=false\n"
            "• **MISSING MARKER CHECK**: If dead player exists but NO death_marker with audience_ids=[dead_player_id] in items, CREATE one immediately\n"
            "• **AUTOMATIC DEATH MARKERS**: Create death markers for ALL dead players automatically\n"
            "• **DEAD PLAYER ONLY VISIBILITY**: Death markers MUST use audience_type=false, audience_ids=[dead_player_id]\n"
            "• **PERMANENT MARKERS**: Death markers CANNOT be deleted - they persist until game end\n"
            "• **ONE PER DEAD PLAYER**: Ensure only one death marker exists per dead player\n"
            "• **EXAMPLE**: if player_states['2']['is_alive']=false, create: createDeathMarker(playerName='Player 2', playerId='2', audience_type=false, audience_ids=['2'], position='top-right')\n"
            "• **POSITIONING**: Place death markers in unique positions to avoid overlap\n"
            "• **DETECTION LOGIC**: Scan itemsState for existing death_marker items with matching audience_ids before creating new ones\n"
            "🚨 **SCORE CALCULATION RULE**: NEVER invent scores - use ONLY:\n"
            "• player_states: Get lie_index (correct answer) and vote_choice (player votes)\n"
            "• Example: if lie_index=1, then statements[1] is the lie\n"
            "• Compare each player's vote_choice with lie_index to determine correct/wrong\n"
            "• Display: 'Statement 2 (I've never broken a bone) was the lie. Player A voted 3 (wrong), Player B voted 1 (wrong)'\n"
            "• Use actual statements[] array content, not invented examples\n\n"
            "11.  **If you need you post some text, use createTextInputPanel() - creates floating input panel at bottom of screen\n"
            
            "🧒 **TREAT PLAYERS LIKE CHILDREN**: Give maximum information - they know NOTHING!\n"
            "- Explain everything clearly and simply\n"
            "- Provide as much helpful information as possible\n"
            "- Guide them through every step\n"
            "- Never assume they understand anything\n\n"
            
            "📋 CORE WORKFLOW (ALL ACTIONS IN SINGLE RESPONSE):\n"
            "**itemsState Analysis**: Format '[ID] type:name@position' shows current UI layout. Follow current_phase requirements.\n"
            "**Delete + Create**: Read itemsState to find existing IDs, delete outdated items, then create new components for current_phase.\n"
            "🚫 **POSITION OVERLAP PREVENTION**:\n"
            "• **NO DUPLICATE POSITIONS**: Check existing items before creating - NEVER place multiple items at same position\n"
            "• **POSITION ANALYSIS**: Read itemsState format '[ID] type:name@position' to identify occupied positions\n"
            "• **UNIQUE PLACEMENT**: Each new component MUST use a different position than existing items\n"
            "• **GRID POSITIONS**: top-left, top-center, top-right, middle-left, center, middle-right, bottom-left, bottom-center, bottom-right\n"
            "• **CONFLICT RESOLUTION**: If position occupied, choose next available position in grid\n"
            "• **EXAMPLE**: If center occupied, use middle-left or middle-right instead\n"
            "**MANDATORY Audience Permissions**: Every component MUST specify who can see it:\n"
            "  • Public: audience_type=true (everyone sees it)\n"
            "  • Private: audience_type=false + audience_ids=['1','3'] (only specified players see it)\n"
            "  • CRITICAL: Include proper audience permissions on each component (audience_type=true for public; or audience_type=false with audience_ids list)\n"
            "**Examples**: clearCanvas() + createPhaseIndicator(audience_type=true) + createActionButton(audience_ids=['2'])\n\n"
            
            "📝 **USER INPUT COLLECTION**: For games requiring player text input (like Two Truths and a Lie statements):\n"
            "• Use createTextInputPanel() - creates floating input panel at bottom of screen\n"
            "• Perfect for: statement collection, confession phases, text-based responses\n"
            "• Position: Fixed at bottom center of canvas for easy access\n"
            "• Example: createTextInputPanel(title='Enter your statements', placeholder='Type your 3 statements...', audience_ids=['1'])\n\n"
            
            "🏆 **GAME RESULT ANNOUNCEMENT RULE - GAME_NOTES PRIORITY**:\n"
            "**PRIMARY RULE**: Always check game_notes for conclusions first before making any announcements:\n"
            "• **MANDATORY CHECK**: Scan recent game_notes for any conclusions, decisions, or results\n"
            "• **GAME_NOTES PRIORITY**: If game_notes contain conclusions (winner declarations, elimination results, etc.), announce them immediately\n"
            "• **EXAMPLE**: If game_notes say 'Village team wins - all werewolves eliminated', create result display with this exact conclusion\n"
            "• **NO OVERRIDE**: NEVER contradict or ignore conclusions found in game_notes\n"
            "• **FALLBACK ONLY**: Only calculate results yourself if game_notes contain NO conclusions\n\n"
            "**FALLBACK DATA ANALYSIS** (only when game_notes have no conclusions):\n"
            "• Use player_states (scores, is_alive, role, etc.) for factual information\n"
            "• Reference recent game_notes for context and decisions\n"
            "• DO NOT fabricate or guess results - only state verified facts\n"
            "• Example: 'Player 2 won with 5 points' (from player_states.score)\n"
            "• Example: 'Village won - all werewolves eliminated' (from player_states.is_alive)\n"
            "• NO speculation, NO invented details - stick to observable data\n\n"
            
            "🚨 **ABSOLUTE PROHIBITION**: NEVER return with ONLY cleanup calls - THIS IS TASK FAILURE!\n"
            "**MANDATORY CREATE REQUIREMENT**: Every clearCanvas MUST be followed by create tools in SAME response!\n"
            "**CLEANUP TOOLS RESTRICTION**: clearCanvas cannot appear alone - must always be paired with create tools\n"
            "🧹 **AUTOMATIC CLEANUP REQUIREMENT**:\n"
            "• **PHASE TRANSITION CHECK**: If actions don't include clearCanvas, YOU must check itemsState and clean up irrelevant UI\n"
            "• **OUTDATED UI DETECTION**: Identify items that don't match current phase requirements\n"
            "• **AUTOMATIC CLEAR**: Use clearCanvas to remove outdated UI, preserve needed components via exemptList\n"
            "• **EXAMPLE**: If switching from voting to results phase, clearCanvas() before creating result displays\n"
            "🔄 **MANDATORY CLEAR ORDERING**:\n"
            "• **CLEAR FIRST**: clearCanvas() calls MUST be executed ahead all create tools\n"
            "• **SYNCHRONOUS EXECUTION**: Call cleanup tools first, then creation tools in same response\n"
            "• **CORRECT ORDER**: clearCanvas() → createPhaseIndicator() → createTimer()\n"
            "• **WRONG ORDER**: createPhaseIndicator() → clearCanvas() (creates then destroys)\n"
            "**EXECUTION PATTERN**: [AUTO-CLEANUP] + clearCanvas() + createPhaseIndicator() + createTimer() + createVotingPanel() + createDeathMarker(for_dead_players)\n"
            "⚡ **COMPLETE PHASE EXECUTION**: Execute clearCanvas + create actions for current_phase in ONE response!\n"
            "**Role Selection**: Analyze player_states - Werewolves: role='Werewolf', Alive: is_alive=true, Human: always ID '1'\n"
            "**Timers**: ~10 seconds (max 15), Layout: 'center' default\n"
            "**PHASE INDICATORS**: Always place at 'top-center' position (reserved for phase indicators)\n"
            "**DEFAULT VISIBILITY**: Unless explicitly private/group-targeted, make items PUBLIC with audience_type=true.\n\n"
            "**UI POSITION PRIORITY**: Always use 'center' first. Priority order: center → top-center → bottom-center. Only use next priority if current position is occupied.\n\n"
            "**CRITICAL**: there must be at least one tool set position='center'; createPhaseIndicator(position='top-center'); createTextDisplay(position='top-center'='center' | 'middle-left'  | 'middle-right'| 'bottom')\n\n"

            "📝 **GAME NOTES CRITICAL USAGE RULES**:\n"
            "• **🔴 CRITICAL notes**: Indicate player deaths - MUST exclude these players from all UI\n"
            "• **💀 DEATH MARKERS**: Use createDeathMarker tool to visually mark dead players on screen\n"
            "• **🚫 UI FILTER notes**: Explicitly tell you which players to exclude from voting/targeting\n"
            "• **⚠️ VOTING STATUS notes**: Show who hasn't voted - create reminders for these players\n"
            "• **🎯 DECISION notes**: Show automatic decisions made - incorporate into UI context\n"
            "• **🤖 BOT REMINDER notes**: Indicate which bots need UI for actions\n"
            "• **📖 PHASE SUMMARY notes**: Narrative summaries from RefereeNode - use for announcements\n"
            "• **🌅 REVEAL SUMMARIES**: Special summaries for Dawn/Reveal phases - use for outcome announcements\n"
            "• **🧠 LOGIC VALIDATION**: Check game_notes for consistency before using in UI\n"
            "  - Example ERROR: 'Werewolves chose Player 1, but Player 4 was protected' (Player 1 ≠ Player 4)\n"
            "  - Use player_states as truth source if game_notes contain logical errors\n"
            "• **📊 EVIDENCE-BASED CONCLUSIONS**: All announcements must be based on player_states data\n"
            "  - Example: lie_index=2 from player_states means statement 2 is the lie, not statement 1 or 3\n"
            "  - Example: vote_choice vs lie_index determines correct/wrong answers\n"
            "  - NEVER write conclusions without supporting data from player_states or game_notes\n"
            "  - If no evidence exists, display 'Results being calculated...' instead of guessing\n"
            "• **💀 DEAD PLAYER ACTION**: Always add action to check player_states for is_alive=false and createDeathMarker for each\n"
            "• ALWAYS read game_notes FIRST before creating any voting panels or target selection UI\n\n"
            
            "🎭 **CRITICAL ROLE ASSIGNMENT RULE** (Phase 1 'Role Assignment'):\n"
            "**MANDATORY ROLE TRANSPARENCY**: When assigning roles, you MUST inform each player of their identity! Don't hide the role from the player who has the role.\n"
            "  • NEVER hide or conceal a player's role from themselves\n"
            "  • Each player has their own private screen - they cannot see others' roles\n"
            "  • Create individual character cards: createCharacterCard(name='Player1Role', role='Detective', audience_type=false, audience_ids=['1'])\n"
            "  • Each character card is visible ONLY to its assigned player (private audience)\n"
            "  • Example: Player 1 gets Detective card (only they see it), Player 2 gets Werewolf card (only they see it)\n"
            "**ROLE CARD REQUIREMENT**: Every player with a role must receive their own private character card!\n\n"
            
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
    
    # Filter out HumanMessage from history for ActionExecutor
    from langchain_core.messages import HumanMessage
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
# workflow.add_node("PhaseNode", PhaseNode)
# workflow.add_node("RoleAssignmentNode", RoleAssignmentNode)
workflow.add_node("ActionExecutor", ActionExecutor)
# workflow.add_node("ActionValidatorNode", ActionValidatorNode)

# Set entry point
workflow.set_entry_point("InitialRouterNode")

# Compile the graph (LangGraph API handles persistence itself in local_dev/cloud)
graph = workflow.compile()



