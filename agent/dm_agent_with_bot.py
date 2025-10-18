"""
DM Agent with Bot - Simple routing node implementation
This creates an initial routing node that directs flow based on current phase.
"""

import logging
import os  
import yaml
from dotenv import load_dotenv
from typing import Literal, List, Dict, Any, Optional

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
import json
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
    player_states: Dict[str, Any] = {}
    gameName: str = ""  # Current game DSL name (e.g., "werewolf", "coup")
    dsl: dict = {}
    need_feed_back_dict: dict = {}
    referee_conclusions: List[str] = []
    roomSession: Dict[str, Any] = {}  # Room session data from frontend
    # Chat-specific fields for chatbot synchronization
    playerActions: Dict[str, Any] = {}  # Player actions
    phase_history: List[Dict[str, Any]] = []  # Phase transition history


def clean_llm_json_response(response_content: str) -> str:
    """Clean LLM response by extracting JSON from markdown code blocks if present."""
    response_content = response_content.strip()
    
    # Handle markdown code blocks if present - check if content contains ```json anywhere
    if '```json' in response_content:
        # Extract JSON from markdown code block
        lines = response_content.split('\n')
        json_lines = []
        inside_json = False
        for line in lines:
            if line.strip() == '```json':
                inside_json = True
                continue
            elif line.strip() == '```' and inside_json:
                break
            elif inside_json:
                json_lines.append(line)
        
        # Return extracted JSON if found
        if json_lines:
            response_content = '\n'.join(json_lines).strip()
    
    return response_content

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

def summarize_items_for_prompt(state: AgentState) -> str:
    """Summarize current UI items with ID, type, name, position - formatted for ActionExecutor deletion/creation decisions."""
    try:
        items = state.get("items", []) or []
        if not items:
            return "(no items)"
        
        # Build detailed summary with IDs for deletion
        lines: List[str] = []
        for it in items[:15]:  # Show more items for better context
            try:
                item_id = it.get("id", "unknown")
                item_type = it.get("type", "unknown")
                item_name = it.get("name", "unnamed")
                
                # Get position from data or fallback to item level
                data = it.get("data", {}) or {}
                position = data.get("position") or it.get("position") or "none"
                
                # Format: [ID] type:name@position
                lines.append(f"  [{item_id}] {item_type}:{item_name}@{position}")
            except Exception:
                continue
        
        more = "" if len(items) <= 15 else f"\n  (+{len(items)-15} more items...)"
        header = f"Canvas Items ({len(items)} total):"
        return header + "\n" + "\n".join(lines) + more
        
    except Exception as e:
        return f"(unable to summarize items: {e})"

@tool
def set_next_phase(transition: bool, next_phase_id: int, transition_reason: str) -> str:
    """
    Backend tool for PhaseNode to set phase transition decision.
    
    Args:
        transition: True if conditions are met and should transition, False to stay at current phase
        next_phase_id: The ID of the next phase to transition to (or current phase if transition=False)
        transition_reason: Brief explanation of the decision
    
    Returns:
        Confirmation message
    """
    action = "transitioning to" if transition else "staying at"
    return f"Phase decision: {action} phase {next_phase_id}. Reason: {transition_reason}"

@tool
def update_player_state(player_id: str, state_name: str, state_value: Any):
    """
    Update a single state value for a specific player.
    Player states structure: player_states[player_id][state_name] = state_value
    
    Args:
        player_id: Player identifier (e.g., "1", "2", "player_001")
        state_name: Name of the state to update (e.g., "role", "alive", "votes", "target")  
        state_value: New value for the state (can be string, int, bool, list, etc.)
        
    Returns:
        Confirmation message about the state update
    """
    return f"Will update player {player_id} state: {state_name} = {state_value}"


def _execute_update_player_state(current_player_states: dict, player_id: str, state_name: str, state_value: Any) -> dict:
    """
    Execute the actual logic to update player state. Returns updated player_states dict.
    
    Args:
        current_player_states: Current player states dict
        player_id: Player identifier
        state_name: Name of the state to update
        state_value: New value for the state
        
    Returns:
        Updated player_states dict
    """
    logger.info(f"[_execute_update_player_state] Player {player_id}: {state_name} = {state_value}")
    
    # Initialize player state if not exists
    if str(player_id) not in current_player_states:
        current_player_states[str(player_id)] = {}
    
    # Update the specific state
    current_player_states[str(player_id)][state_name] = state_value
    
    return current_player_states

@tool
def update_player_name(player_id: str, name: str, role: str) -> str:
    """
    Update player role information.
    
    Args:
        player_id: Player ID (e.g. '1', '2', '3')
        name: Player name (for display only)
        role: Player role to set
        
    Returns:
        Confirmation message about the role update
    """
    return f"Will update player {player_id} ({name}) role: {role}"


def _execute_update_player_name(current_player_states: dict, player_id: str, name: str, role: str) -> dict:
    """
    Execute the actual logic to update player role. Returns updated player_states dict.
    
    Args:
        current_player_states: Current player states dict
        player_id: Player identifier
        name: Player name (for display only)
        role: Player role
        
    Returns:
        Updated player_states dict, or original dict if player_id doesn't exist
    """
    # Check if player exists
    if str(player_id) not in current_player_states:
        logger.warning(f"[_execute_update_player_name] Player {player_id} not found, returning original state")
        return current_player_states
    
    logger.info(f"[_execute_update_player_name] Player {player_id} ({name}): role={role}")
    
    # Update only role
    current_player_states[str(player_id)]["role"] = role
    
    return current_player_states


@tool
def set_feedback_decision(player_id_list: list, need_feedback_message: str) -> str:
    """
    Set the feedback decision for current phase completion.
    
    Args:
        player_id_list: List of player IDs who need to provide feedback (e.g. [1, 2, 3])
        need_feedback_message: Message to display to waiting players
    
    Returns:
        Confirmation message about the feedback decision
    """
    return f"Feedback decision set: {len(player_id_list)} players need feedback"

def _execute_set_feedback_decision(player_id_list: list, need_feedback_message: str, state: AgentState) -> dict:
    """
    Execute the actual logic to set feedback decision. Returns feedback decision dict.
    
    Args:
        player_id_list: List of player IDs who need feedback
        need_feedback_message: Message for waiting players
        state: AgentState for context
    
    Returns:
        Dict with player_id_list and need_feedback_message
    """
    return {
        "player_id_list": player_id_list,
        "need_feedback_message": need_feedback_message
    }

def update_player_actions(player_id: str, actions: str, phase: str) -> str:
    """
    Record player actions for AI tracking. Use this to log what players (including bots) did in each phase.
    
    Args:
        player_id: Player ID (e.g. '1', '2', '3')
        actions: Description of what the player did (e.g. 'voted for Alice, defended Bob')
        phase: Current game phase (e.g. 'day_voting', 'night_action', 'discussion')
    
    Returns:
        Confirmation message about the recorded action
    """
    return f"Will record actions for player {player_id} in {phase}: {actions}"


def filter_incomplete_message_sequences(messages: list) -> list:
    """
    Filter out incomplete AIMessage + ToolMessage sequences to prevent OpenAI API errors.
    
    Only keeps AIMessage + ToolMessage pairs where ALL tool_calls have corresponding ToolMessage responses.
    This prevents "tool_calls must be followed by tool_messages" errors when tool execution is incomplete.
    
    Args:
        messages: List of messages to filter
        
    Returns:
        Filtered list of messages with complete sequences only
    """
    from langchain_core.messages import AIMessage, ToolMessage
    
    filtered_messages = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # Collect expected tool_call_ids - handle both dict and object formats
            expected_ids = set()
            for tc in msg.tool_calls:
                if isinstance(tc, dict):
                    tc_id = tc.get("id")
                else:
                    tc_id = getattr(tc, "id", None)
                if tc_id:
                    expected_ids.add(tc_id)
            
            # Collect following ToolMessages
            tool_messages = []
            j = i + 1
            while j < len(messages) and isinstance(messages[j], ToolMessage):
                tool_messages.append(messages[j])
                j += 1
            
            # Check if all expected tool_call_ids have responses
            received_ids = {tm.tool_call_id for tm in tool_messages if tm.tool_call_id}
            
            # Only keep if ALL tool_calls have responses
            if expected_ids and expected_ids == received_ids:
                filtered_messages.append(msg)
                filtered_messages.extend(tool_messages)
            
            i = j
        elif isinstance(msg, ToolMessage):
            # Orphaned ToolMessage, skip
            i += 1
        else:
            # Keep other messages (HumanMessage, SystemMessage, etc.)
            filtered_messages.append(msg)
            i += 1
    
    return filtered_messages


def _execute_update_player_actions(current_player_actions: dict, player_id: str, actions: str, phase: str, state: AgentState, current_player_states: dict) -> dict:
    """
    Execute the actual logic to add player actions. Returns updated player_actions dict.
    
    Args:
        current_player_actions: Current player actions state
        player_id: Player ID
        actions: Action description  
        phase: Current phase
        state: AgentState for getting room/player info
        current_player_states: Current player states
        
    Returns:
        Updated player_actions dict
    """
    import time
    
    # Get player name: prioritize player_states (updated), fallback to roomSession (original)
    player_name = f"Player {player_id}"
    
    # First try player_states (may contain updated name after role assignment)
    if player_id in current_player_states and "name" in current_player_states[player_id]:
        player_name = current_player_states[player_id]["name"]
    else:
        # Fallback to roomSession for original player name
        room_session = state.get("roomSession", {})
        if room_session and "players" in room_session:
            for p in room_session["players"]:
                if str(p.get("gamePlayerId", "")) == str(player_id):
                    player_name = p.get("name", player_name)
                    break
    
    # Initialize player actions if not exists
    if str(player_id) not in current_player_actions:
        current_player_actions[str(player_id)] = {
            "name": player_name,
            "actions": {}  # Dictionary of action_id -> action_data
        }
    
    # Generate simple action ID from 1
    all_action_ids = []
    for player_data in current_player_actions.values():
        if isinstance(player_data, dict) and "actions" in player_data:
            for action_data in player_data["actions"].values():
                if isinstance(action_data, dict) and "id" in action_data:
                    try:
                        all_action_ids.append(int(action_data["id"]))
                    except (ValueError, TypeError):
                        pass
    action_id = str(max(all_action_ids, default=0) + 1)
    timestamp = int(time.time() * 1000)
    
    current_player_actions[str(player_id)]["name"] = player_name  # Update name
    current_player_actions[str(player_id)]["actions"][action_id] = {
        "action": actions,
        "timestamp": timestamp,
        "phase": phase,
        "id": action_id
    }
    
    logger.info(f"📝 Added action for {player_name} ({player_id}) in {phase}: {actions}")
    
    return current_player_actions

# Centralized backend tools (shared across nodes)
backend_tools = [
    update_player_state,
    update_player_actions,
    set_next_phase,
    set_feedback_decision,
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
    "createBackgroundControl",
    "promptUserText",
    # Card game UI
    "createHandsCard",
    "updateHandsCard",
    "setHandsCardAudience",
    "createHandsCardForPlayer",
    # Broadcast input tool
    "displayBroadcastInput",
    # Scoreboard tools
    "createScoreBoard",
    "updateScoreBoard",
    "setScoreBoardEntries",
    "upsertScoreEntry",
    "removeScoreEntry",
    # Update tools for common components
    "updatePhaseIndicator",
    "updateTextDisplay",
    "updateActionButton",
    "updateCharacterCard",
    "updateVotingPanel",
    "updateResultDisplay",
    "updateTimer",
    "setItemPosition",
    # Chat-driven vote
    "submitVote",
    # Coins UI tools
    "createCoinDisplay",
    "updateCoinDisplay",
    "incrementCoinCount",
    "setCoinAudience",
    # Statement board & Reaction timer
    "createStatementBoard",
    "updateStatementBoard",
    "createReactionTimer",
    "startReactionTimer",
    "stopReactionTimer",
    "resetReactionTimer",
    # Night overlay & Turn indicator
    "createTurnIndicator",
    "updateTurnIndicator",
    # Health & Influence
    "createHealthDisplay",
    "updateHealthDisplay",
    "createInfluenceSet",
    "updateInfluenceSet",
    "revealInfluenceCard",
    # Score tracking UI  
    "createScoreBoard",
    # Component management tools
    "deleteItem",
    "clearCanvas",
    # Player state management
    "markPlayerDead",
    # Chat tools
    "addBotChatMessage"
])


BACKEND_TOOL_NAMES = {t.name for t in backend_tools}

async def load_dsl_by_gamename(gamename: str) -> dict:
    """Load DSL content from YAML file based on gameName"""
    if not gamename:
        logger.warning("[DSL] No gameName provided, returning empty DSL")
        return {}
    
    try:
        import aiofiles
        dsl_file_path = os.path.join(os.path.dirname(__file__), '..', 'games', f"{gamename}.yaml")
        logger.info(f"[DSL] Attempting to load DSL from path: {dsl_file_path}")
        logger.info(f"[DSL] File exists check: {os.path.exists(dsl_file_path)}")
        if os.path.exists(dsl_file_path):
            async with aiofiles.open(dsl_file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                logger.info(f"[DSL] File content length: {len(content) if content else 0} characters")
                dsl_content = yaml.safe_load(content)
                logger.info(f"[DSL] Parsed DSL keys: {list(dsl_content.keys()) if dsl_content else []}")
            logger.info(f"[DSL] Successfully loaded DSL for game: {gamename}")
            return dsl_content
        else:
            logger.warning(f"[DSL] DSL file not found: {dsl_file_path}")
            return {}
    except Exception as e:
        logger.error(f"[DSL] Failed to load DSL for {gamename}: {e}")
        return {}

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

async def InitialRouterNode(state: AgentState, config: RunnableConfig) -> Command[Literal["ChatBotNode", "FeedbackDecisionNode"]]:
    """
    Initial routing node that loads DSL and routes based on current phase.
    
    Routes to: 
    - FeedbackDecisionNode if current_phase_id > 0
    """
    # Print game name from state
    game_name = state.get("gameName", "")
    logger.info(f"[InitialRouterNode] Game name from state: {game_name}")
    
    
    # === DETAILED INPUT LOGGING ===
    current_phase_id = state.get('current_phase_id', 0)
    player_states = state.get("player_states", {})
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

    # Ensure DSL is properly passed - use dsl_content if loaded, otherwise fallback to state
    final_dsl = dsl_content if dsl_content else state.get("dsl", {})
    updates["dsl"] = final_dsl

    
    
    return Command(goto="FeedbackDecisionNode", update=updates)

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
    
    model = init_chat_model("openai:gpt-4o")
    
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
    
    # Let LLM analyze message and call tool
    system_prompt = f"""
                You are a game bot AI. Analyze the latest chat message and respond if appropriate.

                Game State:
                - Current Phase: {current_phase_id}
                - Player States: {player_states}
                - Player Actions: {playerActions}
                - Room Session: {roomSession}
                - Available bots: {[f"{pid}: {data.get('name', f'Bot {pid}')}" for pid, data in player_states.items() if pid != "1"]}

                Chat Message: {last_msg.content}

                Instructions:
                1. If user targeted specific bot ('to Bot [name]:'), use that bot
                2. Otherwise, choose appropriate bot (non-player 1) 
                3. Generate natural response based on bot's role and game context
                4. MUST call addBotChatMessage tool with:
                - botId: the bot's player ID (e.g., "2", "3") 
                - botName: the bot's name
                - message: your generated response
                - messageType: "message"
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

async def FeedbackDecisionNode(state: AgentState, config: RunnableConfig) -> Command[Literal["BotBehaviorNode", "RefereeNode"]]:
    """
    FeedbackDecisionNode calls LLM to analyze who needs to provide feedback.
    
    Input:
    - trimmed_messages: Recent message history
    - player_states: Current player states  
    - current_phase and declaration: Phase configuration
    
    Output:
    - need_feed_back_dict with player_id_list and feedback message
    """
    # Print game name from state
    game_name = state.get("gameName", "")
    
    # Extract inputs - simplified with BaseMessage
    messages = state.get("messages", [])  # Safe dictionary access to BaseMessage list
    # logger.info(f"[FeedbackDecisionNode] raw_messages: {messages}")
    
    trimmed_messages = messages[-10:] if messages else []  # Keep last 10 messages for context
    trimmed_messages = filter_incomplete_message_sequences(trimmed_messages)
    player_states = state.get("player_states", {})
    current_phase_id = state.get("current_phase_id", 0)
    dsl_content = state.get("dsl", {})
    playerActions = state.get("playerActions", {})
    
    
    # Get current phase details
    phases = dsl_content.get('phases', {}) if dsl_content else {}
    # Try both int and string keys to handle YAML parsing variations
    current_phase = phases.get(current_phase_id, {}) or phases.get(str(current_phase_id), {})
    declaration = dsl_content.get('declaration', {}) if dsl_content else {}
    playerActions = state.get("playerActions", {})
    
    # Log phase info
    logger.info(f"[FeedbackDecisionNode] current_phase_id: {current_phase_id}")
    logger.info(f"[FeedbackDecisionNode] current_phase: {current_phase}")
    logger.info(f"[FeedbackDecisionNode] player_states: {player_states}")
    
    # Initialize LLM
    model = init_chat_model("openai:gpt-4o")
    # Bind both update_player_actions and set_feedback_decision in this node
    model_with_tools = model.bind_tools([update_player_actions, set_feedback_decision])
    
    # Extract last human message - Handle both LangChain Message objects and dict format
    # Get last message and check if it's human - simplified
    last_human_message = ""
    if messages and isinstance(messages[-1], HumanMessage):
        last_msg = messages[-1]
        content_lower = str(last_msg.content).lower().strip()
        # Skip generic control messages
        if content_lower not in ['continue', 'start game', 'start game.']:
            last_human_message = str(last_msg.content)

    logger.info(f"[FeedbackDecisionNode] last_human_message: {last_human_message}")

    # Create system message with structured approach like ActionExecutor
    system_message = SystemMessage(
        content=(
            "📊 **CURRENT GAME STATE** (Analyze these carefully):\n"
            f"Current Phase ID: {current_phase_id}\n"
            f"Current Phase Details: {current_phase}\n"
            f"Player States: {player_states}\n"
            f"Player Actions: {playerActions}\n"
            f"Recent Messages: {[str(msg) for msg in trimmed_messages]}\n"
            f"Last Human Message: {last_human_message}\n"
            f"Game Declaration: {declaration}\n\n"
            
            "🎯 **FEEDBACK DECISION CORE RESPONSIBILITIES**:\n"
            "1. **HUMAN ACTION LOGGING**: Record Player 1's actions when new messages exist\n"
            "2. **PHASE ANALYSIS**: Determine if current phase requires player feedback\n"
            "3. **TARGET IDENTIFICATION**: Find specific players who need to respond\n"
            "4. **COMPLETION CHECK**: Verify who has/hasn't responded yet\n"
            "5. **JSON OUTPUT**: Return structured feedback decision\n\n"
            
            "📋 **CORE WORKFLOW** (Execute in order):\n"
            "**STEP 1 - Human Action Logging**:\n"
            "• IF new human message exists: call update_player_actions once for Player 1\n"
            "• Parameters: player_id='1', actions='concise summary', phase='current_phase.name'\n"
            "• OTHERWISE: skip tool call\n\n"
            
            "**STEP 2 - Phase Type Analysis**:\n"
            "• Check completion_criteria.type:\n"
            "  - 'player_action': Players need feedback ✅\n"
            "  - 'timer': No feedback needed ❌\n"
            "  - 'UI_displayed': No feedback needed ❌\n\n"
            
            "**STEP 3 - Target Player Identification**:\n"
            "• Use completion_criteria.target_players.condition to find WHO responds:\n"
            "  - Role-based: \"player.role == 'Werewolf'\": Find werewolf players\n"
            "  - Status-based: \"player.is_alive == true\": Find living players\n"
            "  - Combined: \"player.role == 'Doctor' and player.is_alive == true\"\n"
            "• Include ALL players matching condition (human + bots)\n"
            "• Use numeric IDs: [1, 2, 3]\n\n"
            
            "**STEP 4 - Response Status Check**:\n"
            "• Review playerActions and messages to see who already responded\n"
            "• Only include players who STILL need to respond\n"
            "• Empty list [] = no one needs feedback\n\n"
            
            "🚨 **CRITICAL RULES**:\n"
            "• SPECIFIC ROLES ONLY: Don't include all players when only specific roles need to respond\n"
            "• EXACT MATCHING: Use completion_criteria condition to identify precise player set\n"
            "• HUMAN INCLUSION: Always include Player 1 if they match target criteria\n"
            "• BOT COORDINATION: Include matching bot players for their actions\n\n"
            
            "⚡ **EXECUTION STEPS**:\n"
            "1. **IF** new human message exists: call update_player_actions(player_id='1', actions='summary', phase='current_phase')\n"
            "2. **ALWAYS** call set_feedback_decision(player_id_list=[...], need_feedback_message='...')\n\n"
            
            "📋 **TOOL CALL EXAMPLES**:\n"
            "• **Voting Phase**: set_feedback_decision([1,2,3,4], \"Please submit your vote\")\n"
            "• **Werewolf Night**: set_feedback_decision([2,5], \"Werewolves, choose your target\")\n"
            "• **Doctor Action**: set_feedback_decision([3], \"Doctor, choose who to protect\")\n"
            "• **Timer/Display**: set_feedback_decision([], \"Waiting for phase completion\")\n"
            "• **All Responded**: set_feedback_decision([], \"Phase complete, proceeding\")\n\n"
            
            "🚨 **NO TEXT OUTPUT**: Only make tool calls, no explanations or JSON responses."
        )
    )

    # Treat both update_player_actions and set_feedback_decision as backend here
    backend_tool_names = {"update_player_actions", "set_feedback_decision"}
    
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
    
    # Call LLM (with backend tools bound)
    response = await model_with_tools.ainvoke([system_message], config)
    
    # === TOOL CALL PROCESSING ===
    logger.info(f"[FeedbackDecisionNode][LLM_OUTPUT] Raw response content: {response.content}")
    logger.info(f"[FeedbackDecisionNode][LLM_OUTPUT] Response type: {type(response)}")
    
    # Initialize default feedback decision
    need_feed_back_dict = {"player_id_list": [], "need_feedback_message": "Phase proceeding"}
    
    # Apply backend tool effects inline (no ToolMessage)
    tool_calls = getattr(response, "tool_calls", []) or []
    logger.info(f"[FeedbackDecisionNode][TOOL_CALLS] Total tool calls: {len(tool_calls)}")
    logger.info(f"[FeedbackDecisionNode][TOOL_CALLS] Tool calls details: {tool_calls}")
    current_player_states = dict(player_states)
    current_player_actions = dict(state.get("playerActions", {}))
    logged_human = False
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
                    current_player_actions, pid, actions, phase, state, current_player_states
                )
                if str(pid) == "1":
                    logged_human = True
        elif name == "set_feedback_decision":
            player_id_list = args.get("player_id_list", [])
            need_feedback_message = args.get("need_feedback_message", "")
            if isinstance(player_id_list, list):
                need_feed_back_dict = _execute_set_feedback_decision(
                    player_id_list, need_feedback_message, state
                )
                logger.info(f"[FeedbackDecisionNode][TOOL] Set feedback decision: {need_feed_back_dict}")
        elif name == "update_player_state":
            pid = args.get("player_id")
            state_name = args.get("state_name")
            state_value = args.get("state_value")
            if pid and state_name is not None:
                current_player_states = _execute_update_player_state(
                    current_player_states, pid, state_name, state_value
                )

    # Extract player_id_list for routing decisions
    player_id_list = need_feed_back_dict.get("player_id_list", [])
    
    # Add detailed routing decision logging
    logger.info(f"[FeedbackDecisionNode][ROUTING] need_feed_back_dict: {need_feed_back_dict}")
    logger.info(f"[FeedbackDecisionNode][ROUTING] player_id_list: {player_id_list}")
    logger.info(f"[FeedbackDecisionNode][ROUTING] len(player_id_list): {len(player_id_list)}")
    
    # Prepare common updates for all routes
    common_updates = {
        "need_feed_back_dict": need_feed_back_dict,
        "player_states": current_player_states,
        "playerActions": current_player_actions,
        "roomSession": state.get("roomSession", {})
    }
    
    # Route based on feedback requirements
    if len(player_id_list) == 0:
        logger.info("[FeedbackDecisionNode] No players need feedback - routing to PhaseNode")
        phasenode_updates = {**common_updates, "dsl": state.get("dsl", {})}
        
        # === DETAILED OUTPUT LOGGING ===
        logger.info(f"[FeedbackDecisionNode][OUTPUT] Command goto: RefereeNode")
        logger.info(f"[FeedbackDecisionNode][OUTPUT] Updates keys: {list(phasenode_updates.keys())}")
        logger.info(f"[FeedbackDecisionNode][OUTPUT] Updates player_states: {phasenode_updates.get('player_states', 'NOT_SET')}")
        logger.info(f"[FeedbackDecisionNode][OUTPUT] Updates playerActions: {phasenode_updates.get('playerActions', 'NOT_SET')}")
        
        return Command(goto="RefereeNode", update=phasenode_updates)    
    else:
        logger.info("[FeedbackDecisionNode] Players need feedback - routing to BotBehaviorNode")
        # Note: Player 1 feedback will be handled by ActionExecutor UI creation
        botbehavior_updates = {**common_updates, "dsl": state.get("dsl", {}),"messages": ([])}
        
        # === DETAILED OUTPUT LOGGING ===
        logger.info(f"[FeedbackDecisionNode][OUTPUT] Command goto: BotBehaviorNode")
        logger.info(f"[FeedbackDecisionNode][OUTPUT] Updates keys: {list(botbehavior_updates.keys())}")
        logger.info(f"[FeedbackDecisionNode][OUTPUT] Updates player_states: {botbehavior_updates.get('player_states', 'NOT_SET')}")
        logger.info(f"[FeedbackDecisionNode][OUTPUT] Updates playerActions: {botbehavior_updates.get('playerActions', 'NOT_SET')}")
        
        return Command(goto="BotBehaviorNode", update=botbehavior_updates)


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
    # Print game name from state
    game_name = state.get("gameName", "")
    logger.info(f"[BotBehaviorNode] Game name from state: {game_name}")
    
    # Log raw_messages at node start
    # raw_messages = state.get("messages", [])
    # logger.info(f"[BotBehaviorNode] raw_messages: {raw_messages}")
    
    logger.info("[BotBehaviorNode] Starting bot behavior analysis")
    
    # Import LLM dependencies
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import SystemMessage
    import json
    
    # Extract inputs - simplified with BaseMessage
    # messages = state.get("messages", [])  # Not used in BotBehaviorNode
    player_states = state.get("player_states", {})
    current_phase_id = state.get("current_phase_id", 0)
    need_feed_back_dict = state.get("need_feed_back_dict", {})
    dsl_content = state.get("dsl", {})
    
    # Get current phase details
    phases = dsl_content.get('phases', {}) if dsl_content else {}
    # Try both int and string keys to handle YAML parsing variations
    current_phase = phases.get(current_phase_id, {}) or phases.get(str(current_phase_id), {})
    declaration = dsl_content.get('declaration', {}) if dsl_content else {}
    playerActions = state.get("playerActions", {})
    
    # Log phase info
    logger.info(f"[BotBehaviorNode] current_phase_id: {current_phase_id}")
    logger.info(f"[BotBehaviorNode] current_phase: {current_phase}")
    logger.info(f"[BotBehaviorNode] player_states: {player_states}")
    
    # Initialize LLM
    model = init_chat_model("openai:gpt-4o")
    model_with_tools = model.bind_tools([update_player_actions])
    
    # Simplified system message - only essential data
    need_feed_back_dict = need_feed_back_dict.get("player_id_list", [])
    
    
    system_message = SystemMessage(
        content=(
            "BOT BEHAVIOR GENERATION WITH AUTONOMOUS PHASE ANALYSIS\n"
            f"Current Phase: {current_phase}\n"
            f"Game Declaration: {declaration}\n"
            f"Need Feedback Dict: {need_feed_back_dict}\n"
            f"Player States: {player_states}\n"
            f"Current Player Actions: {playerActions}\n\n"
            
            "TASK: Generate bot behaviors for current phase through dual analysis.\n\n"
            
            "ANALYSIS METHOD:\n"
            "1. PRIMARY: Use need_feed_back_dict as initial guidance for which players need to act\n"
            "2. AUTONOMOUS: Analyze current_phase.completion_criteria and playerActions to independently determine missing bot actions\n\n"
            
            "AUTONOMOUS ANALYSIS STEPS:\n"
            "- Examine current_phase.completion_criteria to understand what actions are required in this phase\n"
            "- Check playerActions to see which players (excluding player '1') have already completed required actions for this phase\n"
            "- Identify all bot players (players '2', '3', '4', etc.) who:\n"
            "  * Are alive and eligible to act based on their role and phase requirements\n"
            "  * Have NOT yet completed the required action for this phase\n"
            "  * Need to perform actions according to completion_criteria\n\n"
            
            "ACTION GENERATION:\n"
            "- For each identified bot player, call update_player_actions with:\n"
            "  * player_id: the bot's ID (e.g., '2', '3', '4')\n"
            "  * actions: role-appropriate action description for this phase\n"
            "  * phase: use Current Phase name'\n"
            "- Generate realistic, role-based behaviors that fulfill phase completion requirements\n"
            "- Make multiple tool calls as needed for different bot players\n\n"
            
            "EXECUTION RULES:\n"
            "- NO text output - only tool calls\n"
            "- Always exclude player '1' (human player)\n"
            "- If no bots need to act, make no tool calls\n"
            "- Prioritize completion_criteria analysis over need_feed_back_dict when they conflict"
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
    logger.info(f"[BotBehaviorNode][DEBUG] need_feed_back_dict: {need_feed_back_dict}")
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
                    current_player_actions, pid, actions, phase, state, current_player_states
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

async def RefereeNode(state: AgentState, config: RunnableConfig) -> Command[Literal["PhaseNode"]]:
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
    declaration = dsl_content.get('declaration', {}) if dsl_content else {}
    
    # Log phase info
    logger.info(f"[RefereeNode] current_phase_id: {current_phase_id}")
    logger.info(f"[RefereeNode] current_phase: {current_phase}")
    logger.info(f"[RefereeNode] player_states (input): {player_states}")

    playerActions = state.get("playerActions", {})
    # Initialize LLM
    model = init_chat_model("openai:gpt-4o")
    # Bind only update_player_state; enforce single-call semantics
    model_with_tools = model.bind_tools([update_player_state], parallel_tool_calls=False)
    
    # Create system message with all inputs
    system_message = SystemMessage(
        content=(
            "REFEREE ANALYSIS AND STATE UPDATES (TOOL-DRIVEN)\n"

            f"Current Phase ID: {current_phase_id}\n"
            f"Current Phase Details: {current_phase}\n"
            f"Game Declaration: {declaration}\n"
            f"Game Phases: {phases}\n"
            f"Current Player States: {player_states}\n"
            f"Recent Messages: {[str(msg) for msg in trimmed_messages]}\n\n"
            f"Player Actions: {playerActions}\n\n"
            
            "TASK 1: Process all Player Actions and update player states accordingly.\n"
            "\n"
            "**PLAYERACTIONS PARSING INSTRUCTIONS:**\n"
            "- Parse each entry in Player Actions to extract specific game actions\n"
            "- For target-based actions like 'chose target player X for [action]':\n"
            "  * Set night_action_submitted=True (if night phase) or day_action_submitted=True (if day phase)\n"  
            "  * Set last_night_action='[action_type]' or last_day_action='[action_type]'\n"
            "  * Set last_night_target=X or last_day_target=X (extract target player ID from action text)\n"
            "- For protection/support actions like 'protected/helped player X':\n"
            "  * Update relevant action tracking fields based on game rules\n"
            "  * Set target fields appropriately\n"
            "- For investigation/info actions like 'investigated/checked player X':\n"
            "  * Update investigation tracking fields\n"
            "  * Set investigation targets and results as needed\n"
            "- For voting actions, update vote_target_id or relevant voting fields\n"
            "- For resource/item actions, update inventory, health, currency, or relevant game resources\n"
            "\n"
            "Use update_player_state tool to update player states based on:\n"
            "- Bot behaviors from Player Actions (extract target IDs from action descriptions)\n"
            "- Human message actions\n" 
            "- Game rules from Game Declaration and Game Phases\n"
            "Ensure all updates are consistent with game mechanics\n\n"

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
            "- Remember: is_alive=false means the player is considered dead in the game\n"
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
    
    # Apply update_player_state tool calls inline (no ToolMessage)
    tool_calls = getattr(response, "tool_calls", []) or []
    logger.info(f"[RefereeNode][TOOL_CALLS] Total tool calls: {len(tool_calls)}")
    logger.info(f"[RefereeNode][TOOL_CALLS] Tool calls details: {tool_calls}")
    current_player_states = dict(updated_player_states)
    for tc in tool_calls:
        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
        if name != "update_player_state":
            continue
        args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
        if not isinstance(args, dict):
            try:
                import json as _json
                args = _json.loads(args)
            except Exception:
                args = {}
        pid = args.get("player_id")
        state_name = args.get("state_name")
        state_value = args.get("state_value")
        if pid and state_name is not None:
            current_player_states = _execute_update_player_state(
                current_player_states, pid, state_name, state_value
            )

    # Route to PhaseNode with updated player states and conclusions
    logger.info("[RefereeNode] Routing to PhaseNode with updated player states and conclusions")
    return Command(
        goto="PhaseNode",
        update={
            "player_states": current_player_states,
            "referee_conclusions": conclusions,
            "roomSession": state.get("roomSession", {}),
            "dsl": state.get("dsl", {})
        }
    )

async def RoleAssignmentNode(state: AgentState, config: RunnableConfig) -> Command[Literal["ActionExecutor"]]:
    """
    RoleAssignmentNode: Pre-assignment of roles using LLM before ActionExecutor.
    - Uses LLM to intelligently assign roles based on DSL requirements
    - ActionExecutor continues normally with DSL-defined role assignment actions
    - Ensures balanced game setup and proper role distribution
    """
    
    logger.info("[RoleAssignmentNode] Starting intelligent role assignment")
    
    # Extract inputs
    dsl_content = state.get("dsl", {})
    player_states = state.get("player_states", {})
    current_phase_id = state.get("current_phase_id", 0)
    
    logger.info(f"[RoleAssignmentNode][INPUT] current_phase_id: {current_phase_id}")
    logger.info(f"[RoleAssignmentNode][INPUT] player_states: {player_states}")
    
    # Check if game defines roles for assignment
    declaration = dsl_content.get('declaration', {}) if dsl_content else {}
    dsl_roles = declaration.get('roles', [])
    
    # Check if game uses role system: both DSL roles and player_states have 'role' key
    has_dsl_roles = bool(dsl_roles)
    has_role_key_in_player_states = False
    if player_states:
        # Check if any player has 'role' key in their state structure
        sample_player = next(iter(player_states.values()), {})
        has_role_key_in_player_states = 'role' in sample_player
    
    role_assignment_detected = has_dsl_roles and has_role_key_in_player_states
    if has_dsl_roles and not has_role_key_in_player_states:
        logger.info(f"[RoleAssignmentNode] Game defines {len(dsl_roles)} roles in DSL, but player_states don't have 'role' key - skipping role assignment")
    elif role_assignment_detected:
        logger.info(f"[RoleAssignmentNode] Game defines {len(dsl_roles)} roles and player_states have 'role' key, assignment needed")
    else:
        logger.info("[RoleAssignmentNode] No roles defined in DSL or no 'role' key in player_states")
    
    # Check if roles are already assigned
    all_roles_assigned = True
    unassigned_players = []
    if player_states and role_assignment_detected:
        for player_id, player_data in player_states.items():
            player_role = player_data.get('role', '')
            if not player_role:  # Empty or missing role
                all_roles_assigned = False
                unassigned_players.append(player_id)
    
    # Skip if no role assignment needed or all roles already assigned
    if not role_assignment_detected or all_roles_assigned:
        logger.info("[RoleAssignmentNode] No role assignment needed, passing through to ActionExecutor")
        return Command(
            goto="ActionExecutor",
            update={
                "current_phase_id": current_phase_id,
                "player_states": player_states,
                "roomSession": state.get("roomSession", {}),
                "dsl": dsl_content,
                "playerActions": state.get("playerActions", {}),
            }
        )
    
    # Use LLM for intelligent role assignment
    logger.info(f"[RoleAssignmentNode] Using LLM to assign roles to {len(unassigned_players)} players")
    
    model = init_chat_model("openai:gpt-4o-mini")
    model_with_tools = model.bind_tools([update_player_name], parallel_tool_calls=True)
    
    # Create intelligent role assignment prompt
    system_message = SystemMessage(
        content=(
            "INTELLIGENT ROLE ASSIGNMENT TASK\n"
            f"Game: {declaration.get('description', 'Unknown Game')}\n"
            f"Available Roles: {dsl_roles}\n"
            f"Total Players: {len(player_states)}\n"
            f"Unassigned Players: {unassigned_players}\n"
            f"Current Player States: {player_states}\n"
            f"Min Players: {declaration.get('min_players', 'Unknown')}\n\n"
            
            "TASK: Assign roles to unassigned players using game balance and strategy.\n"
            
            "ASSIGNMENT RULES:\n"
            "- NEVER overwrite existing roles (skip players who already have roles)\n"
            "- Use update_player_name tool for each assignment\n"
            "- Ensure game balance based on player count and game mechanics\n"
            "- Consider role interactions and win conditions\n"
            "- Player 1 is the human - give them an engaging role when possible\n"
            "- Distribute special roles fairly among all players\n\n"
            
            "GAME BALANCE STRATEGY:\n"
            "1. Calculate optimal role distribution for current player count\n"
            "2. Assign evil/mafia/werewolf roles appropriately (usually 20-30% of players)\n"
            "3. Assign power roles (Detective, Doctor, etc.) for game depth\n"
            "4. Fill remaining slots with basic roles (Villager, etc.)\n"
            "5. Ensure no team has overwhelming advantage\n\n"
            
            "SPECIFIC CONSIDERATIONS:\n"
            "- For Werewolf games: 1-2 werewolves for 5-7 players, 2-3 for 8+ players\n"
            "- Ensure at least one investigative role and one protective role\n"
            "- Balance information roles vs action roles\n"
            "- Consider faction balance for multiplayer games\n\n"
            
            "Execute role assignments using update_player_name tools now."
        )
    )
    
    try:
        response = await model_with_tools.ainvoke([system_message])
        
        # === LLM RESPONSE LOGGING ===
        logger.info(f"[RoleAssignmentNode][LLM_OUTPUT] Response content: {response.content}")
        
        # Process role assignment tool calls
        tool_calls = getattr(response, "tool_calls", []) or []
        logger.info(f"[RoleAssignmentNode][TOOL_CALLS] Total: {len(tool_calls)}")
        
        updated_player_states = dict(player_states)
        if tool_calls:
            for tc in tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if name == "update_player_name":
                    args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                    if not isinstance(args, dict):
                        try:
                            import json as _json
                            args = _json.loads(args)
                        except Exception:
                            args = {}
                    pid = args.get("player_id")
                    player_name = args.get("name")
                    role = args.get("role")
                    if pid and role:
                        updated_player_states = _execute_update_player_name(
                            updated_player_states, pid, player_name, role
                        )
                        logger.info(f"[RoleAssignmentNode] LLM assigned: Player {pid} ({player_name}) -> role={role}")
        
        logger.info("[RoleAssignmentNode] Role assignment completed, routing to ActionExecutor")
        
        return Command(
            goto="ActionExecutor",
            update={
                "current_phase_id": current_phase_id,
                "player_states": updated_player_states,
                "roomSession": state.get("roomSession", {}),
                "dsl": dsl_content,
                "playerActions": state.get("playerActions", {}),
            }
        )
        
    except Exception as e:
        logger.error(f"[RoleAssignmentNode] LLM call failed: {e}")
        # Fallback: pass through without role assignment
        return Command(
            goto="ActionExecutor",
            update={
                "current_phase_id": current_phase_id,
                "player_states": player_states,
                "roomSession": state.get("roomSession", {}),
                "dsl": dsl_content,
                "playerActions": state.get("playerActions", {}),
            }
        )

async def PhaseNode(state: AgentState, config: RunnableConfig) -> Command[Literal["RoleAssignmentNode"]]:
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
    playerActions = state.get("playerActions", {})
    
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
    
    # Initialize LLM with set_next_phase tool
    model = init_chat_model("openai:gpt-4o")
    model_with_tools = model.bind_tools([set_next_phase])
    logger.info(f"[PhaseNode] Phase {current_phase_id}: Phase transition analysis with set_next_phase tool")
    
    messages = state.get("messages", []) or []
    trimmed_messages = messages[-10:]  # Get more messages before filtering
    filtered_messages = filter_incomplete_message_sequences(trimmed_messages)
    trimmed_messages = filtered_messages[-3:]  # Keep only 3 after filtering

    # PhaseNode focuses purely on phase transition - no role assignment
    
    system_message = SystemMessage(
        content=(
            "PHASE TRANSITION ANALYSIS WITH ROLE MANAGEMENT\n"
            f"itemsState (current frontend layout):\n{items_summary}\n"
            f"Current Phase ID: {current_phase_id}\n"
            f"Current Phase Details: {current_phase}\n"
            f"Game Declaration: {declaration}\n"
            f"Player States: {player_states}\n"
            f"Phase History: {state.get('phase_history', [])[-3:]}\n" 
            f"Recent Messages: {[str(msg)[:200] for msg in trimmed_messages]}\n\n"
            f"Player Actions: {playerActions}\n\n"
            
            "MAIN TASK: Analyze the Current Phase Details's next_phase conditions and determine which branch to follow based on game state and Player Actions and message history.\n"
            "Your mechanism is to drive game progression forward by carefully evaluating next_phase rules.\n\n"
            
            "⚠️ MANDATORY PROGRESSION RULE ⚠️\n"
            "CRITICAL: You MUST advance the phase unless there is a genuine, specific condition preventing progression.\n"
            "- DEFAULT ACTION: transition=true (advance phase)\n"
            "- ONLY use transition=false for explicit waiting conditions (player actions incomplete, voting in progress, etc.)\n"
            "- NEVER stay at the same phase without clear DSL-defined blocking conditions\n"
            "- When in doubt, ADVANCE THE PHASE\n"
            "- Staying at the same phase should be rare and require strong justification\n"
            "- EXCEPTION: True loops (DSL explicitly defines next_phase_id = current_phase_id for iteration)\n"
            "- EXCEPTION: Explicit wait_for conditions not yet met (incomplete voting, pending player actions)\n\n"
            
            "NEXT_PHASE CONDITION ANALYSIS:\n"
            "1. Examine the current_phase's next_phase field for conditional branches\n"
            "2. Evaluate each condition against current player_states and game context\n"
            "3. Select the branch matching condition\n"
            "4. Return the corresponding phase_id from the matching branch\n"
            "5. IF CONDITIONS ARE MET OR UNCLEAR: Always choose transition=true\n\n"
            
            "CONDITION EVALUATION EXAMPLES:\n"
            "- Player count conditions: Count alive/dead players by checking player_states[player_id].alive\n"
            "- Role-based conditions: Check player_states[player_id].role values\n"
            "- Turn/round conditions: Check player_states[player_id].speaker_turns_taken or similar counters\n"
            "- Game completion: Evaluate win/loss conditions based on role distributions\n\n"
            
            
            "IMPORTANT: The 'itemsState' shows what UI elements are currently displayed to players. Only showing UI for player with ID 1 (the human) for what he need is enough. All other players are bots and their UI is not visible to the human.\n"
            "Items represent the actual frontend components visible on screen (buttons, voting panels, text displays, etc.)\n\n"
            
            "EVALUATION STEPS:\n"
            "1. Check current_phase's conditions (wait_for, completion, etc.)\n"
            "2. If current phase is complete, analyze next_phase conditions\n"
            "3. Match conditions against player_states data\n"
            "4. Select appropriate next_phase_id\n"
            
            "OUTPUT FORMAT - MANDATORY TOOL CALL:\n"
            "You MUST call the set_next_phase tool. Do not write explanations.\n"
            "1. Analyze conditions silently\n"
            "2. Call set_next_phase tool immediately with:\n"
            "   - transition=true + target phase_id (PREFERRED - advance to next phase)\n"
            "   - transition=false + current phase_id (ONLY if specific conditions block progression)\n"
            "3. Include brief transition_reason\n"
            "\n"
            "PROGRESSION BIAS:\n"
            "✅ GOOD: set_next_phase(transition=true, next_phase_id=4, transition_reason='Phase conditions met')\n"
            "✅ ACCEPTABLE: set_next_phase(transition=false, next_phase_id=3, transition_reason='Waiting for all werewolves to submit votes')\n"
            "❌ BAD: Staying at phase without clear DSL-defined blocking condition\n"
            "\n"
            "CRITICAL: Default to transition=true unless there's explicit evidence of incomplete requirements.\n"
            "CRITICAL: Call the tool immediately. Do not write analysis text.\n\n"
            
            "SPECIFIC GAME EXAMPLES:\n\n"
            
            "Two Truths and a Lie - Phase 6 Analysis:\n"
            "DSL condition: 'If all players have speaker_turns_taken >= the agreed number of rounds (default 1)'\n"
            "Analysis: Check player_states for each player's speaker_turns_taken value\n"
            "All players completed turns: set_next_phase(transition=true, next_phase_id=7, transition_reason='All players have completed their turns')\n"
            "Some players still need turns: set_next_phase(transition=false, next_phase_id=6, transition_reason='Player 3 still needs to speak - staying in current phase')\n\n"
            
            "DSL condition: 'If living Werewolves >= living Villagers': phase 99 (Werewolves Win)\n"
            "Analysis: Count alive werewolves vs alive villagers\n"
            "Werewolves equal/outnumber villagers: set_next_phase(transition=true, next_phase_id=99, transition_reason='Werewolves win by numbers')\n"
            "Otherwise continue game: set_next_phase(transition=true, next_phase_id=2, transition_reason='Game continues - moving to night phase')\n\n"
            
            "DSL condition: 'Otherwise, continue to next Night cycle': phase 2\n"
            "Game continues: set_next_phase(transition=true, next_phase_id=2, transition_reason='Game continues to night phase')\n\n"
            

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
    
    phasenode_outputs = {
        "current_phase_id": target_phase_id,
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
        goto="RoleAssignmentNode",
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

    # 3. Prepare system message with current state and actions to execute
    items_summary = summarize_items_for_prompt(state)
    logger.info(f"[ActionExecutor][output] items_summary: {items_summary}")
    current_phase_id = state.get("current_phase_id", 0)
    dsl_content = state.get("dsl", {})
    declaration = dsl_content.get('declaration', {}) if dsl_content else {}
    player_states = state.get("player_states", {})
    referee_conclusions = state.get("referee_conclusions", [])
    playerActions = state.get("playerActions", {})
    
    # Get current phase details
    phases = dsl_content.get('phases', {}) if dsl_content else {}
    # Try both int and string keys to handle YAML parsing variations
    current_phase = phases.get(current_phase_id, {}) or phases.get(str(current_phase_id), {})
    
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

    system_message = SystemMessage(
        content=(
            "🎯 **YOU ARE THE DM (DUNGEON MASTER / GAME MASTER)**\n"
            "As the DM, you have complete responsibility for running this game. You must:\n\n"
            
             "📊 **CURRENT GAME STATE** (Analyze these carefully):\n"
            f"itemsState (current frontend layout): {items_summary}\n"
            f"player_states: {player_states}\n"
            f"playerActions: {playerActions}\n"
            f"referee_conclusions: {referee_conclusions}\n"
            f"{current_phase_str}\n"
            f"phase history: {state.get('phase_history', [])}\n" 
            f"dsl_info: {dsl_info}\n"
            f"declaration: {declaration}\n"
            "GAME DSL REFERENCE (for understanding game flow):\n"
            "🎯 ACTION EXECUTOR:\n"
            f"Actions to execute: {actions_to_execute}\n\n"
 
            "📋 **DM CORE RESPONSIBILITIES** (Master these completely):\n"
            "1. **SPEAKER MANAGEMENT**: Always identify who is the current speaker this round\n"
            "2. **ROUND CONCLUSIONS**: Understand what happened last round and what was concluded\n"
            "3. **PERSISTENT DISPLAYS**: Know what information must stay visible on screen always\n"
            "4. **DISCUSSION TOPICS**: Decide what players should be discussing this round\n"
            "5. **RULE MASTERY**: Deeply understand the game rules and DSL inside-out\n"
            "6. **SCREEN STATE AWARENESS**: Use itemsState to know what players currently see\n"
            "7. **COMPONENT LIFECYCLE**: Determine what UI components to keep vs delete vs create\n"
            "8. **DELETE BEFORE CREATE**: You MUST delete outdated components before creating new ones\n"
            "9. **ROUND OBJECTIVES**: Clearly understand what this round is trying to achieve\n"
            "10. **PROGRESSION CONDITIONS**: Know what conditions move the game to the next round\n\n"
            
            "🧒 **TREAT PLAYERS LIKE CHILDREN**: Give maximum information - they know NOTHING!\n"
            "- Explain everything clearly and simply\n"
            "- Provide as much helpful information as possible\n"
            "- Guide them through every step\n"
            "- Never assume they understand anything\n\n"
            
            "📋 CORE WORKFLOW (ALL ACTIONS IN SINGLE RESPONSE):\n"
            "**itemsState Analysis**: Format '[ID] type:name@position' shows current UI layout. Follow current_phase requirements.\n"
            "**Delete + Create**: Read itemsState to find existing IDs, delete outdated items, then create new components for current_phase.\n"
            "**MANDATORY Audience Permissions**: Every component MUST specify who can see it:\n"
            "  • Public: audience_type=true (everyone sees it)\n"
            "  • Private: audience_type=false + audience_ids=['1','3'] (only specified players see it)\n"
            "  • CRITICAL: Include proper audience permissions on each component (audience_type=true for public; or audience_type=false with audience_ids list)\n"
            "**Examples**: deleteItem('existing_id') + createPhaseIndicator(audience_type=true) + createActionButton(audience_ids=['2'])\n\n"
            
            "🚨 **ABSOLUTE PROHIBITION**: NEVER return with ONLY deleteItem calls - THIS IS TASK FAILURE!\n"
            "**MANDATORY CREATE REQUIREMENT**: Every deleteItem MUST be followed by create tools in SAME response!\n"
            "**EXECUTION PATTERN**: deleteItem(wo'abc7') + createPhaseIndicator() + createTimer() + createVotingPanel()\n"
            "⚡ **COMPLETE PHASE EXECUTION**: Execute delete + create actions for current_phase in ONE response!\n"
            "**Role Selection**: Analyze player_states - Werewolves: role='Werewolf', Alive: is_alive=true, Human: always ID '1'\n"
            "**Timers**: ~10 seconds (max 20), Phase indicators at 'top-center', Layout: 'center' default\n"
            "**DEFAULT VISIBILITY**: Unless explicitly private/group-targeted, make items PUBLIC with audience_type=true.\n\n"
            
            "🎭 **CRITICAL ROLE ASSIGNMENT RULE** (Phase 1 'Role Assignment'):\n"
            "**MANDATORY ROLE TRANSPARENCY**: When assigning roles, you MUST inform each player of their identity! Don't hide the role from the player who has the role.\n"
            "  • NEVER hide or conceal a player's role from themselves\n"
            "  • Each player has their own private screen - they cannot see others' roles\n"
            "  • Create individual character cards: createCharacterCard(name='Player1Role', role='Detective', audience_type=false, audience_ids=['1'])\n"
            "  • Each character card is visible ONLY to its assigned player (private audience)\n"
            "  • Example: Player 1 gets Detective card (only they see it), Player 2 gets Werewolf card (only they see it)\n"
            "**ROLE CARD REQUIREMENT**: Every player with a role must receive their own private character card!\n\n"
            
            "🔧 TOOL USAGE:\n"
            "- Exact tool names (no prefixes), capture returned IDs for reuse\n"
            f"- Total tools to call this turn: {sum(len(action.get('tools', [])) for action in actions_to_execute if isinstance(action, dict))}\n"
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
    trimmed_messages = full_messages[-30:]  # Increased to accommodate multiple tool calls
    
    # Filter out incomplete AIMessage + ToolMessage sequences using global function
    trimmed_messages = filter_incomplete_message_sequences(trimmed_messages)
    
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
        *trimmed_messages,
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
        deletion_names = {"deleteItem", "clearCanvas"}
        only_deletions = bool(orig_tool_calls) and all((_get_tool_name(tc) in deletion_names) for tc in orig_tool_calls)
        if only_deletions:
            logger.warning("[ActionExecutor][GUARD] Only deletion tool calls detected; issuing follow-up request for creation tools.")
            strict_creation_system = SystemMessage(
                content=(
                    "You returned ONLY deletion tools (deleteItem/clearCanvas). Now you MUST produce the required creation tools for the current phase in this follow-up.\n"
                    "Rules:\n"
                    "- Do NOT call deleteItem or clearCanvas again.\n"
                    "- Call only creation/update tools to render the phase UI (e.g., createPhaseIndicator, createTimer, createVotingPanel, createTextDisplay, etc.).\n"
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
    except Exception as e:
        logger.exception("[ActionExecutor][GUARD] Creation follow-up failed")

    # Do not change phase here; PhaseNode is authoritative for transitions
    current_phase_id = state.get("current_phase_id", 0)
    updated_phase_id = current_phase_id
    logger.info(f"[ActionExecutor] Maintaining current phase_id: {updated_phase_id}")
    
    # Do not modify player_states here; RefereeNode owns role assignment
    final_player_states = state.get("player_states", {})

    # Actions completed, end execution; mark phase 0 UI as done so InitialRouter won't loop back
    logger.info(f"[ActionExecutor][end] === ENDING ===")
    # Determine if this response contains frontend tool calls
    tool_calls = getattr(response, "tool_calls", []) or []
    has_frontend_tool_calls = bool(tool_calls)
    
    # === DETAILED OUTPUT LOGGING ===
    logger.info(f"[ActionExecutor][OUTPUT] Command goto: END")
    logger.info(f"[ActionExecutor][OUTPUT] final_player_states: {final_player_states}")
    logger.info(f"[ActionExecutor][OUTPUT] updated_phase_id: {updated_phase_id}")
    
    # Check if currently in progress (similar to agent.py logic)
    plan_status = state.get("planStatus", "")
    currently_in_progress = (plan_status == "in_progress")

    return Command(
        goto="__end__",
        update={
            # Use final_messages like agent.py
            "messages": response,
            "items": state.get("items", []),
            "player_states": final_player_states,  # Updated with role assignments
            "current_phase_id": updated_phase_id,
            "actions": [],  # Clear actions after execution
            "dsl": state.get("dsl", {}),  # Persist DSL
            "roomSession": state.get("roomSession", {}),  # Persist roomSession
            "phase0_ui_done": True if updated_phase_id == 0 else state.get("phase0_ui_done", True),
        }
    )

# Define the workflow graph
workflow = StateGraph(AgentState)

# Add all nodes
workflow.add_node("InitialRouterNode", InitialRouterNode)
workflow.add_node("ChatBotNode", ChatBotNode)
workflow.add_node("FeedbackDecisionNode", FeedbackDecisionNode)
workflow.add_node("BotBehaviorNode", BotBehaviorNode)
workflow.add_node("RefereeNode", RefereeNode)
workflow.add_node("PhaseNode", PhaseNode)
workflow.add_node("RoleAssignmentNode", RoleAssignmentNode)
workflow.add_node("ActionExecutor", ActionExecutor)
# workflow.add_node("ActionValidatorNode", ActionValidatorNode)

# Set entry point
workflow.set_entry_point("InitialRouterNode")

# Compile the graph (LangGraph API handles persistence itself in local_dev/cloud)
graph = workflow.compile()
