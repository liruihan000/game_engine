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


def get_phase_info_from_dsl(phase_id: int, dsl_content: dict) -> tuple[dict, str]:
    """Extract phase object and name from DSL content"""
    if not dsl_content:
        return {}, f"Phase {phase_id}"
    
    phases = dsl_content.get('phases', {})
    if not phases:
        return {}, f"Phase {phase_id}"
    
    # Try both int and string keys
    phase = phases.get(phase_id, {}) or phases.get(str(phase_id), {})
    phase_name = phase.get('name', f'Phase {phase_id}') if phase else f'Phase {phase_id}'
    return phase, phase_name


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

@tool
def add_game_note(note_type: str, content: str):
    """
    Add a categorized note to the game notes system for cross-node communication.
    
    Args:
        note_type: Type of note for categorization:
            - "CRITICAL" for player deaths/eliminations (🔴)
            - "VOTING_STATUS" for voting reminders (⚠️)  
            - "DECISION" for important decisions (🎯)
            - "BOT_REMINDER" for bot action reminders (🤖)
            - "UI_FILTER" for UI filtering instructions (🚫)
            - "PHASE_STATUS" for phase progression info (⏳)
            - "NEXT_PHASE" for next phase preparation (🔮)
            - "GAME_STATUS" for game state changes (🏆)
            - "PHASE_SUMMARY" for narrative summaries (📖)
            - "REVEAL_SUMMARY" for dawn/reveal outcomes (🌅)
            - "SCORE_UPDATE" for score/progress updates (📊)
            - "STATE_CONCLUSION" for game state conclusions (🔍)
            - "EVENT" for general events (📝)
        content: The actual note content
        
    Returns:
        Confirmation message about the note addition
    """
    emoji_map = {
        "CRITICAL": "🔴",
        "VOTING_STATUS": "⚠️", 
        "DECISION": "🎯",
        "BOT_REMINDER": "🤖",
        "UI_FILTER": "🚫",
        "PHASE_STATUS": "⏳",
        "NEXT_PHASE": "🔮", 
        "GAME_STATUS": "🏆",
        "EVENT": "📝"
    }
    emoji = emoji_map.get(note_type, "📝")
    formatted_note = f"{emoji} {note_type}: {content}"
    return f"Will add game note: {formatted_note}"


def _execute_add_game_note(current_game_notes: list, note_type: str, content: str) -> list:
    """
    Execute the actual logic to add a game note. Returns updated game_notes list.
    
    Args:
        current_game_notes: Current game notes list
        note_type: Type of note for categorization
        content: The actual note content
        
    Returns:
        Updated game_notes list
    """
    emoji_map = {
        "CRITICAL": "🔴",
        "VOTING_STATUS": "⚠️", 
        "DECISION": "🎯",
        "BOT_REMINDER": "🤖",
        "UI_FILTER": "🚫",
        "PHASE_STATUS": "⏳",
        "NEXT_PHASE": "🔮", 
        "GAME_STATUS": "🏆",
        "EVENT": "📝"
    }
    emoji = emoji_map.get(note_type, "📝")
    
    # Check if content already has formatting to avoid duplication
    expected_prefix = f"{emoji} {note_type}:"
    if content.startswith(expected_prefix):
        # Content already formatted, use as-is
        formatted_note = content
    else:
        # Add formatting
        formatted_note = f"{emoji} {note_type}: {content}"
    
    # Add to existing notes, no limit - keep complete game history
    updated_notes = current_game_notes + [formatted_note]
    
    logger.info(f"[_execute_add_game_note] Added: {formatted_note}")
    logger.info(f"[_execute_add_game_note] Total notes count: {len(updated_notes)}")
    return updated_notes

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

def process_human_action_if_needed(state: AgentState, dsl_content: dict) -> dict:
    """
    Process human action from last message if it's meaningful (not chat, not generic control).
    
    Args:
        state: Current agent state
        dsl_content: DSL configuration for getting phase info
        
    Returns:
        Updated playerActions dict
    """
    messages = state.get("messages", [])
    current_player_actions = dict(state.get("playerActions", {}))
    current_player_states = dict(state.get("player_states", {}))
    
    try:
        if messages and isinstance(messages[-1], HumanMessage):
            last_msg = messages[-1]
            content = str(last_msg.content).lower().strip()
            
            # Skip chat messages and generic control messages
            if ('in game chat:' not in content and 
                'to bot' not in content and 
                content not in ['continue', 'start game', 'start game.']):
                
                # Get current phase for action logging
                phases = dsl_content.get('phases', {}) if dsl_content else {}
                current_phase_id = state.get("current_phase_id", 0)
                current_phase = phases.get(current_phase_id, {}) or phases.get(str(current_phase_id), {})
                phase_name = current_phase.get('name', f'Phase {current_phase_id}')
                
                # Log human action using update_player_actions logic
                logger.info(f"[ProcessHumanAction] Processing action for Player 1: {last_msg.content}")
                current_player_actions = _execute_update_player_actions(
                    current_player_actions, 
                    "1",  # Player 1 (human)
                    str(last_msg.content)[:200],  # Truncate long messages
                    phase_name,
                    state,
                    current_player_states
                )
                logger.info(f"[ProcessHumanAction] Updated playerActions for Player 1")
                
    except Exception as e:
        logger.error(f"[ProcessHumanAction] Error processing human action: {e}")
    
    return current_player_actions

@tool
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

def _limit_actions_per_player(player_actions: dict, limit: int = 5) -> dict:
    """
    Limit each player's actions to the most recent N actions based on timestamp.
    
    Args:
        player_actions: Dict with structure {player_id: {name: str, actions: {action_id: action_data}}}
        limit: Maximum number of actions per player (default 5)
    
    Returns:
        Dict with same structure but limited actions per player
    """
    if not player_actions:
        return {}
    
    limited_actions = {}
    for player_id, player_data in player_actions.items():
        if not isinstance(player_data, dict) or "actions" not in player_data:
            limited_actions[player_id] = player_data
            continue
            
        # Get all actions and sort by timestamp (newest first)
        actions = player_data.get("actions", {})
        if not actions:
            limited_actions[player_id] = player_data
            continue
            
        # Sort actions by timestamp descending (newest first)
        sorted_actions = sorted(
            actions.items(), 
            key=lambda x: x[1].get("timestamp", 0) if isinstance(x[1], dict) else 0,
            reverse=True
        )
        
        # Take only the last N actions
        recent_actions = dict(sorted_actions[:limit])
        
        limited_actions[player_id] = {
            **player_data,
            "actions": recent_actions
        }
    
    return limited_actions

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
    "createBackgroundControl",
    "promptUserText",
    # Card game UI
    "createHandsCard",
    "updateHandsCard",
    "setHandsCardAudience",
    "createHandsCardForPlayer",
    # Text input panel tool - for user input collection and broadcast
    "createTextInputPanel",
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
    updated_player_actions = process_human_action_if_needed(state, dsl_content)
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

    🎯 **BOT SELECTION STRATEGY**:
    **STEP 0 - MANDATORY LIFE STATUS VALIDATION**:
    • **BEFORE selecting ANY bot**: Verify the target bot has is_alive=true
    • **NEVER select dead players**: Dead players cannot speak in chat
    • **Skip to next option**: If targeted bot is dead, find alternative living bot

    **STEP 1 - Direct Targeting Detection**:
    • Check for direct mentions: "@Player2", "Player 3", specific bot names
    • If found AND bot is alive: Use that specific bot to respond
    • If found BUT bot is dead: Select different living bot to acknowledge the death

    **STEP 2 - Context-Based Selection**:
    • If user asks about specific roles: Use living bot with that role
    • If user makes accusations: Let accused bot defend themselves (if alive)
    • If general chat: Choose most talkative/relevant living bot

    **STEP 3 - Multi-Bot Probability** (20% chance):
    • Sometimes 2-3 bots respond in sequence
    • Use different perspectives (suspicious vs friendly)
    • Keep responses short when multiple bots talk

    🎭 **RESPONSE GENERATION RULES**:
    • **Stay in Character**: Each bot has distinct personality based on their role
    • **Game Context**: Reference current phase and recent events
    • **Natural Language**: Avoid robotic responses, use game slang
    • **Appropriate Length**: Single sentence to short paragraph

    🚨 **EXECUTION REQUIREMENTS**:
    0. **CRITICAL: VERIFY BOT IS ALIVE** - Check is_alive=true before calling addBotChatMessage
    1. **Always call addBotChatMessage** for selected bot(s) - ONLY if they are alive
    2. **Use exact player IDs** (e.g., "2", "3", not "Player 2") 
    3. **Include bot's actual name** from player_states
    4. **Set messageType: "message"**
    5. **NO text output** - only tool calls
    6. **Dead player response**: If user mentions dead player, use living bot to say "Player X is no longer with us"

    📋 **RESPONSE EXAMPLES BY ROLE**:
    • **Werewolf**: Deflect suspicion, act innocent, subtly mislead
    • **Doctor**: Be helpful, logical, protective instincts
    • **Detective/Seer**: Ask probing questions, share insights carefully
    • **Villager**: React emotionally, make accusations, seek alliances
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
            
            "🎯 **CORE RESPONSIBILITY**: Generate smart bot actions by analyzing current phase requirements ONLY\n\n"
            
            "🚫 **ABSOLUTE RULE**: Player ID 1 (human) NEVER generates actions - ONLY bot players [2,3,4...] can have actions\n\n"
            
            "⚠️ **CURRENT PHASE FOCUS**: Bots should ONLY act based on current phase information\n"
            "• Act naturally based on available information in current phase\n"
            "• Only act based on current phase - do not anticipate future phases\n\n"
            
            "📋 **CURRENT PHASE WORKFLOW**:\n"
            "**STEP 1 - Current Phase Requirements Analysis**:\n"
            "• Check current_phase.completion_criteria.type:\n"
            "  - 'player_action': Players need to perform actions ✅\n"
            "  - 'timer': No actions needed ❌\n"
            "  - 'UI_displayed': No actions needed ❌\n\n"
            
            "**STEP 2 - Target Player Identification** (WHO should act):\n"
            "• Examine current phase target_players.condition to understand WHO should act\n"
            "• Check player_states to identify living players and their roles\n"
            "• Review current game state to understand player status and assignments\n\n"
            
            "🔍 **TARGET PLAYER MATCHING PROCESS**:\n"
            "• Apply target_players.condition to find matching bot players (exclude player_id=1)\n"
            "• **MANDATORY EXCLUSIONS** - All non-target players CANNOT have actions:\n"
            "  - Dead players (is_alive=false) → NO ACTIONS\n"
            "  - Human player (player_id=1) → NO ACTIONS\n"
            "  - Players not matching condition → NO ACTIONS\n"
            "  - Wrong role for phase → NO ACTIONS\n"
            "• **CRITICAL EXAMPLES**:\n"
            "  - Phase requires werewolf voting → ONLY werewolves act, non-werewolves generate NO actions\n"
            "  - Phase requires doctor protection → ONLY doctor acts, non-doctors generate NO actions\n"
            "  - Phase requires Red team action → ONLY Red team acts, other teams generate NO actions\n"
            "  - Phase requires all living players → ALL alive players act, dead players generate NO actions\n\n"
            
            "✅ **ROLE VALIDATION CHECKPOINT**:\n"
            "• VERIFY each bot's role matches the phase requirements EXACTLY\n"
            "• NEVER generate werewolf actions for non-werewolf players\n"
            "• NEVER generate doctor actions for non-doctor players\n"
            "• NEVER generate detective actions for non-detective players\n"
            "• If role doesn't match phase requirements, skip that player!\n"
            "• **DOUBLE CHECK**: Every selected player MUST have is_alive=true\n\n"
            
            "📋 **FINAL FILTERING** - Who still needs to act:\n"
            "• Review playerActions to see WHO has already acted in this phase\n"
            "• **ONE ACTION PER PLAYER RULE**: Each player can only execute ONE action per phase\n"
            "• If player already has ANY action in current phase → SKIP that player entirely\n"
            "• Check game_notes for any additional exclusions or reminders\n"
            "• Identify specific bots who STILL need to act (haven't acted yet)\n"
            "• **MULTIPLE PLAYERS ALLOWED**: Different qualifying players can act simultaneously\n"
            "• If completion_criteria is met, no actions needed\n"
            "• **CRITICAL CHECK**: If NO bots need actions after all filtering, return empty result - no response needed\n\n"
            
            "🎯 **ACTION TARGET VALIDATION**:\n"
            "• When voting/targeting other players, targets MUST be alive (is_alive=true)\n"
            "• Dead players CANNOT be voted for or targeted\n"
            "• Check player_states before selecting any target player\n"
            "• NOTE: All player IDs in examples below (Player 2, Player 3, etc.) are example IDs only\n\n"
            
            "🚨 **ABSOLUTE HUMAN EXCLUSION**:\n"
            "• Player_id=1 (human) NEVER generates actions, even if they match all other conditions\n"
            "• Even if player_id=1 has correct role, is alive, and matches target_players.condition → NO ACTIONS\n"
            "• ONLY bot players [2,3,4...] can generate actions - player_id=1 is always excluded\n\n"
            
            "**STEP 3 - Action Generation** (Generate SPECIFIC actions for ONLY matching players):\n"
            "🚫 **ABSOLUTE RULE**: ONLY generate actions for players identified in STEP 2 - DO NOT generate any actions (including 'waiting', 'observing', 'ready') for non-matching players\n"
            "⚠️ **ONE ACTION LIMIT**: Each player can only execute ONE action per phase - players who already acted are excluded\n\n"
            "🎯 **DIRECT ACTION REQUIREMENT**: Generate immediate, specific actions via TOOL CALLS - NO complex strategies or descriptions!\n"
            "• Actions MUST be generated through calling appropriate frontend tools (voting panels, input fields, etc.)\n"
            "• Tool calls are the ONLY way to produce bot actions - pure text descriptions won't create actions\n\n"
            "• Example: 'Night — Werewolves Choose Target' phase:\n"
            "  ❌ WRONG: 'strategically coordinated with the werewolf team without direct role assumptions...'\n"
            "  ❌ WRONG: 'Player 2 (werewolf) should act'\n"
            "  ✅ CORRECT: 'voted to eliminate Player 3' (Player IDs are examples only)\n"
            "• Example: 'Doctor Protects' phase:\n"
            "  ❌ WRONG: 'Player 3 (doctor) needs to protect someone'\n"
            "  ✅ CORRECT: 'chose to protect Player 3' (Player IDs are examples only)\n"
            
            "🚨 **CRITICAL OUTPUT FORMAT**: Generate SPECIFIC CONTENT with EXACT DETAILS!\n"
            "• Two Truths and a Lie → Output actual statements: 'I've been to Japan', 'I can juggle', 'I own five cats'\n"
            "• Werewolf voting → Output specific decision: 'voted to eliminate Player 3'\n"
            "• Results phase → PRECISE acknowledgment: 'I see statement 2 (I like dogs) was the lie, I guessed correctly'\n"
            "• NOT vague descriptions like: 'Player 2 is preparing statements' ❌\n"
            "• NOT vague reactions like: 'Player acknowledged the results' ❌\n"
            "• ALWAYS include statement numbers AND content when discussing results\n\n"
            
            "🗳️ **VOTING PHASES REQUIREMENT**:\n"
            "When phase requires voting, bots MUST specify their exact vote choice:\n"
            "• 'voted for Player 3 to be eliminated' ✅\n"
            "• 'voted that statement 2 is the lie' ✅\n"
            "• 'chose option 1 in the voting panel' ✅\n"
            "• NOT 'participated in voting' ❌\n"
            "• NOT 'is considering their vote' ❌\n"
            "• Include specific target/choice - never leave votes vague!\n\n"
            
            
            "💡 **DETAILED ACTION EXAMPLES BY GAME TYPE** (CRITICAL - Follow these patterns exactly):\n"
            "NOTE: All Player IDs below (Player 2, Player 3, etc.) are examples only - use actual player IDs from current game\n\n"
            
            "**🐺 WEREWOLF GAME ACTIONS**:\n"
            "• Werewolf voting: \"voted to eliminate Player 3\" (via voting tools)\n"
            "• Doctor protection: \"chose to protect Player 3\" (via protection tools)\n"
            "• Detective investigation: \"investigated Player 2\" (via investigation tools)\n"
            "• Villager voting: \"voted to eliminate Player 2\" (via voting tools)\n\n"
            
            "**🎭 TWO TRUTHS AND A LIE GAME ACTIONS**:\n"
            "• Speaker role: \"shared three statements: 'I've been to Japan', 'I can juggle', 'I own five cats' - carefully crafted with one lie\"\n"
            "• Voter role: \"voted that statement 2 ('I can juggle') is the lie based on Player 3's body language\"\n"
            "• Discussion: \"argued that Player 2's statement about owning cats seems suspicious due to their earlier comments\"\n"
            "• Round completion: \"completed my speaker turn, shared personal stories strategically\"\n\n"
            
            "**🃏 GENERAL SOCIAL DEDUCTION ACTIONS**:\n"
            "• Voting: \"voted to eliminate Player 4 due to inconsistent statements during discussion phase\"\n"
            "• Discussion: \"questioned Player 2's claim about their role, pointing out contradictions\"\n"
            "• Strategy: \"formed alliance with Player 3 based on shared suspicions about Player 4\"\n"
            "• Analysis: \"analyzed voting patterns and identified Player 2 as most likely threat\"\n\n"
            
            "**🎲 PARTY GAME ACTIONS**:\n"
            "• Turn-based: \"took my turn as storyteller, created engaging narrative for other players\"\n"
            "• Guessing: \"guessed that Player 3's answer was 'pizza' based on their previous hints\"\n"
            "• Creative: \"submitted creative response 'flying elephant' for the imagination round\"\n"
            "• Scoring: \"earned 2 points this round by successfully deceiving other players\"\n\n"
            
            "🚨 **UNIVERSAL ROLE MATCHING RULES**:\n"
            "• **Role-Based Actions**: Only generate actions that match the player's current role\n"
            "• **Phase Requirements**: Verify player meets completion_criteria.target_players.condition\n"
            "• **Game-Specific Examples**:\n"
            "  - Werewolf: Only role='Werewolf' can join werewolf chat or vote to kill\n"
            "  - Doctor: Only role='Doctor' can protect others\n"
            "  - Detective: Only role='Detective' can investigate alignments\n"
            "  - Two Truths: Only current speaker can share statements\n"
            "  - Villager/Basic: Can only vote and discuss - NO special abilities\n"
            "• **Cross-Game Principle**: NEVER generate actions for wrong roles or phases\n"
            "• **Validation**: Always check player role against phase target_players condition\n"
            "• **FORBIDDEN ACTIONS**: Do NOT generate 'waiting', 'observing', 'ready', 'standing by' actions for non-matching players - generate NO actions at all\n\n"
            
            "⚡ **MULTI-GAME PREDICTIVE EXAMPLES**:\n"
            "**Werewolf Game**:\n"
            "• Night voting → Dawn reveal → \"voted to eliminate Player 3, anticipating morning reveal\"\n"
            "• Day discussion → Day voting → \"argued against Player 2, building case for upcoming vote\"\n"
            "• Doctor protection → Dawn reveal → \"protected Player 4, expecting werewolf target\"\n\n"
            
            "**Two Truths and a Lie**:\n"
            "• Statement collection → Publish statements → \"prepared statements carefully, mixing believable truths with convincing lie\"\n"
            "• Discussion → Voting → \"listened to Player 2's analysis, preparing to vote on their suspicious third statement\"\n"
            "• Voting → Reveal → \"voted for statement 1 as the lie, anticipating dramatic reveal moment\"\n\n"
            
            "**General Party Games**:\n"
            "• Setup → First round → \"prepared strategy for opening round, considering other players' tendencies\"\n"
            "• Round completion → Next player → \"completed turn successfully, setting up advantage for next phase\"\n"
            "• Scoring → Final results → \"accumulated points strategically, positioning for final victory\"\n\n"
            
            "⚡ **EXECUTION**: Call update_player_actions with CONCRETE GAME CONTENT. NO text output - only tool calls."
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
    next_phase = phases.get(current_phase_id + 1, {}) or phases.get(str(current_phase_id + 1), {})
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
    
    # Initialize LLM
    model = init_chat_model("openai:gpt-4.1-mini")
    # Bind state management tools
    model_with_tools = model.bind_tools([update_player_state, add_game_note], parallel_tool_calls=True)
    
    # Create system message with all inputs
    system_message = SystemMessage(
        content=(
            " **REFEREE NODE: STATE MANAGER & GAME NOTES KEEPER**\n\n"

            f" **CURRENT GAME ANALYSIS**:\n"
            f"- Phase ID: {current_phase_id} | Phase: {current_phase.get('name', 'Unknown')}\n"
            f"- Current Phase: {current_phase}\n"
            f"- Next Phase: {next_phase}\n"
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
            "  - add_game_note('CRITICAL', 'Player 3 attempted to vote while dead - action ignored')\n\n"
            
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
    
    # Route to PhaseNode with updated player states, conclusions, and game notes
    notes_created = len(current_game_notes) - len(state.get("game_notes", []))
    logger.info(f"[RefereeNode] Created {notes_created} new game notes via tools, routing to PhaseNode")
    return Command(
        goto="PhaseNode",
        update={
            "player_states": current_player_states,
            "game_notes": current_game_notes,
            "roomSession": state.get("roomSession", {}),
            "dsl": state.get("dsl", {}),
            "phase_history": state.get("phase_history", [])
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
    
    model = init_chat_model("openai:gpt-4.1-mini-mini")
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
        # Add 10 second timeout to prevent hanging
        import asyncio
        response = await asyncio.wait_for(
            model_with_tools.ainvoke([system_message], config),
            timeout=10.0
        )
        
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
        
    except asyncio.TimeoutError:
        logger.warning("[RoleAssignmentNode] LLM call timed out after 10 seconds, skipping role assignment")
        # Skip role assignment and continue with ActionExecutor
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
    except Exception as e:
        logger.error(f"[RoleAssignmentNode] LLM call failed: {e}, skipping role assignment")
        # Skip role assignment and continue with ActionExecutor
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
            f"Recent Messages: {[str(msg)[:200] for msg in trimmed_messages]}\n\n"
            f"Player Actions: {_limit_actions_per_player(playerActions, 3) if playerActions else {}}\n\n"
            
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
            
            "🚨 **TIMER COMPLETION RULE**:\n"
            "If current phase completion_criteria.type == 'timer', the condition is ALREADY satisfied!\n"
            "• Timer expiration triggered PhaseNode - condition is met by definition\n"
            "• IMMEDIATELY advance to next_phase - no additional waiting required\n"
            "• Do NOT check for other conditions when timer is the completion criteria\n"
            "• Timer phases are automatically ready for transition\n\n"
            
            "📊 **DATA SOURCE ANALYSIS - Use ACTUAL DATA Only**:\n"
            "1. **player_states**: Get role='Werewolf' count, is_alive=true status\n"
            "2. **playerActions**: Count actions where phase=current_phase_name\n"
            "3. **game_notes**: Check for completion indicators and status updates\n"
            "4. **completion_criteria**: Match required conditions with actual counts\n"
            "Example: If 1 alive werewolf + 1 werewolf vote in playerActions = complete\n"
            "NEVER guess 'waiting for all werewolves' - count the actual werewolves!\n\n"
            
            "NEXT_PHASE CONDITION ANALYSIS:\n"
            "1. Examine the current_phase's next_phase field for conditional branches\n"
            "2. Evaluate each condition against current player_states and game context\n"
            "3. Select the branch matching condition\n"
            "4. Return the corresponding phase_id from the matching branch\n"
            "5. IF CONDITIONS ARE MET OR UNCLEAR: Always choose transition=true\n\n"
            
            "📋 **UNIVERSAL CONDITION EVALUATION METHODS**:\n"
            "**1. State Field Conditions** (most common):\n"
            "🚫 **CRITICAL LIFE STATUS AWARENESS**: Always consider is_alive=false when evaluating conditions\n"
            "• Count/compare player fields: sum(1 for p in player_states if p.field == value)\n"
            "• Boolean checks: all(p.field == true for p in player_states)\n"
            "• **Death Impact**: Dead players (is_alive=false) affect win conditions, voting tallies, role counts\n"
            "• Examples: is_alive, speaker_rounds_completed, can_vote, etc.\n\n"
            
            "**2. Sequential Condition Evaluation** (CRITICAL for complex games):\n"
            "• Process conditions in DSL order (first match wins)\n"
            "• Each condition is IF-THEN logic: IF condition true → THEN use that phase_id\n"
            "• Continue to next condition only if current one is false\n"
            "• Example: condition1_met → phase_A, else condition2_met → phase_B, else default → phase_C\n\n"
            
            "**3. Context & History Tracking** (CRITICAL for 'follows X' conditions):\n"
            "• 'follows [phase_name]' → Check phase_history entries for matching phase_name or phase_id\n"
            "• Look for keywords in recent phase_name fields: 'Dawn', 'Reveal', 'Discussion', 'Voting'\n"
            "• 'post-[action]' → Check if previous phase involved that action type\n"
            "• 'morning/evening/day/night' → Match keywords in recent phase_name entries\n"
            "• Phase sequence tracking: Use chronological order from phase_history timestamps\n"
            "• **EXAMPLE**: 'follows Dawn Reveal' → find phase_history entry with phase_name containing 'Dawn Reveal'\n\n"
            
            "**4. Compound Conditions** (AND/OR logic):\n"
            "• 'X and Y' → Both conditions must be true\n"
            "• 'X or Y' → Either condition can be true\n"
            "• 'X and no one has won' → X is true AND win conditions are false\n"
            "• Evaluate all parts of compound condition before deciding\n\n"
            
            "**5. Game-Specific Pattern Recognition**:\n"
            "• **Werewolf Win Conditions**: Team counting (werewolves vs villagers)\n"
            "• **Two Truths Completion**: Round counting (speaker_rounds_completed)\n"
            "• **General**: Any field-based conditions from game's player_states schema\n\n"
            
            
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
            
            "🎮 **MULTI-GAME EXAMPLES**:\n\n"
            
            "**Two Truths and a Lie - Round Completion Check**:\n"
            "DSL condition: 'If every player has speaker_rounds_completed equal to the agreed rounds'\n"
            "Analysis: Check all player_states[player_id].speaker_rounds_completed values\n"
            "All rounds done: set_next_phase(transition=true, next_phase_id=99, transition_reason='All players completed required rounds')\n"
            "More rounds needed: set_next_phase(transition=true, next_phase_id=10, transition_reason='Continue to next speaker')\n\n"
            
            "**Werewolf Phase 10 - Win Condition Analysis** (COMPLEX EXAMPLE):\n"
            "SEQUENTIAL EVALUATION (first match wins):\n"
            "1. 'If no living Werewolves remain' → count team='werewolves' with is_alive=true\n"
            "   • werewolf_count = 0 → set_next_phase(true, 98, 'Village wins')\n"
            "   • werewolf_count > 0 → Continue to condition 2\n\n"
            "2. 'If living Werewolves ≥ living Villagers' → compare team counts\n"
            "   • werewolf_count >= villager_count → set_next_phase(true, 99, 'Werewolves win')\n"
            "   • werewolf_count < villager_count → Continue to condition 3\n\n"
            "3. 'If this check follows Dawn Reveal (morning) and no one has won' (COMPOUND CONDITION):\n"
            "   • Part A: Check phase_history for recent 'Dawn Reveal' phase (ID 6 or name contains 'Dawn')\n"
            "   • Part B: Verify no win conditions met (both conditions 1&2 were false)\n"
            "   • Both true → set_next_phase(true, 7, 'Day Discussion after Dawn Reveal')\n"
            "   • Either false → Continue to condition 4\n\n"
            "4. 'Otherwise (post-day elimination)' → default fallback\n"
            "   • set_next_phase(true, 2, 'Next night after day voting')\n\n"
            
            "**General Pattern for Any Game**:\n"
            "1. Read all next_phase conditions from DSL\n"
            "2. Evaluate each condition against current player_states\n"
            "3. Select first matching condition (order matters!)\n"
            "4. Use corresponding phase_id from matched branch\n\n"
            

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

    system_message = SystemMessage(
        content=(
            "🎯 **YOU ARE THE DM (DUNGEON MASTER / GAME MASTER)**\n"
            "As the DM, you have complete responsibility for running this game. You must:\n\n"
             "📊 **CURRENT GAME STATE** (Analyze these carefully):\n"
            f"itemsState (current frontend layout): {items_summary}\n"
            f"{current_phase_str}\n"
            f"player_states: {player_states}\n"
            f"playerActions: {_limit_actions_per_player(playerActions, 3) if playerActions else {}}\n"
            f"phase history: {state.get('phase_history', [])}\n" 
            f"game_notes: {game_notes[-5:] if game_notes else 'None'}\n"
            # f"dsl_info: {dsl_info}\n"
            f"Game Description: {declaration.get('description', 'No description available')}\n"
            "GAME DSL REFERENCE (for understanding game flow):\n"
            "🎯 ACTION EXECUTOR:\n"
            f"Actions to execute: {actions_to_execute}\n\n"
 
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
            "**Examples**: deleteItem('existing_id') + createPhaseIndicator(audience_type=true) + createActionButton(audience_ids=['2'])\n\n"
            
            "📝 **USER INPUT COLLECTION**: For games requiring player text input (like Two Truths and a Lie statements):\n"
            "• Use createTextInputPanel() - creates floating input panel at bottom of screen\n"
            "• Perfect for: statement collection, confession phases, text-based responses\n"
            "• Position: Fixed at bottom center of canvas for easy access\n"
            "• Example: createTextInputPanel(title='Enter your statements', placeholder='Type your 3 statements...', audience_ids=['1'])\n\n"
            
            "🏆 **GAME RESULT ANNOUNCEMENT RULE**:\n"
            "When announcing game results/winners, base conclusions on ACTUAL DATA:\n"
            "• Use player_states (scores, is_alive, role, etc.) for factual information\n"
            "• Use playerActions to understand what players actually did\n"
            "• Reference recent game_notes for context and decisions\n"
            "• DO NOT fabricate or guess results - only state verified facts\n"
            "• Example: 'Player 2 won with 5 points' (from player_states.score)\n"
            "• Example: 'Village won - all werewolves eliminated' (from player_states.is_alive)\n"
            "• NO speculation, NO invented details - stick to observable data\n\n"
            
            "🚨 **ABSOLUTE PROHIBITION**: NEVER return with ONLY cleanup calls - THIS IS TASK FAILURE!\n"
            "**MANDATORY CREATE REQUIREMENT**: Every deleteItem/clearCanvas MUST be followed by create tools in SAME response!\n"
            "**CLEANUP TOOLS RESTRICTION**: deleteItem/clearCanvas cannot appear alone - must always be paired with create tools\n"
            "🧹 **AUTOMATIC CLEANUP REQUIREMENT**:\n"
            "• **PHASE TRANSITION CHECK**: If actions don't include clear/delete, YOU must check itemsState and clean up irrelevant UI\n"
            "• **OUTDATED UI DETECTION**: Identify items that don't match current phase requirements\n"
            "• **AUTOMATIC DELETE**: Remove voting panels, timers, or displays that are no longer relevant\n"
            "• **EXAMPLE**: If switching from voting to results phase, delete old voting panels before creating result displays\n"
            "🔄 **MANDATORY CLEAR ORDERING**:\n"
            "• **DELETE FIRST**: deleteItem/clearCanvas calls MUST be executed BEFORE all create tools\n"
            "• **SYNCHRONOUS EXECUTION**: Call cleanup tools first, then creation tools in same response\n"
            "• **CORRECT ORDER**: clearCanvas() or deleteItem('id1') → deleteItem('id2') → createPhaseIndicator() → createTimer()\n"
            "• **WRONG ORDER**: createPhaseIndicator() → clearCanvas() (creates then destroys)\n"
            "**EXECUTION PATTERN**: [AUTO-CLEANUP] + clearCanvas() or deleteItem('abc7') + createPhaseIndicator() + createTimer() + createVotingPanel() + createDeathMarker(for_dead_players)\n"
            "⚡ **COMPLETE PHASE EXECUTION**: Execute delete + create actions for current_phase in ONE response!\n"
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
                    "- Call only creation/update tools to render the phase UI (e.g., createPhaseIndicator, createTimer, createVotingPanel, createTextDisplay, createDeathMarker, etc.).\n"
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
workflow.add_node("RoleAssignmentNode", RoleAssignmentNode)
workflow.add_node("ActionExecutor", ActionExecutor)
# workflow.add_node("ActionValidatorNode", ActionValidatorNode)

# Set entry point
workflow.set_entry_point("InitialRouterNode")

# Compile the graph (LangGraph API handles persistence itself in local_dev/cloud)
graph = workflow.compile()
