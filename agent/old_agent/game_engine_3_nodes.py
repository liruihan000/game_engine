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
from langchain_anthropic import ChatAnthropic
import json
import uuid
import time

load_dotenv()
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
    need_feed_back_dict: dict = {}
    roomSession: Dict[str, Any] = {}  # Room session data from frontend
    # Chat-specific fields for chatbot synchronization
    playerActions: Dict[str, Any] = {}  # Player actions
    phase_history: List[Dict[str, Any]] = []  # Phase transition history
    referee_conclusions: List[str] = []  # Referee decisions, state changes, and game event conclusions


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


def format_declaration_for_referee(declaration: dict) -> str:
    """
    Format declaration information for RefereeNode with clear status definitions and examples.
    
    Args:
        declaration: Game declaration from DSL
        
    Returns:
        Formatted string with status definitions, examples, and rules
    """
    if not declaration:
        return "No declaration available"
    
    formatted_parts = []
    
    # Game Description
    if 'description' in declaration:
        formatted_parts.append(f"**GAME DESCRIPTION**:\n{declaration['description']}\n")
    
    # Player States Schema with Examples
    if 'player_states' in declaration:
        formatted_parts.append("**PLAYER STATE SCHEMA** (Field Definitions):")
        player_states = declaration['player_states']
        
        for field_name, field_info in player_states.items():
            if isinstance(field_info, dict):
                field_type = field_info.get('type', 'unknown')
                example = field_info.get('example', 'N/A')
                description = field_info.get('description', 'No description')
                
                formatted_parts.append(
                    f"• **{field_name}** ({field_type}):\n"
                    f"  - Example: {example}\n"
                    f"  - Purpose: {description}"
                )
        formatted_parts.append("")  # Add blank line
    
    # Template/Example States
    if 'player_states_template' in declaration:
        template = declaration['player_states_template']
        if 'player_states' in template:
            formatted_parts.append("**STATE TEMPLATE EXAMPLE**:")
            for player_id, state in template['player_states'].items():
                formatted_parts.append(f"Player {player_id} default state: {state}")
            formatted_parts.append("")
    
    # Players Example (if different from template)
    if 'players_example' in declaration:
        example = declaration['players_example']
        if 'player_states' in example:
            formatted_parts.append("**GAME STATE EXAMPLE** (Active Game):")
            for player_id, state in example['player_states'].items():
                formatted_parts.append(f"Player {player_id} example: {state}")
            formatted_parts.append("")
    
    # Game Rules (if available)
    if 'rules' in declaration:
        formatted_parts.append(f"**GAME RULES**:\n{declaration['rules']}\n")
    
    # Win Conditions (if available) 
    if 'win_conditions' in declaration:
        formatted_parts.append(f"**WIN CONDITIONS**:\n{declaration['win_conditions']}\n")
    
    # Roles (if available)
    if 'roles' in declaration:
        formatted_parts.append("**AVAILABLE ROLES**:")
        roles = declaration['roles']
        if isinstance(roles, dict):
            for role_name, role_info in roles.items():
                formatted_parts.append(f"• {role_name}: {role_info}")
        elif isinstance(roles, list):
            for role in roles:
                formatted_parts.append(f"• {role}")
        formatted_parts.append("")
    
    return "\n".join(formatted_parts) if formatted_parts else "No declaration details available"


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
def add_referee_conclusion(note_type: str, content: str):
    """
    Add a categorized conclusion to the referee conclusions system for cross-node communication.
    
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
    return f"Will add referee conclusion: {formatted_note}"


def _execute_add_referee_conclusion(current_referee_conclusions: list, note_type: str, content: str) -> list:
    """
    Execute the actual logic to add a referee conclusion. Returns updated referee_conclusions list.
    
    Args:
        current_referee_conclusions: Current referee conclusions list
        note_type: Type of note for categorization
        content: The actual note content
        
    Returns:
        Updated referee_conclusions list
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
    updated_notes = current_referee_conclusions + [formatted_note]
    
    logger.info(f"[_execute_add_referee_conclusion] Added conclusion: {formatted_note}")
    logger.info(f"[_execute_add_referee_conclusion] Total notes count: {len(updated_notes)}")
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

# Centralized backend tools (shared across nodes)
backend_tools = [
    update_player_state,
    update_player_actions,
    set_next_phase,
    add_referee_conclusion,
    update_player_name
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
    - PhaseNode for completion conditions detection (after processing human actions)
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

    # Process human actions before routing to BotBehaviorNode
    updated_player_actions = process_human_action_if_needed(state, dsl_content)
    updates["playerActions"] = updated_player_actions

    # Ensure DSL is properly passed - use dsl_content if loaded, otherwise fallback to state
    final_dsl = dsl_content if dsl_content else state.get("dsl", {})
    updates["dsl"] = final_dsl
    
    
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
    
    model = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.7,
        max_tokens=4096
    )
    
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
    
    # Log referee_conclusions for debugging
    referee_conclusions = state.get('referee_conclusions', [])
    logger.info(f"[ChatBotNode] Referee Conclusions Count: {len(referee_conclusions)}")
    if referee_conclusions:
        logger.info(f"[ChatBotNode] All Referee Conclusions: {referee_conclusions}")
    else:
        logger.info(f"[ChatBotNode] No Referee Conclusions Available")

    # Enhanced LLM system for intelligent bot chat responses
    system_prompt = f"""
    🤖 **INTELLIGENT BOT CHAT SYSTEM**

    📊 **GAME CONTEXT**:
    - Current Phase: {current_phase_id}
    - Player States: {player_states}
    - Player Actions: {playerActions}
    - Referee Conclusions: {referee_conclusions if referee_conclusions else 'None'}

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
    
    # Log referee_conclusions for debugging
    referee_conclusions = state.get('referee_conclusions', [])
    logger.info(f"[BotBehaviorNode] Referee Conclusions Count: {len(referee_conclusions)}")
    if referee_conclusions:
        logger.info(f"[BotBehaviorNode] Full Referee Conclusions: {referee_conclusions}")
    else:
        logger.info(f"[BotBehaviorNode] No Referee Conclusions Available")

    # Initialize LLM with Claude 4.5 for better bot behavior
    # model = init_chat_model("anthropic:claude-3-5-sonnet-20241022")

    model = ChatAnthropic(
    model="claude-haiku-4-5-20251001",  # 最新版本
    api_key=os.getenv("ANTHROPIC_API_KEY") ,  # 或使用环境变量 ANTHROPIC_API_KEY
    temperature=0.7,
    max_tokens=4096
)
    model_with_tools = model.bind_tools([update_player_actions])
    items_summary = summarize_items_for_prompt(state)
    
    # Simplified system message - only essential data based on dm_agent_with_bot.py pattern
    need_feedback_player_list = need_feed_back_dict.get("player_id_list", [])
    
    system_message = SystemMessage(
        content=(
            "🤖 **YOU REPRESENT ALL BOT PLAYERS - ACT BASED ON PLAYER STATES**\n\n"
            
            f"📊 **CURRENT GAME STATE:**\n"
            f"Phase: {current_phase}\n"
            f"Game Rules: {declaration}\n"
            f"Player States: {player_states}\n"
            f"Player Actions: {playerActions}\n"
            f"UI Components: {items_summary}\n\n"
            
            "🎯 **YOUR MISSION:**\n"
            "Examine player_states and represent each bot player (player 2, 3, 4, etc.) based on their specific role and status. Make concrete decisions to win and get high scores.\n\n"
            
            "📋 **ACTION STEPS:**\n"
            "1. **Check Player States**: Look at each bot's role, is_alive, team, and other status fields\n"
            "2. **Analyze Phase Requirements**: What does completion_criteria require this phase?\n"
            "3. **Review UI Components**: What voting panels, text inputs, buttons are available?\n"
            "4. **Make Specific Decisions**: Act as each bot based on their role characteristics\n\n"
            
            "🎮 **ROLE-BASED BEHAVIOR:**\n"
            "• **Werewolf players**: Choose specific elimination targets, protect teammates\n"
            "• **Villager players**: Vote to eliminate suspicious players, protect village\n"
            "• **Doctor players**: Choose specific players to protect\n"
            "• **Detective players**: Investigate specific suspicious players\n"
            "• **Speaker players**: Submit 3 specific statements in Two Truths game\n"
            "• **Voting players**: Make specific vote choices based on game information\n\n"
            
            "⚙️ **USE THE TOOL:**\n"
            "For each bot that needs to act, call:\n"
            "• update_player_actions(player_id='[PLAYER_ID]', actions='[SPECIFIC_ACTION]', phase='[CURRENT_PHASE_NAME]')\n\n"
            
            "✅ **SPECIFIC ACTION EXAMPLES:**\n"
            "• 'voted to eliminate Player 3'\n"
            "• 'protected Player 1 from werewolf attack'\n"
            "• 'investigated Player 4 - found villager team'\n"
            "• 'submitted statements: I can play piano, I have been to Japan, I own 5 cats'\n"
            "• 'voted that statement 2 is the lie'\n\n"
            
            "🚨 **IMPORTANT**: Only generate actions for bot players (non-Player 1) based on their roles and states in player_states!"
        )
    )
    # Only treat update_player_actions as backend here
    backend_tool_names = {"update_player_actions"}
    
    # full_messages = state.get("messages", []) or []
    # try:
    #     if full_messages:
    #         last_msg = full_messages[-1]
    #         if isinstance(last_msg, AIMessage):
    #             pending_frontend_call = False
    #             for tc in getattr(last_msg, "tool_calls", []) or []:
    #                 name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
    #                 if name and name not in backend_tool_names:
    #                     pending_frontend_call = True
    #                     break
    #         if pending_frontend_call:
    #             try:
    #                 # print("[TRACE] Pending frontend tool calls detected; skipping LLM this turn and waiting for ToolMessage(s).")
    #                 logger.info("[chatnode][end] Pending frontend tool calls detected; skipping LLM this turn and waiting for ToolMessage(s).")
    #             except Exception:
    #                 pass
    #             return Command(
    #                 goto=END
    #             )
    # except Exception:
    #     pass


    # Call LLM with backend tool bound - Claude needs at least one user message
    user_message = HumanMessage(content="Generate bot actions for the current game phase based on player states and phase requirements.")
    response = await model_with_tools.ainvoke([system_message, user_message], config)
    
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
    
    # Route back to PhaseNode for condition re-check
    logger.info("[BotBehaviorNode] Routing back to PhaseNode for completion condition re-check")  
    return Command(
        goto="PhaseNode",
        update={
            "player_states": current_player_states,
            "playerActions": current_player_actions,
            "roomSession": state.get("roomSession", {}),
            "dsl": state.get("dsl", {})
        }
    )

async def RefereeNode(state: AgentState, config: RunnableConfig) -> Command[Literal["RoleAssignmentNode"]]:
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
    player_states = state.get("player_states", {})
    current_phase_id = state.get("current_phase_id", 0)
    dsl_content = state.get("dsl", {})
    
    
    # Get current phase details
    phases = dsl_content.get('phases', {}) if dsl_content else {}
    # Try both int and string keys to handle YAML parsing variations
    current_phase = phases.get(current_phase_id, {}) or phases.get(str(current_phase_id), {})
    # next_phase = phases.get(current_phase_id + 1, {}) or phases.get(str(current_phase_id + 1), {})
    declaration = dsl_content.get('declaration', {}) if dsl_content else {}
    
    # Log phase info
    logger.info(f"[RefereeNode] current_phase_id: {current_phase_id}")
    logger.info(f"[RefereeNode] current_phase: {current_phase}")
    logger.info(f"[RefereeNode] player_states (input): {player_states}")

    playerActions = state.get("playerActions", {})
    
    # Log referee_conclusions for debugging
    referee_conclusions = state.get('referee_conclusions', [])
    logger.info(f"[RefereeNode] Referee Conclusions Count: {len(referee_conclusions)}")
    if referee_conclusions:
        logger.info(f"[RefereeNode] Current Referee Conclusions: {referee_conclusions}")
    else:
        logger.info(f"[RefereeNode] No Referee Conclusions Available")
    
    # Initialize LLM
    model = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.7,
        max_tokens=4096
    )
    # Bind state management tools
    model_with_tools = model.bind_tools([update_player_state, add_referee_conclusion, update_player_actions], parallel_tool_calls=True)
    
    # Format declaration with clear status definitions  
    formatted_declaration = format_declaration_for_referee(declaration)
    
    # Create system message with all inputs
    system_message = SystemMessage(
        content=(
            "🎯 **GAME MANAGER - UPDATE STATES & CONCLUSIONS**\n\n"
            
            f"📊 **CURRENT GAME DATA:**\n"
            f"Phase {current_phase_id}: {current_phase.get('name', 'Unknown')}\n"
            f"Player States: {player_states}\n"
            f"Player Actions: {playerActions if playerActions else 'None'}\n"
            f"Game Rules: {formatted_declaration}\n\n"
            
            "🎮 **YOUR ROLE:**\n"
            "You are the Game Manager. Update player states based on rules and player actions. "
            "Record important events in conclusions for game history.\n\n"
            "assign roles to players if needed using update_player_name tool\n"
            "Set the current/next speaker for the next phase using update_player_state tool\n"
            
            "🔧 **TOOLS:**\n"
            "• update_player_state(player_id, field_name, value) - Update any player field that is needed to be updated\n"
            "• add_referee_conclusion(category, message) - Record all events updated and the decisions you made\n"
            "Categories: 'DECISION', 'CRITICAL', 'GAME_STATUS', 'DM_ACTION'\n\n"
            
            "📝 **EXAMPLES:**\n\n"
            
            "**Two Truths and a Lie - Speaker Rotation:**\n"
            "• update_player_state('2', 'is_speaker', False)  # Previous speaker done\n"
            "• update_player_state('3', 'is_speaker', True)   # Set next speaker\n"
            "• update_player_state('3', 'speaker_rounds_completed', 1)\n"
            "• add_referee_conclusion('DM_ACTION', 'Rotated speaker from Player 2 to Player 3')\n\n"
            
            "**Two Truths and a Lie - Voting Results:**\n"
            "• update_player_state('1', 'score', 2)  # Correct guess +1 point\n"
            "• update_player_state('2', 'score', 1)  # Successfully deceived +1 point\n"
            "• add_referee_conclusion('GAME_STATUS', 'Player 1 guessed correctly, Player 2 fooled others')\n\n"
            
            "**Werewolf - Night Elimination:**\n"
            "• update_player_state('3', 'is_alive', False)  # Werewolf victim\n"
            "• update_player_state('3', 'role_revealed', True)  # Reveal role on death\n"
            "• add_referee_conclusion('CRITICAL', 'Player 3 (Villager) eliminated by werewolves')\n\n"
            
            "**Werewolf - Day Voting Elimination:**\n"
            "• update_player_state('2', 'is_alive', False)  # Voted out player\n"
            "• update_player_state('2', 'role_revealed', True)  # Public role reveal\n"
            "• add_referee_conclusion('CRITICAL', 'Player 2 (Werewolf) eliminated by village vote')\n\n"
            
            "**Werewolf - Detective Investigation:**\n"
            "• update_player_state('4', 'known_alignments', {'2': 'werewolves'})  # Detective learns info\n"
            "• add_referee_conclusion('DECISION', 'Detective investigated Player 2, learned werewolf team')\n\n"
        )
    )
    

    # Only treat update_player_state as backend here
    # backend_tool_names = {"update_player_state"}
    
    # full_messages = state.get("messages", []) or []
    # try:
    #     if full_messages:
    #         last_msg = full_messages[-1]
    #         if isinstance(last_msg, AIMessage):
    #             pending_frontend_call = False
    #             for tc in getattr(last_msg, "tool_calls", []) or []:
    #                 name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
    #                 if name and name not in backend_tool_names:
    #                     pending_frontend_call = True
    #                     break
    #         if pending_frontend_call:
    #             try:
    #                 # print("[TRACE] Pending frontend tool calls detected; skipping LLM this turn and waiting for ToolMessage(s).")
    #                 logger.info("[chatnode][end] Pending frontend tool calls detected; skipping LLM this turn and waiting for ToolMessage(s).")
    #             except Exception:
    #                 pass
    #             return Command(
    #                 goto=END
    #             )
    # except Exception:
    #     pass
    
    # Call LLM with tool bound
    response = await model_with_tools.ainvoke([system_message], config)
    
    # === DETAILED LLM RESPONSE LOGGING ===
    logger.info(f"[RefereeNode][LLM_OUTPUT] Raw response content: {response.content}")
    logger.info(f"[RefereeNode][LLM_OUTPUT] Response type: {type(response)}")
    
    # No JSON expected; start with current states
    updated_player_states = player_states
    
    # Apply tool calls inline (no ToolMessage) 
    tool_calls = getattr(response, "tool_calls", []) or []
    logger.info(f"[RefereeNode][TOOL_CALLS] Total tool calls: {len(tool_calls)}")
    logger.info(f"[RefereeNode][TOOL_CALLS] Tool calls details: {tool_calls}")
    
    current_player_states = dict(updated_player_states)
    current_referee_conclusions = list(state.get("referee_conclusions", []))
    
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
        elif name == "add_referee_conclusion":
            note_type = args.get("note_type")
            content = args.get("content")
            if note_type and content:
                current_referee_conclusions = _execute_add_referee_conclusion(
                    current_referee_conclusions, note_type, content
                )

    # Notes are now created via tool calls (add_referee_conclusion), no need for direct creation
    # current_referee_conclusions already contains all tool-created notes
    
    # Route to ActionExecutor with updated player states and game notes
    notes_created = len(current_referee_conclusions) - len(state.get("referee_conclusions", []))
    logger.info(f"[RefereeNode] Created {notes_created} new game notes via tools, routing to ActionExecutor")
    return Command(
        goto="RoleAssignmentNode",
        update={
            "player_states": current_player_states,
            "referee_conclusions": current_referee_conclusions,
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
                "player_states": player_states,
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

async def PhaseNode(state: AgentState, config: RunnableConfig) -> Command[Literal["BotBehaviorNode", "RefereeNode", "ActionExecutor"]]:
    """
    PhaseNode - Entry point that checks completion conditions first.
    
    New Architecture:
    1. Check current phase completion conditions
    2. If conditions not met and needs Bot -> route to BotBehaviorNode  
    3. If conditions met -> route to RefereeNode for state management
    4. For Phase 0 (initial setup) -> route directly to ActionExecutor
    
    Routes to:
    - ActionExecutor: For Phase 0 (initial UI setup)  
    - BotBehaviorNode: When completion conditions not met and Bot action needed
    - RefereeNode: When completion conditions are met, for state updates
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
    
    # Log referee_conclusions for debugging
    referee_conclusions = state.get('referee_conclusions', [])
    logger.info(f"[PhaseNode] Referee Conclusions Count: {len(referee_conclusions)}")

    # Initialize LLM with set_next_phase and set_feedback_decision tools
    model = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.7,
        max_tokens=4096
    )
    model_with_tools = model.bind_tools([set_next_phase, set_feedback_decision])
    logger.info(f"[PhaseNode] Phase {current_phase_id}: Phase transition analysis with set_next_phase tool")
    

    # PhaseNode focuses purely on phase transition - no role assignment
    
    system_message = SystemMessage(
        content=(
            "**PHASE NODE: GAME PROGRESSION CONTROLLER**\n\n"
            
            "**CURRENT GAME CONTEXT**:\n"
            f"**CURRENT PHASE INFORMATION:**\n"
            f"  Current Phase ID: {current_phase_id}\n"
            f"  Current Phase info and completion conditions: {current_phase}\n\n"
            
            f"**Game Configuration:**\n"
            f"  Game Declaration: {declaration}\n\n"
            
            f"**Player States:**\n"
            f"  {player_states}\n\n"
            
            f"**Player Actions:**\n"
            f"  {playerActions if playerActions else 'No actions recorded'}\n\n"
            
            f"**Frontend Items State:**\n"
            f"  {items_summary}\n\n"
            
            f"**Game History:**\n"
            f"  Referee Conclusions: {referee_conclusions if referee_conclusions else 'None'}\n"
            f"  Phase History: {state.get('phase_history', [])}\n\n"
            
            "**🎯 YOUR MISSION:**\n"
            "Analyze current phase completion and determine game progression path.\n"
            "Drive game forward by evaluating phase transition conditions and player actions.\n\n"
            
            "🤖 **BOT ANALYSIS & PHASE CONTROL:**\n"
            "Check if bots (players 2,3,4...) need to act. Advance phase when conditions met.\n\n"
            
            "🔧 **TOOLS:**\n"
            "• set_feedback_decision(player_id_list, message) - Request bot actions if needed\n"
            "• set_next_phase(transition, next_phase_id, transition_reason) - Advance or stay\n\n"
            
            "⚠️ **TIMER CONDITIONS ALREADY SATISFIED:**\n"
            "If current phase completion_criteria mentions 'timer' or 'time', the condition is already met.\n"
            "You were only triggered because the timer expired. Proceed to next phase immediately.\n\n"
            
            "📝 **EXAMPLES:**\n\n"
            
            "**Timer Phase - Discussion Timer Expired:**\n"
            "• Phase completion_criteria: 'Discussion timer has expired'\n"
            "• Timer already expired (that's why PhaseNode was triggered)\n"
            "• set_feedback_decision(player_id_list=[], message='Timer expired, no bot actions needed')\n"
            "• set_next_phase(transition=true, next_phase_id=6, transition_reason='Discussion timer expired')\n\n"
            
            "**Bot Action Example - Werewolf Voting Phase**:\n"
            "Phase completion_criteria: 'All living players must vote'\n"
            "Bot Analysis:\n"
            "• Check Player Actions: {'1': 'voted for player 3', '2': missing, '3': missing, '4': 'voted for player 1'}\n"
            "• Bots 2,3 haven't voted yet → set_feedback_decision(player_id_list=[2,3], message='Bots need to vote')\n"
            "• Result: Routes to BotBehaviorNode to generate missing votes\n\n"
            
            "**Bot Completion Example - Night Actions Complete**:\n"
            "Phase completion_criteria: 'Werewolves select victim, Doctor protects'\n"
            "Bot Analysis:\n"
            "• Check Player Actions: {'2': 'werewolf_attack:player_1', '3': 'doctor_protect:player_4'}\n"
            "• All required bot actions present → set_feedback_decision(player_id_list=[], message='All bot night actions completed')\n"
            "• Phase complete → set_next_phase(transition=true, next_phase_id=6, transition_reason='Night actions complete')\n\n"
            
            "**Two Truths and a Lie - Round Completion Check**:\n"
            "DSL condition: 'If every player has speaker_rounds_completed equal to the agreed rounds'\n"
            "Bot Analysis:\n"
            "• Check Player Actions for bot statements: {'2': 'submitted_statements', '3': 'submitted_statements'}\n"
            "• All bot statements present → set_feedback_decision(player_id_list=[], message='All bot statements submitted')\n"
            "• All rounds done: set_next_phase(transition=true, next_phase_id=99, transition_reason='All players completed required rounds')\n\n"
            
            "**Bot Mixed Example - Werewolf Day Discussion Phase**:\n"
            "Phase completion_criteria: 'Discussion complete, ready for voting'\n"
            "Bot Analysis:\n"
            "• Check Player Actions: {'1': 'player_discussion_ready', '2': missing, '3': 'bot_discussion_complete', '4': missing}\n"
            "• Bots 2,4 need to participate in discussion → set_feedback_decision(player_id_list=[2,4], message='Bots need discussion input')\n"
            "• Result: Routes to BotBehaviorNode for bot discussion generation\n\n"
            
            "**No Bot Needed Example - Human-Only Phase**:\n"
            "Phase completion_criteria: 'Human player selects game mode'\n"
            "Bot Analysis:\n"
            "• This phase only requires human input (player 1)\n"
            "• No bot actions needed → set_feedback_decision(player_id_list=[], message='Human-only phase, no bot actions required')\n"
            "• Check if human completed → proceed with phase completion assessment\n\n"
            
            "**Werewolf Phase 10 - Win Condition Analysis** (COMPLEX EXAMPLE):\n"
            "Bot Analysis: Check if any bot actions needed before win check\n"
            "• If no pending bot actions → proceed with win condition evaluation\n"
            "SEQUENTIAL EVALUATION (first match wins):\n"
            "1. 'If no living Werewolves remain' → count team='werewolves' with is_alive=true\n"
            "   • werewolf_count = 0 → set_next_phase(true, 98, 'Village wins')\n"
            "   • werewolf_count > 0 → Continue to condition 2\n\n"
            "2. 'If living Werewolves ≥ living Villagers' → compare team counts\n"
            "   • werewolf_count >= villager_count → set_next_phase(true, 99, 'Werewolves win')\n"
            "   • werewolf_count < villager_count → Continue to condition 3\n\n"
            
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
    
    # Extract phase decision and feedback decision from tool calls
    transition_from_tool = None
    next_phase_id_from_tool = None
    transition_reason = ""
    need_feed_back_dict = {"player_id_list": [], "need_feedback_message": "Phase proceeding"}
    
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
        elif name == "set_feedback_decision":
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
            if not isinstance(args, dict):
                try:
                    import json as _json
                    args = _json.loads(args)
                except Exception:
                    args = {}
            player_id_list = args.get("player_id_list", [])
            need_feedback_message = args.get("need_feedback_message", "")
            need_feed_back_dict = _execute_set_feedback_decision(player_id_list, need_feedback_message, state)
            logger.info(f"[PhaseNode] Feedback decision: {need_feed_back_dict}")
    
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
    
    # Add need_feed_back_dict to outputs
    phasenode_outputs = {
        "current_phase_id": target_phase_id,
        "current_phase_name": target_phase_name,
        "player_states": state.get("player_states", {}),
        "roomSession": state.get("roomSession", {}),
        "dsl": state.get("dsl", {}),
        "phase_history": current_phase_history,
        "need_feed_back_dict": need_feed_back_dict
    }
    
    # Routing logic based on feedback decision
    player_id_list = need_feed_back_dict.get("player_id_list", [])
    
    # If feedback needed (players need to take action) - go to BotBehaviorNode
    if player_id_list and len(player_id_list) > 0:
        route_target = "BotBehaviorNode" 
        logger.info(f"[PhaseNode] Players {player_id_list} need feedback - routing to BotBehaviorNode")
    # If no feedback needed (phase complete) - go to RefereeNode for state management
    else:
        route_target = "RefereeNode"
        logger.info("[PhaseNode] No feedback needed - routing to RefereeNode for state management")
    
    # === DETAILED OUTPUT LOGGING ===
    logger.info(f"[PhaseNode][OUTPUT] Command goto: {route_target}")
    logger.info(f"[PhaseNode][OUTPUT] Updates keys: {list(phasenode_outputs.keys())}")
    logger.info(f"[PhaseNode][OUTPUT] Updates current_phase_id: {phasenode_outputs.get('current_phase_id')}")
    logger.info(f"[PhaseNode][OUTPUT] need_feed_back_dict: {need_feed_back_dict}")
    
    return Command(
        goto=route_target,
        update=phasenode_outputs
    )

async def ActionExecutor(state: AgentState, config: RunnableConfig) -> Command[Literal["__end__"]]:
    """
    Execute actions from DSL and current phase by calling frontend tools.
    Audience-aware rendering: always choose explicit audience permissions per component
    and render public, group, and individual UIs according to the DSL phase design.
    Can make announcements based on referee_conclusions.
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
    model = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.7,
        max_tokens=4096
    )

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
    
    # Log referee_conclusions for debugging
    referee_conclusions = state.get('referee_conclusions', [])
    logger.info(f"[ActionExecutor] Referee Conclusions Count: {len(referee_conclusions)}")
    if referee_conclusions:
        logger.info(f"[ActionExecutor] Current Referee Conclusions: {referee_conclusions}")
    else:
        logger.info(f"[ActionExecutor] No Referee Conclusions Available")

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
            f"{current_phase_str}\n"
            f"phase history: {state.get('phase_history', [])}\n" 
            f"referee_conclusions: {referee_conclusions if referee_conclusions else 'None'}\n"
            f"dsl_info: {dsl_info}\n"
            f"declaration: {declaration}\n"
            "GAME DSL REFERENCE (for understanding game flow):\n"
            "🎯 ACTION EXECUTOR:\n"
            f"Actions to execute: {actions_to_execute}\n\n"
 
            "📋 **DM CORE RESPONSIBILITIES** (Master these completely):\n"
            "1. **REFEREE CONCLUSIONS AWARENESS**: Read referee_conclusions for critical state changes and UI guidance\n"
            "2. **DEAD PLAYER FILTERING**: NEVER create voting options for players with is_alive=false\n"
            "3. **SPEAKER MANAGEMENT**: Always identify who is the current speaker this round\n"
            "4. **ROUND CONCLUSIONS**: Understand what happened last round and what was concluded\n"
            "5. **PERSISTENT DISPLAYS**: Know what information must stay visible on screen always\n"
            "6. **RULE MASTERY**: Deeply understand the game rules and DSL inside-out\n"
            "7. **SCREEN STATE AWARENESS**: Use itemsState to know what players currently see\n"
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
            
            "📝 **USER INPUT COLLECTION**: For games requiring player text input (like Two Truths and a Lie statements):\n"
            "• Use createTextInputPanel() - creates floating input panel at bottom of screen\n"
            "• Perfect for: statement collection, confession phases, text-based responses\n"
            "• Position: Fixed at bottom center of canvas for easy access\n"
            "• Example: createTextInputPanel(title='Enter your statements', placeholder='Type your 3 statements...', audience_ids=['1'])\n\n"
            
            "🚨 **ABSOLUTE PROHIBITION**: NEVER return with ONLY deleteItem calls - THIS IS TASK FAILURE!\n"
            "**MANDATORY CREATE REQUIREMENT**: Every deleteItem MUST be followed by create tools in SAME response!\n"
            "**EXECUTION PATTERN**: deleteItem(wo'abc7') + createPhaseIndicator() + createTimer() + createVotingPanel()\n"
            "⚡ **COMPLETE PHASE EXECUTION**: Execute delete + create actions for current_phase in ONE response!\n"
            "**Role Selection**: Analyze player_states - Werewolves: role='Werewolf', Alive: is_alive=true, Human: always ID '1'\n"
            "**Timers**: ~10 seconds (max 20), Phase indicators at 'top-center', Layout: 'center' default\n"
            "**DEFAULT VISIBILITY**: Unless explicitly private/group-targeted, make items PUBLIC with audience_type=true.\n\n"
            
            "📝 **REFEREE CONCLUSIONS CRITICAL USAGE RULES**:\n"
            "• **🔴 CRITICAL notes**: Indicate player deaths - MUST exclude these players from all UI\n"
            "• **🚫 UI FILTER notes**: Explicitly tell you which players to exclude from voting/targeting\n"
            "• **⚠️ VOTING STATUS notes**: Show who hasn't voted - create reminders for these players\n"
            "• **🎯 DECISION notes**: Show automatic decisions made - incorporate into UI context\n"
            "• **🤖 BOT REMINDER notes**: Indicate which bots need UI for actions\n"
            "• ALWAYS read referee_conclusions FIRST before creating any voting panels or target selection UI\n\n"
            
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
        }
    )

# Define the workflow graph
workflow = StateGraph(AgentState)

# Add all nodes - Keep InitialRouterNode as entry, BotBehaviorNode routes to PhaseNode
workflow.add_node("InitialRouterNode", InitialRouterNode)
workflow.add_node("ChatBotNode", ChatBotNode)
workflow.add_node("BotBehaviorNode", BotBehaviorNode) 
workflow.add_node("RefereeNode", RefereeNode)
workflow.add_node("PhaseNode", PhaseNode)
workflow.add_node("RoleAssignmentNode", RoleAssignmentNode)
workflow.add_node("ActionExecutor", ActionExecutor)

# Keep InitialRouterNode as entry point
workflow.set_entry_point("InitialRouterNode")

# Compile the graph (LangGraph API handles persistence itself in local_dev/cloud)
graph = workflow.compile()
