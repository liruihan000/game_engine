"""
Utility functions for game agent tools.
Helper functions that are not LangChain tools but support tool execution.
"""
import logging
import os
import time
import yaml
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import aiofiles
from .backend_tools import _execute_update_player_actions

logger = logging.getLogger(__name__)




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

def summarize_items_for_prompt(items: list) -> str:
    """Summarize current UI items with ID, type, name, position - formatted for ActionExecutor deletion/creation decisions."""
    try:
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


def process_human_action_if_needed(messages: list, player_actions: dict, player_states: dict, current_phase_id: int, room_session: dict, dsl_content: dict) -> dict:
    """
    Process human action from last message if it's meaningful (not chat, not generic control).
    
    Args:
        messages: List of messages from the conversation
        player_actions: Current player actions dict
        player_states: Current player states dict
        current_phase_id: Current phase ID
        room_session: Room session data containing player info
        dsl_content: DSL configuration for getting phase info
        
    Returns:
        Updated playerActions dict
    """
    current_player_actions = dict(player_actions)
    current_player_states = dict(player_states)
    
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
                current_phase = phases.get(current_phase_id, {}) or phases.get(str(current_phase_id), {})
                phase_name = current_phase.get('name', f'Phase {current_phase_id}')
                
                # Log human action using update_player_actions logic
                logger.info(f"[ProcessHumanAction] Processing action for Player 1: {last_msg.content}")
                current_player_actions = _execute_update_player_actions(
                    current_player_actions, 
                    "1",  # Player 1 (human)
                    str(last_msg.content)[:200],  # Truncate long messages
                    phase_name,
                    room_session,
                    current_player_states
                )
                logger.info(f"[ProcessHumanAction] Updated playerActions for Player 1")
                
    except Exception as e:
        logger.error(f"[ProcessHumanAction] Error processing human action: {e}")
    
    return current_player_actions


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


async def load_dsl_by_gamename(gamename: str) -> dict:
    """Load DSL content from YAML file based on gameName"""
    if not gamename:
        logger.warning("[DSL] No gameName provided, returning empty DSL")
        return {}
    
    try:
        
        dsl_file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'games', f"{gamename}.yaml")
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