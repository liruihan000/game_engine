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

def format_declaration_for_prompt(declaration: dict) -> str:
    """
    Format the DSL declaration section for better readability in prompts.
    
    Args:
        declaration: The declaration section from DSL content
        
    Returns:
        Formatted string with structured declaration information
    """
    try:
        if not declaration:
            return "No declaration available"
        
        lines = []
        
        # Game description (full text)
        desc = declaration.get('description', '')
        if desc:
            lines.append(f"📖 Description: {desc}")
        
        # Basic game info
        if declaration.get('is_multiplayer'):
            min_players = declaration.get('min_players', 'N/A')
            lines.append(f"👥 Multiplayer: Yes (Min: {min_players} players)")
        
        # Roles with details
        roles = declaration.get('roles', [])
        if roles:
            lines.append(f"🎭 Roles ({len(roles)}):")
            for role in roles:
                role_name = role.get('name', 'Unknown')
                role_desc = role.get('description', 'No description')
                lines.append(f"    • {role_name}: {role_desc}")
        
        # Player states fields with details
        player_states = declaration.get('player_states', {})
        if player_states:
            field_count = len(player_states)
            lines.append(f"📊 Player State Fields ({field_count}):")
            for field_name, field_def in player_states.items():
                field_type = field_def.get('type', 'unknown')
                example = field_def.get('example', 'N/A')
                description = field_def.get('description', 'No description')
                lines.append(f"    • {field_name}: {field_type} (e.g., {example}) - {description}")
        
        # Audience groups with details
        audience_groups = declaration.get('audience_groups', {})
        if audience_groups:
            group_count = len(audience_groups)
            lines.append(f"👥 Audience Groups ({group_count}):")
            for group_name, group_def in audience_groups.items():
                description = group_def.get('description', 'No description')
                condition = group_def.get('selection_criteria', 'No criteria')
                lines.append(f"    • {group_name}: {description}")
                lines.append(f"      Criteria: {condition}")
        
        return "\n    ".join(lines) if lines else "Empty declaration"
        
    except Exception as e:
        logger.error(f"Error formatting declaration: {e}")
        return str(declaration)[:300] + ("..." if len(str(declaration)) > 300 else "")


def format_dict_for_prompt(data: dict, indent_level: int = 0) -> str:
    """
    Generic function to format any dictionary for better readability in prompts.
    Recursively handles nested dictionaries and lists.
    
    Args:
        data: Dictionary to format
        indent_level: Current indentation level (for nested structures)
        
    Returns:
        Formatted string with structured dictionary information
    """
    if not data:
        return "{}"
    
    lines = []
    indent = "  " * indent_level
    
    for key, value in data.items():
        if isinstance(value, dict):
            if value:  # Non-empty dict
                lines.append(f"{indent}{key}:")
                nested_lines = format_dict_for_prompt(value, indent_level + 1)
                lines.append(nested_lines)
            else:  # Empty dict
                lines.append(f"{indent}{key}: {{}}")
        elif isinstance(value, list):
            if value:  # Non-empty list
                lines.append(f"{indent}{key}:")
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        lines.append(f"{indent}  [{i}]:")
                        nested_lines = format_dict_for_prompt(item, indent_level + 2)
                        lines.append(nested_lines)
                    else:
                        lines.append(f"{indent}  [{i}]: {item}")
            else:  # Empty list
                lines.append(f"{indent}{key}: []")
        else:
            # Simple value (string, number, boolean, etc.)
            lines.append(f"{indent}{key}: {value}")
    
    return "\n".join(lines)




def format_current_phase_for_prompt(current_phase: dict, current_phase_id: int) -> str:
    """
    Format the current phase information for better readability in prompts.
    
    Args:
        current_phase: Current phase data from DSL
        current_phase_id: Current phase ID
        
    Returns:
        Formatted string with structured current phase information
    """
    try:
        if not current_phase:
            return f"⚠️ Phase {current_phase_id}: No phase data available"
        
        lines = []
        
        # Phase header with ID and name
        phase_name = current_phase.get('name', f'Phase {current_phase_id}')
        lines.append(f"🎯 Current Phase: ID {current_phase_id} - \"{phase_name}\"")
        
        # Phase description (full text)
        description = current_phase.get('description', '')
        if description:
            lines.append(f"📝 Description: {description}")
        
        # Actions/requirements - show all actions with full details
        actions = current_phase.get('actions', [])
        if actions:
            action_count = len(actions)
            lines.append(f"⚙️ Required Actions ({action_count}):")
            for i, action in enumerate(actions, 1):
                if isinstance(action, dict):
                    desc = action.get('description', 'No description')
                    tools = action.get('tools', [])
                    tool_str = f" → Tools: [{', '.join(tools)}]" if tools else ""
                    lines.append(f"    {i}. {desc}{tool_str}")
                else:
                    lines.append(f"    {i}. {str(action)}")
        
        # Completion criteria
        completion_criteria = current_phase.get('completion_criteria', {})
        if completion_criteria:
            criteria_type = completion_criteria.get('type', 'Unknown')
            criteria_desc = completion_criteria.get('description', '')
            
            # Format specific criteria types
            if criteria_type == 'player_action':
                wait_for = completion_criteria.get('wait_for', 'action')
                target_info = completion_criteria.get('target_players', {})
                if target_info:
                    condition = target_info.get('condition', 'Unknown condition')
                    lines.append(f"✅ Completion: {criteria_type} - {wait_for}")
                    lines.append(f"    Target: {condition}")
                else:
                    lines.append(f"✅ Completion: {criteria_type} - {wait_for}")
            elif criteria_type == 'timer':
                lines.append(f"✅ Completion: Timer expires")
            elif criteria_type == 'UI_displayed':
                lines.append(f"✅ Completion: UI components displayed")
            else:
                lines.append(f"✅ Completion: {criteria_type}")
            
            # Add description if different
            if criteria_desc and criteria_desc != criteria_type:
                lines.append(f"    Detail: {criteria_desc}")
        
        # Next phase info
        next_phase = current_phase.get('next_phase')
        if next_phase:
            if isinstance(next_phase, dict):
                next_id = next_phase.get('id', 'Unknown')
                next_name = next_phase.get('name', 'Unknown')
                lines.append(f"➡️ Next Phase: ID {next_id} - \"{next_name}\"")
            else:
                lines.append(f"➡️ Next Phase: {next_phase}")
        
        return "\n".join(lines)
        
    except Exception as e:
        logger.error(f"Error formatting current phase: {e}")
        return f"Phase {current_phase_id}: {str(current_phase)[:200]}{'...' if len(str(current_phase)) > 200 else ''}"


def summarize_items_for_prompt(items) -> str:
    """
    Summarize current UI items with complete readable JSON format for better LLM understanding.
    Shows full item structure including all data fields in formatted JSON.
    
    Args:
        items: Items data structure from game state
        
    Returns:
        Formatted JSON string of all items with complete details
    """
    import json
    
    try:
        # Handle different input types more robustly
        if items is None:
            return "(no items)"
        
        # Convert to list if needed
        if not isinstance(items, (list, tuple)):
            try:
                items = list(items)
            except (TypeError, ValueError):
                return f"(unable to summarize items: invalid type {type(items)})"
        
        if not items:
            return "(no items)"
        
        # Convert items to JSON-serializable format
        serializable_items = []
        for item in items:
            try:
                if isinstance(item, dict):
                    serializable_items.append(item)
                else:
                    # Try to convert to dict if it has attributes
                    item_dict = {}
                    for attr in ['id', 'type', 'name', 'subtitle', 'data']:
                        if hasattr(item, attr):
                            item_dict[attr] = getattr(item, attr)
                    if item_dict:
                        serializable_items.append(item_dict)
                    else:
                        serializable_items.append(str(item))
            except Exception as e:
                serializable_items.append(f"<error parsing item: {e}>")
        
        # Format as readable JSON
        formatted_json = json.dumps(serializable_items, indent=2, ensure_ascii=False, default=str)
        
        return f"Canvas Items ({len(items)} total):\n{formatted_json}"
        
    except Exception as e:
        return f"(unable to summarize items: {e} - type: {type(items)})"


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





def filter_backend_tools_from_messages(full_messages: list) -> list:
    """
    Filter out backend tool calls and tool_use blocks from messages for LLM consumption.
    Removes tool_calls and tool_use blocks while keeping only text content from AIMessage and HumanMessage.
    
    Args:
        full_messages: List of messages from conversation history
        
    Returns:
        List of filtered messages safe for LLM input (no backend tool remnants)
    """
    from langchain_core.messages import AIMessage, HumanMessage
    
    processed_messages = []
    for msg in full_messages:
        if isinstance(msg, AIMessage):
            # Keep AIMessage but remove tool_calls and tool_use blocks from content
            if msg.content:
                if isinstance(msg.content, str) and msg.content.strip():
                    # String content - keep as is
                    processed_messages.append(AIMessage(content=msg.content))
                elif isinstance(msg.content, list) and msg.content:
                    # List content - filter out tool_use blocks, keep text blocks
                    filtered_content = []
                    for item in msg.content:
                        if isinstance(item, dict) and item.get("type") == "tool_use":
                            # Skip tool_use blocks to avoid Claude API errors
                            continue
                        else:
                            # Keep text blocks and other content
                            filtered_content.append(item)
                    
                    # Only add message if it has non-tool content
                    if filtered_content:
                        processed_messages.append(AIMessage(content=filtered_content))
        elif isinstance(msg, HumanMessage):
            # Keep HumanMessage as is
            processed_messages.append(msg)
        # Skip ToolMessage and other types
    
    return processed_messages


def filter_backend_tools_from_response(response, backend_tool_names: set):
    """
    Remove backend tool calls from LLM response to prevent tool_use/tool_result mismatch.
    Keeps only frontend tool calls and cleans content accordingly.
    
    Args:
        response: AIMessage response from LLM
        backend_tool_names: Set of backend tool names to filter out
        
    Returns:
        Cleaned AIMessage with only frontend tool calls
    """
    try:
        from langchain_core.messages import AIMessage
    except ImportError:
        # Return original response if import fails
        return response
    
    if not hasattr(response, 'tool_calls') or not response.tool_calls:
        return response
    
    remaining_tool_calls = []
    for tc in response.tool_calls:
        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
        if name and name not in backend_tool_names:
            remaining_tool_calls.append(tc)
    
    # Update response with only frontend tool calls
    if remaining_tool_calls != response.tool_calls:
        # Clean content: remove backend tool_use blocks
        cleaned_content = response.content
        if isinstance(cleaned_content, list):
            # Filter out backend tool_use blocks from content list
            filtered_content = []
            for item in cleaned_content:
                if isinstance(item, dict) and item.get("type") == "tool_use":
                    tool_name = item.get("name", "")
                    if tool_name not in backend_tool_names:
                        filtered_content.append(item)
                else:
                    filtered_content.append(item)
            cleaned_content = filtered_content
        
        response = AIMessage(content=cleaned_content, tool_calls=remaining_tool_calls)
    
    return response


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