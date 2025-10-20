"""
Game agent tool definitions.
All @tool decorated functions for LangChain integration.
"""
from typing import Any
from langchain.tools import tool



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
            - "PHASE_SUGGESTION" for phase branch suggestions (💡)
            - "BRANCH_RECOMMENDATION" for branch selection advice (🔀)
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
        "PHASE_SUGGESTION": "💡",
        "BRANCH_RECOMMENDATION": "🔀",
        "EVENT": "📝"
    }
    emoji = emoji_map.get(note_type, "📝")
    formatted_note = f"{emoji} {note_type}: {content}"
    return f"Will add game note: {formatted_note}"


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
        "PHASE_SUGGESTION": "💡",
        "BRANCH_RECOMMENDATION": "🔀",
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
   
    # Initialize player state if not exists
    if str(player_id) not in current_player_states:
        current_player_states[str(player_id)] = {}
    
    # Update the specific state
    current_player_states[str(player_id)][state_name] = state_value
    
    return current_player_states


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
           return current_player_states
      
    # Update only role
    current_player_states[str(player_id)]["role"] = role
    
    return current_player_states


def _execute_set_feedback_decision(player_id_list: list, need_feedback_message: str) -> dict:
    """
    Execute the actual logic to set feedback decision. Returns feedback decision dict.
    
    Args:
        player_id_list: List of player IDs who need feedback
        need_feedback_message: Message for waiting players
    
    Returns:
        Dict with player_id_list and need_feedback_message
    """
    return {
        "player_id_list": player_id_list,
        "need_feedback_message": need_feedback_message
    }




def _execute_update_player_actions(current_player_actions: dict, player_id: str, actions: str, phase: str, room_session: dict, current_player_states: dict) -> dict:
    """
    Execute the actual logic to add player actions. Returns updated player_actions dict.
    
    Args:
        current_player_actions: Current player actions state
        player_id: Player ID
        actions: Action description  
        phase: Current phase
        room_session: Room session data containing player info
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
    
    # Generate action ID specific to this player (each player has independent ID sequence)
    player_action_ids = []
    player_actions_dict = current_player_actions[str(player_id)]["actions"]
    for action_data in player_actions_dict.values():
        if isinstance(action_data, dict) and "id" in action_data:
            try:
                player_action_ids.append(int(action_data["id"]))
            except (ValueError, TypeError):
                pass
    action_id = str(max(player_action_ids, default=0) + 1)
    timestamp = int(time.time() * 1000)
    
    current_player_actions[str(player_id)]["name"] = player_name  # Update name
    current_player_actions[str(player_id)]["actions"][action_id] = {
        "action": actions,
        "timestamp": timestamp,
        "phase": phase,
        "id": action_id
    }
    
     
    return current_player_actions