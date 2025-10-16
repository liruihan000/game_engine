#!/usr/bin/env python3
"""
Test script for update_player_actions tool
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json

# Mock AgentState class for testing
class MockAgentState:
    def __init__(self):
        self.roomSession = {
            "players": [
                {"gamePlayerId": "1", "name": "Alice", "isHost": True},
                {"gamePlayerId": "2", "name": "Bob", "isHost": False},
                {"gamePlayerId": "3", "name": "Bot_Charlie", "isHost": False}
            ]
        }
        self.player_states = {
            "4": {"name": "Dave"}
        }
        self.playerActions = {}

def test_update_player_actions():
    """Test the update_player_actions tool function logic"""
    print("🧪 Testing update_player_actions tool logic...")
    
    # Create a mock state
    mock_state = MockAgentState()
    
    print("📊 Initial state:")
    print(f"playerActions: {mock_state.playerActions}")
    print()
    
    # Test logic (simulate the tool function)
    def mock_update_player_actions(state, player_id, actions, phase):
        import time
        
        # Get player name from roomSession or player_states
        player_name = f"Player {player_id}"
        if state.roomSession and "players" in state.roomSession:
            for player in state.roomSession["players"]:
                if str(player.get("gamePlayerId", "")) == str(player_id):
                    player_name = player.get("name", player_name)
                    break
        elif player_id in state.player_states and "name" in state.player_states[player_id]:
            player_name = str(state.player_states[player_id]["name"])
        
        # Update player actions
        state.playerActions[player_id] = {
            "name": player_name,
            "actions": actions,
            "timestamp": int(time.time() * 1000),
            "phase": phase
        }
        
        print(f"📝 Recorded actions for {player_name} ({player_id}) in {phase}: {actions}")
        return f"Recorded actions for {player_name} (ID: {player_id}): {actions}"
    
    # Test 1: Record action for player with roomSession data
    print("🔍 Test 1: Player from roomSession")
    result1 = mock_update_player_actions(mock_state, "1", "voted for Bob, defended Alice", "day_voting")
    print(f"Result: {result1}")
    print(f"Updated playerActions: {json.dumps(mock_state.playerActions, indent=2)}")
    print()
    
    # Test 2: Record action for player from player_states
    print("🔍 Test 2: Player from player_states")
    result2 = mock_update_player_actions(mock_state, "4", "accused Charlie, voted for Charlie", "day_voting")
    print(f"Result: {result2}")
    print(f"Updated playerActions: {json.dumps(mock_state.playerActions, indent=2)}")
    print()
    
    # Test 3: Record action for unknown player
    print("🔍 Test 3: Unknown player")
    result3 = mock_update_player_actions(mock_state, "99", "did mysterious things", "night_action")
    print(f"Result: {result3}")
    print(f"Updated playerActions: {json.dumps(mock_state.playerActions, indent=2)}")
    print()
    
    # Test 4: Update existing player action
    print("🔍 Test 4: Update existing player")
    result4 = mock_update_player_actions(mock_state, "1", "changed vote to Charlie, gave final speech", "day_voting_final")
    print(f"Result: {result4}")
    print(f"Updated playerActions: {json.dumps(mock_state.playerActions, indent=2)}")
    print()
    
    print("✅ All tests completed!")

def test_chat_node_logic():
    """Test the chat node processing logic"""
    print("🧪 Testing chat node processing logic...")
    
    # Mock tool call data
    mock_tool_calls = [
        {
            "name": "update_player_actions",
            "args": {
                "player_id": "2",
                "actions": "cast final vote for Alice",
                "phase": "elimination_phase"
            }
        }
    ]
    
    # Mock state
    mock_state = {
        "playerActions": {},
        "roomSession": {
            "players": [
                {"gamePlayerId": "2", "name": "Bob", "isHost": False}
            ]
        },
        "player_states": {}
    }
    
    # Simulate chat node logic
    current_player_actions = dict(mock_state.get("playerActions", {}))
    
    for tool_call in mock_tool_calls:
        tool_name = tool_call.get("name")
        if tool_name == "update_player_actions":
            tool_args = tool_call.get("args", {})
            player_id = tool_args.get("player_id")
            actions = tool_args.get("actions")
            phase = tool_args.get("phase")
            
            if player_id and actions and phase:
                import time
                
                # Get player name from roomSession
                player_name = f"Player {player_id}"
                room_session = mock_state.get("roomSession", {})
                if room_session and "players" in room_session:
                    for player in room_session["players"]:
                        if str(player.get("gamePlayerId", "")) == str(player_id):
                            player_name = player.get("name", player_name)
                            break
                
                # Update player actions
                current_player_actions[player_id] = {
                    "name": player_name,
                    "actions": actions,
                    "timestamp": int(time.time() * 1000),
                    "phase": phase
                }
                print(f"[chatnode] Recording actions for {player_name} ({player_id}) in {phase}: {actions}")
    
    print(f"Final playerActions: {json.dumps(current_player_actions, indent=2)}")
    print("✅ Chat node logic test completed!")

if __name__ == "__main__":
    test_update_player_actions()
    print()
    test_chat_node_logic()