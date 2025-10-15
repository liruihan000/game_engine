"""
Chatbot Agent - Synchronized with DM Agent State
This creates a chatbot that can role-play as different bot players based on their game state and personality.
"""

import logging
import os
import json
from dotenv import load_dotenv
from typing import Literal, List, Dict, Any, Optional
from typing_extensions import TypedDict
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from copilotkit import CopilotKitState
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Load environment variables
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(env_path)

# Configure logging
logger = logging.getLogger('ChatbotAgent')
logger.handlers.clear()

# Enable verbose logging
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

# File handler
from datetime import datetime
date_str = datetime.now().strftime('%Y%m%d')
log_file = f'/home/lee/canvas-with-langgraph-python/logs/chatbot_agent_{date_str}.log'
os.makedirs(os.path.dirname(log_file), exist_ok=True)

file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.propagate = False
logger.info(f"Chatbot logging to: {log_file}")


class ChatbotState(CopilotKitState):
    """Chatbot state synchronized with DM Agent"""
    # Synchronized fields from DM Agent
    items: List[Dict[str, Any]] = []
    current_phase_id: int = 0
    player_states: Dict[str, Any] = {}
    gameName: str = ""
    dsl: dict = {}
    botbehavior: dict = {}
    roomSession: Dict[str, Any] = {}
    
    # Chat-specific fields
    chat_messages: List[Dict[str, Any]] = []  # Chat history
    active_bot_id: str = ""  # Current bot responding
    chat_context: str = ""  # Current conversation context
    bot_personalities: Dict[str, Dict[str, Any]] = {}  # Bot personality data


async def generate_bot_personality(player_id: str, player_data: dict, game_context: dict) -> dict:
    """Generate personality traits for a bot player based on their role and game context"""
    try:
        role = player_data.get('role', 'Unknown')
        name = player_data.get('name', f'Player {player_id}')
        is_alive = player_data.get('alive', player_data.get('is_alive', True))
        
        # Base personality traits based on common game roles
        personality_templates = {
            'Werewolf': {
                'traits': ['cunning', 'deceptive', 'strategic'],
                'speech_style': 'careful and measured',
                'goals': ['survive', 'eliminate villagers', 'avoid suspicion']
            },
            'Villager': {
                'traits': ['honest', 'suspicious', 'collaborative'],
                'speech_style': 'direct and concerned',
                'goals': ['find werewolves', 'protect innocent players', 'seek truth']
            },
            'Detective': {
                'traits': ['analytical', 'observant', 'logical'],
                'speech_style': 'questioning and methodical',
                'goals': ['investigate suspicious players', 'gather evidence', 'guide village']
            },
            'Doctor': {
                'traits': ['protective', 'wise', 'cautious'],
                'speech_style': 'thoughtful and caring',
                'goals': ['protect important players', 'stay alive', 'help village']
            }
        }
        
        # Get base personality or create generic one
        base_personality = personality_templates.get(role, {
            'traits': ['adaptive', 'social', 'competitive'],
            'speech_style': 'friendly and engaging',
            'goals': ['play well', 'have fun', 'win game']
        })
        
        # Modify personality based on current game state
        if not is_alive:
            base_personality['traits'].append('vengeful')
            base_personality['speech_style'] = 'ghostly and observant'
        
        return {
            'player_id': player_id,
            'name': name,
            'role': role,
            'is_alive': is_alive,
            'traits': base_personality['traits'],
            'speech_style': base_personality['speech_style'],
            'goals': base_personality['goals'],
            'current_mood': 'neutral'
        }
        
    except Exception as e:
        logger.error(f"Error generating bot personality for player {player_id}: {e}")
        return {
            'player_id': player_id,
            'name': f'Player {player_id}',
            'role': 'Unknown',
            'is_alive': True,
            'traits': ['adaptive'],
            'speech_style': 'casual',
            'goals': ['participate'],
            'current_mood': 'neutral'
        }


async def ChatbotNode(state: ChatbotState, config: RunnableConfig) -> Command[Literal["__end__"]]:
    """
    Main chatbot node that responds as different bot players based on game context
    """
    logger.info("[ChatbotNode] Processing chat interaction")
    
    # Extract current game state
    player_states = state.get("player_states", {})
    current_phase_id = state.get("current_phase_id", 0)
    dsl_content = state.get("dsl", {})
    botbehavior = state.get("botbehavior", {})
    chat_messages = state.get("chat_messages", [])
    
    # Get the last human message to respond to
    last_message = ""
    human_player_id = "1"  # Player 1 is always human
    
    if chat_messages:
        last_msg = chat_messages[-1]
        last_message = last_msg.get("content", "")
        sender_id = last_msg.get("sender_id", human_player_id)
    else:
        logger.info("[ChatbotNode] No chat messages to respond to")
        return Command(goto=END, update={})
    
    # Don't respond to bot messages (avoid infinite loops)
    if sender_id != human_player_id:
        logger.info(f"[ChatbotNode] Ignoring bot message from player {sender_id}")
        return Command(goto=END, update={})
    
    # Select which bot should respond (rotate or based on context)
    bot_players = [pid for pid in player_states.keys() if pid != human_player_id]
    if not bot_players:
        logger.warning("[ChatbotNode] No bot players available to respond")
        return Command(goto=END, update={})
    
    # Select bot based on context or random selection
    responding_bot_id = bot_players[len(chat_messages) % len(bot_players)]  # Simple rotation
    bot_data = player_states.get(responding_bot_id, {})
    
    # Generate or update bot personality
    bot_personality = await generate_bot_personality(responding_bot_id, bot_data, dsl_content)
    
    # Initialize LLM
    model = init_chat_model("openai:gpt-4o")
    
    # Get current phase context
    phases = dsl_content.get('phases', {})
    current_phase = phases.get(current_phase_id, {}) or phases.get(str(current_phase_id), {})
    
    # Create system message for bot role-play
    system_message = SystemMessage(
        content=(
            f"CHATBOT ROLE-PLAY INSTRUCTION\n"
            f"You are playing the role of {bot_personality['name']} (Player {responding_bot_id}) in a game.\n\n"
            
            f"GAME CONTEXT:\n"
            f"- Current Phase: {current_phase.get('name', 'Unknown')} (ID: {current_phase_id})\n"
            f"- Your Role: {bot_personality['role']}\n"
            f"- Alive Status: {bot_personality['is_alive']}\n"
            f"- Game Type: {state.get('gameName', 'Unknown')}\n\n"
            
            f"YOUR CHARACTER PROFILE:\n"
            f"- Name: {bot_personality['name']}\n"
            f"- Personality Traits: {', '.join(bot_personality['traits'])}\n"
            f"- Speech Style: {bot_personality['speech_style']}\n"
            f"- Current Goals: {', '.join(bot_personality['goals'])}\n"
            f"- Current Mood: {bot_personality['current_mood']}\n\n"
            
            f"CURRENT GAME STATE:\n"
            f"- All Players: {list(player_states.keys())}\n"
            f"- Player States: {player_states}\n"
            f"- Phase Description: {current_phase.get('description', 'No description')}\n\n"
            
            f"BOT BEHAVIOR CONTEXT:\n"
            f"- Your Recent Behavior: {botbehavior.get(responding_bot_id, 'None specified')}\n"
            f"- All Bot Behaviors: {botbehavior}\n\n"
            
            f"CHAT HISTORY:\n"
            f"{json.dumps(chat_messages[-5:], indent=2)}\n\n"  # Last 5 messages for context
            
            f"LAST HUMAN MESSAGE TO RESPOND TO:\n"
            f'Player {human_player_id}: "{last_message}"\n\n'
            
            f"RESPONSE INSTRUCTIONS:\n"
            f"- Stay completely in character as {bot_personality['name']}\n"
            f"- Use your personality traits and speech style\n"
            f"- Consider your role's knowledge and motivations\n"
            f"- Keep responses conversational and natural (1-2 sentences)\n"
            f"- React appropriately to the current game phase\n"
            f"- If you're dead, speak as a ghost/observer\n"
            f"- Don't reveal information your character wouldn't know\n"
            f"- Be engaging but don't dominate the conversation\n\n"
            
            f"RESPOND AS {bot_personality['name']} WOULD:"
        )
    )
    
    # Generate bot response
    try:
        response = await model.ainvoke([system_message], config)
        bot_response = response.content.strip()
        logger.info(f"[ChatbotNode] Bot {responding_bot_id} ({bot_personality['name']}) responds: {bot_response[:100]}...")
        
        # Create new chat message
        new_message = {
            "sender_id": responding_bot_id,
            "sender_name": bot_personality['name'],
            "content": bot_response,
            "timestamp": datetime.now().isoformat(),
            "message_type": "chat",
            "role": bot_personality['role']
        }
        
        # Update chat messages
        updated_chat_messages = chat_messages + [new_message]
        
        # Update bot personalities cache
        updated_personalities = state.get("bot_personalities", {})
        updated_personalities[responding_bot_id] = bot_personality
        
        logger.info(f"[ChatbotNode] Generated response from {bot_personality['name']}")
        
        return Command(
            goto=END,
            update={
                "chat_messages": updated_chat_messages,
                "active_bot_id": responding_bot_id,
                "bot_personalities": updated_personalities,
                "messages": [AIMessage(content=bot_response)]  # For CopilotKit display
            }
        )
        
    except Exception as e:
        logger.error(f"[ChatbotNode] Error generating bot response: {e}")
        return Command(goto=END, update={})


async def ChatTriggerNode(state: ChatbotState, config: RunnableConfig) -> Command[Literal["ChatbotNode", "__end__"]]:
    """
    Trigger node that decides when to activate chatbot responses
    """
    logger.info("[ChatTriggerNode] Checking for chat triggers")
    
    chat_messages = state.get("chat_messages", [])
    player_states = state.get("player_states", {})
    
    # Check if there are any messages to respond to
    if not chat_messages:
        logger.info("[ChatTriggerNode] No chat messages, ending")
        return Command(goto=END, update={})
    
    # Check if the last message was from a human (player 1)
    last_message = chat_messages[-1]
    sender_id = last_message.get("sender_id", "")
    
    if sender_id == "1":  # Human player
        logger.info("[ChatTriggerNode] Human message detected, triggering chatbot response")
        return Command(goto="ChatbotNode", update={})
    else:
        logger.info("[ChatTriggerNode] Last message was from bot, no response needed")
        return Command(goto=END, update={})


# Define the chatbot workflow
workflow = StateGraph(ChatbotState)

# Add nodes
workflow.add_node("ChatTriggerNode", ChatTriggerNode)
workflow.add_node("ChatbotNode", ChatbotNode)

# Set entry point
workflow.set_entry_point("ChatTriggerNode")

# Compile the graph
chatbot_graph = workflow.compile()

# Export for use in main application
__all__ = ['chatbot_graph', 'ChatbotState', 'generate_bot_personality']