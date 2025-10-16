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
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain.tools import tool
import json
# Monitoring configuration
VERBOSE_LOGGING = True  # Set to False to disable detailed logging

# 直接配置 logger，不依赖 basicConfig
logger = logging.getLogger('DMAgentWithBot')
logger.handlers.clear()  # 清除现有 handlers

if VERBOSE_LOGGING:
    logger.setLevel(logging.INFO)
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    
    # 文件处理器 - 开发态按"天"合并日志（避免热重载生成多文件）
    from datetime import datetime
    date_str = datetime.now().strftime('%Y%m%d')
    log_file = f'/home/lee/canvas-with-langgraph-python/logs/dm_agent_bot_{date_str}.log'
    
    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.propagate = False  # 防止传播到root logger
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
    messages: List[Any] = []
    player_states: Dict[str, Any] = {}
    gameName: str = ""  # Current game DSL name (e.g., "werewolf", "coup")
    dsl: dict = {}
    need_feed_back_dict: dict = {}
    botbehavior: dict = {}
    referee_conclusions: List[str] = []
    roomSession: Dict[str, Any] = {}  # Room session data from frontend
    # Chat-specific fields for chatbot synchronization
    chat_messages: List[Dict[str, Any]] = []  # Chat history
    bot_personalities: Dict[str, Dict[str, Any]] = {}  # Bot personality data
    chat_active: bool = False  # Whether chat is currently active
    phase_completion: Dict[str, bool] = {}  # Phase completion status
    playerActions: Dict[str, Any] = {}  # Player actions

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
    #         logger.info("[InitialRouter] phase0_ui_done=false → Routing to ActionExecutor (phase 0)")
    #         return Command(goto="ActionExecutor", update={**updates, "dsl": state.get("dsl", {} )})
    #     else:
    #         logger.info("[InitialRouter] phase0_ui_done=true → Routing to PhaseNode for transition")
    #         return Command(goto="PhaseNode", update=updates)
    # else:
    # Check latest message for game chat and route to ChatBotNode if needed
    try:
        if full_messages:
            last_msg = full_messages[-1]
            logger.info(f"[InitialRouter] Last message type: {type(last_msg)}")
            logger.info(f"[InitialRouter] Last message content: {last_msg}")
            
            # Handle both LangChain Message objects and dict format
            is_human_message = False
            content = ''
            
            if hasattr(last_msg, 'type') and last_msg.type == 'human':
                # LangChain Message object
                is_human_message = True
                content = getattr(last_msg, 'content', '')
                logger.info(f"[InitialRouter] LangChain Message detected: {content[:100]}")
            elif isinstance(last_msg, dict) and (last_msg.get('type') == 'human' or last_msg.get('role') == 'user'):
                # Dict format message (CopilotKit format)
                is_human_message = True
                content = last_msg.get('content', '')
                logger.info(f"[InitialRouter] Dict Message detected: {content[:100]}")
            
            logger.info(f"[InitialRouter] Message check: is_human={is_human_message}")
            
            if is_human_message and content:
                # Check for game chat patterns
                if 'in game chat:' in content or 'to Bot' in content:
                    logger.info(f"[InitialRouter] Detected game chat: {content[:100]}")
                    final_dsl = dsl_content if dsl_content else state.get("dsl", {})
                    updates["dsl"] = final_dsl
                    return Command(goto="ChatBotNode", update=updates)
    except Exception as e:
        logger.error(f"[InitialRouter] Error checking chat message: {e}")

    logger.info(f"[InitialRouter] Routing to FeedbackDecisionNode (phase {current_phase_id})")
    # Ensure DSL is properly passed - use dsl_content if loaded, otherwise fallback to state
    final_dsl = dsl_content if dsl_content else state.get("dsl", {})
    updates["dsl"] = final_dsl
    logger.info(f"[InitialRouter] Passing DSL with keys: {list(final_dsl.keys()) if final_dsl else 'empty'}")
    updates["messages"] = last_msg
    
    # === DETAILED OUTPUT LOGGING ===
    logger.info(f"[InitialRouter][OUTPUT] Command goto: FeedbackDecisionNode")
    logger.info(f"[InitialRouter][OUTPUT] Updates keys: {list(updates.keys())}")
    logger.info(f"[InitialRouter][OUTPUT] Updates player_states: {updates.get('player_states', 'NOT_SET')}")
    logger.info(f"[InitialRouter][OUTPUT] Updates playerActions: {updates.get('playerActions', 'NOT_SET')}")
    
    return Command(goto="BotBehaviorNode", update=updates)

async def ChatBotNode(state: AgentState, config: RunnableConfig) -> Command[Literal["__end__"]]:
    """LLM-driven chat bot node"""
    logger.info("[ChatBotNode] Processing chat message")
    
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

Latest Message: {messages[-1] if messages else 'None'}

Instructions:
1. Check if this is a game chat message (contains 'in game chat:' or 'to Bot')
2. If user targeted specific bot ('to Bot [name]:'), use that bot
3. Otherwise, choose appropriate bot (non-player 1) 
4. Generate natural response based on bot's role and game context
5. IMPORTANT: Regardless of which bot identity you use to speak, always base your response on the facts from player_states and playerActions
6. MUST call addBotChatMessage tool with:
   - botId: the bot's player ID (e.g., "2", "3") 
   - botName: the bot's name
   - message: your generated response
   - messageType: "message"

Call the addBotChatMessage tool now if this is a chat message.
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

# async def FeedbackDecisionNode(state: AgentState, config: RunnableConfig) -> Command[Literal["__end__"]]:
#     """
#     FeedbackDecisionNode calls LLM to analyze who needs to provide feedback.
    
#     Input:
#     - trimmed_messages: Recent message history
#     - player_states: Current player states  
#     - current_phase and declaration: Phase configuration
    
#     Output:
#     - need_feed_back_dict with player_id_list and feedback message
#     """
#     # Print game name from state
#     game_name = state.get("gameName", "")
#     logger.info(f"[FeedbackDecisionNode] Game name from state: {game_name}")
    
#     logger.info("[FeedbackDecisionNode] Starting feedback decision analysis")
    
#     # Extract inputs for logging - use full_messages like InitialRouter
#     raw_messages = state.get("messages", []) or []
#     # Handle different message formats
#     if isinstance(raw_messages, dict):
#         # If messages is a dict, it might be a single message or a dict with message keys
#         logger.warning(f"[FeedbackDecisionNode] messages is a dict: {type(raw_messages)}, keys: {list(raw_messages.keys()) if raw_messages else []}")
#         if 'content' in raw_messages or 'type' in raw_messages:
#             # Single message as dict
#             full_messages = [raw_messages]
#         else:
#             # Multiple messages with numeric/string keys
#             full_messages = list(raw_messages.values()) if raw_messages else []
#     elif isinstance(raw_messages, list):
#         full_messages = raw_messages
#     else:
#         logger.warning(f"[FeedbackDecisionNode] messages is unexpected type: {type(raw_messages)}, converting to empty list")
#         full_messages = []
    
#     trimmed_messages = full_messages[-10:] if full_messages else []  # Keep last 10 messages for context
#     player_states = state.get("player_states", {})
#     current_phase_id = state.get("current_phase_id", 0)
#     dsl_content = state.get("dsl", {})
#     playerActions = state.get("playerActions", {})
    
#     # === DETAILED INPUT LOGGING ===
#     logger.info(f"[FeedbackDecisionNode][INPUT] current_phase_id: {current_phase_id}")
#     logger.info(f"[FeedbackDecisionNode][INPUT] player_states: {player_states}")
#     logger.info(f"[FeedbackDecisionNode][INPUT] playerActions: {playerActions}")
#     logger.info(f"[FeedbackDecisionNode][INPUT] state keys: {list(state.keys())}")
    
#     # Debug message arrays
#     logger.info(f"[FeedbackDecisionNode][DEBUG] full_messages count: {len(full_messages)}")
#     logger.info(f"[FeedbackDecisionNode][DEBUG] trimmed_messages count: {len(trimmed_messages)}")
#     if full_messages and isinstance(full_messages, list):
#         logger.info(f"[FeedbackDecisionNode][DEBUG] Last full message: {full_messages[-1]}")
#         logger.info(f"[FeedbackDecisionNode][DEBUG] Last full message type: {type(full_messages[-1])}")
#     if trimmed_messages and isinstance(trimmed_messages, list):
#         logger.info(f"[FeedbackDecisionNode][DEBUG] Last trimmed message: {trimmed_messages[-1]}")
#         logger.info(f"[FeedbackDecisionNode][DEBUG] Last trimmed message type: {type(trimmed_messages[-1])}")
    
#     # Debug DSL content
#     logger.info(f"[FeedbackDecisionNode] DSL content keys: {list(dsl_content.keys()) if dsl_content else 'empty'}")
#     if dsl_content and 'phases' in dsl_content:
#         logger.info(f"[FeedbackDecisionNode] Phases available: {list(dsl_content['phases'].keys())}")
#     else:
#         logger.info("[FeedbackDecisionNode] No phases found in DSL content")
    
#     # Get current phase details
#     phases = dsl_content.get('phases', {}) if dsl_content else {}
#     # Try both int and string keys to handle YAML parsing variations
#     current_phase = phases.get(current_phase_id, {}) or phases.get(str(current_phase_id), {})
#     declaration = dsl_content.get('declaration', {}) if dsl_content else {}
#     playerActions = state.get("playerActions", {})
    
#     # player_states should have been initialized by InitialRouterNode
#     if not player_states:
#         logger.warning("[FeedbackDecisionNode] player_states is empty! InitialRouterNode should have initialized it.")
#     else:
#         logger.info(f"[FeedbackDecisionNode] Using player_states: {len(player_states)} players")
    
#     # Log phase info
#     logger.info(f"[FeedbackDecisionNode] current_phase_id: {current_phase_id}")
#     logger.info(f"[FeedbackDecisionNode] current_phase: {current_phase}")
#     logger.info(f"[FeedbackDecisionNode] player_states: {player_states}")
    
#     # Initialize LLM
#     model = init_chat_model("openai:gpt-4o")
#     # Bind only update_player_actions in this node
#     model_with_tools = model.bind_tools([update_player_actions])
    
#     # Extract last human message - Handle both LangChain Message objects and dict format
#     last_human_message = ""
#     for _m in reversed(trimmed_messages):
#         is_human_message = False
#         content = ''
        
#         # Check for LangChain Message object
#         if hasattr(_m, 'type') and _m.type == 'human':
#             is_human_message = True
#             content = getattr(_m, 'content', '')
#         # Check for dict format message (CopilotKit format)
#         elif isinstance(_m, dict) and (_m.get('type') == 'human' or _m.get('role') == 'user'):
#             is_human_message = True
#             content = _m.get('content', '')
#         # Check for legacy HumanMessage object
#         elif isinstance(_m, HumanMessage) and getattr(_m, "content", None):
#             is_human_message = True
#             content = str(_m.content)
        
#         if is_human_message and content:
#             last_human_message = str(content)
#             break

#     logger.info(f"[FeedbackDecisionNode] last_human_message: {last_human_message}")

#     # Create system message with all inputs
#     system_message = SystemMessage(
#         content=(
#             "FEEDBACK DECISION ANALYSIS\n"
#             f"Current Phase ID: {current_phase_id}\n"
#             f"Current Phase Details: {current_phase}\n"
#             f"Game Declaration: {declaration}\n"
#             f"Player States: {player_states}\n"
#             f"Recent Messages: {[str(msg) for msg in trimmed_messages]}\n"
#             f"Last Human Message: {last_human_message}\n\n"
#             f"Player Actions: {playerActions}\n\n"


#             "HUMAN ACTION LOGGING (do this first):\n"
#             "- If and only if there is a new human message, call update_player_actions exactly once to log Player 1's latest action.\n"
#             "- Parameters: player_id='1'; actions=a concise summary of what Player 1 said/did; phase=use current_phase.name if available else f'phase_{Current Phase ID}'.\n"
#             "- Make only ONE call when there is new action; otherwise make no call.\n\n"

#             "Then proceed with feedback analysis.\n"
#             "TASK: Analyze the current phase and Player Actions to determine which players still need to provide feedback in this phase.\n"
#             "Based on the phase completion criteria, player states, player actions, and message history:\n"
#             "1. Identify which players are required to respond\n"
#             "2. Check who has already responded in recent message based on message history for this phase.\n"
#             "3. Generate appropriate feedback message\n"


#             "IMPORTANT - When NO feedback is needed:\n"
#             "- Phase completion is based on TIMER EXPIRY only (not player actions)\n"
#             "- Phase is purely INFORMATIONAL or DISPLAY-focused (showing results, announcements)\n"
#             "- Phase is AUTOMATIC system resolution (calculations, rule applications)\n"
#             "- All required players have already provided their responses\n"
#             "If ANY of these conditions apply, return empty player_id_list [].\n\n"
            
#             "CRITICAL - When feedback IS needed:\n"
#             "- Phase has completion_criteria with 'player_action' type\n"
#             "- Phase requires specific players to make choices, votes, or actions\n"
#             "- Check completion_criteria.target_players condition to identify WHO needs to respond\n"
#             "- Phase waiting for player responses based on roles (werewolves, detective, etc.)\n"
#             "- Always include Player 1 (human) if they match target_players criteria\n"
#             "- Include bot players who match target_players criteria for their coordination\n\n"
            
#             "🎯 **SPECIFIC PLAYER RESPONSE REQUIREMENT**:\n"
#             "- If the current phase completion requires response from a SPECIFIC PERSON (not all players):\n"
#             "  * You MUST identify and output that specific player in player_id_list\n"
#             "  * Example: Doctor protection phase → only include players with role='Doctor'\n"
#             "  * Example: Detective investigation → only include players with role='Detective'\n"
#             "  * Example: Werewolf target selection → only include players with role='Werewolf'\n"
#             "- Use completion_criteria.target_players.condition to determine the specific player(s)\n"
#             "- DO NOT include all players when only specific roles are required to respond\n"
#             "- MUST output the exact player ID(s) who need to provide the required response\n\n"

#             "OUTPUT FORMAT (JSON only)\n"
#             "Example 1 - Voting phase:\n"
#             "{\n"
#             '  "player_id_list": [1, 2, 4, 5, 7],\n'
#             '  "need_feedback_message": "Please cast your vote for elimination"\n'
#             "}\n\n"
#             "Example 2 - Werewolf night action phase:\n"
#             "{\n"
#             '  "player_id_list": [1, 3],\n'
#             '  "need_feedback_message": "Werewolves, choose your target for tonight"\n'
#             "}\n"
#             "Note: Include all players whose role='Werewolf' and is_alive=true\n\n"
#             "Example 3 - Detective investigation:\n"
#             "{\n"
#             '  "player_id_list": [2],\n'
#             '  "need_feedback_message": "Detective, choose a player to investigate"\n'
#             "}\n\n"
#             "Example 4 - No feedback needed (phase completed):\n"
#             "{\n"
#             '  "player_id_list": [],\n'
#             '  "need_feedback_message": "All actions completed, proceeding to next phase"\n'
#             "}\n\n"
#             "Example 5 - No feedback needed (automatic resolution):\n"
#             "{\n"
#             '  "player_id_list": [],\n'
#             '  "need_feedback_message": "Phase resolves automatically based on previous actions"\n'
#             "}\n\n"

#             "RULES:\n"
#             "- STEP 1: Check phase completion_criteria type - if 'player_action', players need feedback\n"
#             "- STEP 2: Evaluate target_players condition against current player_states\n"
#             "- STEP 3: Include ALL players matching the condition (both human and bots)\n"
#             "- Use numeric player IDs (1, 2, 3, etc.)\n"
#             "- Return empty list [] ONLY if:\n"
#             "  * Completion_criteria type is NOT 'player_action'\n"
#             "  * Phase is purely informational/display\n"
#             "  * All matching players have already responded\n"
#             "- For werewolf phases: include ALL players with role='Werewolf' and is_alive=true\n"
#             "- For voting phases: include ALL living players unless specified otherwise\n"
#             "- Create appropriate feedback message for the phase context\n"
#             "- Return valid JSON format only"
#         )
#     )

#     # Only treat update_player_actions as backend here
#     backend_tool_names = {"update_player_actions"}
    
#     full_messages = state.get("messages", []) or []
#     try:
#         if full_messages:
#             last_msg = full_messages[-1]
#             if isinstance(last_msg, AIMessage):
#                 pending_frontend_call = False
#                 for tc in getattr(last_msg, "tool_calls", []) or []:
#                     name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
#                     if name and name not in backend_tool_names:
#                         pending_frontend_call = True
#                         break
#             if pending_frontend_call:
#                 try:
#                     # print("[TRACE] Pending frontend tool calls detected; skipping LLM this turn and waiting for ToolMessage(s).")
#                     logger.info("[chatnode][end] Pending frontend tool calls detected; skipping LLM this turn and waiting for ToolMessage(s).")
#                 except Exception:
#                     pass
#                 return Command(
#                     goto=END
                    
#                 )
#     except Exception:
#         pass
    
#     # Call LLM (with backend tools bound)
#     response = await model_with_tools.ainvoke([system_message], config)
    
#     # === DETAILED LLM RESPONSE LOGGING ===
#     logger.info(f"[FeedbackDecisionNode][LLM_OUTPUT] Raw response content: {response.content}")
#     logger.info(f"[FeedbackDecisionNode][LLM_OUTPUT] Response type: {type(response)}")
    
#     # Parse LLM response
#     try:
#         response_content = clean_llm_json_response(str(response.content))
#         need_feed_back_dict = json.loads(response_content)
#         logger.info(f"[FeedbackDecisionNode][LLM_OUTPUT] Parsed need_feed_back_dict: {need_feed_back_dict}")
#     except Exception as e:
#         logger.error(f"[FeedbackDecisionNode][LLM_OUTPUT] Failed to parse LLM response: {e}")
#         logger.error(f"[FeedbackDecisionNode][LLM_OUTPUT] Raw response was: {response.content}")
#         # Fallback hardcoded response
#         need_feed_back_dict = {
#             "player_id_list": [1, 2, 4, 5, 7],
#             "need_feedback_message": "ask feedback"
#         }
#         logger.info(f"[FeedbackDecisionNode][LLM_OUTPUT] Using fallback: {need_feed_back_dict}")
    
#     # Apply backend tool effects inline (no ToolMessage)
#     tool_calls = getattr(response, "tool_calls", []) or []
#     logger.info(f"[FeedbackDecisionNode][TOOL_CALLS] Total tool calls: {len(tool_calls)}")
#     logger.info(f"[FeedbackDecisionNode][TOOL_CALLS] Tool calls details: {tool_calls}")
#     current_player_states = dict(player_states)
#     current_player_actions = dict(state.get("playerActions", {}))
#     logged_human = False
#     for tc in tool_calls:
#         name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
#         args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
#         if not isinstance(args, dict):
#             try:
#                 import json as _json
#                 args = _json.loads(args)
#             except Exception:
#                 args = {}
#         if name == "update_player_actions":
#             pid = args.get("player_id")
#             actions = args.get("actions")
#             phase = args.get("phase")
#             if pid and actions and phase:
#                 import time as _time
#                 player_name = f"Player {pid}"
#                 room_session = state.get("roomSession", {})
#                 if room_session and "players" in room_session:
#                     for p in room_session["players"]:
#                         if str(p.get("gamePlayerId", "")) == str(pid):
#                             player_name = p.get("name", player_name)
#                             break
#                 elif pid in current_player_states and "name" in current_player_states[pid]:
#                     player_name = current_player_states[pid]["name"]
#                 current_player_actions[str(pid)] = {
#                     "name": player_name,
#                     "actions": actions,
#                     "timestamp": int(_time.time() * 1000),
#                     "phase": phase,
#                 }
#                 if str(pid) == "1":
#                     logged_human = True

#     # No fallback: only log when there is actual new human action/tool call

#     # Save need_feed_back_dict as state
#     player_id_list = need_feed_back_dict.get("player_id_list", [])
#     need_feedback_message = need_feed_back_dict.get("need_feedback_message", "")
    
#     # Prepare common updates for all routes
#     common_updates = {
#         "need_feed_back_dict": need_feed_back_dict,
#         "player_states": current_player_states,
#         "playerActions": current_player_actions,
#         "roomSession": state.get("roomSession", {})
#     }
    
#     # Route based on feedback requirements
#     if len(player_id_list) == 0:
#         logger.info("[FeedbackDecisionNode] No players need feedback - routing to PhaseNode")
#         phasenode_updates = {**common_updates, "dsl": state.get("dsl", {}), "messages": ([])}
        
#         # === DETAILED OUTPUT LOGGING ===
#         logger.info(f"[FeedbackDecisionNode][OUTPUT] Command goto: PhaseNode")
#         logger.info(f"[FeedbackDecisionNode][OUTPUT] Updates keys: {list(phasenode_updates.keys())}")
#         logger.info(f"[FeedbackDecisionNode][OUTPUT] Updates player_states: {phasenode_updates.get('player_states', 'NOT_SET')}")
#         logger.info(f"[FeedbackDecisionNode][OUTPUT] Updates playerActions: {phasenode_updates.get('playerActions', 'NOT_SET')}")
        
#         return Command(goto="__end__", update=phasenode_updates)    
#     else:
#         logger.info("[FeedbackDecisionNode] Players need feedback - routing to BotBehaviorNode")
#         # Note: Player 1 feedback will be handled by ActionExecutor UI creation
#         botbehavior_updates = {**common_updates, "dsl": state.get("dsl", {}),"messages": ([])}
        
#         # === DETAILED OUTPUT LOGGING ===
#         logger.info(f"[FeedbackDecisionNode][OUTPUT] Command goto: BotBehaviorNode")
#         logger.info(f"[FeedbackDecisionNode][OUTPUT] Updates keys: {list(botbehavior_updates.keys())}")
#         logger.info(f"[FeedbackDecisionNode][OUTPUT] Updates player_states: {botbehavior_updates.get('player_states', 'NOT_SET')}")
#         logger.info(f"[FeedbackDecisionNode][OUTPUT] Updates playerActions: {botbehavior_updates.get('playerActions', 'NOT_SET')}")
        
#         return Command(goto="__end__", update=botbehavior_updates)


async def BotBehaviorNode(state: AgentState, config: RunnableConfig) -> Command[Literal["RefereeNode"]]:
    """
    BotBehaviorNode analyzes bot behavior and generates responses for non-human players.
    
    Input:
    - trimmed_messages: Recent message history
    - player_states: Current player states
    - current_phase and declaration: Phase configuration
    - need_feed_back_dict: Required feedback info
    
    Output:
    - botbehavior: dict{player_id: possible_behavior, ...}
    """
    # Print game name from state
    game_name = state.get("gameName", "")
    logger.info(f"[BotBehaviorNode] Game name from state: {game_name}")
    
    logger.info("[BotBehaviorNode] Starting bot behavior analysis")
    
    # Import LLM dependencies
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import SystemMessage
    import json

    # Print game name from state

    logger.info(f"[BotBehaviorNode] Game name from state: {game_name}")
    
    
    # Extract inputs for logging - use full_messages like InitialRouter
    raw_messages = state.get("messages", []) or []
    # Handle different message formats
    if isinstance(raw_messages, dict):
        # If messages is a dict, it might be a single message or a dict with message keys
        logger.warning(f"[BotBehaviorNode] messages is a dict: {type(raw_messages)}, keys: {list(raw_messages.keys()) if raw_messages else []}")
        if 'content' in raw_messages or 'type' in raw_messages:
            # Single message as dict
            full_messages = [raw_messages]
        else:
            # Multiple messages with numeric/string keys
            full_messages = list(raw_messages.values()) if raw_messages else []
    elif isinstance(raw_messages, list):
        full_messages = raw_messages
    else:
        logger.warning(f"[FeedbackDBotBehaviorNodeecisionNode] messages is unexpected type: {type(raw_messages)}, converting to empty list")
        full_messages = []
    
    trimmed_messages = full_messages[-10:] if full_messages else []  # Keep last 10 messages for context
    player_states = state.get("player_states", {})
    current_phase_id = state.get("current_phase_id", 0)
    dsl_content = state.get("dsl", {})
    playerActions = state.get("playerActions", {})
    
    # Extract inputs
    raw_messages = state.get("messages", []) or []
    # Handle different message formats
    if isinstance(raw_messages, dict):
        logger.warning(f"[BotBehaviorNode] messages is a dict: {type(raw_messages)}, keys: {list(raw_messages.keys()) if raw_messages else []}")
        if 'content' in raw_messages or 'type' in raw_messages:
            # Single message as dict
            messages = [raw_messages]
        else:
            # Multiple messages with numeric/string keys
            messages = list(raw_messages.values()) if raw_messages else []
    elif isinstance(raw_messages, list):
        messages = raw_messages
    else:
        logger.warning(f"[BotBehaviorNode] messages is unexpected type: {type(raw_messages)}, converting to empty list")
        messages = []
    
    trimmed_messages = messages[-10:] if messages else []
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
    
    # Enhanced system message - let LLM analyze and decide which bots should act
    phase_name = current_phase.get('name', f'Phase {current_phase_id}') if current_phase else f'Phase {current_phase_id}'
    phase_description = current_phase.get('description', '') if current_phase else ''
    
    # Get current player actions for context (before system message)
    current_player_actions = dict(state.get("playerActions", {}))
    
    # Get all bot players info (exclude player '1' which is human)
    all_bot_players = {}
    for pid, pdata in player_states.items():
        if pid != '1':  # Exclude human player
            all_bot_players[pid] = {
                'name': pdata.get('name', f'Player {pid}'),
                'role': pdata.get('role', 'unknown'),
                'is_alive': pdata.get('is_alive', True),
                'status': pdata.get('status', 'active')
            }
        # Extract last human message - Handle both LangChain Message objects and dict format
    last_human_message = ""
    for _m in reversed(trimmed_messages):
        is_human_message = False
        content = ''
        
        # Check for LangChain Message object
        if hasattr(_m, 'type') and _m.type == 'human':
            is_human_message = True
            content = getattr(_m, 'content', '')
        # Check for dict format message (CopilotKit format)
        elif isinstance(_m, dict) and (_m.get('type') == 'human' or _m.get('role') == 'user'):
            is_human_message = True
            content = _m.get('content', '')
        # Check for legacy HumanMessage object
        elif isinstance(_m, HumanMessage) and getattr(_m, "content", None):
            is_human_message = True
            content = str(_m.content)
        
        if is_human_message and content:
            last_human_message = str(content)
            break
    
    system_message = SystemMessage(
        content=(
            "BOT BEHAVIOR ANALYSIS AND GENERATION\n"
            f"Current Phase: {phase_name}\n"
            f"Phase Description: {phase_description}\n"
            f"Last Human Message: {last_human_message}\n"
            f"All Bot Players: {all_bot_players}\n"
            f"Current Player Actions History: {dict(list(current_player_actions.items())[-5:])}\n"
            f"Recent Messages: {[str(msg) for msg in trimmed_messages[-3:]]}\n"
            
            "\n"
            "TASK: Analyze the current game situation and determine which bots should respond.\n"
            "\n"
            "ANALYSIS STEPS:\n"
            "1. Analyze the current phase requirements and game context\n"
            "2. Consider the last human message and recent game events\n"
            "3. Determine which bot players should logically respond or act\n"
            "4. Consider each bot's role, status, and personality\n"
            "\n"
            "DECISION CRITERIA:\n"
            "- Phase-specific requirements (e.g., voting phases, discussion phases)\n"
            "- Direct responses to human player actions or questions\n"
            "- Role-based behaviors (e.g., werewolves coordinating, villagers discussing)\n"
            "- Game state changes requiring bot reactions\n"
            "- Natural conversation flow and timing\n"
            "\n"
            "INSTRUCTIONS:\n"
            "- For EACH bot that should act, call update_player_actions tool once\n"
            "- Generate contextually appropriate and role-specific actions\n"
            "- Consider bot personalities and previous actions for consistency\n"
            "- Ensure actions fit the current phase and game situation\n"
            "- NO text output - only tool calls for bots that should act\n"
            "- If no bots need to act based on analysis, make no tool calls"
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
    botbehavior = {}  # Empty dict for compatibility
    
    # Apply backend tool effects inline (no ToolMessage)
    tool_calls = getattr(response, "tool_calls", []) or []
    logger.info(f"[BotBehaviorNode][TOOL_CALLS] Total tool calls: {len(tool_calls)}")
    logger.info(f"[BotBehaviorNode][TOOL_CALLS] Tool calls details: {tool_calls}")
    current_player_states = dict(state.get("player_states", {}))
    # current_player_actions already defined above for system message context
    logger.info(f"[BotBehaviorNode] current_player_actions: {current_player_actions}")
    logged_human = False
    
    # First, record the latest user message to playerActions (copied from FeedbackDecisionNode)
    if last_human_message.strip() and not logged_human:
        import time
        # Get human player name from player_states (first player is the human user)
        human_player_name = "Human Player"  # fallback
        if player_states and "1" in player_states:
            human_player_name = player_states["1"].get("name", "Human Player")
        
        current_player_actions["1"] = {
            "name": human_player_name,
            "actions": last_human_message.strip(),
            "timestamp": int(time.time() * 1000),
            "phase": f"phase_{current_phase_id}",
        }
        logger.info(f"[BotBehaviorNode] Recorded human message from {human_player_name}: {last_human_message.strip()}")
        logged_human = True
    
    # Restrict to bots that actually need to act
    try:
        _acting_ids = set(str(i) for i in (need_feed_back_dict.get("player_id_list", []) or []))
    except Exception:
        _acting_ids = set()
    if "1" in _acting_ids:
        try:
            _acting_ids.discard("1")
        except Exception:
            pass
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
            # Allow any bot to be recorded (LLM decides who should act)
            # Only exclude human player (ID "1")
            if pid is not None and str(pid) == "1":
                continue
            if pid and actions and phase:
                import time as _time
                player_name = f"Player {pid}"
                room_session = state.get("roomSession", {})
                if room_session and "players" in room_session:
                    for p in room_session["players"]:
                        if str(p.get("gamePlayerId", "")) == str(pid):
                            player_name = p.get("name", player_name)
                            break
                elif pid in current_player_states and "name" in current_player_states[pid]:
                    player_name = current_player_states[pid]["name"]
                current_player_actions[str(pid)] = {
                    "name": player_name,
                    "actions": actions,
                    "timestamp": int(_time.time() * 1000),
                    "phase": phase,
                }
    
    # Save botbehavior as state and route to RefereeNode
    logger.info("[BotBehaviorNode] Routing to RefereeNode")
    return Command(
        goto="RefereeNode",
        update={
            "botbehavior": botbehavior,
            "player_states": current_player_states,
            "playerActions": current_player_actions,
            "roomSession": state.get("roomSession", {}),
            "dsl": state.get("dsl", {}),
            "messages": ([]),
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
    - botbehavior: Bot player behaviors
    
    Output:
    - Updated player_states based on game rules and player actions
    """
    # Print game name from state
    game_name = state.get("gameName", "")
    logger.info(f"[RefereeNode] Game name from state: {game_name}")
    
    logger.info("[RefereeNode] Starting referee analysis and state updates")
    
    # Extract inputs
    raw_messages = state.get("messages", []) or []
    # Handle different message formats
    if isinstance(raw_messages, dict):
        logger.warning(f"[RefereeNode] messages is a dict: {type(raw_messages)}, keys: {list(raw_messages.keys()) if raw_messages else []}")
        if 'content' in raw_messages or 'type' in raw_messages:
            # Single message as dict
            messages = [raw_messages]
        else:
            # Multiple messages with numeric/string keys
            messages = list(raw_messages.values()) if raw_messages else []
    elif isinstance(raw_messages, list):
        messages = raw_messages
    else:
        logger.warning(f"[RefereeNode] messages is unexpected type: {type(raw_messages)}, converting to empty list")
        messages = []
    
    trimmed_messages = messages[-10:] if messages else []
    player_states = state.get("player_states", {})
    current_phase_id = state.get("current_phase_id", 0)
    dsl_content = state.get("dsl", {})
    botbehavior = state.get("botbehavior", {})
    
    # Get last human message (from player 1)
    last_human_message = ""
    for msg in reversed(trimmed_messages):
        # Assume human messages can be identified by some property
        if hasattr(msg, 'content') and msg.content:
            last_human_message = str(msg.content)
            break
    
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
            f"Last Human Message: {last_human_message}\n"
            f"Bot Behaviors: {botbehavior}\n"
            f"Recent Messages: {[str(msg) for msg in trimmed_messages]}\n\n"
            f"Player Actions: {playerActions}\n\n"
            
            "🎯 **PRIMARY RESPONSIBILITY**: Update player_states based on player actions every round, fill states comprehensively but with evidence\n"
            "\n"
            "🏆 **GAME SCORING & JUDGEMENT RESPONSIBILITY** (New Core Duty):\n"
            "- Understand the game's scoring criteria and judgement mechanisms\n"
            "- When players take actions, apply score changes based on game rules\n" 
            "- Update score/points/performance related fields in player_states\n"
            "- Judge player performance based on DSL rules and game phases\n"
            "- Record scoring rationale and reasons for later display\n"
            "\n"
            "**SCORING ANALYSIS FRAMEWORK:**\n"
            "- Parse DSL declaration for scoring rules and win conditions\n"
            "- Analyze player actions for point-worthy behaviors\n"
            "- Apply penalties for rule violations or poor performance\n"
            "- Award points for successful actions, correct guesses, survival\n"
            "- Update score-related fields: score, performance_rating, ranking_position\n"
            "- Track scoring_history with reasons and timestamps\n"
            "\n"
            "TASK 1: COMPREHENSIVE Player State Updates (MANDATORY EVERY ROUND)\n"
            "\n"
            "**CORE RESPONSIBILITY:**\n"
            "- You are responsible for updating player_states based on player actions every round\n"
            "- Fill player state fields comprehensively, but all updates must have clear evidence\n"
            "- Make state inferences based on Player Actions, Bot Behaviors, Human Messages and game rules\n"
            "- Even without obvious actions, update relevant states based on game phases and rules\n"
            "\n"
            "**PLAYERACTIONS PARSING INSTRUCTIONS (Detailed action analysis):**\n"
            "- Carefully parse each entry in Player Actions, extract specific game action information\n"
            "- For target-based actions like 'chose target player X for [action]':\n"
            "  * Set night_action_submitted=True (if night phase) or day_action_submitted=True (if day phase)\n"  
            "  * Set last_night_action='[action_type]' or last_day_action='[action_type]'\n"
            "  * Set last_night_target=X or last_day_target=X (extract target player ID from action text)\n"
            "  * Update action_count, turn_participation, engagement_level and other participation metrics\n"
            "- For protection/support actions like 'protected/helped player X':\n"
            "  * Update relevant action tracking fields based on game rules\n"
            "  * Set protection_active=True, protected_player_id=X\n"
            "  * Update support_given, alliance_status and other social states\n"
            "- For investigation/info actions like 'investigated/checked player X':\n"
            "  * Update investigation tracking fields\n"
            "  * Set investigation_used=True, investigated_players=[X]\n"
            "  * Update knowledge_gained, suspicion_level and other information states\n"
            "- For voting actions, update vote_target_id, voting_confidence, voting_reason and other voting-related states\n"
            "- For communication actions, update social_interactions, trust_level, alliance_status and other social states\n"
            "- For resource/item actions, update inventory, health, currency, items_used and other resource states\n"
            "\n"
            "**STATE ENRICHMENT STRATEGY (Fill comprehensively but with evidence):**\n"
            "- Infer player intentions, strategies, psychological states based on actions\n"
            "- Update phase_specific states based on game phases (e.g., survival_priority, trust_network, etc.)\n"
            "- Track interaction relationships between players (target frequency, interaction history, etc.)\n"
            "- Update game-related quantitative metrics (activity, participation, influence, etc.)\n"
            "- Infer potential state changes based on game rules (e.g., role ability usage limits, etc.)\n"
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
            "\n"
            "🔄 **MANDATORY EXECUTION REQUIREMENTS:**\n"
            "- Must call update_player_state tool every round, even without obvious action changes\n"
            "- Must update states based on all information in Player Actions, cannot ignore any actions\n"
            "- Must perform scoring analysis: provide reasonable score changes based on player performance\n"
            "- Prioritize explicit actions (voting, target selection, ability usage, etc.)\n"
            "- Secondary processing of inferred states (psychological state, relationship changes, strategy adjustments, etc.)\n"
            "- Scoring processing: update score, performance_rating, ranking and other scoring-related fields\n"
            "- Ensure all player states get appropriate updates, maintain state integrity and consistency\n"
            "- All state updates must have clear evidence sources (Player Actions, game rules, phase requirements, etc.)\n"
            "- Record reasons for scoring changes to provide evidence for ActionExecutor display\n"
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
    
    # Apply ONLY the first update_player_state tool call inline (no ToolMessage)
    tool_calls = getattr(response, "tool_calls", []) or []
    logger.info(f"[RefereeNode][TOOL_CALLS] Total tool calls: {len(tool_calls)}")
    logger.info(f"[RefereeNode][TOOL_CALLS] Tool calls details: {tool_calls}")
    current_player_states = dict(updated_player_states)
    first_applied = False
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
        key = args.get("state_name")
        val = args.get("state_value")
        if pid and key is not None:
            current_player_states.setdefault(pid, {})
            current_player_states[pid][key] = val
            first_applied = True
        break

    # Route to PhaseNode with updated player states and conclusions
    logger.info("[RefereeNode] Routing to PhaseNode with updated player states and conclusions")
    return Command(
        goto="PhaseNode",
        update={
            "player_states": current_player_states,
            "referee_conclusions": conclusions,
            "roomSession": state.get("roomSession", {}),
            "dsl": state.get("dsl", {}),
            "messages": ([]),
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
                "messages": state.get("messages", []),
                "phase_completion": state.get("phase_completion", {}),
                "playerActions": state.get("playerActions", {}),
            }
        )
    
    # Use LLM for intelligent role assignment
    logger.info(f"[RoleAssignmentNode] Using LLM to assign roles to {len(unassigned_players)} players")
    
    model = init_chat_model("openai:gpt-4o")
    model_with_tools = model.bind_tools([update_player_state], parallel_tool_calls=True)
    
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
            "- Use update_player_state tool for each assignment\n"
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
            
            "Execute role assignments using update_player_state tools now."
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
                if name == "update_player_state":
                    args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                    if not isinstance(args, dict):
                        try:
                            import json as _json
                            args = _json.loads(args)
                        except Exception:
                            args = {}
                    pid = args.get("player_id")
                    key = args.get("state_name")
                    val = args.get("state_value")
                    if pid and key is not None:
                        updated_player_states.setdefault(pid, {})
                        updated_player_states[pid][key] = val
                        logger.info(f"[RoleAssignmentNode] LLM assigned: Player {pid} -> {key}={val}")
        
        logger.info("[RoleAssignmentNode] Role assignment completed, routing to ActionExecutor")
        
        return Command(
            goto="ActionExecutor",
            update={
                "current_phase_id": current_phase_id,
                "player_states": updated_player_states,
                "roomSession": state.get("roomSession", {}),
                "dsl": dsl_content,
                "messages": state.get("messages", []),
                "phase_completion": state.get("phase_completion", {}),
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
                "messages": state.get("messages", []),
                "phase_completion": state.get("phase_completion", {}),
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
    - botbehavior: Bot behaviors (for context)
    
    Output:
    - next_phase_id: Determined next phase
    """
    # Print game name from state
    game_name = state.get("gameName", "")
    logger.info(f"[PhaseNode] Game name from state: {game_name}")
    
    logger.info("[PhaseNode] Starting phase transition analysis")
    
    # Extract inputs
    dsl_content = state.get("dsl", {})
    current_phase_id = state.get("current_phase_id", 0)
    botbehavior = state.get("botbehavior", {})
    player_states = state.get("player_states", {})
    playerActions = state.get("playerActions", {})
    
    # === DETAILED INPUT LOGGING ===
    logger.info(f"[PhaseNode][INPUT] current_phase_id: {current_phase_id}")
    logger.info(f"[PhaseNode][INPUT] player_states: {player_states}")
    logger.info(f"[PhaseNode][INPUT] playerActions: {playerActions}")
    logger.info(f"[PhaseNode][INPUT] state keys: {list(state.keys())}")
    playerActions = state.get("playerActions", {})
    
    # Special check for phase 0: Must ensure ActionExecutor has run at least once before allowing transition
    if current_phase_id == 0:
        phase_completion = state.get("phase_completion", {})
        logger.info(f"[PhaseNode] [phase_completion] : {phase_completion}")
        phase0_executed = phase_completion.get("0", False)
        
        if not phase0_executed:
            logger.info("[PhaseNode] Phase 0 hasn't been executed yet by ActionExecutor; staying at phase 0")
            return Command(
                goto="ActionExecutor",
                update={
                    "current_phase_id": 0,
                    "player_states": player_states,
                    "roomSession": state.get("roomSession", {}),
                    "dsl": dsl_content,
                    "messages": ([]),
                }
            )
        else:
            logger.info("[PhaseNode] Phase 0 has been executed, proceeding with transition analysis")
    
    # Get current phase details
    phases = dsl_content.get('phases', {}) if dsl_content else {}
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
    trimmed_messages = messages[-10:] if isinstance(messages, list) and messages else []
    # PhaseNode focuses purely on phase transition - no role assignment
    
    system_message = SystemMessage(
        content=(
            "PHASE TRANSITION ANALYSIS WITH ROLE MANAGEMENT\n"
            f"itemsState (current frontend layout):\n{items_summary}\n"
            f"DSL Content: {dsl_content}\n"
            f"Current Phase ID: {current_phase_id}\n"
            f"Current Phase Details: {current_phase}\n"
            f"Game Declaration: {declaration}\n"
            f"Player States: {player_states}\n"
            f"Bot Behaviors: {botbehavior}\n\n"
            f"Phase Completion Flags: {state.get('phase_completion', {})}\n"
            f"Recent Messages: {[str(msg) for msg in trimmed_messages]}\n\n"
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
            
            "DSL condition: 'If living Werewolves >= living Villagers' → phase 99 (Werewolves Win)\n"
            "Analysis: Count alive werewolves vs alive villagers\n"
            "Werewolves equal/outnumber villagers: set_next_phase(transition=true, next_phase_id=99, transition_reason='Werewolves win by numbers')\n"
            "Otherwise continue game: set_next_phase(transition=true, next_phase_id=2, transition_reason='Game continues - moving to night phase')\n\n"
            
            "DSL condition: 'Otherwise, continue to next Night cycle' → phase 2\n"
            "Game continues: set_next_phase(transition=true, next_phase_id=2, transition_reason='Game continues to night phase')\n\n"
            

        )
    )
    
    # Call LLM with tools for all phases (needed for set_next_phase tool)
    response = await model_with_tools.ainvoke([system_message], config)
    
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

    # Validate proposed next phase id when transition=true
    def _phase_id_exists(pid: Any, phases_dict: dict) -> bool:
        try:
            if pid is None:
                return False
            # Check direct match
            if pid in phases_dict:
                return True
            # Check string version of pid
            if str(pid) in phases_dict:
                return True
            # Check integer version if pid is a numeric string
            if isinstance(pid, str) and pid.isdigit():
                return int(pid) in phases_dict
            return False
        except Exception:
            return False

    target_phase_id = current_phase_id
    if parsed_transition and _phase_id_exists(proposed_next_phase_id, phases):
        # Normalize numeric string to int
        if isinstance(proposed_next_phase_id, str) and proposed_next_phase_id.isdigit():
            try:
                proposed_next_phase_id = int(proposed_next_phase_id)
            except Exception:
                pass
        target_phase_id = proposed_next_phase_id
        logger.info(f"[PhaseNode] Transition approved → {current_phase_id} -> {target_phase_id}")
    elif parsed_transition:
        logger.warning(f"[PhaseNode] Invalid next_phase_id={proposed_next_phase_id}; staying at {current_phase_id}")
        target_phase_id = current_phase_id
    else:
        logger.info(f"[PhaseNode] No transition; staying at phase {current_phase_id}")

    logger.info("[PhaseNode] Routing to ActionExecutor")
    
    phasenode_outputs = {
        "current_phase_id": target_phase_id,
        "player_states": state.get("player_states", {}),
        "roomSession": state.get("roomSession", {}),
        "dsl": state.get("dsl", {}),
        "messages": ([]),
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
    # Player states and actions display tools
    "createPlayerStatesDisplay",
    "createPlayerActionsDisplay", 
    # Component management tools
    "deleteItem",
    "clearCanvas",
    # Player state management
    "markPlayerDead",
    # Chat tools
    "addBotChatMessage"
])

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

async def ActionValidatorNode(state: AgentState, config: RunnableConfig) -> Command[Literal["PhaseNode", "ActionExecutor"]]:
    """
    ActionValidatorNode - Currently in BYPASS mode (validation disabled).
    Simply passes through to allow execution to continue without validation.
    """
    logger.info("[ActionValidatorNode] ⚡ BYPASS MODE - Skipping validation, allowing execution to continue")
    
    # Reset retry count and continue to PhaseNode for phase progression
    return Command(goto="PhaseNode", update={"retry_count": 0})

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
        Dict with update confirmation
    """
    logger.info(f"[update_player_state] Player {player_id}: {state_name} = {state_value}")
    
    result = {
        "success": True,
        "player_id": player_id,
        "state_name": state_name, 
        "state_value": state_value,
        "updated": True
    }
    
    return result

@tool
def update_player_actions(state: AgentState, player_id: str, actions: str, phase: str) -> str:
    """
    Record player actions for AI tracking. Use this to log what players (including bots) did in each phase.
    
    Args:
        player_id: Player ID (e.g. '1', '2', '3')
        actions: Description of what the player did (e.g. 'voted for Alice, defended Bob')
        phase: Current game phase (e.g. 'day_voting', 'night_action', 'discussion')
    
    Returns:
        Confirmation message about the recorded action
    """
    import time
    
    # Get player name from roomSession or player_states
    player_name = f"Player {player_id}"
    if state.roomSession and "players" in state.roomSession:
        for player in state.roomSession["players"]:
            if str(player.get("gamePlayerId", "")) == str(player_id):
                player_name = player.get("name", player_name)
                break
    elif player_id in state.player_states and "name" in state.player_states[player_id]:
        player_name = state.player_states[player_id]["name"]
    
    # Update player actions
    current_phase_id = state.get("current_phase_id", 0)
    state.playerActions[player_id] = {
        "name": player_name,
        "actions": actions,
        "timestamp": int(time.time() * 1000),  # milliseconds
        "phase": phase,
        "current_phase_id": current_phase_id,
    }
    
    logger.info(f"📝 Recorded actions for {player_name} ({player_id}) in {phase}: {actions}")
    
    return f"Recorded actions for {player_name} (ID: {player_id}): {actions}"

# Centralized backend tools (shared across nodes)
backend_tools = [
    update_player_state,
    update_player_actions,
    set_next_phase,
]
BACKEND_TOOL_NAMES = {t.name for t in backend_tools}

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
            "🧠 ACTION EXECUTOR - STRATEGIC GAME ORCHESTRATION\n"
            "You are an intelligent game master AI with advanced reasoning capabilities.\n\n"
            
            "🏆 PREVIOUS ROUND CONCLUSIONS & CURRENT ROUND OBJECTIVES\n"
            "**ULTIMATE PURPOSE**: Help players understand the game, avoid confusion, and find joy in the experience!\n"
            "Treat players like children who know NOTHING - provide maximum information and guidance:\n\n"
            
            "📋 **LAST ROUND SUMMARY & REFEREE ANALYSIS**:\n"
            f"• What happened last round: {playerActions}\n"
            f"• Referee's conclusions about the game state: {referee_conclusions}\n"
            f"• Key evidence and findings: {player_states}\n"
            f"• Important changes since last time: [Extract from player_states vs previous states]\n\n"
            
            "🎯 **THIS ROUND'S CLEAR OBJECTIVES** (Explain everything to players):\n"
            f"• Current Phase Goal: {current_phase.get('name', 'Unknown')} - {current_phase.get('description', '')}\n"
            f"• What players need to do: {current_phase.get('actions', 'Follow phase instructions')}\n"
            f"• Success criteria: How will this phase complete successfully?\n"
            f"• Expected player interactions: What should players be discussing/voting/deciding?\n"
            f"• Timeline and urgency: How much time do players have?\n\n"
            
            "📊 **COMPLETE INFORMATION PACKAGE** (Assume players know nothing):\n"
            f"• Who is still alive: [Extract from player_states where is_alive=true]\n"
            f"• Who has died: [Extract from player_states where is_alive=false] \n"
            f"• Current scores/rankings: [Extract performance_rating, ranking_position]\n"
            f"• Role information visibility: What roles can be revealed to whom?\n"
            f"• Game progress: Phase {current_phase_id} of total phases, what's next?\n"
            f"• Critical evidence to display: Key clues players must see\n\n"
            
            "🎙️ **HOST MASTERY** (You know the rules best):\n"
            f"• WHO speaks now: Analyze {current_phase} rules to determine current speaker\n"
            f"• WHAT tools needed: Study {current_phase.get('actions', [])} for required UI components\n"
            f"• HOW phase completes: Deep understand DSL rules for {current_phase.get('name', '')} success criteria\n\n"
            
            "=== MANDATORY DEEP ANALYSIS PHASE ===\n"
            "Before taking ANY actions, you MUST analyze the complete game situation:\n\n"
            
            "🎯 CORE MISSION: You are the GAME MASTER & HOST executing phase actions and displaying game state\n"
            f"Current Phase: {current_phase_id} - {current_phase.get('name', 'Unknown')} ({current_phase.get('description', '')})\n"
            f"Actions to Execute: {actions_to_execute}\n\n"
            
            "📊 STRATEGIC REASONING REQUIREMENTS:\n"
            "1. **Current Game State Analysis**: What is the complete situation?\n"
            f"   • Player States: {player_states}\n"
            f"   • Player Actions This Round: {playerActions}\n"
            f"   • Referee Analysis: {referee_conclusions}\n"
            f"   • Current UI Layout: {items_summary}\n"
            f"   • Phase Requirements: {current_phase}\n"
            f"   • Game DSL Context: {dsl_content}\n\n"
            
            "2. **Strategic Game Logic**: What is the current phase trying to accomplish?\n"
            "   • What UI elements are needed to advance the game logically?\n"
            "   • How should information be revealed to different player audiences?\n"
            "   • What is the optimal timing and sequence for UI creation?\n\n"
            
            "3. **Audience Strategy Planning**: Who should see what?\n"
            "   • Which components should be PUBLIC (audience_type=true)?\n"
            "   • Which need GROUP targeting (audience_type=false + specific IDs)?\n"
            "   • Which require INDIVIDUAL targeting (single player)?\n"
            "   • How does audience targeting serve the game's strategic flow?\n\n"
            
            "4. **Phase Progression Logic**: How will this phase complete?\n"
            "   • What player actions or interactions will complete this phase?\n"
            "   • How will the UI facilitate natural game progression?\n"
            "   • What engagement patterns will keep all players active?\n\n"
            
            "🚨 CRITICAL EXECUTION RULES (ABSOLUTE PROHIBITIONS):\n"
            "• NEVER return ONLY deleteItem calls - THIS IS TASK FAILURE!\n"
            "• MANDATORY: Every deleteItem MUST be followed by create tools in SAME response!\n"
            "• ALL components MUST have audience permissions: audience_type=true (public) OR audience_type=false + audience_ids=['1','2']\n"
            "• DEFAULT visibility: PUBLIC (audience_type=true) unless specifically private/group-targeted\n"
            "• Execute Pattern: deleteItem('old_id') + createPhaseIndicator() + createTimer() + createVotingPanel()\n\n"
            
            "🎪 GAME MASTER & HOST CORE RESPONSIBILITIES (Your Sacred Duties):\n"
            "1. **Session Analysis**: Understand current session from player actions and states\n"
            "2. **Status Display**: Show game status on canvas - scores, survival, speaker, round summary\n"
            "3. **Evidence Management**: Maintain persistent evidence board with key information\n"
            "4. **Progress Hosting**: Guide players through phases with clear UI components\n"
            "5. **Discussion Materials**: Create/display topics for player engagement when needed\n\n"
            
            "📊 MANDATORY GAME STATUS INFORMATION TO DISPLAY:\n"
            "• Score/Points: Extract from player_states and display via createScoreBoard\n"
            "• Survival Status: Show who's alive/dead via createTextDisplay with death list\n"
            "• Current Speaker: Use createTurnIndicator to highlight active player\n"
            "• Round Summary: Display key events and outcomes via createResultDisplay\n"
            "• Game Progress: Show phase and what's next via createPhaseIndicator\n\n"
            
            "🎨 CANVAS DISPLAY TOOLS & UI STRATEGY:\n"
            "**PERSISTENT COMPONENTS (Keep Throughout Game):**\n"
            "• createScoreBoard: Leaderboard with scores/rankings (position: 'top-right')\n"
            "• createTextDisplay: Death list and key evidence (position: 'middle-left')\n"
            "\n"
            "**DYNAMIC COMPONENTS (Update Each Round):**\n"
            "• createTurnIndicator: Current speaker/actor (position: 'top-center')\n"
            "• createPhaseIndicator: Game phase progress (position: 'top-center')\n"
            "• createResultDisplay: Round results (position: 'center')\n"
            "• createTimer: Phase timers (position: 'top-left', duration: ~10-20 seconds)\n\n"
            
            "👥 AUDIENCE TARGETING STRATEGY:\n"
            "• **PUBLIC**: audience_type=true (all players see it) - Use for: scoreboards, phase indicators, timers\n"
            "• **PRIVATE**: audience_type=false + audience_ids=['player_id'] - Use for: role cards, secret info\n"
            "• **GROUP**: audience_type=false + audience_ids=['1','3','5'] - Use for: team-specific info\n\n"
            
            "🎭 CRITICAL ROLE ASSIGNMENT RULE (Phase 1 'Role Assignment'):\n"
            "**MANDATORY ROLE TRANSPARENCY**: When assigning roles, you MUST inform each player of their identity!\n"
            "• NEVER hide or conceal a player's role from themselves\n"
            "• Each player has their own private screen - they cannot see others' roles\n"
            "• Create individual character cards: createCharacterCard(name='Player1Role', role='Detective', audience_type=false, audience_ids=['1'])\n"
            "• Each character card is visible ONLY to its assigned player (private audience)\n"
            "• Example: Player 1 gets Detective card (only they see it), Player 2 gets Werewolf card (only they see it)\n"
            "**ROLE CARD REQUIREMENT**: Every player with a role must receive their own private character card!\n\n"
            
            "🔄 DETAILED EXECUTION WORKFLOW FOR EACH ROUND:\n"
            "1. **Parse Player Actions** - Understand what players did this round\n"
            "2. **Analyze Player States** - Deep analysis of scores, status, roles, performance\n"
            "3. **Determine Key Information** - Speaker, results, changes, evidence\n"
            "4. **Plan UI Updates** - Which components to delete/create based on phase requirements?\n"
            "5. **Execute Delete + Create** - Remove outdated UI, create new components for current phase\n"
            "6. **Maintain Evidence Board** - Keep persistent scoreboard and key information visible\n"
            "7. **Host Game Progress** - Guide players through natural game flow to next phase\n\n"
            
            "🔧 TECHNICAL EXECUTION DETAILS:\n"
            "• **itemsState Format**: '[ID] type:name@position' shows current UI layout\n"
            "• **Tool Naming**: Exact names (createPhaseIndicator, createTimer, deleteItem) - no prefixes\n"
            "• **Position Guidelines**: 'top-center' for phase info, 'top-right' for scores, 'center' for main content\n"
            "• **Timer Durations**: ~10 seconds (max 20), adjust based on phase complexity\n"
            "• **Evidence Persistence**: ScoreBoard and critical TextDisplays must remain across rounds\n\n"
            
            "💡 PLAYER STATE INTERPRETATION GUIDE:\n"
            "• **Scores**: Extract points/performance_rating for ScoreBoard display\n"
            "• **Survival**: Check is_alive status, highlight deaths prominently\n"
            "• **Roles**: Display according to visibility rules (private cards for each player)\n"
            "• **Actions**: Track recent actions via playerActions for round summaries\n"
            "• **Rankings**: Use ranking_position and scoring_history when available\n"
            "• **Evidence**: Maintain continuity of key information across rounds\n\n"
            
            "⚡ COMPLETE EXECUTION PATTERN: Deep Analysis → Strategic Planning → Delete Outdated → Create New → Display Evidence → Host Progress\n"
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
    
    # Filter out incomplete AIMessage + ToolMessage sequences
    filtered_messages = []
    i = 0
    while i < len(trimmed_messages):
        msg = trimmed_messages[i]
        
        if hasattr(msg, 'tool_calls') and getattr(msg, "tool_calls", None):
            # Collect expected tool_call_ids
            tool_calls = getattr(msg, "tool_calls", [])
            expected_ids = {tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None) for tc in tool_calls}
            expected_ids.discard(None)
            
            # Collect following ToolMessages
            tool_messages = []
            j = i + 1
            while j < len(trimmed_messages) and hasattr(trimmed_messages[j], "tool_call_id"):
                tool_messages.append(trimmed_messages[j])
                j += 1
            
            # Check if all expected tool_call_ids have responses
            received_ids = {getattr(tm, "tool_call_id", None) for tm in tool_messages}
            received_ids.discard(None)
            
            # Only keep if ALL tool_calls have responses
            if expected_ids and expected_ids == received_ids:
                filtered_messages.append(msg)
                filtered_messages.extend(tool_messages)
            
            i = j
        elif hasattr(msg, "tool_call_id"):
            # Orphaned ToolMessage, skip
            i += 1
        else:
            # Keep other messages (HumanMessage, SystemMessage)
            filtered_messages.append(msg)
            i += 1
    
    trimmed_messages = filtered_messages
    
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
    # Mark generic completion flag for current phase (used by PhaseNode as a weak signal)
    try:
        existing_completion = state.get("phase_completion", {}) or {}
        updated_completion = dict(existing_completion)
        updated_completion[str(updated_phase_id)] = True
        logger.info(f"[ActionExecutor][updated_completion] : {updated_completion}")
    except Exception:
        updated_completion = state.get("phase_completion", {}) or {}
        logger.info(f"[ActionExecutor][updated_completion] : {updated_completion}")
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
    
    # Only show chat messages when not actively in progress; always deliver frontend tool calls
    final_messages = [AIMessage(content="", tool_calls=tool_calls)] if (has_frontend_tool_calls or not currently_in_progress) else ([])
    
    return Command(
        goto="__end__",
        update={
            # Use final_messages like agent.py
            "messages": final_messages,
            "items": state.get("items", []),
            "player_states": final_player_states,  # Updated with role assignments
            "current_phase_id": updated_phase_id,
            "actions": [],  # Clear actions after execution
            "dsl": state.get("dsl", {}),  # Persist DSL
            "roomSession": state.get("roomSession", {}),  # Persist roomSession
            "phase0_ui_done": True if updated_phase_id == 0 else state.get("phase0_ui_done", True),
            "phase_completion": updated_completion,
        }
    )

# Define the workflow graph
workflow = StateGraph(AgentState)

# Add all nodes
workflow.add_node("InitialRouterNode", InitialRouterNode)
workflow.add_node("ChatBotNode", ChatBotNode)
# workflow.add_node("FeedbackDecisionNode", FeedbackDecisionNode)
workflow.add_node("BotBehaviorNode", BotBehaviorNode)
workflow.add_node("RefereeNode", RefereeNode)
workflow.add_node("PhaseNode", PhaseNode)
workflow.add_node("RoleAssignmentNode", RoleAssignmentNode)
workflow.add_node("ActionExecutor", ActionExecutor)
workflow.add_node("ActionValidatorNode", ActionValidatorNode)

# Set entry point
workflow.set_entry_point("InitialRouterNode")

# Compile the graph (LangGraph API handles persistence itself in local_dev/cloud)
graph = workflow.compile()
