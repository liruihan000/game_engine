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
    
    # 文件处理器 - 开发态按“天”合并日志（避免热重载生成多文件）
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

async def InitialRouterNode(state: AgentState, config: RunnableConfig) -> Command[Literal["FeedbackDecisionNode"]]:
    """
    Initial routing node that loads DSL and routes based on current phase.
    
    Routes to: 
    - FeedbackDecisionNode if current_phase_id > 0
    """
    # Print game name from state
    game_name = state.get("gameName", "")
    logger.info(f"[InitialRouterNode] Game name from state: {game_name}")

    
    current_phase_id = state.get('current_phase_id', 0)
    logger.info(f"[InitialRouter] Starting with phase_id: {current_phase_id}")
    
    # Define backend tools that don't require frontend interaction
    backend_tool_names = {}
    
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
    logger.info(f"[InitialRouter] Routing to FeedbackDecisionNode (phase {current_phase_id})")
    # Ensure DSL is properly passed - use dsl_content if loaded, otherwise fallback to state
    final_dsl = dsl_content if dsl_content else state.get("dsl", {})
    updates["dsl"] = final_dsl
    logger.info(f"[InitialRouter] Passing DSL with keys: {list(final_dsl.keys()) if final_dsl else 'empty'}")
    updates["messages"] = ([])
    return Command(goto="FeedbackDecisionNode", update=updates)

async def FeedbackDecisionNode(state: AgentState, config: RunnableConfig) -> Command[Literal["BotBehaviorNode", "PhaseNode"]]:
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
    logger.info(f"[FeedbackDecisionNode] Game name from state: {game_name}")
    
    logger.info("[FeedbackDecisionNode] Starting feedback decision analysis")
    
    # Extract inputs for logging
    trimmed_messages = state.get("messages", [])[-10:]
    player_states = state.get("player_states", {})
    current_phase_id = state.get("current_phase_id", 0)
    dsl_content = state.get("dsl", {})
    
    # Debug DSL content
    logger.info(f"[FeedbackDecisionNode] DSL content keys: {list(dsl_content.keys()) if dsl_content else 'empty'}")
    if dsl_content and 'phases' in dsl_content:
        logger.info(f"[FeedbackDecisionNode] Phases available: {list(dsl_content['phases'].keys())}")
    else:
        logger.info("[FeedbackDecisionNode] No phases found in DSL content")
    
    # Get current phase details
    phases = dsl_content.get('phases', {}) if dsl_content else {}
    # Try both int and string keys to handle YAML parsing variations
    current_phase = phases.get(current_phase_id, {}) or phases.get(str(current_phase_id), {})
    declaration = dsl_content.get('declaration', {}) if dsl_content else {}
    
    # player_states should have been initialized by InitialRouterNode
    if not player_states:
        logger.warning("[FeedbackDecisionNode] player_states is empty! InitialRouterNode should have initialized it.")
    else:
        logger.info(f"[FeedbackDecisionNode] Using player_states: {len(player_states)} players")
    
    # Log phase info
    logger.info(f"[FeedbackDecisionNode] current_phase_id: {current_phase_id}")
    logger.info(f"[FeedbackDecisionNode] current_phase: {current_phase}")
    logger.info(f"[FeedbackDecisionNode] player_states: {player_states}")
    
    # Initialize LLM
    model = init_chat_model("openai:gpt-4o")
    
    # Create system message with all inputs
    system_message = SystemMessage(
        content=(
            "FEEDBACK DECISION ANALYSIS\n"
            f"Current Phase ID: {current_phase_id}\n"
            f"Current Phase Details: {current_phase}\n"
            f"Game Declaration: {declaration}\n"
            f"Player States: {player_states}\n"
            f"Recent Messages: {[str(msg) for msg in trimmed_messages]}\n\n"
            
            "TASK: Analyze the current phase and determine which players need to provide feedback.\n"
            "Based on the phase completion criteria, player states, and message history:\n"
            "1. Identify which players are required to respond\n"
            "2. Check who has already responded in recent message based on message history for this phase.\n"
            "3. Generate appropriate feedback message\n"
            "4. The ouput is need_feed_back_dict, which is a dictionary with the following keys:\n"
            "  - player_id_list: a list of player ids that need to provide feedback\n"
            "  - need_feedback_message: a message to the player to provide feedback\n"
            
            "IMPORTANT - When NO feedback is needed:\n"
            "- Phase completion is based on TIME (e.g., 'wait 30 seconds', 'timer expires')\n"
            "- Phase completion is based on UI DISPLAY only (e.g., 'show results', 'display information')\n"
            "- Phase is AUTOMATIC resolution (e.g., 'system calculates', 'auto-proceed')\n"
            "- Phase does NOT explicitly require player input or choices\n"
            "- Phase is purely INFORMATIONAL or DISPLAY-focused\n"
            "- All required players have already provided their responses\n"
            "If ANY of these conditions apply, return empty player_id_list [].\n\n"

            
            "OUTPUT FORMAT (JSON only)\n"
            "Example 1 - Voting phase:\n"
            "{\n"
            '  "player_id_list": [1, 2, 4, 5, 7],\n'
            '  "need_feedback_message": "Please cast your vote for elimination"\n'
            "}\n\n"
            "Example 2 - Night action phase:\n"
            "{\n"
            '  "player_id_list": [3, 6],\n'
            '  "need_feedback_message": "Werewolves, choose your target for tonight"\n'
            "}\n\n"
            "Example 3 - Detective investigation:\n"
            "{\n"
            '  "player_id_list": [2],\n'
            '  "need_feedback_message": "Detective, choose a player to investigate"\n'
            "}\n\n"
            "Example 4 - No feedback needed (phase completed):\n"
            "{\n"
            '  "player_id_list": [],\n'
            '  "need_feedback_message": "All actions completed, proceeding to next phase"\n'
            "}\n\n"
            "Example 5 - No feedback needed (automatic resolution):\n"
            "{\n"
            '  "player_id_list": [],\n'
            '  "need_feedback_message": "Phase resolves automatically based on previous actions"\n'
            "}\n\n"
            
            "RULES:\n"
            "- Only include players who still need to respond\n"
            "- Use numeric player IDs (1, 2, 3, etc.)\n"
            "- Return empty list [] if no feedback is needed\n"
            "- CRITICAL: Return [] for phases that are:\n"
            "  * Time-based completion (timers, delays, automatic progression)\n"
            "  * Display-only phases (showing results, information, announcements)\n"
            "  * System-resolved phases (automatic calculations, rule applications)\n"
            "  * Already completed (all required responses received)\n"
            "  * Purely informational or transitional\n"
            "- Only return player IDs when explicit player choices/actions are required\n"
            "- Create appropriate feedback message for the phase context\n"
            "- Return valid JSON format only"
        )
    )

    backend_tool_names = {}
    
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
    
    # Call LLM
    response = await model.ainvoke([system_message], config)
    
    # Parse LLM response
    try:
        response_content = clean_llm_json_response(str(response.content))
        need_feed_back_dict = json.loads(response_content)
        logger.info(f"[FeedbackDecisionNode] LLM generated: {need_feed_back_dict}")
    except Exception as e:
        logger.error(f"[FeedbackDecisionNode] Failed to parse LLM response: {e}")
        # Fallback hardcoded response
        need_feed_back_dict = {
            "player_id_list": [1, 2, 4, 5, 7],
            "need_feedback_message": "ask feedback"
        }
    
    # Save need_feed_back_dict as state
    player_id_list = need_feed_back_dict.get("player_id_list", [])
    need_feedback_message = need_feed_back_dict.get("need_feedback_message", "")
    
    # Prepare common updates for all routes
    common_updates = {
        "need_feed_back_dict": need_feed_back_dict,
        "player_states": player_states,
        "roomSession": state.get("roomSession", {})
    }
    
    # Route based on feedback requirements
    if len(player_id_list) == 0:
        logger.info("[FeedbackDecisionNode] No players need feedback - routing to PhaseNode")
        return Command(goto="PhaseNode", update={**common_updates, "dsl": state.get("dsl", {}), "messages": ([])})    
    else:
        logger.info("[FeedbackDecisionNode] Players need feedback - routing to BotBehaviorNode")
        # Note: Player 1 feedback will be handled by ActionExecutor UI creation
        return Command(goto="BotBehaviorNode", update={**common_updates, "dsl": state.get("dsl", {}),"messages": ([])})


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
    
    # Extract inputs
    trimmed_messages = state.get("messages", [])[-10:]
    player_states = state.get("player_states", {})
    current_phase_id = state.get("current_phase_id", 0)
    need_feed_back_dict = state.get("need_feed_back_dict", {})
    dsl_content = state.get("dsl", {})
    
    # Get current phase details
    phases = dsl_content.get('phases', {}) if dsl_content else {}
    # Try both int and string keys to handle YAML parsing variations
    current_phase = phases.get(current_phase_id, {}) or phases.get(str(current_phase_id), {})
    declaration = dsl_content.get('declaration', {}) if dsl_content else {}
    
    # Log phase info
    logger.info(f"[BotBehaviorNode] current_phase_id: {current_phase_id}")
    logger.info(f"[BotBehaviorNode] current_phase: {current_phase}")
    logger.info(f"[BotBehaviorNode] player_states: {player_states}")
    
    # Initialize LLM
    model = init_chat_model("openai:gpt-4o")
    
    # Create system message with all inputs
    system_message = SystemMessage(
        content=(
            "BOT BEHAVIOR ANALYSIS\n"
            f"Current Phase ID: {current_phase_id}\n"
            f"Current Phase Details: {current_phase}\n"
            f"Game Declaration: {declaration}\n"
            f"Player States: {player_states}\n"
            f"Recent Messages: {[str(msg) for msg in trimmed_messages]}\n"
            f"Need Feedback Dict: {need_feed_back_dict}\n\n"
            
            "TASK: Generate bot behaviorss for all non-human players without player_id:1 (the human) who need to provide feedback.\n"
            "All of players without player_id:1 are bots.\n"
            "Based on the feedback requirements, analyze each bot player and determine their likely actions:\n"
            "1. Consider each bot's role, current state, and game context\n"
            "2. Generate realistic behavior that fits their character/role\n"
            "3. Ensure behaviors are appropriate for the current phase\n\n"
            "The ouput is botbehavior, which is a dictionary with the following keys:\n"
            "  - player_id: a list of player ids that need to provide feedback\n"
            "  - possible_behavior: a message to the player to provide feedback\n"
            
            "OUTPUT FORMAT (JSON only) - botbehavior:\n"
            "{\n"
            '  "player_id": "possible_behavior",\n'
            '  "player_id": "possible_behavior"\n'
            "}\n"
            "Example 1 - Voting phase behaviors:\n"
            "{\n"
            '  "2": "vote for player 3",\n'
            '  "4": "vote for player 1",\n'
            '  "5": "vote for player 7",\n'
            '  "7": "vote for player 2"\n'
            "}\n\n"
            "Example 2 - Night action behaviors:\n"
            "{\n"
            '  "3": "target player 1 for elimination",\n'
            '  "6": "agree with player 3 target choice"\n'
            "}\n\n"
            "Example 3 - Detective investigation:\n"
            "{\n"
            '  "2": "investigate player 5"\n'
            "}\n\n"
            
            "RULES:\n"
            "- Only generate behaviors for bot players (exclude player 1 - the human)\n"
            "- Use player IDs as string keys (\"2\", \"3\", \"4\", etc.)\n"
            "- Behaviors should be specific actions that fulfill the feedback requirement\n"
            "- Consider player roles, alliances, and survival instincts\n"
            "- Return valid JSON format only\n"
            "- If no bots need to act, return empty object {}"
        )
    )
    
    backend_tool_names = {}
    
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


    # Call LLM
    response = await model.ainvoke([system_message], config)
    
    # Parse LLM response
    try:
        response_content = clean_llm_json_response(str(response.content))
        botbehavior = json.loads(response_content)
        logger.info(f"[BotBehaviorNode] LLM generated botbehavior: {botbehavior}")
    except Exception as e:
        logger.error(f"[BotBehaviorNode] Failed to parse LLM response: {e}")
        # Fallback empty behavior
        botbehavior = {}
    
    # Save botbehavior as state and route to RefereeNode
    logger.info("[BotBehaviorNode] Routing to RefereeNode")
    return Command(
        goto="RefereeNode",
        update={
            "botbehavior": botbehavior,
            "player_states": state.get("player_states", {}),
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
    trimmed_messages = state.get("messages", [])[-10:]
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

    
    # Initialize LLM
    model = init_chat_model("openai:gpt-4o")
    
    # Create system message with all inputs
    system_message = SystemMessage(
        content=(
            "REFEREE ANALYSIS AND STATE UPDATES\n"

            f"Current Phase ID: {current_phase_id}\n"
            f"Current Phase Details: {current_phase}\n"
            f"Game Declaration: {declaration}\n"
            f"Game Phases: {phases}\n"
            f"Current Player States: {player_states}\n"
            f"Last Human Message: {last_human_message}\n"
            f"Bot Behaviors: {botbehavior}\n"
            f"Recent Messages: {[str(msg) for msg in trimmed_messages]}\n\n"
            f"player_states: {player_states}\n"
            
            "TASK: Process all player behaviors including player_id:1 (the human) and the other bot players who need to update the **player states** accordingly (votes, eliminations, role actions, etc.) based on the bot behaviors and the human message and the game rules.\n"
            "Update player states accordingly (votes, eliminations, role actions, etc.) based on the bot behaviors and the human message and the game rules of the Game Declaration and the Game Phases.\n"
            "Ensure all updates are consistent with game mechanics\n\n"

            "ROLE ASSIGNMENT (only if this phase assigns roles): update player_states.role.\n"
            "- Treat the phase as a role-assignment phase if ANY is true:\n"
            "  * Phase name/description contains 'assign' or 'role' (case-insensitive)\n"
            "  * declaration.roles exists AND at least one player has an empty role\n"
            "- When assigning roles:\n"
            "  * Do NOT overwrite non-empty roles (idempotent)\n"
            "  * Distribute special roles at most once each if counts are not specified\n"
            "  * Fill remaining with 'Villager' or the first available role\n"
            "  * Use a deterministic ordering of players to avoid non-determinism across retries\n"
           
           "- The ouput is updated_player_states, which is a updated version of player_states, keep the same keys and values for each player, don't change the keys and values for each player, only update the values that have changed.\n"
            
            
            "OUTPUT FORMAT (JSON only):\n"
            "{\n"
            '  "updated_player_states": { ... },\n'
            '  "conclusions": ["Player 3 was eliminated by majority vote", "No deaths this round"]\n'
            "}\n\n"
            "Example 1 - After voting phase with elimination:\n"
            "{\n"
            '  "updated_player_states": {\n'
            '    "1": {"name": "Alpha", "role": "Werewolf", "is_alive": true, "vote": "3"},\n'
            '    "2": {"name": "Beta", "role": "Detective", "is_alive": true, "vote": "3"},\n'
            '    "3": {"name": "Gamma", "role": "Villager", "is_alive": false, "vote": ""},\n'
            '    "4": {"name": "Delta", "role": "Villager", "is_alive": true, "vote": "3"}\n'
            '  },\n'
            '  "conclusions": ["Player 3 (Gamma) was eliminated by majority vote", "Villagers eliminated a fellow villager"]\n'
            "}\n\n"
            "Example 2 - After night actions with death:\n"
            "{\n"
            '  "updated_player_states": {\n'
            '    "1": {"name": "Alpha", "role": "Werewolf", "is_alive": true, "target": "2"},\n'
            '    "2": {"name": "Beta", "role": "Detective", "is_alive": false, "investigated": "1"},\n'
            '    "3": {"name": "Gamma", "role": "Villager", "is_alive": true}\n'
            '  },\n'
            '  "conclusions": ["Player 2 (Beta) was killed by werewolves", "Detective discovered Alpha is a werewolf before dying"]\n'
            "}\n\n"
            "Example 3 - No major events:\n"
            "{\n"
            '  "updated_player_states": {\n'
            '    "1": {"name": "Alpha", "role": "Werewolf", "is_alive": true, "target": ""},\n'
            '    "2": {"name": "Beta", "role": "Detective", "is_alive": true, "investigated": ""},\n'
            '    "3": {"name": "Gamma", "role": "Villager", "is_alive": true}\n'
            '  },\n'
            '  "conclusions": ["No deaths this round", "All players remain in the game"]\n'
            "}\n\n"
            
            "RULES:\n"
            "- Include ALL players in the updated_player_states\n"
            "- Only modify fields that actually changed due to player actions\n"
            "- Maintain consistency with game rules and phase requirements\n"
            "- Handle eliminations, votes, role abilities, and status changes\n"
            "- Use player IDs as string keys (\"1\", \"2\", \"3\", etc.)\n"
            "- Return complete player state objects for each player\n"
            "- Include conclusions array with key events that happened:\n"
            "  * Player eliminations/deaths (who died and how)\n"
            "  * Game outcomes (who won/lost)\n"
            "  * Important discoveries or revelations\n"
            "  * No events (if nothing significant happened)\n"
            "- Conclusions should be clear narrative statements for display to players\n"
            "- Return valid JSON format only"
        )
    )
    

    backend_tool_names = {}
    
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
    
    # Call LLM
    response = await model.ainvoke([system_message], config)
    
    # Parse LLM response
    try:
        response_content = clean_llm_json_response(str(response.content))
        referee_result = json.loads(response_content)
        updated_player_states = referee_result.get("updated_player_states", player_states)
        conclusions = referee_result.get("conclusions", [])
        logger.info(f"[RefereeNode] player_states (output): {updated_player_states}")
        logger.info(f"[RefereeNode] LLM generated conclusions: {conclusions}")
    except Exception as e:
        logger.error(f"[RefereeNode] Failed to parse LLM response: {e}")
        # Fallback to current player states
        updated_player_states = player_states
        conclusions = []
    
    # Route to PhaseNode with updated player states and conclusions
    logger.info("[RefereeNode] Routing to PhaseNode with updated player states and conclusions")
    return Command(
        goto="PhaseNode",
        update={
            "player_states": updated_player_states,
            "referee_conclusions": conclusions,
            "roomSession": state.get("roomSession", {}),
            "dsl": state.get("dsl", {}),
            "messages": ([]),
        }
    )

async def PhaseNode(state: AgentState, config: RunnableConfig) -> Command[Literal["ActionExecutor"]]:
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
    
    # Initialize LLM
    model = init_chat_model("openai:gpt-4o")
    trimmed_messages = state.get("messages", [])[-10:]
    # Create system message with all inputs
    system_message = SystemMessage(
        content=(
            "PHASE TRANSITION ANALYSIS\n"
            f"itemsState (current frontend layout):\n{items_summary}\n"
            f"DSL Content: {dsl_content}\n"
            f"Current Phase ID: {current_phase_id}\n"
            f"Current Phase Details: {current_phase}\n"
            f"Game Declaration: {declaration}\n"
            f"Player States: {player_states}\n"
            f"Bot Behaviors: {botbehavior}\n\n"
            f"Phase Completion Flags: {state.get('phase_completion', {})}\n"
            f"Recent Messages: {[str(msg) for msg in trimmed_messages]}\n\n"
            
            "TASK: Analyze the current phase's next_phase conditions and determine which branch to follow based on game state.\n"
            "Your mechanism is to drive game progression forward by carefully evaluating next_phase rules.\n\n"
            
            "NEXT_PHASE CONDITION ANALYSIS:\n"
            "1. Examine the current_phase's next_phase field for conditional branches\n"
            "2. Evaluate each condition against current player_states and game context\n"
            "3. Select the branch matching condition\n"
            "4. Return the corresponding phase_id from the matching branch\n\n"
            
            "CONDITION EVALUATION EXAMPLES:\n"
            "- Player count conditions: Count alive/dead players by checking player_states[player_id].alive\n"
            "- Role-based conditions: Check player_states[player_id].role values\n"
            "- Turn/round conditions: Check player_states[player_id].speaker_turns_taken or similar counters\n"
            "- Game completion: Evaluate win/loss conditions based on role distributions\n\n"
            
            "IMPORTANT: The 'itemsState' shows what UI elements are currently displayed to players.\n"
            "Items represent the actual frontend components visible on screen (buttons, voting panels, text displays, etc.)\n\n"
            
            "EVALUATION STEPS:\n"
            "1. Check current_phase's conditions (wait_for, completion, etc.)\n"
            "2. If current phase is complete, analyze next_phase conditions\n"
            "3. Match conditions against player_states data\n"
            "4. Select appropriate next_phase_id\n"
            
            "OUTPUT FORMAT (JSON only):\n"
            "Complete: {\"transition\": true, \"next_phase_id\": X, \"note\": \"criteria met\"}\n"
            "Incomplete: {\"transition\": false, \"note\": \"what's missing\"}\n\n"
            
            "SPECIFIC GAME EXAMPLES:\n\n"
            
            "Two Truths and a Lie - Phase 6 Analysis:\n"
            "DSL condition: 'If all players have speaker_turns_taken >= the agreed number of rounds (default 1)'\n"
            "Analysis: Check player_states for each player's speaker_turns_taken value\n"
            "All players completed turns: {\"transition\": true, \"next_phase_id\": 7, \"note\": \"All players have completed their turns\"}\n"
            "Some players still need turns: {\"transition\": true, \"next_phase_id\": 1, \"note\": \"Player 3 still needs to speak\"}\n\n"
            
            "Werewolf Game - End Condition Analysis:\n"
            "DSL condition: 'If no living Werewolves remain' → phase 99 (Village Wins)\n"
            "Analysis: Count players with role='Werewolf' and alive=true\n"
            "No werewolves alive: {\"transition\": true, \"next_phase_id\": 99, \"note\": \"Village wins - no werewolves remain\"}\n\n"
            
            "DSL condition: 'If living Werewolves >= living non-Werewolves' → phase 99 (Werewolves Win)\n"
            "Analysis: Count alive werewolves vs alive non-werewolves\n"
            "Werewolves equal/outnumber villagers: {\"transition\": true, \"next_phase_id\": 99, \"note\": \"Werewolves win\"}\n\n"
            
            "DSL condition: 'Otherwise, continue to next Night cycle' → phase 10\n"
            "Game continues: {\"transition\": true, \"next_phase_id\": 10, \"note\": \"Game continues to night phase\"}\n\n"
            
            "GENERAL EXAMPLES:\n"
            "All votes received:\n"
            "{\"transition\": true, \"next_phase_id\": 3, \"note\": \"Voting complete, proceeding to results\"}\n\n"
            "Missing player actions:\n"
            "{\"transition\": false, \"note\": \"Waiting for players 2, 4 to vote\"}\n\n"
            "Timer/auto phases ready:\n"
            "{\"transition\": true, \"next_phase_id\": 5, \"note\": \"Display period complete\"}\n\n"
            "Game over condition:\n"
            "{\"transition\": false, \"note\": \"Final results displayed\"}"

        )
    )
    
    # Call LLM
    response = await model.ainvoke([system_message], config)
    logger.info(f"[PhaseNode][output] LLM response: {response}")
    parsed_transition = False
    proposed_next_phase_id = state.get("current_phase_id", 0)
    # Parse LLM response
    try:
        response_content = clean_llm_json_response(str(response.content))
        logger.info(f"[PhaseNode] Cleaned response content: {response_content}")
        phase_decision = json.loads(response_content)
        parsed_transition = bool(phase_decision.get("transition", False))
        proposed_next_phase_id = phase_decision.get("next_phase_id", current_phase_id)
        logger.info(f"[PhaseNode] LLM decision: transition={parsed_transition}, next_phase_id={proposed_next_phase_id}")
        logger.info(f"[PhaseNode] Note: {phase_decision.get('note') or phase_decision.get('reason') or 'No note provided'}")
    except Exception as e:
        logger.error(f"[PhaseNode] Failed to parse LLM response: {e}")
        parsed_transition = False
        proposed_next_phase_id = current_phase_id
        logger.warning(f"[PhaseNode] Using fallback: staying at phase_id {proposed_next_phase_id}")

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
    return Command(
        goto="ActionExecutor",
        update={
            "current_phase_id": target_phase_id,
            "player_states": state.get("player_states", {}),
            "roomSession": state.get("roomSession", {}),
            "dsl": state.get("dsl", {}),
            "messages": ([]),
        }
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
    # Component management tools
    "deleteItem",
    "clearCanvas",
    # Player state management
    "markPlayerDead"
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

def summarize_items_for_prompt(state: AgentState) -> str:
    """Summarize current UI items with type/name/position to help phase decisions."""
    try:
        items = state.get("items", []) or []
        if not items:
            return "(no items)"
        # Build compact summary for up to 10 items
        parts: List[str] = []
        for it in items[:10]:
            try:
                t = it.get("type")
                n = it.get("name")
                d = it.get("data", {}) or {}
                p = d.get("position") or it.get("position")
                parts.append(f"{t}:{n}@{p}")
            except Exception:
                continue
        more = "" if len(items) <= 10 else f" (+{len(items)-10} more)"
        return f"{len(items)} item(s): " + ", ".join(parts) + more
    except Exception:
        return "(unable to summarize items)"

async def ActionExecutor(state: AgentState, config: RunnableConfig) -> Command[Literal["__end__"]]:
    """
    Execute actions from DSL and current phase by calling frontend tools.
    IMPORTANT: This node is responsible for Player 1 (human player) experience only.
    Focus on what Player 1 should see, hear, and interact with.
    Can make announcements based on RefereeNode conclusions.
    """
    # Print game name from state
    game_name = state.get("gameName", "")
    logger.info(f"[ActionExecutor] Game name from state: {game_name}")
    
    logger.info(f"[ActionExecutor][start] ==== start ActionExecutor ====")
    
    # Extract phase info for logging
    current_phase_id = state.get("current_phase_id", 0)
    dsl_content = state.get("dsl", {})
    phases = dsl_content.get('phases', {})
    current_phase = phases.get(current_phase_id, {})
    
    # Log phase info
    logger.info(f"[ActionExecutor] current_phase_id: {current_phase_id}")
    logger.info(f"[ActionExecutor] current_phase: {current_phase}")
    logger.info(f"[ActionExecutor] player_states: {state.get('player_states', {})}")
    
    # Debug: Print entire received state
    logger.info(f"[ActionExecutor][DEBUG] Full received state keys: {list(state.keys())}")
    if "player_states" in state:
        logger.info(f"[ActionExecutor][DEBUG] Received player_states: {state.get('player_states', {})}")
    
    # 1. Define the model
    model = init_chat_model("openai:gpt-4o")

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
    player_states = state.get("player_states", {})
    referee_conclusions = state.get("referee_conclusions", [])
    
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
            f"itemsState (current frontend layout):\n{items_summary}\n"
            f"player_states: {player_states}\n"
            f"referee_conclusions: {referee_conclusions}\n"
            f"{current_phase_str}\n"
            "GAME DSL REFERENCE (for understanding game flow):\n"
            f"{dsl_info}\n"
            "ACTION EXECUTOR INSTRUCTION:\n"
            f"- You have the following actions to execute: {actions_to_execute}\n"
            "- You MUST execute ALL tools listed in the current phase's actions, in this turn. Also consider any extra actions that improve UX (e.g., timers, UI cleanup) and include them.\n"
            "- **CRITICAL**: You MUST call ALL required tools in THIS SINGLE response. Do NOT return until you have made ALL tool calls.\n"
            "- **UI CLEANUP FIRST**: In 90% of cases, start by cleaning up previous UI elements before creating new ones\n"
            "- Use 'clearCanvas' to remove all items, OR use individual 'deleteItem' calls for specific items\n"
            "- For each action in the list, call its specified tools. If an action has tools: [A, B], you must call both A and B.\n"
            "- Make MULTIPLE tool calls in this single response to complete all actions at once.\n"
            "- Use referee_conclusions for announcements when creating text displays (e.g., createTextDisplay with conclusions)\n"
            "- After making ALL tool calls, include a brief, friendly message about what phase/stage is ready.\n"
            "- If no explicit actions provided, generate appropriate UI for the current phase based on DSL.\n"
            
            "ROLE ASSIGNMENT SPECIAL RULES:\n"
            "- If role assignments were generated above, update player_states to include the assigned roles\n"
            
            "PHASE INDICATOR REQUIREMENT:\n"
            "- **EVERY phase MUST update the phase indicator** using createPhaseIndicator\n"
            "- Show current phase name and description to keep players informed\n"
            "- Place phase indicators at 'top-center' position\n"
            "- Include phase-specific information (time remaining, phase objectives, etc.)\n"
            
            "TOOL USAGE RULES:\n"
            "- Tool names must match exactly (no 'functions.' prefix).\n"
            "- When you create items (e.g., via `createTextDisplay`), capture the returned id and reuse it in later calls that require 'itemId'.\n"
            "- To delete multiple items, call 'deleteItem' once per 'itemId'.\n"
            
            "LAYOUT PLACEMENT RULES:\n"
            "- Prefer placing text tools (createTextDisplay, createResultDisplay) at grid position 'center' by default.\n"
            "- If there are already 4 or more items on canvas, distribute subsequent items across: 'top-center', 'middle-left', 'middle-right','bottom-center'.\n"
            "- Keep related elements sensible (e.g., phase indicator near top-center; voting panels center/bottom-center).\n"
            f"- Total tools to call this turn: {sum(len(action.get('tools', [])) for action in actions_to_execute if isinstance(action, dict))}\n"
        )
    )


    backend_tool_names = {}
    
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

    # Log output
    try:
        content_preview = getattr(response, "content", None)
        if isinstance(content_preview, str):
            logger.info(f"[ActionExecutor][OUTPUT] content: {content_preview[:400]}")
        else:
            logger.info(f"[ActionExecutor][OUTPUT] content: (non-text)")
        tool_calls = getattr(response, "tool_calls", []) or []
        if tool_calls:
            for tc in tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                logger.info(f"[ActionExecutor][TOOL_CALL] tool_call name={name} args={args}")
        else:
            logger.info("[ActionExecutor][TOOL_CALL] tool_calls: none")
    except Exception:
        pass

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
    return Command(
        goto=END,
        update={
            # Only deliver assistant messages that contain tool_calls; avoid self-trigger loops
            # 只傳遞工具調用，不傳遞LLM的推理內容
            "messages": [AIMessage(content="你好", tool_calls=getattr(response, "tool_calls", []))] if getattr(response, "tool_calls", None) else [],
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
workflow.add_node("FeedbackDecisionNode", FeedbackDecisionNode)
workflow.add_node("BotBehaviorNode", BotBehaviorNode)
workflow.add_node("RefereeNode", RefereeNode)
workflow.add_node("PhaseNode", PhaseNode)
workflow.add_node("ActionExecutor", ActionExecutor)

# Set entry point
workflow.set_entry_point("InitialRouterNode")

# Compile the graph (LangGraph API handles persistence itself in local_dev/cloud)
graph = workflow.compile()