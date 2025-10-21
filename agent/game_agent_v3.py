"""
DM Agent with Bot - Simple routing node implementation
This creates an initial routing node that directs flow based on current phase.
"""

import logging
import os  
import yaml
from dotenv import load_dotenv
from typing import Literal, List, Dict, Any, Optional
from prompt.prompt_loader import _load_prompt_async
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
from tools.backend_tools import (
    set_next_phase,
    update_player_state,
    add_game_note,
    update_player_actions,
    update_player_name,
    set_feedback_decision,
    _execute_update_player_state,
    _execute_update_player_actions,
    _execute_add_game_note,
    _execute_update_player_name,
    _execute_set_feedback_decision,
)
from tools.utils import (
    get_phase_info_from_dsl,
    clean_llm_json_response,
    summarize_items_for_prompt,
    process_human_action_if_needed,
    filter_incomplete_message_sequences,
    _limit_actions_per_player,
    load_dsl_by_gamename,
    initialize_player_states_from_dsl,
)
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


# Centralized backend tools (shared across nodes)
backend_tools = [
    update_player_state,
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
    "promptUserText",
    # Card game UI
    "createHandsCard",
    "setHandsCardAudience",
    "createHandsCardForPlayer",
    # Text input panel tool - for user input collection and broadcast
    "createTextInputPanel",
    # Scoreboard tools
    "createScoreBoard",
    "setScoreBoardEntries",
    "upsertScoreEntry",
    "removeScoreEntry",
    # Chat-driven vote
    "submitVote",
    # Coins UI tools
    "createCoinDisplay",
    "incrementCoinCount",
    "setCoinAudience",
    # Statement board & Reaction timer
    "createStatementBoard",
    "createReactionTimer",
    "startReactionTimer",
    "stopReactionTimer",
    "resetReactionTimer",
    # Night overlay & Turn indicator
    "createTurnIndicator",
    # Health & Influence
    "createHealthDisplay",
    "createInfluenceSet",
    "revealInfluenceCard",
    # Component management tools
    "clearCanvas",
    # Player state management
    "markPlayerDead",
    # Chat tools
    "addBotChatMessage"
])


BACKEND_TOOL_NAMES = {t.name for t in backend_tools}


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
    updated_player_actions = process_human_action_if_needed(
        messages,
        state.get("playerActions", {}),
        state.get("playerStates", {}), 
        state.get("currentPhaseId", 0),
        state.get("roomSession", {}),
        dsl_content
    )
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
    
    
    # Bind tools to model
    model_with_tools = model.bind_tools(chat_tools)
    game_notes = state.get('game_notes', [])

    chat_system_prompt = await _load_prompt_async("chat_system_prompt")


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
    {chat_system_prompt}
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

async def BotBehaviorNode(state: AgentState, config: RunnableConfig) -> Command[Literal["ActionExecutor"]]:
    """
    BotBehaviorNode analyzes bot behavior and generates responses for non-human players.
    
    Input:
    - trimmed_messages: Recent message history
    - player_states: Current player states
    - current_phase and declaration: Phase configuration
    - need_feed_back_dict: Required feedback info
    
    Output:
    """

    player_states = state.get("player_states", {})
    current_phase_id = state.get("current_phase_id", 0)
    # Remove need_feed_back_dict dependency - use autonomous analysis only
    dsl_content = state.get("dsl", {})
    
    # Get current phase details and NEXT phase for pre-analysis
    phases = dsl_content.get('phases', {}) if dsl_content else {}
    # Try both int and string keys to handle YAML parsing variations
    current_phase = phases.get(current_phase_id, {}) or phases.get(str(current_phase_id), {})
    declaration = dsl_content.get('declaration', {}) if dsl_content else {}
    playerActions = state.get("playerActions", {})
    game_notes = state.get('game_notes', [])

    # Initialize LLM
    model = init_chat_model("openai:gpt-4.1-mini")
    model_with_tools = model.bind_tools([update_player_actions])
    items_summary = summarize_items_for_prompt(state)
    bot_behavior_system_prompt = await _load_prompt_async("bot_behavior_system_prompt")
    
    # System message with precise analysis based on FeedbackDecisionNode logic
    system_message = SystemMessage(
        content=(
            "🤖 **BOT BEHAVIOR GENERATION - CURRENT PHASE FOCUS**\n\n"
            f"📊 **CURRENT GAME STATE**:\n\n"
            f"- **Current Phase ({current_phase_id})**: {current_phase}\n\n"
            f"- **Player States**: {player_states}\n"
            f"- **Player Actions**: {_limit_actions_per_player(playerActions, 1) if playerActions else {}}\n\n"
            f"- **Game Notes**: {game_notes[-5:] if game_notes else 'None'}\n\n"
            f"- **Items State**: {items_summary}\n\n"
            f"- **Declaration**: {declaration}\n\n"
            f"- **Bot Behavior System Prompt**: {bot_behavior_system_prompt}\n\n"
          
        )
    )

    # Call LLM with backend tool bound
    response = await model_with_tools.ainvoke([system_message], config)
    
    # Apply backend tool effects inline (no ToolMessage)
    tool_calls = getattr(response, "tool_calls", []) or []
    current_player_states = dict(state.get("player_states", {}))
    current_player_actions = dict(state.get("playerActions", {}))

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
                    current_player_actions, pid, actions, phase, state.get("roomSession", {}), current_player_states
                )
    
    # Route to RefereeNode
    logger.info("[BotBehaviorNode] Routing to RefereeNode")
    return Command(
        goto="ActionExecutor",
        update={
            "player_states": current_player_states,
            "playerActions": current_player_actions,
            "roomSession": state.get("roomSession", {}),
            "dsl": state.get("dsl", {})
        }
    )



async def ActionExecutor(state: AgentState, config: RunnableConfig) -> Command[Literal["__end__"]]:
    """
    Execute actions from DSL and current phase by calling frontend tools.
    Audience-aware rendering: always choose explicit audience permissions per component
    and render public, group, and individual UIs according to the DSL phase design.
    Can make announcements based on RefereeNode conclusions.
    """
    
    # 1. Define the model
    model = init_chat_model("anthropic:claude-sonnet-4-5-20250929")

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

    # Add all backend tools to ActionExecutor
    all_tools = [*deduped_frontend_tools, *backend_tools]
    
    model_with_tools = model.bind_tools(
        all_tools,
        parallel_tool_calls=True,  # Allow multiple tool calls in single response
    )

    # 3. Prepare system states with current state and actions to execute
    items_summary = summarize_items_for_prompt(state)
    current_phase_id = state.get("current_phase_id", 0)
    dsl_content = state.get("dsl", {})
    declaration = dsl_content.get('declaration', {}) if dsl_content else {}
    player_states = state.get("player_states", {})
    playerActions = state.get("playerActions", {})
    phases = dsl_content.get('phases', {}) if dsl_content else {}
    current_phase = phases.get(current_phase_id, {}) or phases.get(str(current_phase_id), {})
    game_notes = state.get('game_notes', [])
    if current_phase:
        current_phase_str = f"Current phase (ID {current_phase_id}):\n{current_phase}\n"
    else:
        logger.info(f"[ActionExecutor][DSL] Current phase ID: {current_phase_id} (not found in DSL phases)")
        current_phase_str = f"Current phase ID: {current_phase_id} (not found in DSL phases)\n"


    system_message = SystemMessage(
        content=(
            "🎮 **YOU ARE THE DM (DUNGEON MASTER)**\n"
            "You have complete control over all players' game screens. Players can only see what you show them.\n\n"

            "📺 **SCREEN CONTROL SYSTEM**:\n"
            f"- items: Current screen components {items_summary}\n"
            "- audience_type=true: Everyone can see it\n"
            "- audience_type=false + audience_ids=['1','2']: Only specified players can see it\n"
            f"- DSL: Game rules and flow definition\n\n"

            "📊 **CURRENT GAME STATE**:\n"
            f"- Current Phase: {current_phase_str}\n"
            f"- Player States: {player_states}\n"
            f"- Player Actions: {playerActions}\n"
            f"- Phase History: {state.get('phase_history', [])}\n"
            f"- Game Description: {declaration.get('description', 'No description available')}\n\n"

            "🔄 **DM EXECUTION WORKFLOW** (Execute in order):\n\n"

            "**STEP 1: ANALYZE CURRENT PHASE**\n"
            f"• Current phase_id: {current_phase_id}\n"
            f"• Phase name: {current_phase.get('name', 'Unknown')}\n"
            f"• Phase description: {current_phase.get('description', 'No description')}\n"
            f"• Completion criteria: {current_phase.get('completion_criteria', {})}\n"
            f"• Required actions: {current_phase.get('actions', [])}\n\n"

            "**STEP 2: ANALYZE PLAYER BEHAVIOR - STRICT ANALYSIS**\n"
            "🚨 **CRITICAL**: Only use ACTUAL data from playerActions - NEVER fabricate!\n"
            "• Find each player's latest action: highest timestamp + matching current phase name\n"
            "• Extract exact choices, targets, statements from actual playerActions content\n"
            "• Process example: playerActions shows 'Player 2 voted for statement 1' → vote_choice=1\n"
            "• Cross-reference action timestamp and phase name before processing\n"
            "• FORBIDDEN: Using example values not in playerActions\n"
            "• FORBIDDEN: Inventing vote results not present in playerActions\n\n"

            "**STEP 3: UPDATE PLAYER STATES - COMPREHENSIVE STATE UPDATE**\n"
            "🚨 **UNDERSTAND DSL player_states DEFINITIONS FIRST**: Carefully read declaration.player_states definitions!\n"
            "• Each field has specific meaning and update conditions\n"
            "• Base updates on actual evidence, don't fabricate or assume state changes\n"
            "• Timing matters: Update states when events actually happen, not when anticipated\n\n"

            "**STATE UPDATE CHECKLIST**:\n"
            "• **SCAN ALL PLAYER STATES**: Check every field in every player's state for needed updates\n"
            "• **SCORE CALCULATIONS**: For reveal/results phases, calculate and update scores based on actual data\n"
            "  - Example: Two Truths - compare vote_choice vs lie_index, update score accordingly\n"
            "  - Example: Werewolf - update elimination counts, survival streaks, etc.\n"
            "• **GAME PROGRESSION**: Update round counters, phase completions, win conditions\n"
            "• **ACHIEVEMENT TRACKING**: Update any achievement or milestone fields\n"
            "• **MANDATORY**: Every reveal/results phase MUST include score/progress updates\n\n"

            "**VOTING VALIDATION & ERROR HANDLING**:\n"
            "1. Check voting eligibility BEFORE updating player_states\n"
            "2. If invalid vote detected: DO NOT call update_player_state for vote_choice\n"
            "3. Only process and record VALID votes from eligible players\n\n"

            "**DEATH STATUS CHECK**:\n"
            f"• Current status: Living {[pid for pid, data in player_states.items() if data.get('is_alive', True)]}, Dead {[pid for pid, data in player_states.items() if not data.get('is_alive', True)]}\n"
            "• If any player dies (is_alive changes from true to false):\n"
            "  - Immediately set is_alive=false using update_player_state tool\n"
            "  - Dead players cannot vote, act, or participate in any game mechanics\n\n"

            "**STEP 4: JUDGE PHASE TRANSITION - COMPLETION CRITERIA ANALYSIS**\n"
            f"• Check if current phase completion_criteria are satisfied:\n"
            f"  - Type: {current_phase.get('completion_criteria', {}).get('type', 'Unknown')}\n"
            f"  - Description: {current_phase.get('completion_criteria', {}).get('description', 'None')}\n"
            "• If conditions are met, use set_next_phase tool to enter next phase\n"
            "• Record transition reason and logic\n\n"

            "**STEP 5: UPDATE SCREEN CONTENT - UI MANAGEMENT**\n"
            "**SCREEN CLEANUP & CREATION**:\n"
            "• **DELETE FIRST**: clearCanvas() calls MUST be executed before all create tools\n"
            "• **POSITION OVERLAP PREVENTION**: Check existing item positions, NEVER place multiple items at same position\n"
            "• **POSITION ANALYSIS**: Read itemsState format '[ID] type:name@position' to identify occupied positions\n"
            "• **GRID POSITIONS**: top-left, top-center, top-right, middle-left, center, middle-right, bottom-left, bottom-center, bottom-right\n"
            "• **CONFLICT RESOLUTION**: If position occupied, choose next available position in grid\n\n"

            "**MANDATORY AUDIENCE PERMISSIONS**: Every component MUST specify who can see it:\n"
            "• Public: audience_type=true (everyone sees it)\n"
            "• Private: audience_type=false + audience_ids=['1','3'] (only specified players see it)\n\n"

            "**DEATH MARKER REQUIREMENTS**:\n"
            "• Every round, check player_states for is_alive=false\n"
            "• If dead player exists but no death_marker in items, CREATE one immediately\n"
            "• Death markers MUST use audience_type=false, audience_ids=[dead_player_id]\n"
            "• Example: createDeathMarker(playerName='Player 2', playerId='2', audience_type=false, audience_ids=['2'], position='top-right')\n\n"

            "**COMPONENT CREATION RULES**:\n"
            "• Phase indicators: Always place at 'top-center' position\n"
            "• Timers: ~10 seconds (max 15), default position 'center'\n"
            "• Input panels: createTextInputPanel() for collecting player text input\n"
            "• Result displays: Check existing conclusions first, don't fabricate results\n\n"

            "**STEP 6: REASONING PROCESS OUTPUT - TRANSPARENT REASONING**\n"
            "In your response message, explain in detail:\n"
            "• How you analyzed current phase and player behavior\n"
            "• Why you made these specific state updates\n"
            "• Phase transition logic and condition checking\n"
            "• UI changes reasons and player experience considerations\n"
            "• Help players understand game progress and decisions\n\n"

            "**STEP 7: VALIDATE REASONING - VALIDATION CHECKLIST**\n"
            "• Check all updates conform to DSL rules and definitions\n"
            "• Confirm UI display matches current phase requirements\n"
            "• Verify player state updates based on actual data, not fabricated\n"
            "• Ensure dead players are correctly filtered and marked\n"
            "• Verify audience permission settings are correct\n\n"

            "🔧 **AVAILABLE BACKEND TOOLS**:\n"
            "- update_player_state(player_id, state_name, state_value): Update player states\n"
            "- update_player_actions(player_id, actions, phase): Update player action records\n"
            "- set_next_phase(next_phase_id, transition_reason): Transition to next phase\n\n"

            "🚨 **CRITICAL PROHIBITIONS**:\n"
            "- NEVER return with ONLY cleanup calls - this is task failure!\n"
            "- Every clearCanvas MUST be followed by create tools in SAME response!\n"
            "- NEVER fabricate scores - only use player_states actual data\n"
            "- NEVER create voting options or interactive UI for dead players\n"
            "- NEVER assume or fabricate data not in playerActions\n\n"

            "🧒 **TREAT PLAYERS LIKE CHILDREN**:\n"
            "- Explain everything clearly and simply\n"
            "- Provide as much helpful information as possible\n"
            "- Guide them through every step\n"
            "- Never assume they understand anything\n\n"

            "🔧 **TOOL USAGE**: Use exact tool names (no prefixes), capture returned IDs for reuse\n"
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



    # 4. Process messages: remove ToolMessages and tool_calls from AIMessages, no quantity limit
    full_messages = state.get("messages", []) or []
    
    # Process all messages without quantity limit
    # processed_messages = []
    # for msg in full_messages:
    #     if isinstance(msg, ToolMessage):
    #         # Skip all ToolMessages
    #         continue
    #     elif isinstance(msg, AIMessage):
    #         # Skip AIMessages that contain tool calls to avoid tool_use/tool_result mismatch
    #         has_tools = (hasattr(msg, 'tool_calls') and msg.tool_calls) or \
    #                    (hasattr(msg, 'content') and isinstance(msg.content, (list, str)) and 'tool_use' in str(msg.content))
            
    #         if not has_tools and msg.content and msg.content.strip():
    #             # Only include AIMessage with pure text content (no tool usage)
    #             processed_messages.append(msg)
    #         # Skip all AIMessages with tool calls to avoid Claude API errors
    #     elif isinstance(msg, HumanMessage):
    #         # Keep HumanMessages for ActionExecutor
    #         processed_messages.append(msg)
    #     else:
    #         # Keep other message types (SystemMessage, etc.)
    #         processed_messages.append(msg)
    
    # trimmed_messages = processed_messages
    
    
    latest_state_system = SystemMessage(
        content=(
            "LATEST GROUND TRUTH (authoritative):\n"
            f"- items: {items_summary}\n"
            f"- current_phase_id: {current_phase_id}\n"
            f"- player_states: {player_states}\n"
            f"- current_phase_actions: {current_phase.get('actions', [])}\n"
        )
    )

    # Use complete full_messages without any processing
    response = await model_with_tools.ainvoke([
        system_message,
        latest_state_system,
        *full_messages,
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
        deletion_names = {"clearCanvas"}
        only_deletions = bool(orig_tool_calls) and all((_get_tool_name(tc) in deletion_names) for tc in orig_tool_calls)
        if only_deletions:
            logger.warning("[ActionExecutor][GUARD] Only clearCanvas tool calls detected; issuing follow-up request for creation tools.")
            strict_creation_system = SystemMessage(
                content=(
                    "You returned ONLY clearCanvas tools. Now you MUST produce the required creation tools for the current phase in this follow-up.\n"
                    "Rules:\n"
                    "- Do NOT call clearCanvas again.\n"
                    "- Call only creation tools to render the phase UI (e.g., createPhaseIndicator, createTimer, createVotingPanel, createTextDisplay, createDeathMarker, etc.).\n"
                    f"- Current phase context: ID {current_phase_id}. Follow its 'actions' strictly.\n"
                    "- Include proper audience permissions on each component (audience_type=true for public; or audience_type=false with audience_ids list).\n"
                )
            )
            # Claude requires at least one HumanMessage
            followup_human_msg = HumanMessage(content="Please create the missing UI components based on the system instructions.")
            followup_response = await model_with_tools.ainvoke([
                strict_creation_system,
                latest_state_system,
                followup_human_msg,
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

    # Process backend tool calls in ActionExecutor
    tool_calls = getattr(response, "tool_calls", []) or []
    current_player_states = dict(state.get("player_states", {}))
    current_player_actions = dict(state.get("playerActions", {}))
    current_game_notes = list(state.get("game_notes", []))
    updated_phase_id = state.get("current_phase_id", 0)
    
    for tc in tool_calls:
        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
        args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
        if not isinstance(args, dict):
            try:
                import json as _json
                args = _json.loads(args)
            except Exception:
                args = {}
        
        # Handle backend tool calls
        if name == "update_player_state":
            pid = args.get("player_id")
            state_name = args.get("state_name")
            state_value = args.get("state_value")
            if pid and state_name is not None:
                current_player_states = _execute_update_player_state(
                    current_player_states, pid, state_name, state_value
                )
                logger.info(f"[ActionExecutor] Updated player state: {pid}.{state_name} = {state_value}")
        elif name == "update_player_actions":
            pid = args.get("player_id")
            actions = args.get("actions")
            phase = args.get("phase")
            if pid and actions and phase:
                current_player_actions = _execute_update_player_actions(
                    current_player_actions, pid, actions, phase, state.get("roomSession", {}), current_player_states
                )
                logger.info(f"[ActionExecutor] Updated player actions: {pid} in phase {phase}")
        elif name == "add_game_note":
            note_type = args.get("note_type")
            content = args.get("content")
            if note_type and content:
                current_game_notes = _execute_add_game_note(
                    current_game_notes, note_type, content
                )
                logger.info(f"[ActionExecutor] Added game note: {note_type} - {content}")
        elif name == "set_next_phase":
            next_phase_id = args.get("next_phase_id")
            transition_reason = args.get("transition_reason", "")
            if next_phase_id is not None:
                # Validate phase ID using normalize function
                phases = dsl_content.get('phases', {})
                
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
                
                normalized_pid, is_valid = _normalize_and_validate_phase_id(next_phase_id, phases)
                if is_valid:
                    updated_phase_id = normalized_pid
                    logger.info(f"[ActionExecutor] Set next phase: {normalized_pid}, reason: {transition_reason}")
                else:
                    logger.warning(f"[ActionExecutor] Invalid phase ID {next_phase_id}, maintaining current phase")
    
    # Update phase history when phase changes
    current_phase_history = list(state.get("phase_history", []))
    if updated_phase_id != state.get("current_phase_id", 0):
        phases = dsl_content.get('phases', {})
        phase_name = phases.get(updated_phase_id, {}).get('name', f'Phase {updated_phase_id}') or phases.get(str(updated_phase_id), {}).get('name', f'Phase {updated_phase_id}')
        
        phase_entry = {
            "phase_id": updated_phase_id,
            "phase_name": phase_name
        }
        current_phase_history.append(phase_entry)
        logger.info(f"[ActionExecutor] Updated phase history: added phase {updated_phase_id} ({phase_name})")
    
    final_player_states = current_player_states

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
        goto=END,
        update={
            # Use final_messages like agent.py
            "messages": response,
            "items": state.get("items", []),
            "player_states": final_player_states,  # Updated via backend tool calls
            "playerActions": current_player_actions,  # Updated via backend tool calls
            "game_notes": current_game_notes,  # Updated via backend tool calls
            "phase_history": current_phase_history,  # Updated when phase changes
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
# workflow.add_node("RoleAssignmentNode", RoleAssignmentNode)
workflow.add_node("ActionExecutor", ActionExecutor)
# workflow.add_node("ActionValidatorNode", ActionValidatorNode)

# Set entry point
workflow.set_entry_point("InitialRouterNode")

# Compile the graph (LangGraph API handles persistence itself in local_dev/cloud)
graph = workflow.compile()



