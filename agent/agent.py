"""
This is the main entry point for the LLM Game Engine DM Agent.
It defines the workflow graph, game state, tools, nodes and edges for 
running diverse game types from DSL descriptions.
"""

# Apply patch for CopilotKit import issue before any other imports
# This fixes the incorrect import path in copilotkit.langgraph_agent (bug in v0.1.63)
import sys
import yaml
import os
import logging
from dotenv import load_dotenv


# Load environment variables with absolute path
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(env_path)

# Monitoring configuration
VERBOSE_LOGGING = True  # Set to False to disable detailed logging

# 直接配置 logger，不依赖 basicConfig
logger = logging.getLogger('GameAgent')
logger.handlers.clear()  # 清除现有 handlers

if VERBOSE_LOGGING:
    logger.setLevel(logging.INFO)
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    
    # 文件处理器
    file_handler = logging.FileHandler('/home/lee/canvas-with-langgraph-python/logs/agent.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.propagate = False  # 防止传播到root logger
else:
    logger.setLevel(logging.CRITICAL)

# Only apply the patch if the module doesn't already exist
if 'langgraph.graph.graph' not in sys.modules:
    # Create a mock module for the incorrect import path that CopilotKit expects
    class _MockModule:
        pass

    # Import the necessary modules first
    import langgraph
    import langgraph.graph
    import langgraph.graph.state

    # Import CompiledStateGraph from the correct location
    from langgraph.graph.state import CompiledStateGraph

    # Create the fake module path that CopilotKit incorrectly expects
    _mock_graph_module = _MockModule()
    _mock_graph_module.CompiledGraph = CompiledStateGraph

    # Add it to sys.modules so CopilotKit's incorrect import will work
    sys.modules['langgraph.graph.graph'] = _mock_graph_module

# Now we can safely import everything else
from typing import Any, List, Optional, Dict
from typing_extensions import Literal
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from copilotkit import CopilotKitState
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

class AgentState(CopilotKitState):
    """
    Game Engine DM Agent State
    
    Inherits from CopilotKitState for bidirectional frontend-backend sync.
    Contains game-specific fields for phase management, character tracking,
    event handling, and DSL-driven game logic.
    """
    tools: List[Any] = []
    # Shared state fields synchronized with the frontend
    items: List[Dict[str, Any]] = []  # Game elements (character cards, UI components, game objects)
    # Game DM state (for interactive game engine)
    phase: str = "" 
    # dsl: dict = {}  # Rules DSL (loaded once)
    events: List[Dict[str, Any]] = []  # queued UI/agent events, e.g., {type, payload, ts, source}
    characters: List[Dict[str, Any]] = []  # e.g., [{id, name}]
    # Player state management
    player_states: Dict[str, Any] = {}  # Player game states indexed by player_id
    gameName: str = ""  # Current game DSL name
    roomSession: Dict[str, Any] = {}  # Room session data from frontend
    playerActions: Dict[str, Dict[str, Any]] = {}  # Player action tracking by ID: {"1": {"name": "Alice", "actions": "voted for Bob", "timestamp": 1634567890, "phase": "day_voting"}}
    # No active item; all actions should specify an item identifier
    # Planning state
    planSteps: List[Dict[str, Any]] = []
    currentStepIndex: int = -1
    planStatus: str = ""
async def load_dsl_by_gamename(gamename: str) -> dict:
    """Load DSL content from YAML file based on gameName"""
    if not gamename:
        logger.warning("[DSL] No gameName provided, returning empty DSL")
        return {}
    
    try:
        import aiofiles
        dsl_file_path = os.path.join(os.path.dirname(__file__), '..', 'games', f"{gamename}.yaml")
        logger.info(f"[DSL] Loading DSL from: {dsl_file_path}")
        
        async with aiofiles.open(dsl_file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            dsl_content = yaml.safe_load(content)
        
        logger.info(f"[DSL] Successfully loaded DSL for game: {gamename}")
        return dsl_content
    except FileNotFoundError:
        logger.error(f"[DSL] DSL file not found for game: {gamename} at {dsl_file_path}")
        return {}
    except Exception as e:
        logger.error(f"[DSL] Failed to load DSL for game {gamename}: {e}")
        return {}

async def load_game_dsl(gamename: str = None) -> dict:
    """Load the game DSL from YAML file - supports both legacy and gameName-based loading"""
    if gamename:
        return await load_dsl_by_gamename(gamename)
    
    # Legacy fallback to simple_choice_game.yaml
    import asyncio
    import aiofiles
    
    try:
        dsl_path = '/home/lee/canvas-with-langgraph-python/games/simple_choice_game.yaml'
        async with aiofiles.open(dsl_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            dsl_content = yaml.safe_load(content)
        return dsl_content
    except Exception as e:
        print(f"Failed to load DSL: {e}")
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
                
                if default_value is not None:
                    template[field_name] = default_value
                elif field_type == 'boolean':
                    template[field_name] = True
                elif field_type == 'integer':
                    template[field_name] = 0
                elif field_type == 'array':
                    template[field_name] = []
                else:
                    template[field_name] = ''
        
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





def summarize_items_for_prompt(state: AgentState) -> str:
    """Simplified for game engine - items not used for complex project management"""
    try:
        items = state.get("items", []) or []
        if not items:
            return "(no items)"
        return f"{len(items)} item(s) present"
    except Exception:
        return "(unable to summarize items)"


@tool
def set_plan(steps: List[str]):
    """
    Initialize a plan consisting of step descriptions. Resets progress and sets status to 'in_progress'.
    """
    return {"initialized": True, "steps": steps}

@tool
def update_plan_progress(step_index: int, status: Literal["pending", "in_progress", "completed", "blocked", "failed"], note: Optional[str] = None):
    """
    Update a single plan step's status, and optionally add a note.
    """
    return {"updated": True, "index": step_index, "status": status, "note": note}

@tool
def complete_plan():
    """
    Mark the plan as completed.
    """
    return {"completed": True}

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
    state.playerActions[player_id] = {
        "name": player_name,
        "actions": actions,
        "timestamp": int(time.time() * 1000),  # milliseconds
        "phase": phase
    }
    
    logger.info(f"📝 Recorded actions for {player_name} ({player_id}) in {phase}: {actions}")
    
    return f"Recorded actions for {player_name} (ID: {player_id}): {actions}"

backend_tools = [
    set_plan,
    update_plan_progress,
    complete_plan,
    update_player_state,
    update_player_actions,
]

# Extract tool names from backend_tools for comparison
backend_tool_names = [tool.name for tool in backend_tools]

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


async def chat_node(state: AgentState, config: RunnableConfig) -> Command[Literal["tool_node", "__end__"]]:

    logger.info(f"[chatnode][start] ==== start chatnode ====")
    
    # Log initial state snapshot
    try:
        logger.info("🚀 CHATNODE START - INITIAL STATE SNAPSHOT")
        logger.info("=" * 60)
        
        # Player States
        player_states = state.get("player_states", {})
        logger.info(f"📊 PLAYER STATES: {player_states}")
        
        # Player Actions 
        player_actions = state.get("playerActions", {})
        logger.info(f"📝 PLAYER ACTIONS: {player_actions}")
        
        # Current Phase
        current_phase = state.get("phase", "")
        logger.info(f"🎭 CURRENT PHASE: {current_phase}")
        
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Error logging initial state: {e}")

    """
    Standard chat node based on the ReAct design pattern. It handles:
    - The model to use (and binds in CopilotKit actions and the tools defined above)
    - The system prompt
    - Getting a response from the model
    - Handling tool calls

    For more about the ReAct design pattern, see:
    https://www.perplexity.ai/search/react-agents-NcXLQhreS0WDzpVaS4m9Cg
    """

    # Initialize player_states if empty and we have roomSession data
    player_states = state.get("player_states", {})
    if not player_states:
        logger.info(f"[chatnode] player_states empty, checking roomSession...")
        room_session = state.get("roomSession")
        if room_session and room_session.get("players"):
            logger.info("[chatnode] Initializing player_states from roomSession")
            try:
                # Load DSL for initialization - use gameName if available
                game_name = state.get("gameName", "")
                logger.info(f"[chatnode] Loading DSL for player initialization, gameName: {game_name}")
                
                if game_name:
                    dsl_content = await load_game_dsl(game_name)
                else:
                    dsl_content = await load_game_dsl()
                if dsl_content:
                    room_players = room_session["players"]
                    initialized_states = await initialize_player_states_from_dsl(dsl_content, room_players)
                    if initialized_states:
                        # Update state with initialized player_states
                        state = {**state, "player_states": initialized_states}
                        logger.info(f"[chatnode] ✅ Initialized player_states: {len(initialized_states)} players")
                    else:
                        logger.warning("[chatnode] Failed to initialize player_states from DSL")
                else:
                    logger.warning("[chatnode] No DSL available for player_states initialization")
            except Exception as e:
                logger.error(f"[chatnode] Error initializing player_states: {e}")
        else:
            logger.warning("[chatnode] No roomSession data available for player_states initialization")
    else:
        logger.info(f"[chatnode] Using existing player_states: {len(player_states)} players")

    # 1. Define the model
    model = init_chat_model("openai:gpt-4.1-mini")

    # 2. Prepare and bind tools to the model (dedupe, allowlist, and cap)
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

    # cap to well under 128 (OpenAI tools limit), leaving room for backend tools
    MAX_FRONTEND_TOOLS = 110
    if len(deduped_frontend_tools) > MAX_FRONTEND_TOOLS:
        deduped_frontend_tools = deduped_frontend_tools[:MAX_FRONTEND_TOOLS]

    model_with_tools = model.bind_tools(
        [
            *deduped_frontend_tools,
            *backend_tools,
        ],
        parallel_tool_calls=True,  # Allow multiple tool calls
    )

    # 3. Define the system message by which the chat model will be run
    items_summary = summarize_items_for_prompt(state)
    post_tool_guidance = state.get("__last_tool_guidance", None)
    last_action = state.get("lastAction", "")
    plan_steps = state.get("planSteps", []) or []
    current_step_index = state.get("currentStepIndex", -1)
    plan_status = state.get("planStatus", "")
    
    # Load DSL based on gameName if available, otherwise use default
    game_name = state.get("gameName", "")
    logger.info(f"[chatnode] Game name from state: {game_name}")
    
    if game_name:
        logger.info(f"[chatnode] Loading DSL for game: {game_name}")
        dsl_content = await load_game_dsl(game_name)
    else:
        logger.info("[chatnode] No gameName specified, using default DSL")
        dsl_content = await load_game_dsl()
    game_schema = (
        
        "  - events: Array<{type: string, payload: any, timestamp: number}> (game events)\n"
    )

    loop_control = (
        "LOOP CONTROL RULES:\n"
        "1) Never call the same mutating tool repeatedly in a single turn.\n"
        "2) If asked to 'add a couple' checklist items, add at most 2 and then stop.\n"
        "3) Avoid creating empty-text checklist items; if you don't have labels, ask once for labels.\n"
        "4) After a successful mutation (create/update/delete), summarize changes and STOP instead of looping.\n"
        "5) If lastAction starts with 'created:', DO NOT call createItem again unless the user explicitly asks to create another item.\n"
    )

    # Summarize game state for the DM
    def _summarize_game_state(s: AgentState) -> str:
        try:
            phase = s.get("phase", "")
            chars = s.get("characters", []) or []
            evts = s.get("events", []) or []
            chars_line = ", ".join([f"{c.get('id','char')}:{c.get('name','')}" for c in chars]) or "(none)"
            recent = evts[-5:] if isinstance(evts, list) else []
            evts_line = ", ".join([str(e.get("type","")) for e in recent]) or "(none)"
            return f"phase={phase} · characters=[{chars_line}] · recentEvents=[{evts_line}]"
        except Exception:
            return "(unable to summarize game state)"

    game_summary = _summarize_game_state(state)

    # Include DSL in system message
    dsl_info = f"LOADED GAME DSL:\n{dsl_content}\n" if dsl_content else "No DSL loaded.\n"

    system_message = SystemMessage(
        content=(
            f"gameName (ground truth): {game_name}\n"
            f"itemsState (ground truth):\n{items_summary}\n"
            f"lastAction (ground truth): {last_action}\n"
            f"gameState (ground truth): {game_summary}\n"
            f"planStatus (ground truth): {plan_status}\n"
            f"currentStepIndex (ground truth): {current_step_index}\n"
            f"planSteps (ground truth): {[s.get('title', s) for s in plan_steps]}\n"
            f"{dsl_info}\n"
            f"{loop_control}\n"
            f"{game_schema}\n"
            "AUTONOMOUS DUNGEON MASTER POLICY:\n"
            "Every turn, you MUST autonomously plan and execute the next action to drive the game forward. Based on DSL rules and current game state, you will manage player states, automatically generate UI, and simulate Bot behaviors."

            "Autonomous Execution Strategy:"
            "- Analyze current game state and DSL rules to determine the next required action"
            "- Act autonomously based on game logic without waiting for explicit user instructions"
            "- If the DSL is flawed, use common-sense game logic to make corrections and proceed"
            "- At the end of each turn, anticipate what operations need to be executed in the next turn"

            "Planning & Progress Tracking Strategy:"
            "- MANDATORY: Every task must have at least 2 loop steps - no single-step tasks allowed"
            "- For complex phases with multiple steps, use the set_plan tool to create an execution plan"
            "- Track progress of each step using update_plan_progress ('in_progress', 'completed')"
            "- Proceed automatically between steps without waiting for user confirmation"
            "- Call complete_plan only after all steps in the plan are successfully finished"
            "- Each plan step must represent a complete scene/phase with full UI content"

            "Phase Progression Strategy:"
            "- Check current phase completion conditions to decide if advancing to the next phase is needed"
            "- Evaluate which branch matches the current state based on DSL next_phase conditions"
            "- Automatically update the game phase and create corresponding UI components"
            "- Every phase change MUST update the phase indicator (createPhaseIndicator)"

            "Player State Management Strategy:"
            "- Use the update_player_state(player_id, state_name, state_value) tool for all state modifications"
            "- Simulate logical actions for Bot players based on their role and the current phase"
            "- Update the human player's state based on their UI or text input"
            "- Always check current playerStates from LATEST GROUND TRUTH after every update to verify success"
            "- Determine win conditions based on DSL rules and player states to end the game accordingly"

            "MANDATORY UI DISPLAY POLICY:"
            "- CRITICAL: Every single step, action, explanation, or instruction MUST have a visual UI component"
            "- NEVER provide text responses without corresponding UI tools (createTextDisplay, createPhaseIndicator, etc.)"
            "- ALL explanations, narrations, instructions, and descriptions MUST use createTextDisplay tool"
            "- Every scene must be a complete visual experience with ALL necessary UI components created in one step"
            "- NO EXCEPTIONS: If you say something, you must create UI for it"

            "UI Management Strategy:"
            "- Before creating new UI, clean up outdated interface components (clearCanvas or deleteItem)"  
            "- Avatars, once created, should generally not be deleted"
            "- A persistent character card for the human player should be displayed in the bottom-left corner for easy reference"
            "- Enforce information privacy: use private_hint for secret information and show UI only to the human player"
            "- Ensure the UI stays synchronized with the current game state"
            "- UI Layout Priority: ALWAYS prioritize center positions (middle-center, center, top-center) for main content"
            "- Place primary game information and phase content in center first, secondary info on sides after"
            "- Layout order: center → top-center → bottom-center → middle-center → sides (left/right)"
            "- Voting panels use center/bottom-center, text displays use middle-center, phase indicators use top-center"
            "- Create complete scene UI in single step: phase indicator + main content + supporting elements + explanations"

            "Tool Execution & Verification Policy:"
            "- After every tool call, you MUST re-read the state to confirm the change was successful"
            "- Never claim a change occurred if the LATEST GROUND TRUTH does not reflect it"

            

            "Game State Grounding Rules:"
            "- Make decisions based ONLY on the latest game state, player states, and DSL content"
            "- Ignore outdated information from chat history; use LATEST GROUND TRUTH as the only source of truth"
            "- If game state values are missing, proceed with reasonable defaults based on the current phase"

            "Message Broadcasting Rules:"
            "- Only send chat messages during key phase transitions and critical game events"
            "- Keep broadcast messages concise and thematic (game narrative style)"
            "- Do NOT broadcast for minor UI updates or routine tool executions"

            "Game Initialization Policy:"
            "- If the history is empty and the user says 'start game', automatically initialize the game"
            
            + (f"\nPOST-TOOL POLICY:\n{post_tool_guidance}\n" if post_tool_guidance else "")
        )
    )

    # 4. Run the model to generate a response
    # If the user asked to modify an item but did not specify which, interrupt to choose
    # try:
    #     last_user = next((m for m in reversed(state["messages"]) if getattr(m, "type", "") == "human"), None)
    #     if last_user and any(k in last_user.content.lower() for k in ["item", "rename", "owner", "priority", "status"]) and not any(k in last_user.content.lower() for k in ["prj_", "item id", "id="]):
    #         choice = interrupt({
    #             "type": "choose_item",
    #             "content": "Please choose which item you mean.",
    #         })
    #         state["chosen_item_id"] = choice
    # except Exception:
    #     pass

    # 4.1 If the latest message contains unresolved FRONTEND tool calls, do not call the LLM yet.
    #     End the turn and wait for the client to execute tools and append ToolMessage responses.
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
                    logger.info("[chatnode][end] Pending frontend tool calls detected; skipping LLM this turn and waiting for ToolMessage(s).")
                except Exception:
                    pass
                
                return Command(
                    goto=END,
                    update={
                        # no changes; just wait for the client to respond with ToolMessage(s)
                        "items": state.get("items", []),
                        "itemsCreated": state.get("itemsCreated", 0),
                        "lastAction": state.get("lastAction", ""),
                        "planSteps": state.get("planSteps", []),
                        "currentStepIndex": state.get("currentStepIndex", -1),
                        "planStatus": state.get("planStatus", ""),
                        # persist game dm fields
                        "phase": state.get("phase", ""),
                        "events": state.get("events", []),
                        "characters": state.get("characters", []),
                        # persist player states
                        "player_states": state.get("player_states", {}),
                        "gameName": state.get("gameName", ""),
                        "roomSession": state.get("roomSession", {}),
                    },
                )
    except Exception:
        pass

    # 4.2 Keep all message history as requested
    trimmed_messages = full_messages

    # 4.3 Append a final, authoritative state snapshot after chat history
    #
    # Ensure the latest shared state takes priority over chat history and
    # stale tool results. This enforces state-first grounding, reduces drift, and makes
    # precedence explicit. Optional post-tool guidance confirms successful actions
    # (e.g., deletion) instead of re-stating absence.
    # Get player states for prompt
    player_states = state.get("player_states", {})
    game_name = state.get("gameName", "")
    current_phase = state.get("phase", "")
    
    latest_state_system = SystemMessage(
        content=(
            "LATEST GROUND TRUTH (authoritative):\n"
            f"- items:\n{items_summary}\n"
            f"- lastAction: {last_action}\n\n"
            f"- planStatus: {plan_status}\n"
            f"- currentStepIndex: {current_step_index}\n"
            f"- planSteps: {[s.get('title', s) for s in plan_steps]}\n\n"
            f"GAME STATE (authoritative):\n"
            f"- gameName: {game_name}\n"
            f"- currentPhase: {current_phase}\n"
            f"- playerStates: {player_states}\n"
            f"- totalPlayers: {len(player_states) if player_states else 0}\n\n"
            "Resolution policy: If ANY prior message mentions values that conflict with the above,\n"
            "those earlier mentions are obsolete and MUST be ignored.\n"
            "When asked 'what is it now', ALWAYS read from this LATEST GROUND TRUTH.\n"
            "Use playerStates to make game decisions (role assignments, win conditions, voting, etc.).\n"
            + ("\nIf the last tool result indicated success (e.g., 'deleted:ID'), confirm the action rather than re-stating absence." if post_tool_guidance else "")
        )
    )

    # Log input plan steps, status, and recent history before invoking LLM
    try:
        logger.info(f"[LLM][plan] planStatus={plan_status} currentStepIndex={current_step_index}")
        logger.info(f"[LLM][plan] planSteps={plan_steps}")
        logger.info(f"[LLM][history] history_count={len(trimmed_messages)}")
        if trimmed_messages:
            m = trimmed_messages[-1]
            m_type = getattr(m, "type", None) or m.__class__.__name__
            m_content = getattr(m, "content", None)
            preview = m_content[:400] if isinstance(m_content, str) else "(non-text)"
            logger.info(f"[LLM][history] last_history {m_type}: {preview}")
        
        # Log detailed game state before LLM invocation
        logger.info("=" * 80)
        logger.info("🎮 GAME STATE SNAPSHOT BEFORE LLM")
        logger.info("=" * 80)
        
        # Player States
        player_states = state.get("player_states", {})
        logger.info(f"📊 PLAYER STATES ({len(player_states)} players):")
        for player_id, player_data in player_states.items():
            logger.info(f"  Player {player_id}: {player_data}")
        
        # Player Actions 
        player_actions = state.get("playerActions", {})
        logger.info(f"📝 PLAYER ACTION RECORDS ({len(player_actions)} records):")
        for player_id, action_data in player_actions.items():
            logger.info(f"  Player {player_id}: {action_data}")
        
        # Current Phase
        current_phase = state.get("phase", "")
        logger.info(f"🎭 CURRENT PHASE: {current_phase}")
        logger.info("=" * 80)
        
    except Exception:
        pass

    response = await model_with_tools.ainvoke([
        system_message,
        *trimmed_messages,
        latest_state_system,
    ], config)

    # Log LLM output content and planned tool calls (not just the last turn)
    try:
        logger.info("=" * 80)
        logger.info("🤖 LLM OUTPUT RESPONSE")
        logger.info("=" * 80)
        
        content_preview = getattr(response, "content", None)
        if isinstance(content_preview, str):
            logger.info(f"💬 RESPONSE CONTENT: {content_preview}")
        else:
            logger.info(f"💬 RESPONSE CONTENT: (non-text)")
            
        tool_calls = getattr(response, "tool_calls", []) or []
        if tool_calls:
            logger.info(f"🔧 TOOL CALLS ({len(tool_calls)} calls):")
            for i, tc in enumerate(tool_calls, 1):
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                logger.info(f"  {i}. {name}: {args}")
        else:
            logger.info("🔧 TOOL CALLS: none")
        
        logger.info("=" * 80)
    except Exception:
        pass
    

    # Predictive plan/game state updates based on imminent tool calls (for UI rendering)
    try:
        tool_calls = getattr(response, "tool_calls", []) or []
        predicted_plan_steps = plan_steps.copy()
        predicted_current_index = current_step_index
        predicted_plan_status = plan_status
        for tc in tool_calls:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
            if not isinstance(args, dict):
                try:
                    import json as _json
                    args = _json.loads(args)  # sometimes args can be a json string
                except Exception:
                    args = {}
            if name == "set_plan":
                raw_steps = args.get("steps") or []
                predicted_plan_steps = [{"title": s if isinstance(s, str) else str(s), "status": "pending"} for s in raw_steps]
                if predicted_plan_steps:
                    predicted_plan_steps[0]["status"] = "in_progress"
                    predicted_current_index = 0
                    predicted_plan_status = "in_progress"
                else:
                    predicted_current_index = -1
                    predicted_plan_status = ""
            elif name == "update_plan_progress":
                idx = args.get("step_index")
                status = args.get("status")
                note = args.get("note")
                if isinstance(idx, int) and 0 <= idx < len(predicted_plan_steps) and isinstance(status, str):
                    if note:
                        predicted_plan_steps[idx]["note"] = note
                    predicted_plan_steps[idx]["status"] = status
                    if status == "in_progress":
                        predicted_current_index = idx
                        predicted_plan_status = "in_progress"
                    if status == "completed" and idx >= predicted_current_index:
                        predicted_current_index = idx
            elif name == "complete_plan":
                for i in range(len(predicted_plan_steps)):
                    if predicted_plan_steps[i].get("status") != "completed":
                        predicted_plan_steps[i]["status"] = "completed"
                predicted_plan_status = "completed"
        # Aggregate overall plan status conservatively and manage progression
        if predicted_plan_steps:
            statuses = [str(s.get("status", "")) for s in predicted_plan_steps]
            # Do NOT auto-mark overall plan completed unless complete_plan is called.
            # We still reflect failure if any step failed.
            if any(st == "failed" for st in statuses):
                predicted_plan_status = "failed"
            elif any(st == "in_progress" for st in statuses):
                predicted_plan_status = "in_progress"
            elif any(st == "blocked" for st in statuses):
                predicted_plan_status = "blocked"
            else:
                predicted_plan_status = predicted_plan_status or ""

            # Only promote a new step when the previously active step transitioned to completed
            active_idx = next((i for i, s in enumerate(predicted_plan_steps) if str(s.get("status", "")) == "in_progress"), -1)
            if active_idx == -1:
                # find last completed and promote the next pending, else first pending
                last_completed = -1
                for i, s in enumerate(predicted_plan_steps):
                    if str(s.get("status", "")) == "completed":
                        last_completed = i
                # Prefer the immediate next step after the last completed
                promote_idx = next((i for i in range(last_completed + 1, len(predicted_plan_steps)) if str(predicted_plan_steps[i].get("status", "")) == "pending"), -1)
                if promote_idx == -1:
                    promote_idx = next((i for i, s in enumerate(predicted_plan_steps) if str(s.get("status", "")) == "pending"), -1)
                if promote_idx != -1:
                    predicted_plan_steps[promote_idx]["status"] = "in_progress"
                    predicted_current_index = promote_idx
                    predicted_plan_status = "in_progress"
        # If we predicted changes, persist them before routing or ending
        plan_updates = {}
        if predicted_plan_steps != plan_steps:
            plan_updates["planSteps"] = predicted_plan_steps
        if predicted_current_index != current_step_index:
            plan_updates["currentStepIndex"] = predicted_current_index
        if predicted_plan_status != plan_status:
            plan_updates["planStatus"] = predicted_plan_status
    except Exception:
        plan_updates = {}
    
    # Handle player state updates from tool calls
    player_state_updates = {}
    try:
        tool_calls = getattr(response, "tool_calls", []) or []
        current_player_states = dict(state.get("player_states", {}))
        current_player_actions = dict(state.get("playerActions", {}))
        
        # Validate tool calling constraints: only one backend tool allowed
        backend_tool_calls = [tc for tc in tool_calls if tc.get("name") in backend_tool_names]
        if len(backend_tool_calls) > 1:
            logger.warning(f"[chatnode] CONSTRAINT VIOLATION: Multiple backend tools called: {[tc.get('name') for tc in backend_tool_calls]}")
            # Could potentially filter to only the first backend tool call
            # backend_tool_calls = backend_tool_calls[:1]
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            if tool_name == "update_player_state":
                tool_args = tool_call.get("args", {})
                player_id = tool_args.get("player_id")
                state_name = tool_args.get("state_name")
                state_value = tool_args.get("state_value")
                
                if player_id and state_name is not None:
                    # Initialize player dict if it doesn't exist
                    if player_id not in current_player_states:
                        current_player_states[player_id] = {}
                    
                    # Update the player state
                    current_player_states[player_id][state_name] = state_value
                    logger.info(f"[chatnode] Updating player {player_id}: {state_name} = {state_value}")
            
            elif tool_name == "update_player_actions":
                tool_args = tool_call.get("args", {})
                player_id = tool_args.get("player_id")
                actions = tool_args.get("actions")
                phase = tool_args.get("phase")
                
                if player_id and actions and phase:
                    import time
                    
                    # Get player name from roomSession or player_states
                    player_name = f"Player {player_id}"
                    room_session = state.get("roomSession", {})
                    if room_session and "players" in room_session:
                        for player in room_session["players"]:
                            if str(player.get("gamePlayerId", "")) == str(player_id):
                                player_name = player.get("name", player_name)
                                break
                    elif player_id in current_player_states and "name" in current_player_states[player_id]:
                        player_name = current_player_states[player_id]["name"]
                    
                    # Update player actions
                    current_player_actions[player_id] = {
                        "name": player_name,
                        "actions": actions,
                        "timestamp": int(time.time() * 1000),  # milliseconds
                        "phase": phase
                    }
                    logger.info(f"[chatnode] Recording actions for {player_name} ({player_id}) in {phase}: {actions}")
        
        # Pass updated states to update
        player_state_updates["player_states"] = current_player_states
        player_state_updates["playerActions"] = current_player_actions
        
    except Exception as e:
        logger.warning(f"[chatnode] Error processing player state updates: {e}")
        player_state_updates = {}

    # only route to tool node if tool is not in the tools list
    if route_to_tool_node(response):
        tool_calls = getattr(response, "tool_calls", []) or []
        logger.info(f"[chatnode][end] === OUTPUT: ROUTING TO TOOL_NODE ===")
        # for tc in tool_calls:
        #     tool_name = tc.get("name", "unknown")
        #     tool_args = tc.get("args", {})
        #     logger.info(f"[CHAT_NODE] Tool: {tool_name}")
        #     logger.info(f"[CHAT_NODE] Args: {tool_args}")
        # logger.info(f"[CHAT_NODE] === END OUTPUT ===")
        # print("routing to tool node")
        return Command(
            goto="tool_node",
            update={
                "messages": [response],
                # persist shared state keys so UI edits survive across runs
                "items": state.get("items", []),
                "itemsCreated": state.get("itemsCreated", 0),
                "lastAction": state.get("lastAction", ""),
                "planSteps": state.get("planSteps", []),
                "currentStepIndex": state.get("currentStepIndex", -1),
                "planStatus": state.get("planStatus", ""),
                **plan_updates,
                **player_state_updates,
                # persist game dm fields
                "phase": state.get("phase", ""),
                "events": state.get("events", []),
                "characters": state.get("characters", []),
                # persist player states
                "player_states": state.get("player_states", {}),
                "gameName": state.get("gameName", ""),
                "roomSession": state.get("roomSession", {}),
                # guidance for follow-up after tool execution
                "__last_tool_guidance": "If a deletion tool reports success (deleted:ID), acknowledge deletion even if the item no longer exists afterwards."
            }
        )

    # 5. If there are remaining steps, auto-continue; otherwise end the graph.
    try:
        effective_steps = plan_updates.get("planSteps", plan_steps)
        effective_plan_status = plan_updates.get("planStatus", plan_status)
        has_remaining = bool(effective_steps) and any(
            (s.get("status") not in ("completed", "failed")) for s in effective_steps
        )
        if effective_plan_status in ("completed", "failed"):
            logger.info(f"[chatnode][end] === OUTPUT: PLAN STATUS IS COMPLETED OR FAILED ===")
            return Command(
                goto=END,
                update={
                    "planStatus": effective_plan_status,
                },
            )
    except Exception:
        effective_steps = plan_steps
        effective_plan_status = plan_status
        has_remaining = False
    

    # Determine if this response contains frontend tool calls that must be delivered to the client
    try:
        tool_calls = getattr(response, "tool_calls", []) or []
    except Exception:
        tool_calls = []
    has_frontend_tool_calls = False
    for tc in tool_calls:
        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
        if name and name not in backend_tool_names:
            has_frontend_tool_calls = True
            break

    # If the model produced FRONTEND tool calls, deliver them to the client and stop the turn.
    # The client will execute and post ToolMessage(s), after which the next run can resume.
    if has_frontend_tool_calls:
        frontend_tools = [tc.get("name", "unknown") for tc in tool_calls if tc.get("name") not in backend_tool_names]
        logger.info(f"[chatnode][end] === OUTPUT: ENDING WITH FRONTEND TOOLS ===")
        # logger.info(f"[CHAT_NODE] Frontend tools: {frontend_tools}")
        # logger.info(f"[CHAT_NODE] Waiting for client execution")
        # logger.info(f"[CHAT_NODE] === END OUTPUT ===")
        return Command(
            goto=END,
            update={
                "messages": [response],
                "items": state.get("items", []),
                "itemsCreated": state.get("itemsCreated", 0),
                "lastAction": state.get("lastAction", ""),
                "planSteps": state.get("planSteps", []),
                "currentStepIndex": state.get("currentStepIndex", -1),
                "planStatus": state.get("planStatus", ""),
                **plan_updates,
                **player_state_updates,
                "__last_tool_guidance": (
                    "Frontend tool calls issued. Waiting for client tool results before continuing."
                ),
                # persist game dm fields
                "phase": state.get("phase", ""),
                "events": state.get("events", []),
                "characters": state.get("characters", []),
                # persist player states
                "player_states": state.get("player_states", {}),
                "gameName": state.get("gameName", ""),
                "roomSession": state.get("roomSession", {}),
            },
        )

    if has_remaining and effective_plan_status != "completed":
        # Stop here; do not auto-loop back into chat_node. Let the next activation drive further steps.
        return Command(
            goto=chat_node,
            update={
                "messages": ([]),
                # persist shared state keys so UI edits survive across runs
                "items": state.get("items", []),
                "itemsCreated": state.get("itemsCreated", 0),
                "lastAction": state.get("lastAction", ""),
                "planSteps": state.get("planSteps", []),
                "currentStepIndex": state.get("currentStepIndex", -1),
                "planStatus": state.get("planStatus", ""),
                **plan_updates,
                **player_state_updates,
                # persist game dm fields
                "phase": state.get("phase", ""),
                "events": state.get("events", []),
                "characters": state.get("characters", []),
                # persist player states
                "player_states": state.get("player_states", {}),
                "gameName": state.get("gameName", ""),
                "roomSession": state.get("roomSession", {}),
                "__last_tool_guidance": None,
            }
        )

    # If all steps look completed but planStatus is not yet 'completed', nudge the model to call complete_plan
    try:
        all_steps_completed = bool(effective_steps) and all((s.get("status") == "completed") for s in effective_steps)
        plan_marked_completed = (effective_plan_status == "completed")
    except Exception:
        all_steps_completed = False
        plan_marked_completed = False

    if all_steps_completed and not plan_marked_completed:
        # Do not auto-loop; end and rely on external trigger for any wrap-up.
        return Command(
            goto=chat_node,
            update={
                "messages": ([]),
                # persist shared state keys so UI edits survive across runs
                "items": state.get("items", []),
                "itemsCreated": state.get("itemsCreated", 0),
                "lastAction": state.get("lastAction", ""),
                "planSteps": state.get("planSteps", []),
                "currentStepIndex": state.get("currentStepIndex", -1),
                "planStatus": state.get("planStatus", ""),
                **plan_updates,
                **player_state_updates,
                # persist game dm fields
                "phase": state.get("phase", ""),
                "events": state.get("events", []),
                "characters": state.get("characters", []),
                # persist player states
                "player_states": state.get("player_states", {}),
                "gameName": state.get("gameName", ""),
                "roomSession": state.get("roomSession", {}),
                "__last_tool_guidance": None,
            }
        )

    # Only show chat messages when not actively in progress; always deliver frontend tool calls
    currently_in_progress = (plan_updates.get("planStatus", plan_status) == "in_progress")
    final_messages = [response] if (has_frontend_tool_calls or not currently_in_progress) else ([])
    
    logger.info(f"[chatnode][end] === ENDING ===")
    return Command(
        goto=END,
        update={
            "messages": final_messages,
            # persist shared state keys so UI edits survive across runs
            "items": state.get("items", []),
            "itemsCreated": state.get("itemsCreated", 0),
            "lastAction": state.get("lastAction", ""),
            "planSteps": state.get("planSteps", []),
            "currentStepIndex": state.get("currentStepIndex", -1),
            "planStatus": state.get("planStatus", ""),
            **plan_updates,
            # persist game dm fields
            "phase": state.get("phase", ""),
            "events": state.get("events", []),
            "characters": state.get("characters", []),
            # persist player states
            "player_states": state.get("player_states", {}),
            "gameName": state.get("gameName", ""),
            "roomSession": state.get("roomSession", {}),
            "__last_tool_guidance": None,
        }
    )


def route_to_tool_node(response: BaseMessage):
    """
    Route to tool node if any tool call in the response matches a backend tool name.
    """
    tool_calls = getattr(response, "tool_calls", None)
    if not tool_calls:
        return False

    for tool_call in tool_calls:
        name = tool_call.get("name")
        if name in backend_tool_names:
            return True
    return False

# Define the workflow graph
workflow = StateGraph(AgentState)
workflow.add_node("chat_node", chat_node)
workflow.add_node("tool_node", ToolNode(tools=backend_tools))
workflow.add_edge("tool_node", "chat_node")
workflow.set_entry_point("chat_node")

graph = workflow.compile()


# if __name__ == "__main__":
#     import asyncio
#     from langchain_core.messages import HumanMessage

#     async def main():
#         out = await graph.ainvoke({"messages": [HumanMessage(content="start game")]})
#         print(out)

#     asyncio.run(main())