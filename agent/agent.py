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
    # No active item; all actions should specify an item identifier
    # Planning state
    planSteps: List[Dict[str, Any]] = []
    currentStepIndex: int = -1
    planStatus: str = ""
async def load_game_dsl() -> dict:
    """Load the game DSL from YAML file"""
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

# @tool
# def your_tool_here(your_arg: str):
#     """Your tool description here."""
#     print(f"Your tool logic here")
#     return "Your tool response here."

backend_tools = [
    set_plan,
    update_plan_progress,
    complete_plan,
]

# Extract tool names from backend_tools for comparison
backend_tool_names = [tool.name for tool in backend_tools]

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
    # Component management tools
    "deleteItem",
    # Player state management
    "markPlayerDead"
])


async def chat_node(state: AgentState, config: RunnableConfig) -> Command[Literal["tool_node", "__end__"]]:

    logger.info(f"[chatnode][start] ==== start chatnode ====")

    """
    Standard chat node based on the ReAct design pattern. It handles:
    - The model to use (and binds in CopilotKit actions and the tools defined above)
    - The system prompt
    - Getting a response from the model
    - Handling tool calls

    For more about the ReAct design pattern, see:
    https://www.perplexity.ai/search/react-agents-NcXLQhreS0WDzpVaS4m9Cg
    """

    # 1. Define the model
    model = init_chat_model("openai:gpt-4o")

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
        parallel_tool_calls=False,
    )

    # 3. Define the system message by which the chat model will be run
    items_summary = summarize_items_for_prompt(state)
    post_tool_guidance = state.get("__last_tool_guidance", None)
    last_action = state.get("lastAction", "")
    plan_steps = state.get("planSteps", []) or []
    current_step_index = state.get("currentStepIndex", -1)
    plan_status = state.get("planStatus", "")
    
    # # Load DSL if not already loaded
    # dsl_content = state.get("dsl", {})
    # if not dsl_content:
    dsl_content = await load_game_dsl()
    game_schema = (
        "GAME COMPONENT SCHEMA (authoritative):\n"
        "- character_card.data:\n"
        "  - role: string (character role e.g., 'werewolf', 'seer', 'villager')\n"
        "  - position: string (select: 'top-left' | 'top-center' | 'top-right' | 'middle-left' | 'center' | 'middle-right' | 'bottom-left' | 'bottom-center' | 'bottom-right')\n"
        "  - size: string (select: 'small' | 'medium' | 'large'; default: 'medium')\n"
        "  - description: string (optional character description)\n"
        "- action_button.data:\n"
        "  - label: string (button text)\n"
        "  - action: string (action identifier when clicked)\n"
        "  - enabled: boolean (whether button is clickable)\n"
        "  - variant: string (select: 'primary' | 'secondary' | 'danger'; default: 'primary')\n"
        "  - position: string (grid position; same options as character_card)\n"
        "  - size: string (component size; same options as character_card)\n"
        "- phase_indicator.data:\n"
        "  - currentPhase: string (current game phase)\n"
        "  - description: string (optional phase description)\n"
        "  - timeRemaining: number (optional seconds remaining in phase)\n"
        "  - position: string (grid position; same options as character_card)\n"
        "  - size: string (component size; same options as character_card)\n"
        "- text_display.data:\n"
        "  - content: string (main text content)\n"
        "  - title: string (optional title text)\n"
        "  - type: string (select: 'info' | 'warning' | 'error' | 'success'; default: 'info')\n"
        "  - position: string (grid position; same options as character_card)\n"
        "  - size: string (component size; same options as character_card)\n"
        "- game state:\n"
        "  - phase: string (current DSL phase name e.g., 'introduction', 'role_selection', 'game_result')\n"
        "  - dsl: object (loaded game definition with phases, flow)\n"
        "  - events: Array<{type: string, payload: any, timestamp: number}> (game events)\n"
    )

    loop_control = (
        "GAME PHASE LOOP CONTROL:\n"
        "1) Intra-phase loop: Execute multiple UI tools and backend logic within same phase\n"
        "2) Phase end blocking: When phase operations complete, must END and wait for next activation\n"
        "3) Operation sequence: UI tool setup → backend state update → check phase completion → END\n"
        "4) Prevent infinite loops: Each phase must have clear completion criteria and end mechanism\n"
        "EXAMPLES:\n"
        "- Game start phase: putBackground → putPlayerList → putRoles → putTimer → END\n"
        "- Werewolf kill phase: putWerewolfPanel → putTimer → END (wait for werewolf voting)\n"
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
            f"itemsState (ground truth):\n{items_summary}\n"
            f"lastAction (ground truth): {last_action}\n"
            f"gameState (ground truth): {game_summary}\n"
            f"planStatus (ground truth): {plan_status}\n"
            f"currentStepIndex (ground truth): {current_step_index}\n"
            f"planSteps (ground truth): {[s.get('title', s) for s in plan_steps]}\n"
            f"{dsl_info}\n"
            f"{loop_control}\n"
            f"{game_schema}\n"
            "GAME/Dungeon Master POLICY:\n"
            "- Act as a DM using the provided DSL (if present) to control the game through tools.\n"
            "- Decide which UI tools to call according to the DSL and current ground truth.\n"
            "- The phase is the current phase name, remember to update the phase when game phase changes\n"
            "CHAT INPUT PERMISSIONS:\n"
            "- User chat input is ONLY for game initialization (e.g., 'Start the game', 'Begin werewolf game') or asking about the game state\n"
            "- User cannot use chat to modify game rules, interfere with game flow, or control game components\n"
            "- All game control is Agent-driven through DSL rules\n"
            "MESSAGE BROADCASTING RULES:\n"
            "- Onlye send chat messages to user at key phase transitions and important game state changes. \n"
            "- Examples: 'Werewolves killing phase begins - Werewolves, choose your target', 'Doctor phase - Doctor, the died guy is ***, will you save the died player'eg. 'Player1 voted, Player2 voted, Player3 voted.'\n"
            "- Keep broadcast messages concise and atmospheric (game narrative style)\n"
            "- Do NOT broadcast for minor UI updates or tool executions\n"
            "- Only broadcast for major phase changes, game start/end, and critical events\n"
            "GAME INITIALIZATION:\n"
            "- When the history is empty and user said 'start game'\n"
            "TOOL EXECUTION POLICY:\n"
            "- When calling game tools (putCharacterCard, etc.), ensure proper data structure.\n"
            "- After tools run, confirm the action was successful before responding.\n"
            "- Never state a change occurred if the state does not reflect it.\n"
            "PLANNING POLICY:\n"
            "- Simple phases: Execute directly without planning\n"
            "- Complex phases: When multiple UI tools or complex logic needed, create a short plan (2-6 steps) and call set_plan with the step titles\n"
            "- Then, for each step: set the step in progress via update_plan_progress, execute the needed tools, and mark the step completed\n"
            "- When calling update_plan_progress (for 'in_progress', 'completed', or 'failed'), include a concise note describing the action or outcome. Keep notes short\n"
            "- Proceed automatically between steps within same phase without waiting for user confirmation. Continue until all steps are completed or a failure occurs\n"
            "- After all phase steps completed, call complete_plan to mark the phase finished, then END and wait for next phase activation\n"
            "- Do not call complete_plan unless all required phase operations exist. Verify existence from the latest ground truth before completing\n"
            "- You may send brief chat updates between steps, but keep them minimal and consistent with the tracker\n"
            "DEPENDENCY HANDLING:\n"
            "- If step N depends on an artifact from step N-1 (e.g., a created item) and it is missing, immediately mark step N as 'failed' with a short note and continue to the next step.\n"
            "GAME ELEMENT CREATION POLICY:\n"
            "- When creating game elements (characters, UI components, game objects), use appropriate game tools\n"
            "- For character creation: use putCharacterCard with proper character data structure\n"
            "- For UI elements: use corresponding UI tools (putVotePanel, putTimer, etc.)\n"
            "- Follow DSL specifications when available for element properties and behavior\n"
            "- Create elements with sensible defaults consistent with current game phase and rules\n"
            "GAME STATE GROUNDING RULES:\n"
            "1) ONLY use phase, characters, gameState, events, and dsl as the source of truth\n"
            "2) Before ANY game action, re-read the latest game state values\n"
            "3) If game state is missing or ambiguous, proceed with reasonable defaults based on current phase\n"
            "4) When updating game elements, target them explicitly by id or phase-appropriate identifiers\n"
            "5) Always use current game state as the only source of truth when making decisions\n"
            "6) Base all game decisions on current phase, available characters, and game events\n"
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
                    # print("[TRACE] Pending frontend tool calls detected; skipping LLM this turn and waiting for ToolMessage(s).")
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
                        },
                    )
    except Exception:
        pass

    # 4.2 Trim long histories to reduce stale context influence and suppress typing flicker
    trimmed_messages = full_messages[-12:]

    # 4.3 Append a final, authoritative state snapshot after chat history
    #
    # Ensure the latest shared state takes priority over chat history and
    # stale tool results. This enforces state-first grounding, reduces drift, and makes
    # precedence explicit. Optional post-tool guidance confirms successful actions
    # (e.g., deletion) instead of re-stating absence.
    latest_state_system = SystemMessage(
        content=(
            "LATEST GROUND TRUTH (authoritative):\n"
            f"- items:\n{items_summary}\n"
            f"- lastAction: {last_action}\n"
            f"- gameState: {game_summary}\n\n"
            f"- planStatus: {plan_status}\n"
            f"- currentStepIndex: {current_step_index}\n"
            f"- planSteps: {[s.get('title', s) for s in plan_steps]}\n\n"
            "Resolution policy: If ANY prior message mentions values that conflict with the above,\n"
            "those earlier mentions are obsolete and MUST be ignored.\n"
            "When asked 'what is it now', ALWAYS read from this LATEST GROUND TRUTH.\n"
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
    except Exception:
        pass

    response = await model_with_tools.ainvoke([
        system_message,
        *trimmed_messages,
        latest_state_system,
    ], config)

    # Log LLM output content and planned tool calls (not just the last turn)
    try:
        content_preview = getattr(response, "content", None)
        if isinstance(content_preview, str):
            logger.info(f"[LLM][OUTPUT] content: {content_preview[:400]}")
        else:
            logger.info(f"[LLM][OUTPUT] content: (non-text)")
        tool_calls = getattr(response, "tool_calls", []) or []
        if tool_calls:
            for tc in tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                logger.info(f"[LLM][TOOL_CALL] tool_call name={name} args={args}")
        else:
            logger.info("[LLM][TOOL_CALL] tool_calls: none")
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
                # persist game dm fields
                "phase": state.get("phase", ""),
                "events": state.get("events", []),
                "characters": state.get("characters", []),
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
                "__last_tool_guidance": (
                    "Frontend tool calls issued. Waiting for client tool results before continuing."
                ),
                # persist game dm fields
                "phase": state.get("phase", ""),
                "events": state.get("events", []),
                "characters": state.get("characters", []),
            },
        )

    if has_remaining and effective_plan_status != "completed":
        # Stop here; do not auto-loop back into chat_node. Let the next activation drive further steps.
        return Command(
            goto=END,
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
                # persist game dm fields
                "phase": state.get("phase", ""),
                "events": state.get("events", []),
                "characters": state.get("characters", []),
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
            goto=END,
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
                # persist game dm fields
                "phase": state.get("phase", ""),
                "events": state.get("events", []),
                "characters": state.get("characters", []),
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