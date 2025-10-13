"""
Minimal DM agent (two-LLM-node graph):
- chat1: decides if current phase is finished; if yes, directly updates state.phase (no backend tool); if not, pass reasons to chat2 via state without rendering to frontend
- chat2: performs follow-up reasoning/execution based on chat1's handoff; ends the run without rendering by default

Notes:
- Internal hops use messages: [] to avoid frontend rendering; only explicit END with AIMessage would render
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, BaseMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from typing_extensions import Literal

from langgraph.graph import StateGraph, END
from langgraph.types import Command


class DMState(dict):
    """Simple state container.

    Keys we rely on (all optional):
    - messages: List[BaseMessage]
    - phase: str
    - dsl: dict
    - handoff: dict | None (internal data passed from chat1 to chat2)
    """


# ------------------------------ Nodes -------------------------------

async def chat1(state: DMState, config: RunnableConfig) -> Command[Literal["chat2", "__end__"]]:
    """LLM node 1 - decide phase completion and route.

    - If phase is complete: directly update state.phase (no tool) and END
    - Else: write a succinct handoff note for chat2 (no frontend render) → goto chat2
    """
    model = init_chat_model("openai:gpt-4o")

    # build prompts (structured output expectation)
    phase = str(state.get("phase", "")).strip()
    dsl_present = bool(state.get("dsl"))
    system_content = (
        "You are DM chat1. Goals:\n"
        "1) Decide if the current phase is complete based on latest messages and state\n"
        "2) If complete, OUTPUT JSON ONLY: {\"transition\": true, \"next_phase\": <string>, \"note\": <short>}\n"
        "3) If not complete, OUTPUT JSON ONLY: {\"transition\": false, \"note\": <short why not>}\n"
        "No extra text, JSON only.\n"
        f"Current phase: {phase or '(unset)'}\n"
        f"DSL loaded: {dsl_present}\n"
        "When transition is valid, do NOT call tools. We will update state ourselves.\n"
    )

    messages: List[BaseMessage] = list(state.get("messages", []) or [])
    response = await model.ainvoke([SystemMessage(content=system_content), *messages], config)

    # Parse JSON-only content
    import json
    raw = str(getattr(response, "content", "")).strip()
    data: Dict[str, Any]
    try:
        data = json.loads(raw) if raw else {}
    except Exception:
        data = {"transition": False, "note": raw[:120]}

    transition = bool(data.get("transition"))
    next_phase = str(data.get("next_phase", "")).strip()
    note = str(data.get("note", "")).strip()

    if transition and next_phase:
        # Directly update phase; do not render to frontend
        return Command(
            goto=END,
            update={
                "messages": [],
                "phase": next_phase,
                "handoff": None,
                "dsl": state.get("dsl", None),
            },
        )

    # Otherwise set internal handoff for chat2; do not render to frontend
    handoff_note = note or raw
    return Command(
        goto="chat2",
        update={
            "messages": [],
            "handoff": {"note": handoff_note} if handoff_note else {"note": "pending"},
            "phase": state.get("phase", ""),
            "dsl": state.get("dsl", None),
        },
    )


async def chat2(state: DMState, config: RunnableConfig) -> Command[Literal["__end__"]]:
    """LLM node 2 - act upon chat1's handoff (internal), then END.

    By default we do not render to frontend (messages: []). If you want to broadcast,
    change messages to include an AIMessage with a summary.
    """
    model = init_chat_model("openai:gpt-4o")
    handoff = state.get("handoff", {}) or {}
    system_content = (
        "You are DM chat2. You receive a short internal handoff note from chat1\n"
        "and should perform minimal follow-up reasoning. Keep outputs minimal.\n"
        f"Handoff: {handoff}\n"
        "Do not call any tools. Return a very short acknowledgement.\n"
    )

    messages: List[BaseMessage] = list(state.get("messages", []) or [])
    response = await model.ainvoke([SystemMessage(content=system_content), *messages], config)

    # End run without rendering to frontend by default (messages: [])
    return Command(
        goto=END,
        update={
            "messages": [],
            # clear handoff after consumption
            "handoff": None,
            "phase": state.get("phase", ""),
            "dsl": state.get("dsl", None),
            # uncomment the following to render a brief message
            # "messages": [AIMessage(content=str(getattr(response, "content", "")).strip())],
        },
    )


# ----------------------------- Workflow -----------------------------

workflow = StateGraph(DMState)
workflow.add_node("chat1", chat1)
workflow.add_node("chat2", chat2)

# Entry point
workflow.set_entry_point("chat1")

graph = workflow.compile()


