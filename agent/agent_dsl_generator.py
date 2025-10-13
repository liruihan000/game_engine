#!/usr/bin/env python3
"""
DSL Generator Agent
Converts natural language game descriptions into structured YAML DSL files.
"""
import os
import sys
import yaml
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TypedDict, Annotated, Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# ========== Setup Logging ==========
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"dsl_generator_{timestamp}.log"

logger = logging.getLogger("DSLGenerator")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
logger.addHandler(file_handler)
logger.info(f"Logging to: {log_file}")

# ========== State Definition ==========
class DSLGeneratorState(TypedDict):
    """State for DSL Generator Agent"""
    messages: Annotated[list, add_messages]
    game_description: str  # User's full game description
    dsl_content: dict      # Generated DSL structure
    refinement_count: int  # Number of refinement iterations
    status: str            # "collecting", "generating", "refining", "complete"

# ========== DSL Generator Node ==========
async def dsl_generator_node(state: DSLGeneratorState):
    """
    Main node that generates DSL from user description.
    """
    logger.info("[DSLGenerator][start] ==== Generating DSL ====")
    
    game_desc = state.get("game_description", "")
    current_dsl = state.get("dsl_content", {})
    status = state.get("status", "collecting")
    
    # Initialize LLM
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.3,
        streaming=True
    )
    
    # Build system prompt
    system_prompt = SystemMessage(
        content=(
            "You are an expert game designer assistant that converts natural language game descriptions into structured YAML DSL.\n\n"
            "DSL STRUCTURE SPECIFICATION:\n"
            "```yaml\n"
            "0:  # Phase ID (integer, starting from 0)\n"
            "  name: \"Phase Name\"\n"
            "  description: \"Detailed description of what happens in this phase\"\n"
            "  actions:\n"
            "    - description: \"Clear previous phase UI elements\"\n"
            "      tools: [\"clear_canvas\"]  # Always first action in each phase\n"
            "    - description: \"What DM does next\"\n"
            "      tools: [\"tool_name1\", \"tool_name2\"]  # Can use multiple tools\n"
            "  completion_criteria:\n"
            "    type: \"timer\" | \"player_action\" | \"final\"\n"
            "    # For timer:\n"
            "    duration_seconds: 30\n"
            "    # For player_action:\n"
            "    wait_for: \"single_player_choice\" | \"all_players_vote\" | \"all_players_speak\"\n"
            "    valid_actions: [\"action1\", \"action2\"]\n"
            "  next_phase:\n"
            "    id: 1\n"
            "    name: \"Next Phase Name\"\n"
            "  logic:  # Optional, for branching\n"
            "    - if: \"condition description\"\n"
            "      next_phase:\n"
            "        id: 2\n"
            "        name: \"Alternative Phase\"\n"
            "```\n\n"
            "AVAILABLE TOOLS:\n"
            "- phase_indicator: Display current phase at top-center\n"
            "- text_display: Show narrative text\n"
            "- character_card: Display character info\n"
            "- action_button: Create clickable buttons\n"
            "- vote_panel: Voting interface\n"
            "- timer_display: Countdown timer\n"
            "- status_indicator: Show game status\n"
            "- player_list: Display player roster\n"
            "- clear_canvas: Remove all UI elements\n"
            "- deleteItem: Remove specific UI element\n\n"
            "DESIGN PRINCIPLES:\n"
            "1. Each phase should have atomic, clear actions\n"
            "2. First action in each phase should clear previous UI (except phase 0)\n"
            "3. Each phase requests player feedback at most ONCE\n"
            "4. Game end is a separate final phase\n"
            "5. Use appropriate completion_criteria for each phase\n"
            "6. Phase IDs should be sequential integers starting from 0\n\n"
            "YOUR TASK:\n"
            "- Analyze the game description\n"
            "- Break it down into logical phases\n"
            "- For each phase, define: name, description, actions (with tools), completion_criteria, next_phase\n"
            "- Add branching logic if the game requires it\n"
            "- Output ONLY valid YAML that follows the specification above\n"
            "- Do NOT include any explanations or markdown code fences, just the YAML content\n"
        )
    )
    
    # Get messages history
    messages = state.get("messages", [])
    
    # Build prompt
    if status == "collecting" or status == "generating":
        user_prompt = f"Generate a complete game DSL for the following game:\n\n{game_desc}"
    else:  # refining
        user_prompt = f"Current DSL:\n```yaml\n{yaml.dump(current_dsl, allow_unicode=True, sort_keys=False)}```\n\nUser feedback: {messages[-1].content if messages else 'Please refine the DSL.'}"
    
    # Call LLM
    response = await model.ainvoke([system_prompt, HumanMessage(content=user_prompt)])
    
    logger.info(f"[DSLGenerator][LLM] Generated DSL length: {len(response.content)} chars")
    
    # Parse YAML
    try:
        dsl_dict = yaml.safe_load(response.content)
        logger.info(f"[DSLGenerator][SUCCESS] Parsed DSL with {len(dsl_dict)} phases")
        
        return {
            "messages": [response],
            "dsl_content": dsl_dict,
            "status": "complete"
        }
    except yaml.YAMLError as e:
        logger.error(f"[DSLGenerator][ERROR] YAML parsing failed: {e}")
        return {
            "messages": [AIMessage(content=f"Failed to generate valid YAML. Error: {e}\n\nPlease describe your game again.")],
            "status": "collecting"
        }

# ========== Save DSL Node ==========
async def save_dsl_node(state: DSLGeneratorState):
    """
    Save generated DSL to file.
    """
    logger.info("[SaveDSL][start] ==== Saving DSL ====")
    
    dsl_content = state.get("dsl_content", {})
    if not dsl_content:
        return {
            "messages": [AIMessage(content="No DSL to save. Please generate a DSL first.")]
        }
    
    # Ask for filename
    messages = state.get("messages", [])
    last_message = messages[-1].content if messages else ""
    
    # Extract filename from message or generate default
    if "save as" in last_message.lower() or "filename" in last_message.lower():
        # Try to extract filename
        filename = last_message.lower().replace("save as", "").replace("filename", "").strip()
        filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
        if not filename.endswith(".yaml"):
            filename += ".yaml"
    else:
        # Generate default filename
        game_name = dsl_content.get(0, {}).get("name", "game").lower().replace(" ", "_")
        filename = f"{game_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    
    # Save to games directory
    games_dir = Path(__file__).parent.parent / "games"
    games_dir.mkdir(exist_ok=True)
    file_path = games_dir / filename
    
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(dsl_content, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    
    logger.info(f"[SaveDSL][SUCCESS] Saved to {file_path}")
    
    return {
        "messages": [AIMessage(content=f"✅ DSL saved to: `{file_path}`\n\nYou can now use this file with the game agent!")]
    }

# ========== Build Graph ==========
def build_graph():
    """Build the DSL generator graph."""
    workflow = StateGraph(DSLGeneratorState)
    
    # Add nodes
    workflow.add_node("generate_dsl", dsl_generator_node)
    workflow.add_node("save_dsl", save_dsl_node)
    
    # Define edges
    workflow.add_edge(START, "generate_dsl")
    
    # Conditional edge: if user wants to save, go to save_dsl, otherwise end
    def should_save(state: DSLGeneratorState) -> Literal["save_dsl", "__end__"]:
        messages = state.get("messages", [])
        last_msg = messages[-1].content.lower() if messages else ""
        if "save" in last_msg:
            return "save_dsl"
        return END
    
    workflow.add_conditional_edges(
        "generate_dsl",
        should_save,
        {
            "save_dsl": "save_dsl",
            END: END
        }
    )
    workflow.add_edge("save_dsl", END)
    
    # Compile
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    
    logger.info("[Graph] DSL Generator graph compiled successfully")
    return graph

# ========== CLI Interface ==========
async def interactive_mode():
    """Run interactive CLI mode."""
    graph = build_graph()
    
    print("=" * 60)
    print("DSL Generator Agent - Interactive Mode")
    print("=" * 60)
    print("\nDescribe your game in natural language, and I'll generate a DSL for you.")
    print("Commands:")
    print("  - Type your game description")
    print("  - 'refine' to make changes to the generated DSL")
    print("  - 'save' or 'save as <filename>' to save the DSL")
    print("  - 'quit' to exit")
    print("=" * 60)
    
    config = {"configurable": {"thread_id": "dsl_gen_1"}}
    state = {
        "messages": [],
        "game_description": "",
        "dsl_content": {},
        "refinement_count": 0,
        "status": "collecting"
    }
    
    while True:
        user_input = input("\n👤 You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\n👋 Goodbye!")
            break
        
        # Update state
        state["messages"].append(HumanMessage(content=user_input))
        
        if state["status"] == "collecting":
            state["game_description"] = user_input
            state["status"] = "generating"
        
        # Run graph
        try:
            result = await graph.ainvoke(state, config)
            
            # Get last AI message
            ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
            if ai_messages:
                last_ai_msg = ai_messages[-1].content
                print(f"\n🤖 Agent: {last_ai_msg}")
            
            # Update state
            state = result
            
            # Show DSL preview if generated
            if result.get("dsl_content") and result.get("status") == "complete":
                print("\n📄 Generated DSL Preview:")
                print("─" * 60)
                dsl_yaml = yaml.dump(result["dsl_content"], allow_unicode=True, sort_keys=False)
                print(dsl_yaml[:500] + ("..." if len(dsl_yaml) > 500 else ""))
                print("─" * 60)
                print("\nType 'save' to save this DSL, or provide feedback to refine it.")
        
        except Exception as e:
            logger.error(f"[Error] {e}", exc_info=True)
            print(f"\n❌ Error: {e}")

# ========== Main Entry ==========
if __name__ == "__main__":
    import asyncio
    
    if len(sys.argv) > 1:
        # Direct mode: generate from command line argument
        game_description = " ".join(sys.argv[1:])
        print(f"Generating DSL for: {game_description}")
        
        graph = build_graph()
        config = {"configurable": {"thread_id": "dsl_gen_cli"}}
        state = {
            "messages": [HumanMessage(content=game_description)],
            "game_description": game_description,
            "dsl_content": {},
            "refinement_count": 0,
            "status": "generating"
        }
        
        result = asyncio.run(graph.ainvoke(state, config))
        
        # Print result
        if result.get("dsl_content"):
            print("\n" + "=" * 60)
            print("Generated DSL:")
            print("=" * 60)
            print(yaml.dump(result["dsl_content"], allow_unicode=True, sort_keys=False))
    else:
        # Interactive mode
        asyncio.run(interactive_mode())

