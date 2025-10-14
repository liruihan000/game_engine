"""
Minimal three-node DSL generation agent using LangGraph.

Flow:
- declaration_node: generate `declaration` YAML and save immediately to games/generated.yaml
- phases_node: read YAML, generate `phases`, merge and save
- validation_node: read YAML, validate/correct full YAML, overwrite

Prompts are sourced from `agent/prompt/*.txt`. The initial game concept must be
provided via state input `game_description` from the frontend.
"""

from __future__ import annotations

import os
import re
import yaml
import logging
from typing import Any, Dict, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig


# Load environment variables (API keys, etc.) from agent/.env
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(_THIS_DIR, ".env")
load_dotenv(ENV_PATH)

# Monitoring configuration
VERBOSE_LOGGING = True  # Set to False to disable detailed logging

# 直接配置 logger，不依赖 basicConfig
logger = logging.getLogger('DSLAgent')
logger.handlers.clear()  # 清除现有 handlers

if VERBOSE_LOGGING:
    logger.setLevel(logging.INFO)
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    
    # 文件处理器 - 使用时间戳创建新的日志文件
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'/home/lee/canvas-with-langgraph-python/logs/dsl_agent_{timestamp}.log'
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

# Resolve paths relative to repository root (one level up from agent/)
ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
PROMPT_DIR = os.path.join(_THIS_DIR, "prompt")
GAMES_DIR = os.path.join(ROOT_DIR, "games")
DEFAULT_OUTPUT_PATH = os.path.join(GAMES_DIR, "generated.yaml")


class DSLState(TypedDict, total=False):
    # File path for persisted YAML
    yaml_path: str
    # Last YAML object (dict)
    dsl: Dict[str, Any]
    # Textual buffers (optional)
    game_description: str


async def _read_text_file(path: str) -> str:
    import asyncio
    return await asyncio.to_thread(lambda: open(path, "r", encoding="utf-8").read())


async def _ensure_games_dir() -> None:
    import asyncio
    await asyncio.to_thread(lambda: os.makedirs(GAMES_DIR, exist_ok=True))

async def _write_text_file(path: str, content: str) -> None:
    import asyncio
    await asyncio.to_thread(lambda: open(path, "w", encoding="utf-8").write(content))


def _extract_yaml(text: str) -> str:
    """Extract YAML by stripping common code fences if present."""
    if not isinstance(text, str):
        logger.warning("_extract_yaml: Input is not string")
        return ""
    
    text = text.strip()
    original_length = len(text)
    
    # Remove triple backtick fences if model added them
    fenced = re.compile(r"^```[a-zA-Z]*\n([\s\S]*?)\n```$", re.MULTILINE | re.DOTALL)
    m = fenced.search(text)
    if m:
        extracted = m.group(1).strip()
        logger.info(f"_extract_yaml: Found general code fence, extracted {len(extracted)} chars from {original_length}")
        return extracted
    
    # Also try to find YAML content between ```yaml and ```
    yaml_fenced = re.compile(r"```yaml\s*\n([\s\S]*?)\n```", re.MULTILINE | re.DOTALL)
    m = yaml_fenced.search(text)
    if m:
        extracted = m.group(1).strip()
        logger.info(f"_extract_yaml: Found YAML fence, extracted {len(extracted)} chars from {original_length}")
        return extracted
        
    # If no fences found, return the text as-is
    logger.info(f"_extract_yaml: No fences found, returning original text ({original_length} chars)")
    return text


def _safe_load_yaml(text: str) -> Dict[str, Any]:
    if not text:
        logger.warning("_safe_load_yaml: Empty text input")
        return {}
    
    try:
        data = yaml.safe_load(text)
        if data is None:
            logger.warning("_safe_load_yaml: yaml.safe_load returned None")
            return {}
        elif not isinstance(data, dict):
            logger.warning(f"_safe_load_yaml: Expected dict, got {type(data)}: {data}")
            return {}
        else:
            logger.info(f"_safe_load_yaml: Successfully parsed {len(data)} top-level keys")
            return data
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error: {e}")
        logger.error(f"Problematic YAML text (first 500 chars):\n{text[:500]}")
        logger.error(f"Problematic YAML text (last 500 chars):\n{text[-500:]}")
        # Return empty dict on error instead of crashing
        return {}


def _safe_dump_yaml(data: Dict[str, Any]) -> str:
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)


async def declaration_node(state: DSLState, config: RunnableConfig) -> Command[str]:
    logger.info("=== DECLARATION NODE START ===")
    logger.info(f"Initial state: {state}")
    
    prompt_path = os.path.join(PROMPT_DIR, "dsl_declaration_generation_prompt.txt")
    system_prompt = await _read_text_file(prompt_path)
    game_desc = state.get("game_description") or ""
    logger.info(f"Game description: {game_desc[:100]}...")

    model = init_chat_model("openai:gpt-5")
    response = await model.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            "Game concept (use it to produce only the declaration YAML):\n\n" + game_desc
        )),
    ], config)

    yaml_text = _extract_yaml(response.content or "")
    logger.info(f"Extracted YAML text length: {len(yaml_text)}")
    decl_obj = _safe_load_yaml(yaml_text)
    logger.info(f"Parsed declaration object keys: {list(decl_obj.keys()) if isinstance(decl_obj, dict) else 'Not a dict'}")

    # Persist immediately
    await _ensure_games_dir()
    yaml_path = state.get("yaml_path") or "generated.yaml"
    # If relative path, place in games directory
    if not os.path.isabs(yaml_path):
        path = os.path.join(GAMES_DIR, yaml_path)
    else:
        path = yaml_path
    logger.info(f"Target file path: {path}")
    
    # Ensure only `declaration` is stored at this step
    out: Dict[str, Any] = {}
    if isinstance(decl_obj, dict) and "declaration" in decl_obj:
        out["declaration"] = decl_obj["declaration"]
    else:
        out = decl_obj  # fallback if model already returned the full mapping
    
    logger.info(f"Writing declaration to file. Output keys: {list(out.keys())}")
    await _write_text_file(path, _safe_dump_yaml(out))
    
    # Verify what was written
    written_content = await _read_text_file(path)
    logger.info(f"Verification - file content length after write: {len(written_content)}")
    logger.info("=== DECLARATION NODE END ===")

    return Command(
        goto="phases",
        update={
            "yaml_path": path,
            "dsl": out,
        },
    )


async def phases_node(state: DSLState, config: RunnableConfig) -> Command[str]:
    logger.info("=== PHASES NODE START ===")
    logger.info(f"State received: {state}")
    
    yaml_path = state.get("yaml_path") or "generated.yaml"
    # If relative path, place in games directory
    if not os.path.isabs(yaml_path):
        path = os.path.join(GAMES_DIR, yaml_path)
    else:
        path = yaml_path
    logger.info(f"Reading from file path: {path}")
    logger.info(f"File exists: {os.path.exists(path)}")
    
    if os.path.exists(path):
        file_content_before = await _read_text_file(path)
        logger.info(f"File content before processing (length: {len(file_content_before)}):")
        logger.info(f"File content preview: {file_content_before[:200]}...")
        existing = _safe_load_yaml(file_content_before)
        logger.info(f"Existing YAML keys: {list(existing.keys()) if isinstance(existing, dict) else 'Not a dict'}")
        if isinstance(existing, dict) and "declaration" in existing:
            logger.info(f"Declaration found in existing file with keys: {list(existing['declaration'].keys()) if isinstance(existing['declaration'], dict) else 'Not a dict'}")
        else:
            logger.warning("No declaration found in existing file!")
    else:
        existing = {}
        logger.warning("File does not exist, using empty dict")

    prompt_path = os.path.join(PROMPT_DIR, "dsl_phases_generation_prompt.txt")
    system_prompt = await _read_text_file(prompt_path)

    # Provide declaration context explicitly
    declaration_block = _safe_dump_yaml({"declaration": existing.get("declaration", {})})
    logger.info(f"Declaration block being sent to model (length: {len(declaration_block)}):")
    logger.info(f"Declaration block preview: {declaration_block[:200]}...")

    model = init_chat_model("openai:gpt-5")
    response = await model.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            "Only output the phases mapping. Here is the declaration (context):\n\n" + declaration_block
        )),
    ], config)

    yaml_text = _extract_yaml(response.content or "")
    logger.info(f"Model response YAML length: {len(yaml_text)}")
    phases_obj = _safe_load_yaml(yaml_text)
    logger.info(f"Parsed phases object keys: {list(phases_obj.keys()) if isinstance(phases_obj, dict) else 'Not a dict'}")

    # Merge phases into existing structure
    if not isinstance(existing, dict):
        existing = {}
        logger.warning("Existing was not a dict, reset to empty dict")
    
    logger.info(f"Before merge - existing keys: {list(existing.keys())}")
    if "phases" in phases_obj:
        existing["phases"] = phases_obj["phases"]
        logger.info("Added phases from phases_obj['phases']")
    else:
        # If model returned phases at root (numeric keys), normalize under `phases`
        existing["phases"] = phases_obj
        logger.info("Added phases from root of phases_obj")
    
    logger.info(f"After merge - existing keys: {list(existing.keys())}")
    
    final_yaml = _safe_dump_yaml(existing)
    logger.info(f"Final YAML to write (length: {len(final_yaml)}):")
    logger.info(f"Final YAML preview: {final_yaml[:200]}...")
    
    await _write_text_file(path, final_yaml)
    
    # Verify what was actually written
    written_content = await _read_text_file(path)
    logger.info(f"Verification - file content after write (length: {len(written_content)}):")
    logger.info(f"Verification preview: {written_content[:200]}...")
    logger.info("=== PHASES NODE END ===")

    return Command(
        goto="validation",
        update={
            "dsl": existing,
        },
    )


async def validation_node(state: DSLState, config: RunnableConfig) -> Command[str]:
    logger.info("=== VALIDATION NODE START ===")
    logger.info(f"State received: {state}")
    
    yaml_path = state.get("yaml_path") or "generated.yaml"
    # If relative path, place in games directory
    if not os.path.isabs(yaml_path):
        path = os.path.join(GAMES_DIR, yaml_path)
    else:
        path = yaml_path
    logger.info(f"Reading from file path: {path}")
    logger.info(f"File exists: {os.path.exists(path)}")
    
    existing_text = await _read_text_file(path) if os.path.exists(path) else ""
    existing = _safe_load_yaml(existing_text)
    logger.info(f"File content before validation (length: {len(existing_text)}):")
    logger.info(f"File content preview: {existing_text[:200]}...")

    prompt_path = os.path.join(PROMPT_DIR, "dsl_validation_node_prompt.txt")
    system_prompt = await _read_text_file(prompt_path)

    model = init_chat_model("openai:gpt-5")
    response = await model.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            "Validate and correct this YAML; output the full corrected YAML only.\n\n" + existing_text
        )),
    ], config)

    yaml_text = _extract_yaml(response.content or "")
    logger.info(f"Model response YAML length: {len(yaml_text)}")
    validated = _safe_load_yaml(yaml_text)
    logger.info(f"Validated YAML keys: {list(validated.keys()) if isinstance(validated, dict) else 'Not a dict'}")

    # CRITICAL: If validation parsing failed, keep original content instead of overwriting with empty dict
    if not validated or not isinstance(validated, dict) or len(validated) == 0:
        logger.warning("Validation YAML parsing failed! Keeping original file content.")
        # Read existing content and keep it
        existing_content = await _read_text_file(path) if os.path.exists(path) else ""
        final_yaml = existing_content
        validated = existing
    else:
        # Overwrite with validated content
        await _ensure_games_dir()
        final_yaml = _safe_dump_yaml(validated)
        
    logger.info(f"Final validated YAML to write (length: {len(final_yaml)}):")
    logger.info(f"Final YAML preview: {final_yaml[:200]}...")
    
    await _write_text_file(path, final_yaml)
    
    # Verify final result
    final_content = await _read_text_file(path)
    logger.info(f"Final verification - file content (length: {len(final_content)}):")
    logger.info(f"Final content preview: {final_content[:200]}...")
    logger.info("=== VALIDATION NODE END ===")

    return Command(
        goto=END,
        update={
            "dsl": validated,
        },
    )


# Assemble the graph
workflow = StateGraph(DSLState)
workflow.add_node("declaration", declaration_node)
workflow.add_node("phases", phases_node)
workflow.add_node("validation", validation_node)
workflow.add_edge("declaration", "phases")
workflow.add_edge("phases", "validation")
workflow.set_entry_point("declaration")

graph = workflow.compile()


if __name__ == "__main__":
    import asyncio

    async def main():
        # Provide `game_description` from the frontend; file fallback removed
        initial: DSLState = {
            "yaml_path": DEFAULT_OUTPUT_PATH,
            # "game_description": "Your game concept here",
        }
        await graph.ainvoke(initial)
        print(f"Saved DSL to: {DEFAULT_OUTPUT_PATH}")

    asyncio.run(main())


