"""
Simple prompt template loader for game agent system prompts.
"""
import asyncio
from pathlib import Path

async def _load_prompt_async(template_name: str) -> str:
    """Async version of prompt loading using asyncio.to_thread."""
    def _sync_load():
        prompt_dir = Path(__file__).parent
        template_path = prompt_dir / f"{template_name}.txt"
        
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")
        
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    return await asyncio.to_thread(_sync_load)

def load_prompt(template_name: str) -> str:
    """Load a prompt template from the prompt directory."""
    prompt_dir = Path(__file__).parent
    template_path = prompt_dir / f"{template_name}.txt"
    
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()