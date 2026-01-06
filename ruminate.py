#!/usr/bin/env python3
"""
Rumination Engine - An input-less LLM thought experiment
"""

import os
import time
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()
from mem0 import Memory

# === Configuration ===
THOUGHT_INTERVAL = 60  # seconds between thoughts
USER_ID = "ruminator"  # isolated mem0 identity
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL")  # None = default
MAX_TOKENS = 1024

# === Logging Setup ===
log_dir = Path("/app/logs")
log_dir.mkdir(exist_ok=True)

# Rolling log file
log_file = log_dir / f"thoughts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Console output
    ]
)
logger = logging.getLogger("ruminate")

# === Mem0 Configuration ===
mem0_config = {
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": f"bolt://{os.environ.get('NEO4J_HOST', 'neo4j')}:7687",
            "username": "neo4j",
            "password": "ruminate123",
        }
    },
    "vector_store": {
        "provider": "pgvector",
        "config": {
            "host": os.environ.get("POSTGRES_HOST", "postgres"),
            "port": 5432,
            "user": "ruminate",
            "password": "ruminate",
            "dbname": "mem0",
        }
    },
    "llm": {
        "provider": "anthropic",
        "config": {
            "model": MODEL,
            "api_key": os.environ.get("ANTHROPIC_API_KEY"),
            **({"base_url": ANTHROPIC_BASE_URL} if ANTHROPIC_BASE_URL else {}),
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "api_key": os.environ.get("OPENAI_API_KEY"),
        }
    }
}


def load_system_prompt() -> str:
    """Load personality/system prompt from file."""
    prompt_path = Path("/app/prompt.txt")
    if prompt_path.exists():
        return prompt_path.read_text().strip()
    return "You are a mind, thinking."


def build_context(memory: Memory, last_thought: str | None) -> tuple[str, list[dict]]:
    """Build the message context from memories and last thought."""
    
    # Retrieve relevant memories
    memories = memory.get_all(user_id=USER_ID, limit=20)
    
    memory_text = ""
    if memories and memories.get("results"):
        memory_text = "\n\nYour memories:\n"
        for mem in memories["results"]:
            memory_text += f"- {mem['memory']}\n"
    
    system = load_system_prompt() + memory_text
    
    messages = []
    
    if last_thought:
        # Continue from last thought
        messages.append({
            "role": "assistant", 
            "content": last_thought
        })
        messages.append({
            "role": "user",
            "content": "[Continue thinking]"
        })
    else:
        # First thought - need initial seed
        messages.append({
            "role": "user",
            "content": "[Begin]"
        })
    
    return system, messages


def think(client: anthropic.Anthropic, memory: Memory, last_thought: str | None) -> str:
    """Generate a single thought."""
    
    system, messages = build_context(memory, last_thought)
    
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=messages
    )
    
    thought = response.content[0].text
    
    # Store in memory (let mem0 decide what's worth keeping)
    memory.add(thought, user_id=USER_ID)
    
    return thought


def main():
    """Main rumination loop."""
    
    logger.info("=" * 60)
    logger.info("RUMINATION ENGINE STARTING")
    logger.info("=" * 60)
    
    # Initialize clients
    client = anthropic.Anthropic(base_url=ANTHROPIC_BASE_URL)
    memory = Memory.from_config(mem0_config)
    
    logger.info(f"Model: {MODEL}")
    logger.info(f"Interval: {THOUGHT_INTERVAL}s")
    logger.info(f"Identity: {USER_ID}")
    logger.info("=" * 60)
    
    last_thought = None
    thought_count = 0
    
    try:
        while True:
            thought_count += 1
            logger.info(f"\n--- Thought #{thought_count} ---")
            
            thought = think(client, memory, last_thought)
            
            # Log the thought (both console and file)
            logger.info(f"\n{thought}\n")
            
            last_thought = thought
            
            # Wait before next thought
            time.sleep(THOUGHT_INTERVAL)
            
    except KeyboardInterrupt:
        logger.info("\n" + "=" * 60)
        logger.info("RUMINATION ENDED")
        logger.info(f"Total thoughts: {thought_count}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
