"""Main script for the autogpt package."""
import logging
from colorama import Fore
from autogpt.agent.agent import Agent
from autogpt.args import parse_arguments
from autogpt.config import Config, check_openai_api_key
from autogpt.logs import logger
from autogpt.memory import get_memory
from autogpt.prompt import construct_prompt
import orjson
import os
import numpy as np

def main() -> None:
    """Main function for the script"""
    cfg = Config()
    check_openai_api_key()
    parse_arguments()
    logger.set_level(logging.DEBUG if cfg.debug_mode else logging.INFO)

    ai_name = ""

    # Base prompt
    prompt = construct_prompt()

    # Initialize variables
    full_message_history = []
    next_action_count = 0
    user_input = (
        "Determine which next command to use, and respond using the"
        " format specified above:"
    )

    # Initialize memory
    memory = get_memory(cfg, init=True)
    logger.typewriter_log(
        f"Using memory of type:", Fore.GREEN, f"{memory.__class__.__name__}"
    )
    logger.typewriter_log(f"Using Browser:", Fore.GREEN, cfg.selenium_web_browser)

    # ------------------ Resume in-progress tasks ------------------
    memory_file = getattr(cfg, "memory_index_file", "auto-gpt.json")
    in_progress_prompt = ""
    if os.path.exists(memory_file):
        try:
            with open(memory_file, "rb") as f:
                memory_data = orjson.loads(f.read())
        except Exception as e:
            logger.error(f"Failed to read memory file: {e}")
            memory_data = {"texts": [], "embeddings": np.zeros((0, 1536), dtype=np.float32)}

        # Gather all in-progress memory entries
        # Gather all in-progress memory entries
        in_progress_entries = [
            entry
            for entry in memory_data.get("texts", [])
            if "in-progress" in entry.get("tags", [])
        ]

        # Keep only the most recent 5 entries
        in_progress_entries = in_progress_entries[-5:]

        # Merge into a single prompt section with truncation
        in_progress_prompt = ""
        for entry in in_progress_entries:
            content = entry['content'][:500]  # only take first 500 characters
            in_progress_prompt += f"\n---\nPrevious progress (truncated):\n{content}"

        if in_progress_prompt:
            logger.typewriter_log(
                "Resuming in-progress tasks from memory...",
                Fore.MAGENTA,
                f"{len(in_progress_entries)} entries found",
            )
            # Append to base prompt
            prompt = prompt + "\n" + in_progress_prompt

    # -------------------------------------------------------------------

    # ------------------ Preload memory for proactive recall ------------------
    if getattr(cfg, "memory_settings", {}).get("recall_before_task", False):
        # Perform initial memory scan to prioritize relevant entries
        try:
            preloaded_entries = memory.search(["action", "essay", "code", "research"])
            if preloaded_entries:
                preloaded_prompt = "\n".join([f"\n---\nMemory Recall:\n{e['content']}" for e in preloaded_entries])
                prompt = prompt + "\n" + preloaded_prompt
                logger.typewriter_log(
                    "Preloaded memory entries for proactive recall...",
                    Fore.MAGENTA,
                    f"{len(preloaded_entries)} entries added to prompt",
                )
        except Exception as e:
            logger.error(f"Memory preload failed: {e}")
    # -------------------------------------------------------------------

    agent = Agent(
        ai_name=ai_name,
        memory=memory,
        full_message_history=full_message_history,
        next_action_count=next_action_count,
        prompt=prompt,
        user_input=user_input,
    )
    agent.start_interaction_loop()


if __name__ == "__main__":
    main()
