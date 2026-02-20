"""
Conversational Summary Command for Auto-GPT v0.6.44
Handles user questions based on memory and streams output manually.
"""

from autogpt.config import Config
from autogpt.chat import chat_with_ai
from autogpt.logs import logger
import time

cfg = Config()

def conversational_summary(prompt: str, memory=None, full_message_history=None, conversational_mode: bool = True) -> str:
    """
    Generate a conversational summary response based on the agent's memory.

    Args:
        prompt (str): The user's natural question or request.
        memory: The memory provider object used by the agent.
        full_message_history: Optional chat history to include in context.
        conversational_mode: If True, only summarize from memory without triggering file operations.

    Returns:
        str: The generated conversational response.
    """
    if memory is None:
        logger.error("Memory object is required for conversational_summary.")
        return "Error: No memory available."

    # ------------------ Gather relevant memory ------------------
    try:
        relevant_entries = memory.get_relevant(prompt, k=10)  # v0.6.44 compatible
    except Exception as e:
        logger.error(f"Error fetching relevant memory: {e}")
        relevant_entries = []

    context_text = "\n".join([entry["content"] for entry in relevant_entries])

    if not context_text:
        context_text = "I have no prior knowledge of this topic. Answering based on general knowledge.\n"

    # ------------------ Prepare the AI prompt ------------------
    conversation_prompt = (
        f"You are the assistant. Use the following knowledge to answer the user question in a conversational way.\n\n"
        f"Knowledge Base:\n{context_text}\n\n"
        f"User Question: {prompt}\n\n"
        f"Answer conversationally and in detail:"
    )

    # ------------------ Generate response ------------------
    print("\n[Agent starts responding...]")
    try:
        # v0.6.44 compatible: chat_with_ai does not accept `memory` or `stream`
        response = chat_with_ai(
            prompt=conversation_prompt,
            user_input=prompt,
            full_message_history=full_message_history or [],
            permanent_memory=memory,
            token_limit=cfg.fast_token_limit
        )

        # ------------------ Simulate streaming output ------------------
        print(response, flush=True)
        final_output = response

        print("\n[Response complete]\n")
        return final_output

    except Exception as e:
        logger.error(f"Error generating conversational summary: {e}")
        return f"Error generating conversational summary: {e}"
