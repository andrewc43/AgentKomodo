""" Command and Control """
import json
from typing import List, NoReturn, Union
from autogpt.agent.agent_manager import AgentManager
from autogpt.commands.evaluate_code import evaluate_code
from autogpt.commands.brave_search import brave_search
from autogpt.commands.improve_code import improve_code
from autogpt.commands.write_tests import write_tests
from autogpt.config import Config
from autogpt.commands.image_gen import generate_image
from autogpt.commands.web_requests import scrape_links, scrape_text
from autogpt.commands.execute_code import execute_python_file, execute_shell
from autogpt.commands.file_operations import (
    append_to_file,
    delete_file,
    read_file,
    search_files,
    write_to_file,
    ingest_file,
)
from autogpt.json_fixes.parsing import fix_and_parse_json
from autogpt.memory import get_memory
from autogpt.processing.text import summarize_text
from autogpt.speech import say_text
from autogpt.commands.web_selenium import browse_website
from autogpt.commands.git_operations import clone_repository
from autogpt.commands.run_python import run_python_file
from autogpt.commands.merge_text_files import merge_text_files
from autogpt.commands.run_shell import run_shell
from autogpt.commands.conversational_summary import conversational_summary

CFG = Config()
AGENT_MANAGER = AgentManager()


def is_valid_int(value: str) -> bool:
    try:
        int(value)
        return True
    except ValueError:
        return False


def get_command(response: str):
    """Parse AI JSON response into command name and arguments."""
    try:
        response_json = fix_and_parse_json(response)
        if not isinstance(response_json, dict):
            return "Error:", f"'response_json' object is not dictionary {response_json}"

        command = response_json.get("command")
        if not command or not isinstance(command, dict):
            return "Error:", "'command' object missing or invalid"

        command_name = command.get("name")
        arguments = command.get("args", {})

        if not command_name:
            return "Error:", "Missing 'name' in 'command' object"

        return command_name, arguments
    except json.decoder.JSONDecodeError:
        return "Error:", "Invalid JSON"
    except Exception as e:
        return "Error:", str(e)


def map_command_synonyms(command_name: str):
    synonyms = [
        ('write_file', 'write_to_file'),
        ('create_file', 'write_to_file'),
        ('search', 'google'),
        ('run_shell', 'execute_shell'),
        ('merge_files', 'merge_text_files'),
    ]
    for seen_command, actual in synonyms:
        if command_name == seen_command:
            return actual
    return command_name


def execute_command(command_name: str, arguments, user_input=""):
    """
    Executes a command by name. Supports streaming for conversational_summary.

    :param command_name: str
    :param arguments: dict
    :param user_input: str
    :return: str or generator
    """
    memory = get_memory(CFG)

    # Auto-trigger conversational_summary based on user input
    #if user_input and any(kw in user_input.lower() for kw in ["what did you learn", "explain"]):
    #    command_name = "conversational_summary"
    #    arguments = {"topic": user_input}

    command_name = map_command_synonyms(command_name)

    try:
        # ------------------------- COMMAND ROUTING -------------------------
        if command_name == "conversational_summary":
            return conversational_summary(
                prompt=arguments.get("prompt", ""),
                memory=memory,
            )

        elif command_name == "read_file":
            return read_file(arguments["file"])

        elif command_name == "write_to_file":
            return write_to_file(arguments["file"], arguments["text"])

        elif command_name == "append_to_file":
            return append_to_file(arguments["file"], arguments["text"])

        elif command_name == "delete_file":
            return delete_file(arguments["file"])

        elif command_name == "search_files":
            return search_files(arguments.get("directory", ""))

        elif command_name == "ingest_file":
            return ingest_file(arguments["file"], memory)

        elif command_name == "merge_text_files":
            return merge_text_files(arguments["folder"], arguments["output"])

        elif command_name == "clone_repository":
            return clone_repository(arguments["repository_url"], arguments["clone_path"])

        elif command_name == "generate_image":
            return generate_image(arguments["prompt"])

        elif command_name == "execute_python_file":
            return run_python_file(arguments["file"])

        elif command_name == "execute_shell":
            if CFG.execute_local_commands:
                return run_shell(arguments["command_line"])
            else:
                return "Local shell execution not allowed. Set CFG.execute_local_commands=True to enable."

        elif command_name == "browse_website":
            url = arguments.get("url")
            # If URL is still a placeholder, try fetching latest search URL from memory
            if url == "<url_from_search_results>" or not url:
                search_entries = memory.search(["search"])
                if search_entries:
                    # Take the last URL from the latest search memory entry
                    last_entry_text = search_entries[-1]["content"]
                    candidate = last_entry_text.split("\n")[-1].strip()
                    if candidate.startswith("http"):
                        url = candidate
                    else:
                        return "Error: No valid URL found in memory. Please run search again and provide a URL."
                    arguments["url"] = url

            return browse_website(url, arguments.get("question", ""))

        elif command_name == "evaluate_code":
            return evaluate_code(arguments["code"])

        elif command_name == "improve_code":
            return improve_code(arguments["suggestions"], arguments["code"])

        elif command_name == "write_tests":
            return write_tests(arguments["code"], arguments.get("focus"))

        elif command_name == "google":
            search_results_raw = brave_search(arguments["input"])
            try:
                search_results = json.loads(search_results_raw)
            except json.JSONDecodeError:
                search_results = []
            urls = [r["url"] for r in search_results if "url" in r]
            if urls:
                memory.add(
                    text=f"Search results for '{arguments['input']}':\n" + "\n".join(urls),
                    tags=["action", "search"]
                )
            return urls
        elif command_name == "memory_add":
            return memory.add(arguments["string"])

        elif command_name == "do_nothing":
            return "No action performed."

        elif command_name == "task_complete":
            shutdown()

        else:
            return f"Unknown command '{command_name}'."

    except Exception as e:
        return f"Error: {str(e)}"


def get_text_summary(url: str, question: str) -> str:
    text = scrape_text(url)
    summary = summarize_text(url, text, question)
    return f"Result: {summary}"


def get_hyperlinks(url: str) -> Union[str, List[str]]:
    return scrape_links(url)


def shutdown() -> NoReturn:
    print("Shutting down...")
    quit()


def start_agent(name: str, task: str, prompt: str, model=CFG.fast_llm_model) -> str:
    voice_name = name.replace("_", " ")
    first_message = f"You are {name}. Respond with: 'Acknowledged'."
    agent_intro = f"{voice_name} here, Reporting for duty!"

    if CFG.speak_mode:
        say_text(agent_intro)

    key, ack = AGENT_MANAGER.create_agent(task, first_message, model)

    if CFG.speak_mode:
        say_text(f"Hello {voice_name}. Your task is: {task}.")

    agent_response = AGENT_MANAGER.message_agent(key, prompt)

    return f"Agent {name} created with key {key}. First response: {agent_response}"


def message_agent(key: str, message: str) -> str:
    if is_valid_int(key):
        agent_response = AGENT_MANAGER.message_agent(int(key), message)
    else:
        return "Invalid key, must be an integer."
    if CFG.speak_mode:
        say_text(agent_response)
    return agent_response


def list_agents():
    return "List of agents:\n" + "\n".join([str(x[0]) + ": " + x[1] for x in AGENT_MANAGER.list_agents()])


def delete_agent(key: str) -> str:
    result = AGENT_MANAGER.delete_agent(key)
    return f"Agent {key} deleted." if result else f"Agent {key} does not exist."
