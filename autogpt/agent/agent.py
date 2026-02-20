from colorama import Fore, Style
from autogpt.app import execute_command, get_command
from datetime import datetime
from autogpt.chat import chat_with_ai, create_chat_message
from autogpt.config import Config
from autogpt.json_fixes.bracket_termination import attempt_to_fix_json_by_finding_outermost_brackets
from autogpt.logs import logger, print_assistant_thoughts
from autogpt.speech import say_text
from autogpt.spinner import Spinner
from autogpt.utils import clean_input
from autogpt.commands.conversational_summary import conversational_summary
from collections.abc import Iterator


class Agent:
    """Agent class for interacting with Auto-GPT."""

    def __init__(self, ai_name, memory, full_message_history, next_action_count, prompt, user_input):
        self.ai_name = ai_name
        self.memory = memory
        self.full_message_history = full_message_history
        self.next_action_count = next_action_count
        self.prompt = prompt
        self.user_input = user_input
        self.cfg = Config()
        self.user_prompt_mode = False  # Flag for handling user natural questions

    def start_interaction_loop(self):
        loop_count = 0
        command_name = None
        arguments = None
        if not hasattr(self, "_recent_commands"):
            self._recent_commands = []

        sig = (
            command_name,
            hashlib.md5(json.dumps(arguments, sort_keys=True).encode()).hexdigest()
        )

        self._recent_commands.append(sig)
        self._recent_commands = self._recent_commands[-6:]

        if self._recent_commands.count(sig) >= 3:
            print("[loop guard] Detected repeated command. Forcing strategy shift.")
            self.full_message_history.append({
                "role": "system",
                "content": "Loop detected. You have repeated the same command multiple times. Change strategy or request user input."
            })

        while True:
            loop_count += 1

            # Continuous mode limit
            if self.cfg.continuous_mode and self.cfg.continuous_limit > 0 and loop_count > self.cfg.continuous_limit:
                logger.typewriter_log("Continuous Limit Reached: ", Fore.YELLOW, f"{self.cfg.continuous_limit}")
                break

            # ------------------ Pre-task memory recall ------------------
            if getattr(self.cfg, "memory_settings", {}).get("recall_before_task", False):
                try:
                    relevant_entries = self.memory.search(["action", "essay", "code", "research"])
                    if relevant_entries:
                        recall_prompt = "\n".join([f"\n---\nMemory Recall:\n{e['content']}" for e in relevant_entries])
                        self.prompt += "\n" + recall_prompt
                        logger.typewriter_log("Proactively recalled relevant memory...", Fore.MAGENTA, f"{len(relevant_entries)} entries added")
                except Exception as e:
                    logger.error(f"Memory recall failed: {e}")
            # ------------------------------------------------------------

            # ------------------ Handle human questions (conversational mode) ------------------
            if self.user_prompt_mode:
                # Directly handle conversational summary without triggering file commands
                print(f"{self.ai_name}: Thinking...", flush=True)
                response = conversational_summary(
                    prompt=self.user_input,
                    memory=self.memory,
                    conversational_mode=True  # Skip file writing, only summarize memory
                )
                if hasattr(response, "__iter__") and not isinstance(response, str):
                    for chunk in response:
                        print(chunk, flush=True)
                else:
                    print(response, flush=True)

                # Reset user prompt
                self.user_input = ""
                self.user_prompt_mode = False
                continue  # Skip normal command execution for this loop


                # Ask AI for next command
            with Spinner("Thinking... "):
                assistant_reply = chat_with_ai(
                    self.prompt,
                    self.user_input,
                    self.full_message_history,
                    self.memory,
                    self.cfg.fast_token_limit,
                )

            print_assistant_thoughts(self.ai_name, assistant_reply)

                # Parse command
            try:
                command_name, arguments = get_command(
                    attempt_to_fix_json_by_finding_outermost_brackets(assistant_reply)
                )
                if self.cfg.speak_mode:
                    say_text(f"I want to execute {command_name}")
            except Exception as e:
                logger.error("Error parsing command: \n", str(e))

            # ------------------ User interaction for non-continuous mode ------------------
            if not self.cfg.continuous_mode and self.next_action_count == 0:
                self.user_input = ""
                logger.typewriter_log(
                    "NEXT ACTION: ",
                    Fore.CYAN,
                    f"COMMAND = {Fore.CYAN}{command_name}{Style.RESET_ALL}  ARGUMENTS = {Fore.CYAN}{arguments}{Style.RESET_ALL}",
                )
                print(
                    "Enter 'y' to authorise command, 'y -N' to run N continuous commands, 'n' to exit program, or type a question for the agent...",
                    flush=True,
                )
                while True:
                    console_input = clean_input(Fore.MAGENTA + "Input:" + Style.RESET_ALL)
                    if console_input.lower().rstrip() == "y":
                        self.user_input = "GENERATE NEXT COMMAND JSON"
                        break
                    elif console_input.lower().startswith("y -"):
                        try:
                            self.next_action_count = abs(int(console_input.split(" ")[1]))
                            self.user_input = "GENERATE NEXT COMMAND JSON"
                        except ValueError:
                            print("Invalid input format. Please enter 'y -n'.")
                            continue
                        break
                    elif console_input.lower() == "n":
                        self.user_input = "EXIT"
                        break
                    else:
                        # Any other input is treated as a user question
                        self.user_input = console_input
                        self.user_prompt_mode = True
                        break

                if self.user_input == "GENERATE NEXT COMMAND JSON":
                    logger.typewriter_log("-=-=-=-=-=-=-= COMMAND AUTHORISED BY USER -=-=-=-=-=-=-=", Fore.MAGENTA, "")
                elif self.user_input == "EXIT":
                    print("Exiting...", flush=True)
                    break

            # ------------------ Execute command ------------------
            if command_name is not None and command_name.lower().startswith("error"):
                result = f"Command {command_name} threw the following error: {arguments}"
            elif command_name == "human_feedback":
                result = f"Human feedback: {self.user_input}"
            else:
                cmd_result = execute_command(command_name, arguments, user_input=self.user_input)

                # NEW: handle generators for streaming output
                is_stream = isinstance(cmd_result, Iterator) and not isinstance(cmd_result, (str, bytes))
                if is_stream:
                    result_text = ""
                    for chunk in cmd_result:
                        print(chunk, flush=True)  # stream to console
                        result_text += chunk + "\n"
                    result = result_text.strip()
                else:
                    result = str(cmd_result)

                if self.next_action_count > 0:
                    self.next_action_count -= 1

            # ------------------ Memory entry ------------------
            tags = ["action"]
            long_task_keywords = ["essay", "report", "article", "story"]
            if any(word in self.prompt.lower() for word in long_task_keywords):
                if getattr(self.cfg, "essay_settings", {}).get("in_progress_tagging", True):
                    tags.append("essay")
                    if getattr(self.cfg, "memory_settings", {}).get("auto_tag_in_progress", True):
                        tags.append("in-progress")

            memory_entry = self.memory.add(
                text=f"Assistant Reply: {assistant_reply if not self.user_prompt_mode else 'USER PROMPT MODE'}\nResult: {result}\nHuman Feedback: {self.user_input}",
                tags=tags,
                task_id=f"{tags[0]}_{loop_count}_{int(datetime.utcnow().timestamp())}"
            )

            # ------------------ Mark done if task finished ------------------
            task_finished = command_name == "task_complete" or self.user_input.lower() in ["essay complete", "finish task"]
            if task_finished and getattr(self.cfg, "memory_settings", {}).get("auto_tag_done", True):
                for entry in self.memory.search(["in-progress"]):
                    if "in-progress" in entry["tags"]:
                        entry["tags"].remove("in-progress")
                        entry["tags"].append("done")
                if getattr(self.memory, "save_on_every_action", True):
                    self.memory.save()

            # ------------------ Append result to message history ------------------
            if result is not None:
                self.full_message_history.append(create_chat_message("system", result))
                logger.typewriter_log("SYSTEM: ", Fore.YELLOW, result)
            else:
                self.full_message_history.append(create_chat_message("system", "Unable to execute command"))
                logger.typewriter_log("SYSTEM: ", Fore.YELLOW, "Unable to execute command")

            # ------------------ Proactive browsing ------------------
            if getattr(self.cfg, "browsing_settings", {}).get("enable_browsing", False) and getattr(self.cfg, "browsing_settings", {}).get("proactive_search", False):
                self.user_input += "\nSEARCH_WEB_PROACTIVELY"

            # ------------------ Plan ahead ------------------
            if getattr(self.cfg, "behavioral_modifiers", {}).get("plan_ahead", False):
                self.prompt += "\nPlan your next steps carefully before acting."
