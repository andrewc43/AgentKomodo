import subprocess
from pathlib import Path

WORKING_DIRECTORY = Path(__file__).parent.parent / "auto_gpt_workspace"

def run_shell(command: str) -> str:
    """Executes shell command in workspace."""
    if ";" in command or "&&" in command or "|" in command:
        return "Error: Multiple commands not allowed for safety."
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKING_DIRECTORY,
            capture_output=True,
            text=True,
            timeout=60
        )
        output = result.stdout
        errors = result.stderr
        if errors:
            return f"STDOUT:\n{output}\n\nSTDERR:\n{errors}"
        return output
    except Exception as e:
        return f"Error executing shell command: {str(e)}"
