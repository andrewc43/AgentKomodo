import subprocess
import os
from pathlib import Path

WORKING_DIRECTORY = Path(__file__).parent.parent /  "auto_gpt_workspace"

def run_python_file(path: str) -> str:
    print(f"Executing file '{path}' in workspace '{WORKING_DIRECTORY}'")

    if not path.endswith("py"):
        return "Error: Invalid file type. Only .py files are allowed."

    full_path = os.path.join(WORKING_DIRECTORY, path)

    if not os.path.isfile(full_path):
        return f"Error: file {path} does not exist."

    try:
        result = subprocess.run(
            ["python3", full_path],
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
        return f"Error executing file: {str(e)}"
