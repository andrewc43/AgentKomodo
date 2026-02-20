import os

def merge_text_files(folder: str, output: str) -> str:
    try:
        files = sorted([f for f in os.listdir(folder) if f.endswith(".txt")])
        with open(output, "w") as out:
            for f in files:
                with open(os.path.join(folder, f), "r") as part:
                    out.write(part.read() + "\n\n")
        return f"Merged {len(files)} files into {output}"
    except Exception as e:
        return f"Error: {str(e)}"
