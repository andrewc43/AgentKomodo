from autogpt.commands.file_operations import safe_join, split_file, WORKING_DIRECTORY
from autogpt.memory.local import READ_PROGRESS

def stream_file(filename: str, chunk_size: int = 2000, overlap: int = 200) -> str:
    """
    Stream a file in chunks directly back to the agent/user.
    Each call returns the next chunk.
    """
    global READ_PROGRESS

    try:
        filepath = safe_join(WORKING_DIRECTORY, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Split file into chunks
        chunks = list(split_file(content, max_length=chunk_size, overlap=overlap))
        num_chunks = len(chunks)

        # Initialize progress
        if filename not in READ_PROGRESS:
            READ_PROGRESS[filename] = 0

        idx = READ_PROGRESS[filename]

        if idx >= num_chunks:
            return f"End of file reached for {filename}."

        chunk = chunks[idx]
        READ_PROGRESS[filename] += 1

        # Return chunk as a "chatty" response
        return f"Reading {filename}, part {idx + 1} of {num_chunks}:\n{chunk}"

    except Exception as e:
        return f"Error reading file: {str(e)}"
