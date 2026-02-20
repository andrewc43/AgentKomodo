import dataclasses
import os
from typing import Any, List, Optional, Tuple
from datetime import datetime
import numpy as np
import orjson

from autogpt.memory.base import MemoryProviderSingleton, get_ada_embedding

EMBED_DIM = 1536
SAVE_OPTIONS = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SERIALIZE_DATACLASS


def create_default_embeddings():
    return np.zeros((0, EMBED_DIM)).astype(np.float32)


@dataclasses.dataclass
class CacheContent:
    texts: List[dict] = dataclasses.field(default_factory=list)
    embeddings: np.ndarray = dataclasses.field(default_factory=create_default_embeddings)


class LocalCache(MemoryProviderSingleton):
    """A class that stores the memory in a local file."""

    def __init__(self, cfg) -> None:
        """Initialize the memory instance."""
        self.cfg = cfg
        self.filename = f"{cfg.memory_index}.json"
        self.save_on_every_action = getattr(cfg, "MEMORY_SAVE_ON_EVERY_ACTION", True)
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "rb") as f:
                    file_content = f.read()
                    if not file_content.strip():
                        file_content = b"{}"

                    loaded = orjson.loads(file_content)
                    self.data = CacheContent(**loaded)
            except orjson.JSONDecodeError:
                print(f"Error: The file '{self.filename}' is not valid JSON.")
                self.data = CacheContent()
        else:
            print(
                f"Warning: The file '{self.filename}' does not exist. Memory will be initialized empty."
            )
            self.data = CacheContent()

    def add(self, text: str, tags: list = None, task_id: str = None):
        """
        Add structured text entry to memory with embedding.

        Args:
            text (str): the text content
            tags (list): optional tags for search/context
            task_id (str): optional unique identifier for this task
        """
        if "Command Error:" in text:
            return ""

        tags = tags or []

        # Auto-add "in-progress" if essay or long task
        if "in-progress" not in tags and getattr(self.cfg, "memory_settings", {}).get("auto_tag_in_progress", True):
            if any(word in text.lower() for word in ["essay", "report", "article", "story"]):
                tags.append("in-progress")

        memory_entry = {
            "id": task_id or f"action_{len(self.data.texts)}_{int(datetime.utcnow().timestamp())}",
            "tags": tags,
            "timestamp": datetime.utcnow().isoformat(),
            "content": text
        }

        self.data.texts.append(memory_entry)

        # Embedding
        embedding = get_ada_embedding(text)
        vector = np.array(embedding).astype(np.float32)
        vector = vector[np.newaxis, :]
        self.data.embeddings = np.concatenate([self.data.embeddings, vector], axis=0)

        # Save immediately if enabled
        if self.save_on_every_action:
            self.save()

        return memory_entry

    def mark_done(self, task_tags: list = None):
        """Mark all matching tasks as done by replacing 'in-progress' with 'done'."""
        task_tags = task_tags or ["in-progress"]
        updated = 0
        for entry in self.data.texts:
            if any(tag in entry.get("tags", []) for tag in task_tags):
                if "in-progress" in entry["tags"]:
                    entry["tags"].remove("in-progress")
                if "done" not in entry["tags"]:
                    entry["tags"].append("done")
                updated += 1
        if updated > 0 and self.save_on_every_action:
            self.save()
        return updated

    def save(self):
        """Save memory to disk."""
        with open(self.filename, "wb") as f:
            out = orjson.dumps(self.data, option=SAVE_OPTIONS)
            f.write(out)

    def search(self, query_tags: list):
        """Return all entries that match any of the given tags."""
        results = [e for e in self.data.texts if any(tag in e.get("tags", []) for tag in query_tags)]
        return results

    def clear(self) -> str:
        """Clear all memory."""
        self.data = CacheContent()
        return "Memory cleared."

    def get(self, data: str) -> Optional[List[Any]]:
        """Return the most relevant entry for the given data."""
        return self.get_relevant(data, 1)

    def get_relevant(self, text: str, k: int) -> List[Any]:
        """
        Compute similarity scores and return top-k entries, with a minimum similarity threshold.

        Args:
            text: str
            k: int
        """
        if len(self.data.texts) == 0:
            return []

        embedding = get_ada_embedding(text)
        scores = np.dot(self.data.embeddings, embedding)

        # Sort indices by similarity (highest first)
        top_k_indices = np.argsort(scores)[-k:][::-1]

        # Optional: filter by threshold but always return up to k results
        THRESHOLD = 0.15  # tweak this between 0.1-0.2 as needed
        filtered = [self.data.texts[i] for i in top_k_indices if scores[i] >= THRESHOLD]

        # If filtering removed too many, return at least top-k closest
        if len(filtered) < k:
            for i in top_k_indices:
                if self.data.texts[i] not in filtered:
                    filtered.append(self.data.texts[i])
                if len(filtered) >= k:
                    break

        return filtered

    def get_stats(self) -> Tuple[int, Tuple[int, ...]]:
        """Return memory statistics."""
        return len(self.data.texts), self.data.embeddings.shape
