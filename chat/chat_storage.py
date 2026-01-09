"""Chat history storage module."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
from config import CHAT_HISTORY_DIR


def ensure_storage_dir():
    """Create the chat history directory if it doesn't exist."""
    CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def create_chat(backend: str = "", model: str = "") -> dict:
    """Create a new chat and return its metadata."""
    ensure_storage_dir()

    chat_id = str(uuid.uuid4())
    now = datetime.now().isoformat()

    chat_data = {
        "id": chat_id,
        "title": "New Chat",
        "created_at": now,
        "updated_at": now,
        "backend": backend,
        "model": model,
        "messages": []
    }

    save_chat(chat_id, chat_data)
    return chat_data


def save_chat(chat_id: str, chat_data: dict):
    """Save chat data to file."""
    ensure_storage_dir()

    chat_data["updated_at"] = datetime.now().isoformat()
    file_path = CHAT_HISTORY_DIR / f"{chat_id}.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(chat_data, f, ensure_ascii=False, indent=2)


def load_chat(chat_id: str) -> Optional[dict]:
    """Load chat data from file."""
    file_path = CHAT_HISTORY_DIR / f"{chat_id}.json"

    if not file_path.exists():
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def delete_chat(chat_id: str) -> bool:
    """Delete a chat file."""
    file_path = CHAT_HISTORY_DIR / f"{chat_id}.json"

    if file_path.exists():
        file_path.unlink()
        return True
    return False


def list_chats() -> list[dict]:
    """List all chats with metadata (without full message content)."""
    ensure_storage_dir()

    chats = []
    for file_path in CHAT_HISTORY_DIR.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                chats.append({
                    "id": data.get("id", file_path.stem),
                    "title": data.get("title", "Untitled"),
                    "created_at": data.get("created_at", ""),
                    "updated_at": data.get("updated_at", ""),
                    "backend": data.get("backend", ""),
                    "model": data.get("model", ""),
                    "message_count": len(data.get("messages", []))
                })
        except (json.JSONDecodeError, IOError):
            continue

    # Sort by updated_at, newest first
    chats.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return chats


def update_chat_title(chat_id: str, title: str) -> bool:
    """Update the title of a chat."""
    chat_data = load_chat(chat_id)
    if chat_data:
        chat_data["title"] = title
        save_chat(chat_id, chat_data)
        return True
    return False


def add_message(chat_id: str, role: str, content: str, reasoning: str = "") -> bool:
    """Add a message to an existing chat."""
    chat_data = load_chat(chat_id)
    if chat_data is None:
        return False

    message = {"role": role, "content": content}
    if reasoning:
        message["reasoning"] = reasoning

    chat_data["messages"].append(message)

    # Update title from first user message if still default
    if chat_data["title"] == "New Chat" and role == "user":
        # Take first 50 chars of the message
        chat_data["title"] = content[:50] + ("..." if len(content) > 50 else "")

    save_chat(chat_id, chat_data)
    return True


if __name__ == "__main__":
    # Test the storage
    print("Creating new chat...")
    chat = create_chat("vllm", "Qwen3-8B-AWQ")
    print(f"Created: {chat['id']}")

    print("Adding messages...")
    add_message(chat["id"], "user", "Hello!")
    add_message(chat["id"], "assistant", "Hi! How can I help?", "User greeted me")

    print("Loading chat...")
    loaded = load_chat(chat["id"])
    print(f"Messages: {len(loaded['messages'])}")

    print("Listing chats...")
    chats = list_chats()
    for c in chats:
        print(f"  - {c['title']} ({c['id'][:8]}...)")

    print("Cleaning up...")
    delete_chat(chat["id"])
    print("Done!")
