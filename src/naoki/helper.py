"""Between-loop helpers: exit-word check and the memory archive."""

from config import MEMORY_DIR
import orjson


def handle_user_input(user: str) -> bool:
    """False means stop the run: user typed an exit word."""
    exit_words = ("quit", "exit", "bye")
    return user.strip().lower() not in exit_words


def archive(bot, written: int, CONFIG) -> int:
    """Append human/ai messages newer than `written` to memory.jsonl.

    `written` is the caller's watermark (the message count at last flush), so
    only the current turn gets written; the new count comes back for the next
    call.
    """
    messages = bot.app.get_state(CONFIG).values["messages"]
    MEMORY_DIR.parent.mkdir(parents=True, exist_ok=True)
    with MEMORY_DIR.open("ab") as f:
        for m in messages[written:]:
            if m.type not in ("human", "ai") or not m.content.strip():
                continue
            f.write(orjson.dumps({"role": m.type, "content": m.content}) + b"\n")
    return len(messages)
