"""Long-term memory tools."""

from langchain_core.tools import tool

from rag import get_store


@tool
def recall(query: str, limit: int = 5) -> str:
    """Search your long-term memory of past conversations.

    Use when the user asks about earlier chats, past decisions, or anything
    from a previous session. Args: query (what to look up), limit (max turns).
    """
    # similarity_search, not the threshold retriever: she chose to search,
    # so show what is actually there instead of filtering at SCORE_MIN
    docs = get_store().similarity_search(query, k=limit)
    if not docs:
        return "(no matching memories)"
    return "\n\n".join(
        f"[turn {d.metadata.get('turn')}]\n{d.page_content}" for d in docs
    )
