"""Chat memory as vectors: transcript jsonl -> per-exchange documents ->
chunks -> Chroma. index() runs after every good turn; get_retriever() is
what agent.retrieve() calls at the top of each prompt."""

import logging
import orjson
from functools import lru_cache
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from config import EMBED_MODEL, MEMORY_DIR, DB_DIR

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 3
SCORE_MIN = 0.4  # similarity floor for the threshold retriever (Chroma cosine, 0-1)
SKIP_ROLES = {"tool"}  # tool output is noise in memory prose

logging.getLogger("langchain_core.vectorstores.base").setLevel(logging.ERROR)


def load_doc(file):
    messages = []
    with open(file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                messages.append(orjson.loads(line))

    exchanges, current = [], []
    for m in messages:
        if m["role"] == "human" and current:
            exchanges.append(current)
            current = []
        current.append(m)
    if current:
        exchanges.append(current)

    docs = []
    for i, ex in enumerate(exchanges):
        lines = [
            f"{m['role']}: {m['content']}"
            for m in ex
            if m["content"].strip() and m["role"] not in SKIP_ROLES
        ]
        if not lines:
            continue
        docs.append(
            Document(
                page_content="\n".join(lines),
                metadata={"source": str(file), "turn": i, "num_messages": len(ex)},
            )
        )
    return docs


def chunk_doc(docs):
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(docs)


@lru_cache(maxsize=1)
def get_store():
    """The Chroma store, built once per process (embedding calls are lazy)."""
    return Chroma(
        collection_name="chat_memory",
        embedding_function=OllamaEmbeddings(model=EMBED_MODEL, keep_alive=30),
        persist_directory=str(DB_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def index(file=MEMORY_DIR) -> int:
    """Index everything not yet in the store and return how many chunks were added.

    Idempotent per chunk: the turn-based ids are checked against the store
    first, so re-indexing the whole file on every turn is cheap.
    """
    chunks = chunk_doc(load_doc(file))

    seen, ids = {}, []
    for c in chunks:
        turn = c.metadata["turn"]
        n = seen.get(turn, 0)
        seen[turn] = n + 1
        ids.append(f"{c.metadata['source']}-{turn}-{n}")

    store = get_store()
    have = set(store.get(ids=ids, include=[])["ids"])
    todo = [(c, i) for c, i in zip(chunks, ids) if i not in have]
    if not todo:
        return 0
    store.add_documents([c for c, _ in todo], ids=[i for _, i in todo])
    return len(todo)


def get_retriever(k=TOP_K):
    """Threshold retriever: up to TOP_K docs, none below SCORE_MIN similarity."""
    return get_store().as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": SCORE_MIN},
    )
