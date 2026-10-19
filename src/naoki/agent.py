"""The agent graph — retrieve -> naoki -> (tools loop) — and its streamed display.

Each turn prints itself: reasoning collapses into an overwrite-in-place
status line, answer text streams token by token, and a dim timing line
closes the turn. All chunks accumulate into one AIMessageChunk so the
graph state sees a complete reply. Plain ANSI only, no UI dependency.
"""

from time import perf_counter

from langchain_core.messages import AIMessageChunk, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from config import MODEL, SYSTEM
from rag import get_retriever
from tools import TOOLS

CONFIG = {"configurable": {"thread_id": "naoki"}}

_DIM = "\033[2m"
_DIM_ITALIC = "\033[2;3m"
_RESET = "\033[0m"


def _clip(text, limit: int = 120) -> str:
    one_line = " ".join(str(text).split())
    return one_line if len(one_line) <= limit else one_line[: limit - 1] + "…"


class State(MessagesState):
    context: str


retriever = get_retriever()


class Naoki:
    def __init__(self):
        self.naoki = ChatOllama(
            model=MODEL,
            keep_alive="30m",  # hold the model warm in ollama between turns
            reasoning=False,
        ).bind_tools(list(TOOLS.values()))
        self._tool_node = ToolNode(list(TOOLS.values()))

        builder = StateGraph(State)
        builder.add_node("retrieve", self.retrieve)
        builder.add_node("naoki", self.turn)
        builder.add_node("tools", self.log_tools)
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "naoki")
        builder.add_conditional_edges("naoki", self.route_tools)
        builder.add_edge("tools", "naoki")
        self.app = builder.compile(checkpointer=MemorySaver())

    def retrieve(self, state: State):
        docs = retriever.invoke(state["messages"][-1].content)
        return {"context": "\n\n".join(d.page_content for d in docs)}

    def route_tools(self, state: State):
        destination = tools_condition(state)
        if destination == "tools":
            for call in state["messages"][-1].tool_calls:
                print(f"{_DIM}● {call['name']} {_clip(call['args'])}{_RESET}")
        return destination

    def log_tools(self, state: State):
        result = self._tool_node.invoke(state)
        for m in result["messages"]:
            print(f"{_DIM}↳ {m.name}: {_clip(m.content, 160)}{_RESET}")
        return result

    def turn(self, state: State):
        """One model hop: stream and display it, return the accumulated reply.

        Runs again after every tool round; the tools loop reaches back here
        through route_tools until the model answers without tool calls.
        """
        prefix = [SystemMessage(content=SYSTEM)]
        if state.get("context"):
            prefix.append(
                SystemMessage(
                    content="Relevant past conversations (use only if helpful):\n"
                    + state["context"]
                )
            )
        messages = prefix + state["messages"][-5:]

        reply = AIMessageChunk(content="")
        t0 = perf_counter()
        print()  # gap between prompt and this turn's output
        thinking_line = False  # the dim status line is on screen, awaiting replacement
        answered = False
        try:
            for chunk in self.naoki.stream(messages):
                reasoning = chunk.additional_kwargs.get("reasoning_content", "")
                # reasoning can trickle in after text on some models; never
                # repaint the status line once the answer has started
                if reasoning and not answered:
                    elapsed = perf_counter() - t0
                    print(
                        f"\x1b[2K\r{_DIM_ITALIC}● thinking… {elapsed:.0f}s{_RESET}",
                        end="",
                        flush=True,
                    )
                    thinking_line = True
                if chunk.content or chunk.tool_call_chunks:
                    if chunk.content and not answered:
                        if thinking_line:
                            print(
                                f"\x1b[2K\r{_DIM}"
                                f"thought for {perf_counter() - t0:.1f}s{_RESET}"
                            )
                        answered = True
                    if chunk.content:
                        print(chunk.content, end="", flush=True)
                reply += chunk
        except BaseException:
            if answered or thinking_line:
                print()  # never leave a dangling partial line for the prompt
            raise
        if answered:
            print()
            line = f"⏱ {perf_counter() - t0:.2f}s"
            tokens = (reply.response_metadata or {}).get("eval_duration")
            if tokens:
                line += f" · tokens {tokens / 1_000_000_000:.2f}s"
            print(f"{_DIM}{line}{_RESET}")
        return {"messages": [reply]}
