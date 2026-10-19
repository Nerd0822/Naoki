"""Chat loop: plain input prompt, streamed replies, /-commands for voice.

Stdlib interface: you type, the reply streams raw. Voice features live
behind slash commands, all handled here without going near the graph:
  /voice, /again, /mic, /stop   speech controls
  /tts, /stt <flag> [<value>]   list or set engine flags (session only;
                                config.py decides what runs next time)
"""

import os
import select
import sys
import termios
import tty
from time import perf_counter

from langchain_core.messages import HumanMessage

import stt
import tts
from agent import CONFIG, Naoki
from config import ART, MODEL
from helper import archive, handle_user_input
from rag import index

_RESET = "\033[0m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_RED = "\033[31m"
_BOLD_RED = "\033[1;31m"
_BOLD_MAGENTA = "\033[1;35m"


def note(text: str) -> None:
    print(f"{_DIM}● {text}{_RESET}")


def error(message: str) -> None:
    print(f"{_BOLD_RED}✖{_RESET} {_RED}{message}{_RESET}")


def banner() -> None:
    print()
    print(ART.strip())
    print(
        f"{_BOLD_MAGENTA}Naoki{_RESET} {_DIM}· personal assistant · "
        f"{MODEL or 'ollama'}{_RESET}"
    )
    print(
        f"{_DIM}quit | exit | bye to leave · ctrl+c interrupts · "
        f"/tts /stt /voice /again /mic /stop{_RESET}"
    )
    print()


def prompt() -> str:
    """One prompt turn. Ctrl+C raises KeyboardInterrupt, Ctrl+D EOFError."""
    return input(f"{_BOLD_MAGENTA}❯{_RESET} ")


def _last_ai_reply(bot) -> str | None:
    """The most recent assistant text in the thread (tool calls carry none)."""
    messages = bot.app.get_state(CONFIG).values.get("messages", [])
    for message in reversed(messages):
        if message.type == "ai" and message.content:
            return message.content
    return None


def _say(text: str) -> None:
    """Speak at the prompt level; a Ctrl+C just mutes her mid-sentence."""
    try:
        tts.speak(text)
    except KeyboardInterrupt:
        print()
        note("stopped talking")


def _wait_enter(timeout: float) -> bool:
    """Wait until Enter is pressed (True) or the timeout lapses (False).

    Also accepts alt+x. Ctrl+C raises KeyboardInterrupt, Ctrl+D EOFError.
    """
    if not sys.stdin.isatty():
        raise RuntimeError("push-to-talk needs an interactive terminal")
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        deadline = perf_counter() + timeout
        while True:
            remaining = deadline - perf_counter()
            if not _wait_readable(fd, min(0.25, remaining)):
                if remaining <= 0:
                    return False
                continue
            chunk = os.read(fd, 64)
            if b"\r" in chunk or b"\n" in chunk:
                return True
            if b"\x04" in chunk:
                raise EOFError
            if chunk == b"\x1bx" or chunk == b"\x1bX":
                return True
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _wait_readable(fd: int, timeout: float) -> bool:
    ready, _, _ = select.select([fd], [], [], max(0.0, timeout))
    return bool(ready)


def _talk() -> str:
    """Push-to-talk round. Returns clean text, or "" when it failed/cancelled."""
    try:
        stt.start()
        print(f"{_CYAN}●{_RESET}{_DIM} recording… [enter] stops · auto-cap {stt.STT_MAX_SECONDS}s{_RESET}")
        _wait_enter(stt.STT_MAX_SECONDS + 5.0)  # Enter, alt+x, or the auto-cap
        wav, _seconds, _ = stt.stop()
        if not stt.loaded():
            note("warm-up: first transcription fetches the whisper model")
        text = stt.transcribe(wav)
    except RuntimeError as exc:
        stt.abort()
        error(str(exc))
        return ""
    except (KeyboardInterrupt, EOFError):
        stt.abort()
        error("recording cancelled")
        return ""
    print(f"{_DIM}● heard: {text}{_RESET}")
    return text


def _command(line: str) -> None:
    """/tts and /stt: bare = list every flag, `<flag>` = show it,
    `<flag> <value>` = set it (session only; config.py rules the next run)."""
    parts = line[1:].split()
    if not parts:
        error("usage: /tts [<flag> <value>] · /stt [<flag> <value>]")
        return
    ns, *rest = parts
    target = {"tts": tts, "stt": stt}.get(ns.lower())
    if target is None:
        error(f"unknown command /{ns} -- try /tts or /stt")
        return
    settings = target.flags()
    if not rest:
        for flag, value in settings.items():
            note(f"{ns} {flag} = {value}")
        return
    name = rest[0].lower()
    if name not in settings:
        error(f"/{ns} flags: {', '.join(sorted(settings))}")
        return
    if len(rest) == 1:
        note(f"{ns} {name} = {settings[name]}")
        return
    try:
        note(f"{ns}: {target.set_flag(name, ' '.join(rest[1:]))}")
    except ValueError as exc:
        error(str(exc))


def main():
    banner()
    bot = Naoki()
    written = len(bot.app.get_state(CONFIG).values.get("messages", []))  # archive watermark
    last_reply = None

    while True:
        try:
            user = prompt()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        # /commands never reach the graph
        if user.startswith("/"):
            words = user[1:].split()
            if not words:
                error("commands: /tts /stt /voice /again /mic /stop")
                continue
            word, *rest = words
            word = word.lower()
            if word == "voice":
                tts.set_enabled(not tts.ENABLED)
                note("she is talking now" if tts.ENABLED else "she is muted now")
                continue
            if word == "again":
                if last_reply:
                    _say(last_reply)
                else:
                    note("nothing said yet")
                continue
            if word == "stop":
                tts.stop()
                note("audio stopped")
                continue
            if word == "mic":
                if tts.ENABLED:
                    tts.stop()  # never record over her own voice
                user = _talk()
                if not user:
                    continue
            elif word in ("tts", "stt"):
                _command(user)
                continue
            else:
                error(f"unknown /{word} -- commands: /tts /stt /voice /again /mic /stop")
                continue

        if not handle_user_input(user):
            break
        if not user.strip():
            continue

        try:
            bot.app.invoke({"messages": [HumanMessage(content=user.strip())]}, CONFIG)
        except KeyboardInterrupt:
            note("interrupted")
            continue
        except Exception as e:
            error(f"{type(e).__name__}: {e}")
            continue

        written = archive(bot, written, CONFIG)
        try:
            index()
        except Exception as e:
            error(f"index failed: {type(e).__name__}: {e}")

        last_reply = _last_ai_reply(bot)
        if last_reply and tts.ENABLED:
            _say(last_reply)


if __name__ == "__main__":
    try:
        main()
    finally:
        tts.shutdown()
