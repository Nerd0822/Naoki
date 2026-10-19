"""Shell tools."""

import subprocess
import tempfile
from datetime import datetime
from itertools import count
from pathlib import Path

from langchain_core.tools import tool

from config import SHELL_TIMEOUT

# Background jobs are session-local on purpose: the registry (and the output
# files) live in the process, so anything that should survive a restart needs
# a plan of its own.
JOBS_DIR = Path(tempfile.gettempdir()) / "naoki_jobs"
JOBS: dict = {}
_counter = count(1)
MAX_JOBS = 3


@tool
def run_command(command: str) -> str:
    """Run a shell command and return its stdout, stderr, and exit code.

    Blocks until the command finishes, so an interactive command (vim, top, ssh
    with no host) hangs until the timeout kills it. Args: command.
    """
    # ponytail: SHELL_TIMEOUT is the only hang guard -- a denylist of
    # interactive commands would fail fast instead of waiting it out. Add one
    # if the wait actually annoys.
    r = subprocess.run(
        command, shell=True, capture_output=True, text=True, timeout=SHELL_TIMEOUT
    )
    return f"exit={r.returncode}\n{r.stdout}{r.stderr}"


@tool
def start_job(command: str) -> str:
    """Run a shell command in the background; the chat stays free while it runs.

    Use for anything slow -- downloads, builds, batch runs. Returns a job id;
    check progress with check_job, all jobs with list_jobs. Args: command.
    """
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    running = sum(j["proc"].poll() is None for j in JOBS.values())
    if running >= MAX_JOBS:
        return f"(too many jobs: {running} running, max {MAX_JOBS} -- wait for some to finish)"
    job_id = next(_counter)
    with (JOBS_DIR / f"job{job_id}.out").open("w") as out:
        proc = subprocess.Popen(
            command, shell=True, stdout=out, stderr=subprocess.STDOUT
        )
    now = datetime.now().strftime("%H:%M:%S")
    JOBS[job_id] = {"proc": proc, "command": command, "log": JOBS_DIR / f"job{job_id}.out", "started": now}
    return f"job {job_id} started at {now} ({running + 1} of {MAX_JOBS} slots): {command[:80]}"


@tool
def check_job(job_id: int) -> str:
    """Check a background job: running or finished, with its output so far.

    Args: job_id (from start_job).
    """
    job = JOBS.get(job_id)
    if job is None:
        return f"(no job {job_id} in this session -- see list_jobs)"
    out = job["log"].read_text()[-2000:] or "(no output yet)"
    if job["proc"].poll() is None:
        return f"job {job_id} running since {job['started']}:\n{out}"
    return f"job {job_id} finished, exit={job['proc'].returncode}:\n{out}"


@tool
def list_jobs() -> str:
    """List this session's background jobs with their status."""
    if not JOBS:
        return "(no jobs started this session)"
    lines = []
    for jid in sorted(JOBS):
        proc = JOBS[jid]["proc"]
        state = "running" if proc.poll() is None else f"finished, exit={proc.returncode}"
        lines.append(f"job {jid} [{state}] {JOBS[jid]['command'][:80]}")
    return "\n".join(lines)
