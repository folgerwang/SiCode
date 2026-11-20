"""SiCode: a coding assistant in the terminal, in the style of Claude Code.

Requests run on the local Qwen3.8-Flash-Next (qwen.exe), which reads, edits and runs things in the current directory
with its own tool loop. A request that starts with [c] goes to Claude over the Claude API instead: Claude plans,
hands self-contained tasks to the local model and reviews the result. "[l] task" or a plain "task" is local.

usage: sicode [--mode ask|edits|auto] [--dir PATH] [--model claude-opus-5-5] [--effort high] [-p "[c] request"]
"""
import argparse
import base64
import difflib
import getpass
import json
import os
import re
import subprocess
import sys
import time

from . import __version__, planners, sysinfo
from .tools import MEDIA_EXTS, TOOLS, Workspace, clip, read_clipboard_media
from .worker import (DEFAULT_EXE, DEFAULT_MODEL, DIRECT_PROMPT, WORKER_PROMPT, Engine, local_models, parse_reply,
                     run_task)

DIM, BOLD, CYAN, YELLOW, RED, GREEN, OFF = "\033[2m", "\033[1m", "\033[36m", "\033[33m", "\033[31m", "\033[32m", "\033[0m"

CONFIG = os.path.join(os.path.expanduser("~"), ".sicode", "config.json")
INSTRUCTION_FILES = ("SICODE.md", "CLAUDE.md", "AGENTS.md")
MODEL_ALIASES = {"opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5", "haiku": "claude-haiku-5-5",
                 "fable": "claude-fable-5-1"}
# models whose declined requests the API can re-run on a fallback model
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1"}
MODES = {"ask": "asks before each file change and each shell command",
         "edits": "applies file changes, asks before shell commands",
         "auto": "applies file changes and runs shell commands without asking"}

HELP = f"""{BOLD}Requests{OFF}
  task             runs on the local model (same as [l] task)
  [c] task         cloud: Claude (or Codex) does the work itself
  [m] task         mix: Claude (or Codex) splits the job; steps marked [l] run on the local model, [c] it does itself
                   ([h] is the same)
  @path            in a request: attaches that file's content
  !command         runs a shell command yourself, outside any model
  a line ending in \\ continues on the next line; Ctrl+C stops the current reply
{BOLD}Commands{OFF}
  /help            this text
  /clear           start new conversations (local and Claude)
  /compact         replace the local conversation by its summary, to free context
  /context         models, context use, tokens spent, memory, mode
  /status          RAM, VRAM and the VRAM cache rate
  /vram [GB]       the most VRAM that may be in use on the card (0: no cap)
  /dir [path]      show or switch the working folder (starts new conversations)
  /diff            git diff of the workspace
  /undo            put back the files changed by the last request
  /init            have the local model write SICODE.md, the project notes loaded into every conversation
  /mode [name]     permission mode: ask, edits, auto
  /model [name]    Claude model for [c]: opus, sonnet, haiku, fable or a full model id
  /local [number]  the local model: lists the installed ones, or switches to one
  /planner [name]  who plans [c] requests: api, claude-code, codex, agy, codebuddy, tcloud
  /providers       show activated cloud providers & account usage; /providers enable|disable <id> | auto
  /cloud [0-100]   cloud AI share for this project (0=local, 100=cloud only & unload local, default 10)
  /paste           attach image, video or audio from clipboard (or press Alt+V while typing)
  /think           show or hide the local model's reasoning
  /key             set the Claude API key
  /update [check]  check GitHub for a newer SiCode release (and install it)
  /exit            quit"""

INIT_REQUEST = """Explore this repository and write SICODE.md at its root: the notes an assistant should read before \
working here. Cover what the project is, the layout (the directories and files that matter), how to build, run and \
test it (exact commands), and conventions visible in the code (naming, formatting, how errors are handled). Only \
state what you verified by reading files. Keep it under 60 lines."""

COMPACT_REQUEST = """Summarise this conversation so the work can continue from the summary alone: what the user \
asked for, what was done (files and functions changed, commands run and their results), decisions made, and what is \
still open. Do not call a tool."""


def have_credentials():
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or os.path.isdir(os.path.expanduser("~/.config/anthropic")))


def load_saved_key():
    """Puts the key saved by an earlier run into the environment, unless the environment already has one."""
    if have_credentials():
        return
    try:
        with open(CONFIG, encoding="utf-8") as f:
            key = json.load(f).get("anthropic_api_key")
    except (OSError, ValueError):
        return
    if key:
        os.environ["ANTHROPIC_API_KEY"] = key


def ask_for_key():
    """Asks for a Claude API key (typed hidden) and saves it for later runs. Returns False if none was given."""
    if not sys.stdin.isatty():
        return False
    print(f"{BOLD}Claude API key{OFF} {DIM}(for [c] requests; from console.anthropic.com; "
          f"Enter to skip and use only the local model){OFF}")
    try:
        key = getpass.getpass("  key: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if not key:
        print(f"{DIM}  skipped; /key asks again{OFF}")
        return False
    os.environ["ANTHROPIC_API_KEY"] = key
    try:
        save_config(anthropic_api_key=key)
        print(f"{DIM}  saved to {CONFIG}{OFF}")
    except OSError as e:
        print(f"{DIM}  not saved ({e}); it is used for this run only{OFF}")
    return True


def vram_cap(a):
    """The VRAM cap in GB: --vram-cap, else the saved setting, else 0 (none)."""
    if a.vram_cap is not None:
        return max(0.0, a.vram_cap)
    try:
        return max(0.0, float(load_config().get("vram_cap_gb") or 0))
    except (TypeError, ValueError):
        return 0.0


def load_config():
    try:
        with open(CONFIG, encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg if isinstance(cfg, dict) else {}
    except (OSError, ValueError):
        return {}


def save_config(**values):
    """Merges values into ~/.sicode/config.json."""
    cfg = {**load_config(), **values}
    os.makedirs(os.path.dirname(CONFIG), exist_ok=True)
    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    os.chmod(CONFIG, 0o600)


ARCHITECT_PROMPT = f"""You are the lead engineer in SiCode, a coding assistant. The user works in a repository \
(the workspace) and talks to you.

{planners.SPLIT}

For a job with more than one step, call `update_plan` first and again whenever a state changes. Write the plan as \
{planners.PLAN_FORMAT}

To run an [l] step, call `delegate` with the complete brief: the goal and why it matters, the files or areas \
involved, the constraints, what done looks like and how to check it, and what to report back. One coherent change \
per brief, in sequence when later steps depend on earlier ones. After a task that changes code, look at the diff \
that `delegate` returns, and read the code or run the tests yourself when the change is risky or the report is \
vague. If the work is wrong, send a follow-up task that says exactly what to fix, or fix it yourself as a [c] step.

To do a [c] step, use your own tools: `read_file`, `list_files`, `search`, `write_file`, `edit_file` and \
`run_command`. The user sees each file change and shell command, yours and the worker's, and may decline one; if \
they do, do not retry it, ask what they want instead.

Answer questions that need no work on the repository directly, without a plan. Tell the user plainly what was done \
and by whom, what was verified and how, and what is still open. Keep your replies short."""

DELEGATE = {
    "name": "delegate",
    "description": "Hand one self-contained task to the local worker model and wait for it to finish. Returns the "
                   "worker's report, the files it changed and the diff. The worker starts with no context: the task "
                   "text is everything it knows.",
    "input_schema": {"type": "object", "additionalProperties": False, "required": ["task"],
                     "properties": {"task": {"type": "string", "description": "The complete brief for the worker."}}},
}
ARCHITECT_TOOLS = [DELEGATE] + TOOLS  # hybrid: Claude hands [l] steps to the worker and does [c] steps itself

# [c]: Claude over the API does the work itself, with the same tools and permissions as the local model
SOLO_API_PROMPT = """You are SiCode, a coding assistant working in the user's repository (the workspace). Carry \
out the request yourself with the tools: read the relevant files before changing them, make the changes the \
request asks for and no others, and check your work when there is a way to (build, run the tests). For a job with \
several steps, call update_plan first and again as steps finish. The user sees each file change and shell command \
and may decline one; if they do, do not retry it, ask what they want instead. Keep the final reply short: what \
you did or found, what you ran and its result, and anything unfinished."""


def valid_input(tool, args):
    """Checks a tool input against its schema (eager input streaming does not validate on the server)."""
    schema = tool["input_schema"]
    if not isinstance(args, dict) or any(k not in args for k in schema["required"]):
        return False
    types = {"string": str, "integer": int}
    return all(k in schema["properties"] and isinstance(v, types[schema["properties"][k]["type"]])
               and not isinstance(v, bool) for k, v in args.items())


def short(args):
    text = ", ".join(f"{k}={str(v).splitlines()[0][:70] if str(v) else ''!r}" for k, v in args.items()
                     if k not in ("content", "new_text", "old_text", "plan"))
    return text[:160]


def ask(question):
    """One-line question to the user; no answer (input closed, Ctrl+C) counts as no."""
    try:
        return input(question).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return "n"


def show_diff(path, old, new, limit=60):
    if old is None:
        lines = [f"+{line}" for line in new.split("\n")]
        print(f"{CYAN}  new file {path}{OFF} {DIM}({len(lines)} lines){OFF}")
    else:
        lines = list(difflib.unified_diff(old.replace("\r\n", "\n").split("\n"), new.replace("\r\n", "\n").split("\n"),
                                          lineterm="", n=2))[2:]
        added = sum(1 for x in lines if x.startswith("+"))
        removed = sum(1 for x in lines if x.startswith("-"))
        print(f"{CYAN}  edit {path}{OFF} {DIM}(+{added} -{removed}){OFF}")
    for line in lines[:limit]:
        colour = GREEN if line.startswith("+") else RED if line.startswith("-") else DIM
        print(f"    {colour}{line[:200]}{OFF}")
    if len(lines) > limit:
        print(f"    {DIM}... {len(lines) - limit} more lines{OFF}")


class ReplyPrinter:
    """Prints a local reply as it is written: the reasoning as a short note (or dimmed, with show_thinking), the
    answer as text, and nothing of the tool-call markup (the call is announced when it runs)."""

    NOTE = "  thinking ..."

    def __init__(self, show_thinking=False):
        self.show_thinking = show_thinking
        self.reset()

    def reset(self):
        self.buf, self.all, self.state, self.wrote, self.note = "", "", "think", False, False

    def _clear_note(self):
        if self.note:
            print("\r" + " " * len(self.NOTE) + "\r", end="")
            self.note = False

    def _emit(self, text):
        if not self.wrote:
            text = text.lstrip("\n")
        if text:
            print(text, end="", flush=True)
            self.wrote = True

    def feed(self, delta):
        self.buf += delta
        self.all += delta
        if self.state == "think":
            i = self.buf.find("</think>")
            if i < 0:
                if self.show_thinking:
                    keep = min(len(self.buf), 8)  # may be the start of </think>
                    print(f"{DIM}{self.buf[:len(self.buf) - keep]}{OFF}", end="", flush=True)
                    self.buf = self.buf[len(self.buf) - keep:]
                else:
                    if not self.note:
                        print(f"{DIM}{self.NOTE}{OFF}", end="", flush=True)
                        self.note = True
                    self.buf = self.buf[-8:]
                return
            if self.show_thinking:
                print(f"{DIM}{self.buf[:i].rstrip()}{OFF}")
            self._clear_note()
            self.buf, self.state = self.buf[i + 8:], "answer"
        if self.state == "answer":
            i = self.buf.find("<tool_call>")
            if i >= 0:
                self._emit(self.buf[:i].rstrip())
                self.buf, self.state = "", "call"
                return
            hold = max((k for k in range(1, 11) if self.buf.endswith("<tool_call>"[:k])), default=0)
            # trailing whitespace waits too: before a tool call or the end of the reply it is dropped
            ready = self.buf[:len(self.buf) - hold].rstrip()
            self._emit(ready)
            self.buf = self.buf[len(ready):]

    def finish(self):
        self._clear_note()
        if self.state == "answer":
            self._emit(self.buf.rstrip())
        elif self.state == "think" and not self.show_thinking:  # no </think> at all: the whole text is the answer
            self._emit(self.all.strip())
        if self.wrote:
            print()
        self.reset()


class App:
    def __init__(self, a):
        self.a = a
        self.mode = a.mode
        self.allowed = set()     # first words of shell commands the user allowed for the session
        self.ws = Workspace(a.dir, self.approve_command, self.approve_edit, self.show_plan)
        self.engine = None
        self.messages = []       # Claude's conversation
        self.usage = {"in": 0, "out": 0, "cache_read": 0, "local": 0}
        self.client = None       # the Claude API client, created by the first [c] request
        self.anthropic = None
        self.local_open = False  # the engine holds a local conversation that a follow-up can continue
        self.summary = ""        # /compact: carried into the next local conversation
        self.undo = []           # (request, {path: content before}) per request that changed files
        self.show_thinking = False
        self.quiet_tools = False  # a delegated task: tool calls are indented under the task
        self.by = ""             # hybrid: who is acting now, "c" (the cloud model) or "l" (the local model)
        self.instructions_name, self.instructions = self.load_instructions()
        cfg = load_config()
        self.enabled_planners = dict(cfg.get("enabled_planners") or {})
        for k in planners.PLANNERS:
            if k not in self.enabled_planners:
                self.enabled_planners[k] = (k == "api" and have_credentials()) or bool(planners.find(k))
        self.planner_usage = {}
        raw_usage = cfg.get("planner_usage") or {}
        for k in planners.PLANNERS:
            u = raw_usage.get(k) or {}
            self.planner_usage[k] = {"used": int(u.get("used", 0)), "quota_left": float(u.get("quota_left", 100.0)),
                                     "quota_manual": bool(u.get("quota_manual", False))}
        self.mix_auto = bool(cfg.get("mix_auto", True))
        self.planner_models = dict(cfg.get("planner_models") or {})
        self.planner_efforts = dict(cfg.get("planner_efforts") or {})
        ratios = {os.path.realpath(k): max(0, min(100, int(v))) for k, v in (cfg.get("project_cloud_ratio") or {}).items()}
        self.project_cloud_ratio = ratios
        if getattr(a, "cloud", None) is not None:
            self.cloud_ratio = max(0, min(100, int(a.cloud)))
        else:
            self.cloud_ratio = ratios.get(os.path.realpath(self.ws.root), 10)
        self.pending_media = []
        self.planner = a.planner if a.planner != "auto" else cfg.get("planner") or \
            planners.pick_primary_planner(self.enabled_planners, self._effective_usage(), have_credentials())
        self.planner_chat = None  # the cloud CLI conversation, when [c] goes through an account login

    def _effective_usage(self):
        detected = planners.detect_account_usage()
        eff = {}
        for k in planners.PLANNERS:
            u = dict(self.planner_usage.get(k) or {"used": 0, "quota_left": 100.0, "quota_manual": False})
            d = detected.get(k) or {}
            u["used_5h"] = int(d.get("used_5h", 0))
            u["used_24h"] = int(d.get("used_24h", 0))
            if not u.get("quota_manual"):
                u["quota_left"] = float(d.get("est_quota_left", 100.0))
            eff[k] = u
        return eff

    def _record_planner_usage(self, planner_id):
        u = self.planner_usage.setdefault(planner_id, {"used": 0, "quota_left": 100.0, "quota_manual": False})
        u["used"] = int(u.get("used", 0)) + 1
        try:
            save_config(planner_usage=self.planner_usage)
        except OSError:
            pass

    def _providers_summary(self, primary):
        avail = set(planners.available_planners(have_credentials()))
        eff = self._effective_usage()
        lines = []
        for k, label in planners.PLANNERS.items():
            if k == primary or k not in avail or not self.enabled_planners.get(k, False):
                continue
            m = self.planner_models.get(k) or (self.a.model if k == "api" else "default")
            ef = self.planner_efforts.get(k) or (self.a.effort if k == "api" else "default")
            q = eff.get(k, {}).get("quota_left", 100.0)
            lines.append(f'- "{k}" ({label}): model={m}, effort={ef}, usage_left={q:.0f}%')
        return "\n".join(lines)

    # --- project notes ---

    def load_instructions(self):
        for name in INSTRUCTION_FILES:
            p = os.path.join(self.ws.root, name)
            if os.path.isfile(p):
                with open(p, encoding="utf-8", errors="replace") as f:
                    return name, clip(f.read(), 12000)
        return None, ""

    # --- permissions and what the tools show ---

    def approve_edit(self, path, old, new):
        if self.by == "c":
            print(f"  {CYAN}[c]{OFF}", end="")
        show_diff(path, old, new)
        if self.mode != "ask":
            return True
        answer = ask("  apply? [y]es / [a]lways for file changes / [n]o: ")
        if answer.startswith("a"):
            self.mode = "edits"
            print(f"{DIM}  mode: edits ({MODES['edits']}){OFF}")
        return answer[:1] in ("y", "a")

    def approve_command(self, command):
        print(f"{YELLOW}  {'[c] ' if self.by == 'c' else ''}$ {command}{OFF}")
        words = command.split()
        first = words[0].lower() if words else ""
        if self.mode == "auto" or first in self.allowed:
            return True
        answer = ask(f"  run? [y]es / [a]lways for '{first}' commands / [n]o: ")
        if answer.startswith("a"):
            self.allowed.add(first)
        return answer[:1] in ("y", "a")

    def show_plan(self, plan):
        for step in planners.parse_plan(plan):
            mark = {"done": "[x]", "doing": "[~]", "todo": "[ ]"}[step["state"]]
            line = f"{mark} " + (f"[{step['who']}] " if step["who"] else "") + step["text"]
            colour = GREEN if line.startswith("[x]") else YELLOW if line.startswith("[~]") else ""
            print(f"    {colour}{line}{OFF}")

    def show_call(self, name, args):
        pad = "    " if self.quiet_tools else "  [c] " if self.by == "c" else "  "
        if name in ("write_file", "edit_file", "run_command", "update_plan"):
            if name == "update_plan":
                print(f"{pad}{CYAN}plan{OFF}")
            return  # these announce themselves: the diff, the command line, the plan
        print(f"{pad}{CYAN}{name}{OFF}{DIM}({short(args)}){OFF}", flush=True)

    def show_result(self, result):
        if result.startswith("error"):
            print(f"{DIM}      {result.splitlines()[0][:150]}{OFF}")

    def status_lines(self):
        """RAM, VRAM and the VRAM cache rate, as the page shows them in its top bar."""
        gb = lambda b: f"{b / 1073741824:.1f}"
        out = []
        ram, vram = sysinfo.ram(), sysinfo.vram()
        if ram:
            mine = sysinfo.process_ram(self.engine.proc.pid) if self.engine else None
            out.append(f"  RAM             {gb(ram[0])} of {gb(ram[1])} GB in use"
                       + (f" (the local model holds {gb(mine)} GB)" if mine else ""))
        cap = vram_cap(self.a)
        if vram:
            out.append(f"  VRAM            {gb(vram[0])} of {gb(vram[1])} GB in use, all programs"
                       + (f"; cap {cap:g} GB" if cap else "; no cap (/vram sets one)"))
        c = self.engine.cache if self.engine else None
        if c and c["hits"] + c["misses"]:
            last = f"{100 * c['last']:.0f}% in the last reply, " if c["last"] is not None else ""
            out.append(f"  VRAM cache rate {last}{100 * c['hits'] / (c['hits'] + c['misses']):.0f}% since the model "
                       f"loaded ({c['cached']:,} of {c['experts']:,} experts in VRAM)")
        else:
            out.append("  VRAM cache rate shown after the first local reply")
        return out

    def unload_engine(self):
        """Closes the local model so that the next request loads it again (with a new model or VRAM cap)."""
        if self.engine:
            self.engine.close()
            self.engine, self.local_open = None, False

    # --- the local model ---

    def start_engine(self):
        if self.engine is None:
            print(f"{DIM}loading Qwen3.8-Flash-Next ...{OFF}", flush=True)
            t0 = time.time()
            self.engine = Engine(self.a.exe, self.a.worker_model, self.a.backend, vram_cap_gb=vram_cap(self.a))
            print(f"{DIM}local model ready in {time.time() - t0:.0f} s (context {self.engine.ctx:,} tokens){OFF}")
        return self.engine

    def local_turn(self, text, prompt=DIRECT_PROMPT):
        engine = self.start_engine()
        printer = ReplyPrinter(self.show_thinking)

        def on_event(kind, data):
            if kind == "delta":
                printer.feed(data)
            elif kind == "reply":
                printer.finish()
            elif kind == "call":
                self.show_call(*data)
            elif kind == "result":
                self.show_result(data)
            elif kind == "stats":
                self.usage["local"] += data["gen_tokens"]

        fresh = not self.local_open
        if fresh and self.summary:
            text = f"Summary of the conversation so far:\n{self.summary}\n\nThe user's next request:\n{text}"
            self.summary = ""
        t0 = time.time()
        self.local_open = True  # an interrupted reply is closed by the engine, so the conversation can go on
        try:
            _, stats = run_task(engine, self.ws, text, on_event, effort=self.a.worker_effort, fresh=fresh,
                                prompt=prompt, instructions=self.instructions, stream=True)
        except KeyboardInterrupt:
            printer.finish()
            print(f"{YELLOW}[interrupted]{OFF}")
            return
        speed = stats["gen_tokens"] / max(stats["gen_s"], 1e-9)
        rate = engine.cache["last"]
        cache = f", VRAM cache rate {100 * rate:.0f}%" if rate is not None else ""
        print(f"{DIM}[local: {stats['steps']} steps, {stats['gen_tokens']:,} tokens at {speed:.0f} tok/s, "
              f"context {stats['ctx_used']:,} of {engine.ctx:,}{cache}, {time.time() - t0:.0f} s]{OFF}")

    def compact(self):
        if not self.local_open:
            print(f"{DIM}[no local conversation to compact]{OFF}")
            return
        print(f"{DIM}summarising the local conversation ...{OFF}", flush=True)
        try:
            reply = self.engine.turn(COMPACT_REQUEST)
        except KeyboardInterrupt:
            print(f"{YELLOW}[interrupted; nothing was compacted]{OFF}")
            return
        before = reply["ctx_used"]
        self.summary = parse_reply(reply["text"])[1]
        self.local_open = False
        print(f"{DIM}[compacted: {before:,} tokens of context replaced by a summary of "
              f"{len(self.summary.split()):,} words, used from the next request]{OFF}")

    # --- [c]: Claude is the architect and delegates to the local model ---

    def git_diff(self, paths=()):
        try:
            r = subprocess.run(["git", "diff", "--stat", "--patch", "--", *paths], cwd=self.ws.root,
                               capture_output=True, encoding="utf-8", errors="replace", timeout=60)
            return r.stdout if r.returncode == 0 else ""
        except (OSError, subprocess.TimeoutExpired):
            return ""

    def delegate(self, task):
        engine = self.start_engine()
        self.ws.changed.clear()
        print(f"  {CYAN}[l] delegate{OFF} {DIM}{task.strip().splitlines()[0][:150]}{OFF}")

        def on_event(kind, data):
            if kind == "call":
                self.show_call(*data)
            elif kind == "result":
                self.show_result(data)
            elif kind == "stats":
                self.usage["local"] += data["gen_tokens"]

        t0 = time.time()
        self.local_open = False  # the engine now holds this task, not the user's local conversation
        self.quiet_tools = True
        by, self.by = self.by, "l"
        try:
            report, stats = run_task(engine, self.ws, task, on_event, effort=self.a.worker_effort,
                                     prompt=WORKER_PROMPT, instructions=self.instructions)
        finally:
            self.quiet_tools = False
            self.by = by
        print(f"{DIM}    done: {stats['steps']} steps, {stats['gen_tokens']:,} tokens, {time.time() - t0:.0f} s{OFF}")
        changed = sorted(self.ws.changed)
        out = "WORKER REPORT\n" + report + "\n\nFILES CHANGED\n" + ("\n".join(changed) if changed else "none")
        diff = self.git_diff(changed) if changed else ""
        if diff:
            out += "\n\nDIFF (git, against the last commit; new untracked files are not shown)\n" + clip(diff, 12000)
        return out

    def run_tool(self, block):
        tool = next((t for t in [DELEGATE] + TOOLS if t["name"] == block.name), None)
        if tool is None or not valid_input(tool, block.input):
            return {"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                    "content": "INVALID_JSON: the tool input did not match the schema; send the call again."}
        if block.name == "delegate":
            result = self.delegate(block.input["task"])
        else:
            self.show_call(block.name, block.input)
            result = self.ws.call(block.name, block.input)
            self.show_result(result)
        return {"type": "tool_result", "tool_use_id": block.id, "content": result}

    def architect_turn(self, text, solo=False):
        self.messages.append({"role": "user", "content": text})
        tools = [{**t, "eager_input_streaming": True} for t in (TOOLS if solo else ARCHITECT_TOOLS)]
        system = SOLO_API_PROMPT if solo else ARCHITECT_PROMPT
        if self.instructions:
            system += f"\n\n# Project instructions ({self.instructions_name})\n\n{self.instructions}"
        extra = {}
        if self.a.model in FALLBACK_MODELS:
            # a request the safety classifiers decline is re-run on a fallback model in the same call
            extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        while True:
            with self.client.beta.messages.stream(
                model=self.a.model, max_tokens=32000, system=system, tools=tools, messages=self.messages,
                thinking={"type": "adaptive"}, output_config={"effort": self.a.effort},
                cache_control={"type": "ephemeral"}, **extra,
            ) as stream:
                wrote = False
                for piece in stream.text_stream:
                    print(piece, end="", flush=True)
                    wrote = True
                if wrote:
                    print()
                msg = stream.get_final_message()
            u = msg.usage
            self.usage["in"] += u.input_tokens + (u.cache_creation_input_tokens or 0)
            self.usage["cache_read"] += u.cache_read_input_tokens or 0
            self.usage["out"] += u.output_tokens
            self.messages.append({"role": "assistant", "content": msg.content})
            if msg.stop_reason == "pause_turn":
                continue
            if msg.stop_reason == "refusal":
                print(f"{RED}[Claude declined this request]{OFF}")
                return
            if msg.stop_reason == "max_tokens":
                print(f"{RED}[Claude's reply hit the length limit]{OFF}")
                return
            calls = [b for b in msg.content if b.type == "tool_use"]
            if msg.stop_reason != "tool_use" or not calls:
                return
            self.messages.append({"role": "user", "content": [self.run_tool(b) for b in calls]})

    def claude_ready(self):
        if self.client is None:
            if not have_credentials() and not ask_for_key():
                print(f"{RED}No Claude API key, so [c] is unavailable: set ANTHROPIC_API_KEY or run sicode in a "
                      f"terminal and enter one. Requests without [c] run on the local model.{OFF}")
                return False
            import anthropic
            self.anthropic = anthropic
            self.client = anthropic.Anthropic()
        return True

    def delegate_cloud(self, task_text, meta):
        prov = (meta.get("provider") or "").strip().lower()
        avail = set(planners.available_planners(have_credentials()))
        if prov not in avail or not self.enabled_planners.get(prov, False):
            return self.delegate(task_text)
        model = meta.get("model") or self.planner_models.get(prov) or None
        effort = meta.get("effort") or self.planner_efforts.get(prov) or None
        print(f"  {CYAN}[{prov}]{OFF} {task_text.splitlines()[0][:80]}", flush=True)
        snap = self.ws.snapshot()
        try:
            if prov == "api":
                if not self.claude_ready():
                    return "error: no Claude API key"
                msg = self.client.messages.create(
                    model=model or self.a.model, max_tokens=2048,
                    messages=[{"role": "user", "content": task_text}],
                )
                reply = "".join(b.text for b in msg.content if b.type == "text")
            else:
                chat = planners.Planner(prov, self.ws.root, model=model, effort=effort,
                                        solo=True, allow_shell=self.mode == "auto")
                self.planner_chat = chat
                try:
                    reply = chat.ask(task_text)
                finally:
                    self.planner_chat = None
            self._record_planner_usage(prov)
            for path, old, new in self.ws.changes_since(snap):
                if new is None:
                    print(f"  {CYAN}[{prov}] deleted {path}{OFF}")
                else:
                    print(f"  {CYAN}[{prov}]{OFF}", end="")
                    show_diff(path, old, new)
            return reply or "(no reply)"
        except Exception as e:
            return f"error from {prov}: {e}"

    def login_turn(self, text, solo=False, planner_id=None, media_items=None):
        """[c] and [h] through a cloud CLI, signed in with the user's account."""
        p = planner_id or self.planner
        imgs = [m["path"] for m in (media_items or []) if str(m.get("mime", "")).startswith("image/")]
        chat = self.planner_chat
        if chat is None or chat.planner != p or chat.solo != solo or imgs:
            chat = self.planner_chat = planners.Planner(
                p, self.ws.root,
                model=self.planner_models.get(p) or None,
                effort=self.planner_efforts.get(p) or None,
                solo=solo, allow_shell=self.mode == "auto",
                providers_info=self._providers_summary(p),
                cloud_ratio=self.cloud_ratio, images=imgs,
            )
        chat.allow_shell = self.mode == "auto"
        print(f"{DIM}  {planners.PLANNERS[p]} is {'working' if solo else 'planning'} ...{OFF}", flush=True)
        snap = [None]

        def before_turn():  # the tool edits on its own: compare the workspace around each of its turns
            snap[0] = self.ws.snapshot()

        def after_turn():
            self._record_planner_usage(p)
            for path, old, new in self.ws.changes_since(snap[0]):
                mark = "" if solo else f"  {CYAN}[c]{OFF}"
                if new is None:
                    print(f"{mark}{CYAN}  deleted {path}{OFF}")
                else:
                    print(mark, end="")
                    show_diff(path, old, new)

        def on_plan(steps):
            self.show_plan("\n".join({"done": "[x]", "doing": "[~]", "todo": "[ ]"}[x["state"]] +
                                     (f" [{x['who']}] " if x["who"] else " ") + x["text"] for x in steps))

        try:
            planners.run(chat, text, self.delegate, lambda t: print(t, flush=True), on_plan=on_plan,
                         before_turn=before_turn, after_turn=after_turn, delegate_cloud=self.delegate_cloud)
        except KeyboardInterrupt:
            chat.stop()
            print(f"\n{YELLOW}[interrupted]{OFF}")

    def claude_turn(self, text, solo=False, media_items=None):
        active_p = planners.pick_task_planner(text, self.enabled_planners, self._effective_usage(),
                                              have_credentials(), media=media_items) if self.mix_auto else self.planner
        if active_p != "api":
            return self.login_turn(text, solo, planner_id=active_p, media_items=media_items)
        if not self.claude_ready():
            return
        self._record_planner_usage("api")
        mark = len(self.messages)
        an = self.anthropic
        self.by = "" if solo else "c"
        try:
            imgs = [m for m in (media_items or []) if str(m.get("mime", "")).startswith("image/")]
            if imgs:
                blocks = []
                for m in imgs:
                    try:
                        with open(m["path"], "rb") as f:
                            b64 = base64.b64encode(f.read()).decode("ascii")
                        blocks.append({"type": "image", "source": {"type": "base64", "media_type": m["mime"], "data": b64}})
                    except OSError:
                        continue
                blocks.append({"type": "text", "text": text})
                self.messages.append({"role": "user", "content": blocks})
                self.architect_turn(text, solo)
            else:
                self.architect_turn(text, solo)
        except KeyboardInterrupt:
            del self.messages[mark:]
            print(f"\n{YELLOW}[interrupted; that request was dropped from Claude's conversation]{OFF}")
        except an.AuthenticationError:
            del self.messages[mark:]
            print(f"{RED}The Claude API rejected the key. Enter a valid one with /key.{OFF}")
        except an.RateLimitError:
            del self.messages[mark:]
            print(f"{RED}Rate limited by the Claude API; try again in a minute.{OFF}")
        except an.APIStatusError as e:
            del self.messages[mark:]
            print(f"{RED}Claude API error {e.status_code}: {e.message}{OFF}")
        except an.APIConnectionError:
            del self.messages[mark:]
            print(f"{RED}Could not reach the Claude API (network).{OFF}")
        self.by = ""
        u = self.usage
        print(f"{DIM}[Claude: {u['in']:,} in, {u['cache_read']:,} cached, {u['out']:,} out | "
              f"local: {u['local']:,} tokens]{OFF}")

    # --- one request ---

    def attach(self, text, media_items=None):
        """Appends the content of each @path that names a workspace file."""
        out = text
        for ref in dict.fromkeys(re.findall(r"(?<!\S)@([^\s@]+)", text)):
            try:
                p = self.ws.path(ref)
            except ValueError:
                continue
            if os.path.isfile(p):
                mime = MEDIA_EXTS.get(os.path.splitext(p)[1].lower())
                if mime:
                    out += f"\n\n<attached_media path=\"{self.ws.rel(p)}\" mime=\"{mime}\" />"
                else:
                    with open(p, encoding="utf-8", errors="replace") as f:
                        out += f"\n\n<file path=\"{self.ws.rel(p)}\">\n{clip(f.read())}\n</file>"
                print(f"{DIM}  attached {self.ws.rel(p)}{OFF}")
        for m in (media_items or []):
            rel = os.path.relpath(m["path"], self.ws.root) if m["path"].startswith(self.ws.root) else m["path"]
            out += f"\n\n<attached_media path=\"{rel}\" mime=\"{m['mime']}\" />"
        return out

    def turn(self, text, prompt=DIRECT_PROMPT):
        """Routes one request: "[c]" cloud, "[h]"/"[m]" mix, "[l]" local, or by project cloud_ratio."""
        tag, rest = text[:3].lower(), text[3:].strip()
        if tag == "[m]":  # mix: the other name of the hybrid mode
            tag = "[h]"
        if tag not in ("[c]", "[l]", "[h]"):
            tag = "[l]" if self.cloud_ratio <= 0 else "[c]" if self.cloud_ratio >= 100 else "[h]"
            rest = text
        media_items = list(self.pending_media)
        self.pending_media.clear()
        if media_items and tag == "[l]":
            tag = "[h]"
            print(f"{DIM}Media attached: routing to multimodal Cloud AI.{OFF}")
        if not rest and not media_items:
            return
        if not rest and media_items:
            rest = "Please inspect the attached media."
        self.ws.begin_turn()
        try:
            if tag == "[c]" and self.engine is not None:
                self.close()
                self.local_open = False
                print(f"{DIM}Cloud 100%: unloaded local model to free RAM and VRAM.{OFF}")
            if tag in ("[c]", "[h]"):
                self.claude_turn(self.attach(rest, media_items), solo=tag == "[c]", media_items=media_items)
            else:
                self.local_turn(self.attach(rest, media_items), prompt)
        finally:
            if self.ws.backups:
                self.undo.append((rest.splitlines()[0][:60], self.ws.backups))
                self.ws.begin_turn()

    # --- commands ---

    def command(self, text):
        """Runs a /command. Returns False to quit."""
        name, _, arg = text.partition(" ")
        arg = arg.strip()
        if name in ("/exit", "/quit"):
            return False
        if name == "/help":
            print(HELP)
        elif name == "/clear":
            self.messages.clear()
            self.planner_chat = None
            self.local_open, self.summary = False, ""
            print(f"{DIM}[conversations cleared]{OFF}")
        elif name == "/compact":
            self.compact()
        elif name == "/context":
            u = self.usage
            used = self.engine.ctx_used if self.engine and self.local_open else 0
            ctx = f"{used:,} of {self.engine.ctx:,} tokens" if self.engine else "model not loaded yet"
            notes = self.instructions_name or "none (/init writes SICODE.md)"
            local = next((m["label"] for m in local_models((self.a.worker_model,)) if os.path.normcase(m["path"]) ==
                          os.path.normcase(self.a.worker_model)), self.a.worker_model)
            cloud = planners.PLANNERS[self.planner] + (f", {self.a.model}, effort {self.a.effort}"
                                                       if self.planner == "api" else "")
            print(f"  local model     {local}\n  local context   {ctx}\n  local tokens    {u['local']:,} written\n"
                  f"  cloud model     {cloud}\n"
                  f"  Claude API      {u['in']:,} in, {u['cache_read']:,} cached, {u['out']:,} out, "
                  f"{len(self.messages)} messages\n"
                  + "\n".join(self.status_lines()) + "\n"
                  f"  mode            {self.mode}: {MODES[self.mode]}\n  project notes   {notes}\n"
                  f"  workspace       {self.ws.root}")
        elif name == "/status":
            print("\n".join(self.status_lines()))
        elif name == "/vram":
            if arg:
                try:
                    gb = max(0.0, float(arg))
                except ValueError:
                    print(f"{RED}give the cap in GB, or 0 for none{OFF}")
                    return True
                changed = gb != vram_cap(self.a)
                self.a.vram_cap = gb
                try:
                    save_config(vram_cap_gb=gb)
                except OSError:
                    pass
                if changed and self.engine:
                    self.unload_engine()
                    print(f"{DIM}[the local model reloads with the next request]{OFF}")
            cap = vram_cap(self.a)
            print(f"{DIM}VRAM cap: {f'{cap:g} GB in use on the card, counting other programs' if cap else 'none'}{OFF}")
        elif name == "/dir":
            if arg:
                root = os.path.realpath(os.path.expanduser(arg.strip('"')))
                if not os.path.isdir(root):
                    print(f"{RED}{root} is not a folder{OFF}")
                    return True
                # a new working folder is a new job: both conversations start afresh
                self.ws = Workspace(root, self.approve_command, self.approve_edit, self.show_plan)
                self.messages.clear()
                self.planner_chat = None
                self.local_open, self.summary, self.undo = False, "", []
                self.instructions_name, self.instructions = self.load_instructions()
            notes = f"project notes {self.instructions_name}" if self.instructions_name else "no project notes (/init)"
            print(f"{DIM}workspace {self.ws.root}; {notes}{OFF}")
        elif name == "/diff":
            print(self.git_diff() or f"{DIM}[no changes against the last commit, or not a git repository]{OFF}")
        elif name == "/undo":
            if not self.undo:
                print(f"{DIM}[nothing to undo]{OFF}")
            else:
                label, backups = self.undo.pop()
                print(f"{DIM}[put back {', '.join(self.ws.restore(backups))} (changed by: {label})]{OFF}")
        elif name == "/init":
            self.turn("[l] " + INIT_REQUEST)
            self.instructions_name, self.instructions = self.load_instructions()
        elif name == "/mode":
            if arg in MODES:
                self.mode = arg
            elif arg:
                print(f"{RED}unknown mode; choose ask, edits or auto{OFF}")
            print(f"{DIM}mode: {self.mode} ({MODES[self.mode]}){OFF}")
        elif name == "/model":
            if arg:
                self.a.model = MODEL_ALIASES.get(arg.lower(), arg)
            print(f"{DIM}Claude model for [c]: {self.a.model}{OFF}")
        elif name == "/planner":
            if arg in planners.PLANNERS:
                self.planner = arg
                try:
                    save_config(planner=arg)
                except OSError:
                    pass
            elif arg:
                print(f"{RED}unknown planner; choose {', '.join(planners.PLANNERS)}{OFF}")
            for k, v in planners.PLANNERS.items():
                found = "" if k == "api" or planners.find(k) else " (its tool is not installed)"
                print(f"  {'*' if k == self.planner else ' '} {k:12s} {v}{found}")
        elif name == "/local":
            models = local_models((self.a.worker_model,))
            if arg.isdigit() and 1 <= int(arg) <= len(models):
                self.a.worker_model = models[int(arg) - 1]["path"]
                try:
                    save_config(local_model=self.a.worker_model)
                except OSError:
                    pass
                self.unload_engine()  # the next request loads the chosen model
            elif arg:
                print(f"{RED}give the number of a model from the list{OFF}")
            for i, m in enumerate(models, 1):
                on = os.path.normcase(m["path"]) == os.path.normcase(self.a.worker_model)
                print(f"  {'*' if on else ' '} {i}  {m['label']}{' (default)' if m['default'] else ''}")
        elif name == "/think":
            self.show_thinking = not self.show_thinking
            print(f"{DIM}[local reasoning {'shown' if self.show_thinking else 'hidden'}]{OFF}")
        elif name == "/key":
            if ask_for_key():
                self.client = None  # the next [c] request builds a client with the new key
        elif name == "/cloud":
            if not arg:
                print(f"cloud share: {self.cloud_ratio}% (0=local, 100=cloud only)")
            else:
                try:
                    self.cloud_ratio = max(0, min(100, int(arg.rstrip("%"))))
                    self.project_cloud_ratio[os.path.realpath(self.ws.root)] = self.cloud_ratio
                    save_config(project_cloud_ratio=self.project_cloud_ratio)
                    if self.cloud_ratio >= 100 and self.engine is not None:
                        self.close()
                        self.local_open = False
                        print(f"{DIM}unloaded local model to free RAM and VRAM{OFF}")
                    print(f"cloud share: {self.cloud_ratio}%")
                except ValueError:
                    print(f"{RED}usage: /cloud [0-100]{OFF}")
        elif name == "/providers":
            parts = arg.split()
            if len(parts) == 2 and parts[0] in ("enable", "disable") and parts[1] in planners.PLANNERS:
                self.enabled_planners[parts[1]] = (parts[0] == "enable")
                save_config(enabled_planners=self.enabled_planners)
            elif arg == "auto":
                self.mix_auto = not self.mix_auto
                save_config(mix_auto=self.mix_auto)
                print(f"mix auto-select: {'on' if self.mix_auto else 'off'}")
            avail = set(planners.available_planners(have_credentials()))
            eff = self._effective_usage()
            primary = planners.pick_primary_planner(self.enabled_planners, eff, have_credentials())
            for k, label in planners.PLANNERS.items():
                st_str = "not installed" if k not in avail else ("primary" if k == primary else ("activated" if self.enabled_planners.get(k) else "disabled"))
                u = eff[k]
                print(f"  {k:<12} {st_str:<14} {u['quota_left']:5.1f}% left  (5h: {u['used_5h']}, 24h: {u['used_24h']})  {DIM}{label}{OFF}")
        elif name == "/paste":
            items = read_clipboard_media(os.path.join(self.ws.root, ".sicode", "uploads"))
            if not items:
                print(f"{DIM}no image, video or audio found in clipboard{OFF}")
            else:
                self.pending_media.extend(items)
                print(f"{DIM}attached from clipboard: {', '.join(it['name'] for it in items)}{OFF}")
        elif name == "/update":
            from . import updater
            print(f"{DIM}checking GitHub for SiCode updates (current v{__version__}) ...{OFF}")
            info = updater.check_update()
            if info.get("error"):
                print(f"{RED}update check failed: {info['error']}{OFF}")
            elif not info.get("has_update"):
                print(f"{GREEN}SiCode v{__version__} is up to date (latest: v{info['latest']}).{OFF}")
            else:
                print(f"{CYAN}New version available: v{info['latest']} ({info.get('wheel_name') or info['tag']}){OFF}")
                if arg == "check":
                    print(f"{DIM}Run /update or `sicode --update` to install.{OFF}")
                else:
                    ans = input("Install update now? [Y/n] ").strip().lower()
                    if ans in ("", "y", "yes"):
                        res = updater.apply_update(info.get("wheel_url"))
                        if res["ok"]:
                            print(f"{GREEN}Updated to v{info['latest']}. Restart sicode to use the new version.{OFF}")
                        else:
                            print(f"{RED}Update failed:{OFF}\n{res['output']}")
        else:
            print(f"{RED}unknown command {name}; /help lists them{OFF}")
        return True

    def read_request(self):
        if os.name == "nt" and sys.stdin.isatty():
            import msvcrt
            lines = []
            while True:
                prompt_str = f"\n{GREEN}>{OFF} " if not lines else f"{DIM}.{OFF} "
                sys.stdout.write(prompt_str)
                sys.stdout.flush()
                buf = []
                while True:
                    ch = msvcrt.getwch()
                    if ch in ("\r", "\n"):
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                        break
                    if ch == "\x03":
                        raise KeyboardInterrupt
                    if ch in ("\x08", "\x7f"):
                        if buf:
                            buf.pop()
                            sys.stdout.write("\b \b")
                            sys.stdout.flush()
                        continue
                    if ch == "\x1b":
                        if msvcrt.kbhit():
                            nxt = msvcrt.getwch()
                            if nxt.lower() == "v":
                                items = read_clipboard_media(os.path.join(self.ws.root, ".sicode", "uploads"))
                                if items:
                                    self.pending_media.extend(items)
                                    sys.stdout.write(f" {DIM}[+{len(items)} media]{OFF}")
                                    sys.stdout.flush()
                        continue
                    if ch in ("\x00", "\xe0"):
                        if msvcrt.kbhit():
                            sc = msvcrt.getwch()
                            if sc == "/":
                                items = read_clipboard_media(os.path.join(self.ws.root, ".sicode", "uploads"))
                                if items:
                                    self.pending_media.extend(items)
                                    sys.stdout.write(f" {DIM}[+{len(items)} media]{OFF}")
                                    sys.stdout.flush()
                        continue
                    buf.append(ch)
                    sys.stdout.write(ch)
                    sys.stdout.flush()
                line = "".join(buf)
                if line.endswith("\\"):
                    lines.append(line[:-1])
                    continue
                lines.append(line)
                return "\n".join(lines).strip()
        lines = []
        while True:
            line = input(f"\n{GREEN}>{OFF} " if not lines else f"{DIM}.{OFF} ")
            if line.endswith("\\"):
                lines.append(line[:-1])
                continue
            lines.append(line)
            return "\n".join(lines).strip()

    def repl(self):
        notes = f"project notes {self.instructions_name}" if self.instructions_name else "no project notes (/init)"
        who = f"Claude {self.a.model}" if self.planner == "api" else planners.PLANNERS[self.planner]
        print(f"{BOLD}SiCode{OFF} {__version__}  requests run on the local model; [m] mix, [c] cloud: {who}\n"
              f"{DIM}workspace {self.ws.root}\nmode {self.mode}: {MODES[self.mode]}; {notes}\n"
              f"/help lists the commands{OFF}")
        if self.planner == "api" and not have_credentials():  # an account login needs no key
            ask_for_key()
        while True:
            try:
                text = self.read_request()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            if text.strip().lower() in ("exit", "quit"):  # typed without the slash
                break
            try:
                if text.startswith("/"):
                    if not self.command(text):
                        break
                elif text.startswith("!"):
                    subprocess.run(text[1:], shell=True, cwd=self.ws.root)
                else:
                    self.turn(text)
            except KeyboardInterrupt:
                print(f"\n{YELLOW}[interrupted]{OFF}")
            except RuntimeError as e:
                print(f"{RED}{e}{OFF}")
        self.close()

    def close(self):
        if self.engine:
            self.engine.close()


def main():
    ap = argparse.ArgumentParser(prog="sicode", description="Requests run on the local Qwen3.8-Flash-Next. Start one with "
                                 "[c] to have Claude or Codex do it instead, or with [m] (mix) to have it split the job "
                                 "between itself and the local model.")
    ap.add_argument("--dir", default=os.getcwd(), help="workspace (default: the current directory)")
    ap.add_argument("--mode", default="ask", choices=list(MODES),
                    help="ask before file changes and shell commands (default), only before commands (edits), or never (auto)")
    ap.add_argument("--yes", action="store_true", help="the same as --mode auto")
    ap.add_argument("--model", default="claude-opus-5-5", help="Claude model for [c] requests")
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"],
                    help="reasoning effort of Claude")
    ap.add_argument("--worker-effort", default="low", choices=["low", "medium", "xhigh"],
                    help="reasoning effort of the local model")
    ap.add_argument("--exe", default=DEFAULT_EXE, metavar="PATH", help="the engine (default: the bundled qwen.exe)")
    ap.add_argument("--worker-model", default=None, metavar="PATH",
                    help="folder of the local model (default: the saved choice, else the 4-bit Qwen3.8-Flash-Next)")
    ap.add_argument("--vram-cap", type=float, default=None, metavar="GB",
                    help="the most VRAM that may be in use on the card, counting other programs (default: the saved "
                         "setting, else no cap); the local model sizes its expert cache to stay under it")
    ap.add_argument("--backend", default="vulkan", choices=["vulkan", "cuda", "cpu"], help="where the local model runs")
    ap.add_argument("-p", "--prompt", help="run one request and exit; may start with [l] local, [c] cloud or [m] mix")
    ap.add_argument("--planner", default="auto", choices=["auto", *planners.PLANNERS],
                    help="who plans [c] requests: the Claude API key, or an account login through a local CLI "
                         "(default: auto-selected by remaining usage and task)")
    ap.add_argument("--cloud", type=int, default=None, help="cloud AI share %% for this session (0=local, 100=cloud only)")
    ap.add_argument("--web", action="store_true", help="open SiCode as a page in the browser instead of the terminal")
    ap.add_argument("--port", type=int, default=8765, help="port of the --web page (default 8765; 0 picks a free one)")
    ap.add_argument("--no-browser", action="store_true", help="with --web: print the address, do not open it")
    ap.add_argument("--download-model", action="store_true",
                    help="download the local model (about 115 GB) to ~/.sicode/models and exit")
    ap.add_argument("--update", action="store_true",
                    help="check GitHub for a newer SiCode release and install it")
    ap.add_argument("--version", action="version", version="sicode " + __version__)
    a = ap.parse_args()
    if a.update:
        from . import updater
        info = updater.check_update()
        if info.get("error"):
            print("Update check failed: " + info["error"])
            return
        if not info.get("has_update"):
            print(f"SiCode v{__version__} is already up to date (latest: v{info['latest']}).")
            return
        print(f"Updating SiCode v{__version__} -> v{info['latest']} ({info.get('wheel_name') or info['tag']}) ...")
        res = updater.apply_update(info.get("wheel_url"))
        print(res["output"])
        return
    if a.download_model:
        from .worker import download_model
        print("downloaded to " + download_model())
        return
    if a.yes:
        a.mode = "auto"
    if not a.worker_model:  # the choice made with /local or in the page's Settings, while that folder exists
        saved = load_config().get("local_model")
        a.worker_model = saved if saved and os.path.exists(saved) else DEFAULT_MODEL
    a.model = MODEL_ALIASES.get(a.model.lower(), a.model)
    os.system("")  # turns on ANSI colours in the Windows console
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8", errors="replace")
    if a.web:
        from .web import serve
        return serve(a)
    load_saved_key()
    app = App(a)
    if a.prompt:
        try:
            if a.prompt.startswith("/"):  # a command such as /status, run once
                app.command(a.prompt)
            else:
                app.turn(a.prompt)
        finally:
            app.close()
    else:
        app.repl()


if __name__ == "__main__":
    main()
