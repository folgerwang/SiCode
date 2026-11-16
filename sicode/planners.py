"""Planners that use an account login instead of an API key.

The cloud model can be the vendor's own command-line tool, which signs in with the user's subscription: Claude Code
(`claude`, a Claude account) or Codex (`codex`, a ChatGPT account). SiCode never sees the account's credentials: it
runs the tool non-interactively in the workspace and reads its reply.

In the hybrid mode the tool splits the job into steps marked [l] (the local model) and [c] (itself). It cannot call
SiCode's tools, so the hand-off is done in text: it ends a reply with a <delegate> block, SiCode runs that task on
the local model and sends the report back as the next message of the same session. Its own steps it does with its
own file tools; SiCode compares the workspace before and after each of its turns to show what it changed.
"""
import json
import os
import re
import shutil
import subprocess
import threading
import time

PLANNERS = {
    "api": "Claude API key",
    "claude-code": "Claude account (through Claude Code)",
    "codex": "ChatGPT account (through Codex)",
    "agy": "Google account (through Antigravity)",
    "codebuddy": "Tencent CodeBuddy CLI",
    "tcloud": "Tencent Cloud AI CLI",
}

PLANNER_BINS = {
    "claude-code": "claude",
    "codex": "codex",
    "agy": "agy",
    "codebuddy": "codebuddy",
    "tcloud": "tcloud",
}

SPLIT = """Two of you do the work, and every step of the job goes to one of you:

- [l] the local model: Qwen3.8-Flash-Next (4-bit) on the user's own GPU. Free to run, but slow (about 35 tokens per \
second) and clearly less capable than you. Each task you hand it starts a fresh worker with no memory of earlier \
tasks or of this conversation; it works alone with read, write, edit, search and shell tools until it reports back. \
Its report describes what it did; treat that as a claim, not as proof.
- [c] you. Your tokens cost the user money or quota, the worker's cost nothing.

Give a step to [l] when a careful but ordinary engineer could do it from a written brief: mechanical or repetitive \
edits, boilerplate, tests for behaviour you specify, exploration ("find where X is handled and report the functions \
involved"), running builds, tests and other commands, and anything long but well defined. Take a step as [c] when \
it needs your judgement or precision: the design, a subtle bug, a change that must be right across several files, \
anything where a wrong edit is costly, a final review, or an edit so small that writing the brief would cost more \
than doing it. When in doubt between the two, prefer [l] and review its result."""

PLAN_FORMAT = """one step per line, each starting with its state and who does it: "[ ] [l] step" or "[ ] [c] step". \
States: [ ] to do, [~] in progress, [x] done."""


def hybrid_prompt(can_run, providers_info="", cloud_ratio=10):
    """The system prompt for the primary cloud planner in the hybrid (Mix) mode."""
    run = ("You can also run shell commands yourself." if can_run else
           "You cannot run shell commands yourself: builds, tests and other commands are always [l] steps.")
    r = max(0, min(100, int(cloud_ratio)))
    if r <= 15:
        bias = f"Project allocation setting: {100 - r}% Local / {r}% Cloud. Strongly prefer [l]: delegate almost all edits, implementation, and tests to the local model; reserve [c] only for planning and final review."
    elif r <= 40:
        bias = f"Project allocation setting: {100 - r}% Local / {r}% Cloud. Prefer [l]: delegate most coding and verification to the local model, keeping only complex design or tricky bugs as [c]."
    elif r <= 60:
        bias = f"Project allocation setting: {100 - r}% Local / {r}% Cloud. Balanced split between [l] and [c]."
    elif r <= 85:
        bias = f"Project allocation setting: {100 - r}% Local / {r}% Cloud. Prefer [c]: handle most design and implementation in the cloud, delegating only routine commands or bulk boilerplate to [l]."
    else:
        bias = f"Project allocation setting: {100 - r}% Local / {r}% Cloud. Strongly prefer [c]: do almost all work directly in the cloud (and other activated cloud providers), using [l] only if strictly needed."
    extra_cloud = ""
    if providers_info:
        extra_cloud = f"""

Other activated cloud providers (ranked by usage left) are also available for specialized steps:
{providers_info}
To hand a step to another cloud provider (with an optional model or thinking effort: low, medium, high, xhigh), use:
<delegate provider="codex" effort="high">
The complete brief for that provider.
</delegate>
Use <delegate> (without provider) for the free local model [l]."""
    return f"""You are the lead engineer in SiCode, a coding assistant. The user works in a repository (your working \
directory) and talks to you.

{SPLIT}

{bias}{extra_cloud}

For a job with more than one step, start with the plan and send it again, complete, whenever a state changes:

<plan>
[ ] [l] first step
[ ] [c] second step
</plan>

In a plan block write {PLAN_FORMAT}

To run an [l] step, end your reply with exactly one block of this form:

<delegate>
The complete brief for the worker.
</delegate>

SiCode then runs it and sends you the worker's report and the list of changed files as the next message. Write each \
brief so that someone who has seen nothing can do it: the goal and why it matters, the files or areas involved, the \
constraints, what done looks like and how to check it, and what to report back. One coherent change per brief.

To do a [c] step, do it yourself in this turn with your own tools: read, search, edit and write files in the \
working directory. {run} SiCode shows the user the files you changed.

A reply without a <delegate> block ends your turn and is shown to the user. Answer questions that need no work on \
the repository directly, without a plan. In your final reply tell the user plainly what was done and by whom, what \
was verified and how, and what is still open. Keep it short."""


SOLO_PROMPT = """You are working through SiCode, a coding assistant, in the user's repository (your working \
directory). Carry out the request yourself: read the relevant files before changing them, make the changes the \
request asks for and no others, and check your work when you have a way to. If you cannot run commands here, say \
which checks are still to be run. Keep the final reply short: what you did or found, what you ran and its result, \
and anything unfinished."""


def cloud_orchestrate_prompt(can_run, providers_info):
    """System prompt for 100% Cloud mode when multiple cloud providers/models are activated."""
    run = "You may also run shell commands yourself." if can_run else "Avoid running destructive shell commands."
    return f"""{SOLO_PROMPT} {run}

Other activated cloud providers (ranked by usage left) are also available for specialized sub-tasks:
{providers_info}

If a sub-task benefits from another provider, model, or thinking effort (low, medium, high, xhigh), you may end a reply with one block:
<delegate provider="codex" effort="high">
The complete brief for that provider.
</delegate>
SiCode will run that step on the chosen provider and send you its report. A reply without <delegate> finishes your turn."""

DELEGATE_RE = re.compile(r"<delegate([^>]*)>\s*(.*?)\s*</delegate>", re.S)
ATTR_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')
PLAN_RE = re.compile(r"<plan>\s*(.*?)\s*</plan>", re.S)
STEP_RE = re.compile(r"^\s*(?:[-*]\s*)?(?:\[([ xX~])\]\s*)?(?:\[([lLcC])\]\s*)?(.*\S)\s*$")
MAX_HANDOFFS = 12


_AGY_MODELS_CACHE = None


def agy_models():
    """Return [{id, label}] from `agy models`, cached for the process lifetime."""
    global _AGY_MODELS_CACHE
    if _AGY_MODELS_CACHE is not None:
        return _AGY_MODELS_CACHE
    exe = find("agy")
    if not exe:
        _AGY_MODELS_CACHE = []
        return _AGY_MODELS_CACHE
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        out = subprocess.check_output([exe, "models"], stderr=subprocess.DEVNULL, timeout=5,
                                      creationflags=flags).decode("utf-8", errors="replace")
        models = []
        for line in out.splitlines():
            if "\t" in line:
                mid, label = line.split("\t", 1)
                mid, label = mid.strip(), label.strip()
                if mid:
                    models.append({"id": mid, "label": label or mid})
        _AGY_MODELS_CACHE = models
    except Exception:
        _AGY_MODELS_CACHE = []
    return _AGY_MODELS_CACHE


_EFFORT_SUFFIX_RE = re.compile(r"-(low|medium|high|xhigh|max)$")
_EFFORT_LABEL_RE = re.compile(r"\s*\((Low|Medium|High|XHigh|Max)\)$")


def agy_base_models():
    """Return deduplicated base models [{id, label}] from `agy models`."""
    seen = {}
    for m in agy_models():
        base_id = _EFFORT_SUFFIX_RE.sub("", m["id"])
        base_label = _EFFORT_LABEL_RE.sub("", m["label"])
        if base_id not in seen:
            seen[base_id] = {"id": base_id, "label": base_label}
    return list(seen.values())


def _resolve_agy_model(model, effort):
    """Map a base model + effort (e.g. gemini-4-hydrogen + high) to an exact `agy models` id."""
    if not model:
        return None
    raw = [m["id"] for m in agy_models()]
    if model in raw:
        return model
    if effort and f"{model}-{effort}" in raw:
        return f"{model}-{effort}"
    for fallback in ("medium", "high", "low", "xhigh", "max"):
        if f"{model}-{fallback}" in raw:
            return f"{model}-{fallback}"
    return model


def find(planner):
    """Path of the planner's command-line tool, or None."""
    return shutil.which(PLANNER_BINS.get(planner, ""))


def available_planners(have_key):
    """Return list of planner ids that are currently installed/configured."""
    return [k for k in PLANNERS if (k == "api" and have_key) or bool(find(k))]


def _count_jsonl_timestamps(path, field, scale, now):
    c5 = c24 = 0
    if not os.path.isfile(path):
        return 0, 0
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    ts = float(json.loads(line).get(field, 0)) / scale
                except (ValueError, TypeError, AttributeError):
                    continue
                if ts >= now - 18000:
                    c5 += 1
                if ts >= now - 86400:
                    c24 += 1
    except OSError:
        pass
    return c5, c24


def _count_recent_dirs(root, now):
    c5 = c24 = 0
    if not os.path.isdir(root):
        return 0, 0
    try:
        for entry in os.scandir(root):
            try:
                mt = entry.stat().st_mtime
                if mt >= now - 18000:
                    c5 += 1
                if mt >= now - 86400:
                    c24 += 1
            except OSError:
                continue
    except OSError:
        pass
    return c5, c24


def detect_account_usage():
    """Inspect local CLI history/session files to estimate recent usage per provider."""
    now = time.time()
    home = os.path.expanduser("~")
    counts = {
        "claude-code": _count_jsonl_timestamps(os.path.join(home, ".claude", "history.jsonl"), "timestamp", 1000.0, now),
        "codex": _count_jsonl_timestamps(os.path.join(home, ".codex", "history.jsonl"), "ts", 1.0, now),
        "agy": _count_recent_dirs(os.path.join(home, ".gemini", "antigravity-cli", "brain"), now),
        "codebuddy": _count_recent_dirs(os.path.join(home, ".codebuddy"), now),
        "tcloud": _count_recent_dirs(os.path.join(home, ".tcloud"), now),
    }
    out = {}
    for k in PLANNERS:
        u5, u24 = counts.get(k, (0, 0))
        est_left = max(5.0, round(100.0 - min(95.0, u5 * 2.5), 1)) if u5 > 0 else 100.0
        out[k] = {"used_5h": u5, "used_24h": u24, "est_quota_left": est_left}
    return out


def pick_primary_planner(enabled_map, usage_map, have_key):
    """Pick the enabled, installed planner with the most usage left (highest quota_left, fewest used calls)."""
    return pick_task_planner("", enabled_map, usage_map, have_key)


def pick_task_planner(task_text, enabled_map, usage_map, have_key, media=None):
    """Pick the best enabled, installed cloud provider considering remaining usage, task type, and media."""
    avail = [k for k in available_planners(have_key) if enabled_map.get(k, True)]
    if not avail:
        avail = available_planners(have_key)
    if not avail:
        return "claude-code"
    low = (task_text or "").lower()
    is_code = any(w in low for w in ("fix", "bug", "test", "refactor", "implement", "build", "compile", "error", "api", "function"))
    is_design = any(w in low for w in ("ui", "design", "css", "html", "docs", "readme", "explain", "review", "search", "plan", "architecture"))
    has_video = any(str(m.get("mime", "")).startswith("video/") for m in (media or []))
    has_image = any(str(m.get("mime", "")).startswith("image/") for m in (media or []))
    order = list(PLANNERS.keys())

    def score(k):
        u = (usage_map or {}).get(k) or {}
        quota = float(u.get("quota_left", 100.0))
        used = int(u.get("used", 0))
        affinity = 0.0
        if has_video:
            if k == "agy":
                affinity += 25.0
            elif k in ("claude-code", "api", "codex"):
                affinity += 5.0
        elif has_image:
            if k in ("agy", "claude-code", "codex", "api"):
                affinity += 10.0
        if is_code and k in ("codex", "claude-code", "codebuddy"):
            affinity += 5.0
        elif is_design and k in ("agy", "claude-code"):
            affinity += 5.0
        # Usage left dominates; affinity breaks ties and steers media to capable models.
        return (-(quota + affinity), used, order.index(k))

    return min(avail, key=score)


def default_planner(have_key):
    """An account login when its tool is installed, else the API key."""
    for p in ("claude-code", "codex", "agy", "codebuddy", "tcloud"):
        if find(p):
            return p
    return "api" if have_key else "claude-code"


def parse_plan(text):
    """[{state: todo|doing|done, who: "l"|"c"|"", text}] from plan lines such as "[~] [l] run the tests"."""
    steps = []
    for line in text.strip().splitlines():
        m = STEP_RE.match(line)
        if not m:
            continue
        mark = m.group(1)
        state = "done" if mark in ("x", "X") else "doing" if mark == "~" else "todo"
        steps.append({"state": state, "who": (m.group(2) or "").lower(), "text": m.group(3)})
    return steps


def split_reply_meta(text):
    """(text shown to the user, task for the worker or None, plan text or None, delegate attrs dict)."""
    plan = PLAN_RE.search(text)
    if plan:
        text = text[:plan.start()] + text[plan.end():]
    m = DELEGATE_RE.search(text)
    if not m:
        return text.strip(), None, plan.group(1) if plan else None, {}
    attrs = {k.lower(): v for k, v in ATTR_RE.findall(m.group(1) or "")}
    return text[:m.start()].strip(), m.group(2).strip(), plan.group(1) if plan else None, attrs


def split_reply(text):
    """(text shown to the user, task for the worker or None, plan text or None)."""
    shown, task, plan, _ = split_reply_meta(text)
    return shown, task, plan


class Planner:
    """One conversation with a cloud CLI tool, run non-interactively in the workspace."""

    def __init__(self, planner, cwd, model=None, effort=None, session=None, solo=False, allow_shell=False, raw=False,
                 providers_info="", cloud_ratio=10, images=None):
        self.planner, self.cwd, self.model, self.effort = planner, cwd, model, effort
        self.solo = solo                # the tool does the work itself instead of handing it to the local model
        self.allow_shell = allow_shell  # solo: it may also run shell commands
        self.raw = raw                  # single-turn non-coding prompt (e.g. watchdog check)
        self.providers_info = providers_info
        self.cloud_ratio = cloud_ratio
        self.images = list(images or [])
        self.session = session   # the tool's own session id, to continue the conversation
        self.proc = None

    def stop(self):
        if self.proc and self.proc.poll() is None:
            from .tools import _kill_tree
            _kill_tree(self.proc.pid)
            try:
                self.proc.kill()
            except OSError:
                pass

    def _run(self, args, prompt, should_stop):
        exe = find(self.planner)
        if not exe:
            name = PLANNERS.get(self.planner, self.planner)
            raise RuntimeError(f"{name} is not installed or not on PATH; install it and sign in, or choose another "
                               "planner")
        # the point is the account login: a key in the environment would make the tool bill the API instead
        env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        self.proc = subprocess.Popen([exe, *args], cwd=self.cwd, env=env, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)
        try:
            if prompt is not None:
                self.proc.stdin.write(prompt.encode("utf-8"))
            self.proc.stdin.close()
            # the pipes are drained by helpers so a long reply cannot block the tool
            chunks, errs = [], []
            t1 = threading.Thread(target=lambda: chunks.append(self.proc.stdout.read()), daemon=True)
            t2 = threading.Thread(target=lambda: errs.append(self.proc.stderr.read()), daemon=True)
            t1.start(); t2.start()
            while self.proc.poll() is None:
                if should_stop():
                    self.stop()
                    raise KeyboardInterrupt
                time.sleep(0.2)
            t1.join(5); t2.join(5)
        except KeyboardInterrupt:
            self.stop()
            raise
        text = b"".join(chunks).decode("utf-8", errors="replace")
        err = b"".join(errs).decode("utf-8", errors="replace")
        return self.proc.returncode, text, err

    def _system_prompt(self, can_run=None):
        run = self.allow_shell if can_run is None else can_run
        if self.solo:
            return cloud_orchestrate_prompt(run, self.providers_info) if self.providers_info else SOLO_PROMPT
        return hybrid_prompt(run, self.providers_info, self.cloud_ratio)

    def ask(self, prompt, should_stop=lambda: False):
        """Sends one message and returns the planner's reply text."""
        if self.planner == "claude-code":
            if self.raw:
                args = ["-p", "--output-format", "json", "--tools", ""]
            else:
                tools = "Read,Grep,Glob,Edit,Write" + (",Bash" if self.allow_shell else "")
                args = ["-p", "--output-format", "json", "--tools", tools, "--permission-mode", "acceptEdits",
                        "--append-system-prompt", self._system_prompt(self.allow_shell)]
                if self.allow_shell:
                    args += ["--allowedTools", "Bash"]
            if self.session:
                args += ["--resume", self.session]
            if self.model:
                args += ["--model", self.model]
            if self.effort:
                args += ["--effort", self.effort]
            code, text, err = self._run(args, prompt, should_stop)
            data = None
            for line in reversed(text.strip().splitlines()):
                try:
                    data = json.loads(line)
                    break
                except ValueError:
                    continue
            if not isinstance(data, dict):
                raise RuntimeError("Claude Code gave no reply" + (": " + (err or text).strip()[-400:] if err or text else ""))
            self.session = data.get("session_id") or self.session
            if data.get("is_error"):
                raise RuntimeError("Claude Code: " + str(data.get("result") or data.get("subtype") or "error")[:400]
                                   + " (run `claude` once in a terminal to sign in, if that is the problem)")
            return str(data.get("result") or "")
        if self.planner == "codex":
            if self.session:
                args = ["exec", "resume", self.session, "-", "--json", "--skip-git-repo-check"]
            else:
                args = ["exec", "--json", "--skip-git-repo-check", "-"]
                if not self.raw:
                    prompt = self._system_prompt(True) + "\n\n# The user's request\n\n" + prompt
            # Codex edits and runs commands inside its own sandbox, limited to the workspace
            args += ["-c", 'sandbox_mode="workspace-write"']
            if self.model:
                args += ["-m", self.model]
            if self.effort:
                args += ["-c", f'model_reasoning_effort="{self.effort}"']
            for img in self.images:
                args += ["-i", img]
            code, text, err = self._run(args, prompt, should_stop)
            reply, failure = [], ""
            for line in text.splitlines():
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("type") == "thread.started":
                    self.session = e.get("thread_id") or self.session
                elif e.get("type") == "item.completed" and e.get("item", {}).get("type") == "agent_message":
                    reply.append(e["item"].get("text", ""))
                elif e.get("type") in ("error", "turn.failed"):
                    failure = json.dumps(e)[:400]
            if not reply:
                raise RuntimeError("Codex gave no reply: " + (failure or err.strip()[-400:] or f"exit code {code}")
                                   + " (run `codex login` in a terminal to sign in, if that is the problem)")
            return reply[-1]
        if self.planner == "agy":
            if not self.session and not self.raw:
                prompt = self._system_prompt(self.allow_shell) + "\n\n# The user's request\n\n" + prompt
            args = ["--output-format", "json", "--mode", "accept-edits"]
            if self.allow_shell:
                args.append("--dangerously-skip-permissions")
            if self.session:
                args += ["--conversation", self.session]
            resolved_model = _resolve_agy_model(self.model, self.effort)
            if resolved_model:
                args += ["--model", resolved_model]
            if self.effort:
                args += ["--effort", self.effort]
            args.append(f"--print={prompt}")
            code, text, err = self._run(args, None, should_stop)
            data = None
            for line in reversed(text.strip().splitlines()):
                try:
                    data = json.loads(line)
                    break
                except ValueError:
                    continue
            if not isinstance(data, dict):
                raise RuntimeError("Antigravity gave no reply" + (": " + (err or text).strip()[-400:] if err or text else ""))
            self.session = data.get("conversation_id") or self.session
            if data.get("status") != "SUCCESS" and not data.get("response"):
                raise RuntimeError("Antigravity: " + str(data.get("error") or data.get("status") or "error")[:400])
            return str(data.get("response") or "")
        if self.planner in ("codebuddy", "tcloud"):
            if not self.session and not self.raw:
                prompt = self._system_prompt(self.allow_shell) + "\n\n# The user's request\n\n" + prompt
            if self.planner == "codebuddy":
                args = ["-p", "--output-format", "json"]
                if self.session:
                    args += ["--resume", self.session]
                if self.model:
                    args += ["--model", self.model]
                code, text, err = self._run(args, prompt, should_stop)
            else:
                args = ["ai", "chat", "--prompt", prompt, "--output", "json"]
                if self.model:
                    args += ["--model", self.model]
                code, text, err = self._run(args, None, should_stop)
            data = None
            for line in reversed(text.strip().splitlines()):
                try:
                    data = json.loads(line)
                    break
                except ValueError:
                    continue
            if isinstance(data, dict):
                self.session = data.get("session_id") or self.session
                for k in ("result", "response", "text", "content", "output"):
                    if data.get(k):
                        return str(data[k])
            if text.strip():
                return text.strip()
            raise RuntimeError(f"{PLANNERS[self.planner]} gave no reply" + (": " + err.strip()[-400:] if err.strip() else f" (exit code {code})"))
        raise RuntimeError(f"unknown planner {self.planner}")


def run(planner, text, delegate, show, should_stop=lambda: False, on_plan=lambda steps: None,
        before_turn=lambda: None, after_turn=lambda: None, delegate_cloud=None):
    """One user request. The tool's replies go to `show`, its plan to `on_plan(steps)`, its [l] tasks to
    `delegate(task) -> report`. before_turn / after_turn bracket every run of the tool, which may edit files."""
    def ask(prompt):
        before_turn()
        try:
            return planner.ask(prompt, should_stop)
        finally:
            after_turn()

    reply = ask(text)
    if planner.solo and not planner.providers_info:
        show(reply.strip() or "[no reply]")
        return
    for _ in range(MAX_HANDOFFS):
        shown, task, plan, meta = split_reply_meta(reply)
        if plan:
            on_plan(parse_plan(plan))
        if shown:
            show(shown)
        if task is None:
            return
        prov = (meta.get("provider") or "").strip().lower()
        if prov and prov != "local" and delegate_cloud is not None:
            report = delegate_cloud(task, meta)
            reply = ask(f"Provider {prov} has finished the delegated task.\n\n" + report)
        else:
            report = delegate(task)
            reply = ask("The worker has finished the [l] task.\n\n" + report)
    show("[stopped: more hand-offs than SiCode allows in one request]")
