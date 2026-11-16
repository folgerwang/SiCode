"""File and shell tools shared by Claude and the local model. Every path is confined to the workspace."""
import fnmatch
import os
import re
import signal
import subprocess
import tempfile
import time

MAX_OUT = 20000  # characters of tool output handed back to a model
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "build", "dist", ".vs", "models", ".pytest_cache"}
MEDIA_EXTS = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
    ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
    ".mkv": "video/x-matroska", ".avi": "video/x-msvideo",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4", ".aac": "audio/aac",
    ".flac": "audio/flac", ".ogg": "audio/ogg", ".opus": "audio/ogg", ".wma": "audio/x-ms-wma",
}


def read_clipboard_media(dest_dir):
    """Read image or copied image/video/audio files from the Windows clipboard into dest_dir."""
    if os.name != "nt":
        return []
    import json as _json
    os.makedirs(dest_dir, exist_ok=True)
    clip_png = os.path.join(dest_dir, f"clip_{int(time.time() * 1000)}.png")
    ps = (
        "Add-Type -AssemblyName System.Windows.Forms,System.Drawing; "
        "$out = @(); "
        "if ([System.Windows.Forms.Clipboard]::ContainsFileDropList()) { "
        "  foreach ($f in [System.Windows.Forms.Clipboard]::GetFileDropList()) { $out += $f } "
        "} elseif ([System.Windows.Forms.Clipboard]::ContainsImage()) { "
        "  $img = [System.Windows.Forms.Clipboard]::GetImage(); "
        f"  $img.Save('{clip_png.replace(chr(39), chr(39)*2)}', [System.Drawing.Imaging.ImageFormat]::Png); "
        f"  $out += '{clip_png.replace(chr(39), chr(39)*2)}'; "
        "} "
        "$out | ConvertTo-Json -Compress"
    )
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-STA", "-Command", ps],
                           capture_output=True, text=True, timeout=8, creationflags=flags)
    except (OSError, subprocess.TimeoutExpired):
        return []
    raw = (r.stdout or "").strip()
    if not raw:
        return []
    try:
        paths = _json.loads(raw)
    except ValueError:
        return []
    if isinstance(paths, str):
        paths = [paths]
    items = []
    for p in paths:
        if not isinstance(p, str) or not os.path.isfile(p):
            continue
        ext = os.path.splitext(p)[1].lower()
        mime = MEDIA_EXTS.get(ext)
        if mime:
            items.append({"name": os.path.basename(p), "mime": mime, "path": p})
    return items


def clip(text, limit=MAX_OUT):
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n... [{len(text) - limit} characters omitted] ...\n" + text[-half:]


def _kill_tree(pid):
    """Terminate a process and all of its descendants."""
    if not pid:
        return
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except OSError:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass


class Workspace:
    """The project directory and the tools that act on it.

    approve_command(command) -> bool and approve_edit(path, old, new) -> bool are asked before a shell command
    runs and before a file changes (old is None for a new file). show_plan(text) displays the task list.
    """

    def __init__(self, root, approve_command, approve_edit=lambda path, old, new: True, show_plan=lambda text: None,
                 should_stop=lambda: False):
        self.root = os.path.realpath(root)
        self.approve_command = approve_command
        self.approve_edit = approve_edit
        self.show_plan = show_plan
        self.should_stop = should_stop
        self._active_proc = None
        self.changed = set()   # files written since the caller last cleared it
        self.backups = {}      # path -> content before the first change since begin_turn() (None: did not exist)

    def begin_turn(self):
        self.backups = {}

    def path(self, rel):
        p = os.path.realpath(os.path.join(self.root, rel))
        if p != self.root and not p.startswith(self.root + os.sep):
            raise ValueError(f"{rel} is outside the workspace")
        return p

    def rel(self, p):
        return os.path.relpath(p, self.root).replace("\\", "/")

    def walk(self):
        for d, dirs, files in os.walk(self.root):
            dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
            for f in sorted(files):
                yield os.path.join(d, f)

    def _store(self, p, old, new):
        """Writes `new` to p after asking; returns None, or the error text for the model."""
        r = self.rel(p)
        if not self.approve_edit(r, old, new):
            return "error: the user declined this change"
        self.backups.setdefault(r, old)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(new)
        self.changed.add(r)
        return None

    def restore(self, backups):
        """Puts files back to the contents in `backups` (None deletes the file). Returns the paths restored."""
        for r, old in backups.items():
            p = self.path(r)
            if old is None:
                if os.path.exists(p):
                    os.remove(p)
            else:
                os.makedirs(os.path.dirname(p), exist_ok=True)  # the file may have been deleted with its folder
                with open(p, "w", encoding="utf-8", newline="") as f:
                    f.write(old)
        return sorted(backups)

    def snapshot(self, max_files=8000, max_size=1_000_000):
        """The text files of the workspace, {path: content}: taken before a tool that edits on its own runs."""
        snap = {}
        for p in self.walk():
            if len(snap) >= max_files:
                break
            try:
                if os.path.getsize(p) <= max_size:
                    with open(p, encoding="utf-8", newline="") as f:
                        snap[self.rel(p)] = f.read()
            except (UnicodeDecodeError, OSError):
                continue
        return snap

    def changes_since(self, snap):
        """[(path, old, new)] for files that differ from the snapshot (old None: new file, new None: deleted).

        Records them like the changes of SiCode's own tools, so they can be undone.
        """
        now = self.snapshot()
        out = []
        for r in sorted(set(snap) | set(now)):
            old, new = snap.get(r), now.get(r)
            if old == new:
                continue
            if new is None and os.path.exists(self.path(r)):
                continue  # still there, only no longer readable as text
            out.append((r, old, new))
            self.backups.setdefault(r, old)
            self.changed.add(r)
        return out

    # --- tools: each returns the text shown to the model ---

    def read_file(self, path, offset=1, limit=400):
        with open(self.path(path), encoding="utf-8", errors="replace") as f:
            lines = f.read().split("\n")
        offset, limit = max(1, int(offset)), max(1, int(limit))
        part = lines[offset - 1:offset - 1 + limit]
        out = "\n".join(f"{offset + i}\t{line}" for i, line in enumerate(part))
        if offset - 1 + limit < len(lines):
            out += f"\n[{len(lines)} lines in total; read more with offset={offset + limit}]"
        return clip(out) if out else "[empty file]"

    def write_file(self, path, content):
        p = self.path(path)
        old = None
        if os.path.exists(p):
            with open(p, encoding="utf-8", errors="replace", newline="") as f:
                old = f.read()
        return self._store(p, old, content) or f"wrote {self.rel(p)} ({content.count(chr(10)) + 1} lines)"

    def edit_file(self, path, old_text, new_text):
        p = self.path(path)
        with open(p, encoding="utf-8", newline="") as f:
            old = f.read()
        crlf = "\r\n" in old
        text = old.replace("\r\n", "\n") if crlf else old
        n = text.count(old_text)
        if n == 0:
            return "error: old_text was not found in the file (it must match exactly, including indentation)"
        if n > 1:
            return f"error: old_text occurs {n} times; include more surrounding lines so it is unique"
        text = text.replace(old_text, new_text)
        return self._store(p, old, text.replace("\n", "\r\n") if crlf else text) or f"edited {self.rel(p)}"

    def list_files(self, pattern="*"):
        hits = [r for r in (self.rel(p) for p in self.walk())
                if fnmatch.fnmatch(r, pattern) or fnmatch.fnmatch(os.path.basename(r), pattern)]
        if not hits:
            return "no files match"
        return clip("\n".join(hits[:500]) + (f"\n[{len(hits) - 500} more]" if len(hits) > 500 else ""))

    def search(self, pattern, glob="*"):
        rx = re.compile(pattern)
        out = []
        for p in self.walk():
            r = self.rel(p)
            if not (fnmatch.fnmatch(r, glob) or fnmatch.fnmatch(os.path.basename(r), glob)):
                continue
            try:
                if os.path.getsize(p) > 2_000_000:
                    continue
                with open(p, encoding="utf-8") as f:
                    for i, line in enumerate(f, 1):
                        if rx.search(line):
                            out.append(f"{r}:{i}: {line.rstrip()[:300]}")
                            if len(out) >= 200:
                                return clip("\n".join(out) + "\n[stopped at 200 matches]")
            except (UnicodeDecodeError, OSError):
                continue
        return clip("\n".join(out)) if out else "no matches"

    def kill_command(self):
        proc = self._active_proc
        if proc is not None and proc.poll() is None:
            _kill_tree(proc.pid)

    def run_command(self, command, timeout=300):
        if not self.approve_command(command):
            return "error: the user declined to run this command"
        # Redirect to a temp file rather than a pipe: on Windows, `start /b ...` children inherit stdout/stderr
        # handles and would otherwise keep a pipe open indefinitely after the top-level shell exits.
        limit = max(1, int(timeout))
        with tempfile.TemporaryFile(mode="w+b") as out_file:
            kw = {"shell": True, "cwd": self.root, "stdin": subprocess.DEVNULL,
                  "stdout": out_file, "stderr": subprocess.STDOUT}
            if os.name == "nt":
                kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            else:
                kw["start_new_session"] = True
            proc = subprocess.Popen(command, **kw)
            self._active_proc = proc
            deadline = time.monotonic() + limit
            try:
                while proc.poll() is None:
                    if self.should_stop():
                        _kill_tree(proc.pid)
                        raise KeyboardInterrupt
                    if time.monotonic() >= deadline:
                        _kill_tree(proc.pid)
                        return f"error: the command did not finish within {timeout} s"
                    time.sleep(0.1)
                out_file.seek(0)
                output = out_file.read().decode("utf-8", errors="replace")
            finally:
                if self._active_proc is proc:
                    self._active_proc = None
                if proc.poll() is None:
                    _kill_tree(proc.pid)
        return clip(f"exit code {proc.returncode}\n{output}".rstrip())

    def update_plan(self, plan):
        self.show_plan(plan)
        return "plan updated"

    def call(self, name, args):
        fn = {"read_file": self.read_file, "write_file": self.write_file, "edit_file": self.edit_file,
              "list_files": self.list_files, "search": self.search, "run_command": self.run_command,
              "update_plan": self.update_plan}.get(name)
        if fn is None:
            return f"error: unknown tool {name}"
        try:
            return fn(**args)
        except Exception as e:  # the model reads the error and corrects itself
            return f"error: {type(e).__name__}: {e}"


def _tool(name, description, props, required):
    return {"name": name, "description": description,
            "input_schema": {"type": "object", "properties": props, "required": required, "additionalProperties": False}}


S, I = {"type": "string"}, {"type": "integer"}
# Tool definitions in the Claude API shape; the worker converts them to the Qwen chat template's shape.
TOOLS = [
    _tool("read_file", "Read a text file from the workspace. Returns numbered lines.",
          {"path": {**S, "description": "Path relative to the workspace root."},
           "offset": {**I, "description": "First line to read, from 1. Default 1."},
           "limit": {**I, "description": "Number of lines. Default 400."}}, ["path"]),
    _tool("write_file", "Create a file or replace its whole content. Use edit_file for a change to an existing file.",
          {"path": S, "content": {**S, "description": "The complete new content of the file."}}, ["path", "content"]),
    _tool("edit_file", "Replace one exact occurrence of old_text in a file with new_text.",
          {"path": S, "old_text": {**S, "description": "Text to replace; must occur exactly once in the file."},
           "new_text": S}, ["path", "old_text", "new_text"]),
    _tool("list_files", "List workspace files whose path or name matches a glob pattern such as *.py or src/*.cpp.",
          {"pattern": {**S, "description": "Glob pattern. Default *."}}, []),
    _tool("search", "Search file contents with a regular expression. Returns path:line: text.",
          {"pattern": {**S, "description": "Python regular expression."},
           "glob": {**S, "description": "Only search files matching this glob. Default *."}}, ["pattern"]),
    _tool("run_command", "Run a shell command in the workspace root (Windows cmd) and return its exit code and output. "
          "Use it to build and to run tests.",
          {"command": S, "timeout": {**I, "description": "Seconds. Default 300."}}, ["command"]),
    _tool("update_plan", "Show the user your task list for a job with several steps, and update it as steps finish. "
          "Send the whole list each time.",
          {"plan": {**S, "description": "One step per line, each starting with [ ] (to do), [~] (in progress) or "
                                        "[x] (done)."}}, ["plan"]),
]
