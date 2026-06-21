"""
mistral.rs Python code executor.

This script is embedded into the mistralrs-code-exec Rust crate and runs as a
persistent subprocess. It communicates with the Rust host via line-delimited
JSON over stdin/stdout.

Launch with: python3 -u <this_script> <work_dir>

Features:
- Persistent namespace across execute calls (Jupyter-like)
- Last-expression capture (if the final statement is an expression, eval it)
- Matplotlib figure capture (saved as base64 PNG)
- PIL Image capture from last expression
- Pandas DataFrame repr from last expression
- stdout/stderr capture
- Clean SIGINT handling (KeyboardInterrupt preserves session)
"""

import sys
import os
import json
import io
import ast
import traceback
import time
import base64
import signal

# Keep a reference to the real stdout before user code can redirect it.
_real_stdout = sys.stdout


class _BlockedStdin:
    """Stub that raises an error if user code tries to read from stdin."""

    def read(self, *a, **kw):
        raise RuntimeError("input() / stdin.read() is not supported in code execution")

    def readline(self, *a, **kw):
        raise RuntimeError("input() / stdin.read() is not supported in code execution")

    def readlines(self, *a, **kw):
        raise RuntimeError("input() / stdin.read() is not supported in code execution")

    def __iter__(self):
        raise RuntimeError("iterating stdin is not supported in code execution")

    def __next__(self):
        raise RuntimeError("iterating stdin is not supported in code execution")


_blocked_stdin = _BlockedStdin()

work_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
os.chdir(work_dir)

namespace = {"__builtins__": __builtins__, "__name__": "__main__"}

# matplotlib forced to a non-interactive backend; savefig/show hooks capture figures even after plt.close().
_captured_figures = []

# When True, savefig captures route to _animation_frames (video) instead of _captured_figures (images).
_in_animation_save = False
_animation_frames = []

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    import matplotlib.animation as _animation

    _orig_savefig = matplotlib.figure.Figure.savefig
    _orig_show = _plt.show
    _orig_animation_save = _animation.Animation.save

    def _hooked_savefig(self, *args, **kwargs):
        """Intercept savefig to also capture the figure for the model."""
        if _in_animation_save:
            _capture_animation_frame(self)
        else:
            _capture_single_figure(self)
        return _orig_savefig(self, *args, **kwargs)

    def _hooked_show(*args, **kwargs):
        """Intercept plt.show() to capture all open figures."""
        for fig_num in _plt.get_fignums():
            _capture_single_figure(_plt.figure(fig_num))
        return _orig_show(*args, **kwargs)

    def _hooked_animation_save(self, *args, **kwargs):
        """Intercept Animation.save to capture frames as video."""
        global _in_animation_save
        _in_animation_save = True
        try:
            return _orig_animation_save(self, *args, **kwargs)
        finally:
            _in_animation_save = False

    matplotlib.figure.Figure.savefig = _hooked_savefig
    _plt.show = _hooked_show
    _animation.Animation.save = _hooked_animation_save
except ImportError:
    _orig_savefig = None

_HAS_MATPLOTLIB = _orig_savefig is not None


def _capture_single_figure(fig):
    """Render a figure to PNG and append to the capture list."""
    try:
        buf = io.BytesIO()
        _orig_savefig(fig, buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        _captured_figures.append(
            {"format": "png", "data_base64": base64.b64encode(buf.read()).decode()}
        )
    except Exception:
        pass


def _capture_animation_frame(fig):
    """Render a figure to PNG bytes and append to the animation frames list."""
    try:
        buf = io.BytesIO()
        _orig_savefig(fig, buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        _animation_frames.append(buf.read())
    except Exception:
        pass


def capture_matplotlib_figures():
    """Return all captured figures and also grab any still-open ones."""
    images = list(_captured_figures)
    _captured_figures.clear()

    # Also capture any figures still open that weren't saved/shown.
    try:
        import matplotlib.pyplot as plt

        seen = {img["data_base64"] for img in images}
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            buf = io.BytesIO()
            _orig_savefig(fig, buf, format="png", bbox_inches="tight", dpi=150)
            buf.seek(0)
            data = base64.b64encode(buf.read()).decode()
            if data not in seen:
                images.append({"format": "png", "data_base64": data})
                seen.add(data)
        plt.close("all")
    except Exception:
        pass

    return images


def detect_pil_image(obj):
    """If obj is a PIL Image, return it as base64 PNG."""
    try:
        from PIL import Image

        if isinstance(obj, Image.Image):
            buf = io.BytesIO()
            obj.save(buf, format="PNG")
            buf.seek(0)
            return {
                "format": "png",
                "data_base64": base64.b64encode(buf.read()).decode(),
            }
    except ImportError:
        pass
    except Exception:
        pass
    return None


def detect_dataframe(obj):
    """If obj is a pandas DataFrame/Series, return its repr and type name."""
    try:
        import pandas as pd

        if isinstance(obj, (pd.DataFrame, pd.Series)):
            return repr(obj), type(obj).__name__
    except ImportError:
        pass
    return None, None


# Formats we read as utf-8 text. Anything else is treated as binary and
# base64-encoded.
_TEXT_FORMATS = frozenset(
    {
        "csv",
        "tsv",
        "json",
        "geojson",
        "xml",
        "yaml",
        "yml",
        "toml",
        "md",
        "markdown",
        "html",
        "htm",
        "svg",
        "latex",
        "tex",
        "sql",
        "python",
        "py",
        "rust",
        "rs",
        "txt",
        "text",
        "log",
        "vega",
        "vega-lite",
    }
)

# Format → mime hints so the Rust side can classify without sniffing.
_FORMAT_MIME = {
    "csv": "text/csv",
    "tsv": "text/tab-separated-values",
    "json": "application/json",
    "geojson": "application/json",
    "xml": "application/xml",
    "yaml": "application/yaml",
    "yml": "application/yaml",
    "toml": "application/toml",
    "md": "text/markdown",
    "markdown": "text/markdown",
    "html": "text/html",
    "htm": "text/html",
    "svg": "image/svg+xml",
    "latex": "application/x-tex",
    "tex": "application/x-tex",
    "sql": "application/sql",
    "python": "text/x-python",
    "py": "text/x-python",
    "rust": "text/x-rust",
    "rs": "text/x-rust",
    "txt": "text/plain",
    "text": "text/plain",
    "log": "text/plain",
    "vega": "application/json",
    "vega-lite": "application/json",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "mp4": "video/mp4",
    "webm": "video/webm",
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "pdf": "application/pdf",
    "parquet": "application/x-parquet",
    "zip": "application/zip",
}


# Max size of a declared output file we'll read into memory. Larger files become an error placeholder. Override via MISTRALRS_MAX_OUTPUT_BYTES.
MAX_OUTPUT_BYTES = int(
    os.environ.get("MISTRALRS_MAX_OUTPUT_BYTES", str(256 * 1024 * 1024))
)


def _read_output_file(entry):
    """Read one declared output. Returns a dict matching the Rust ExecuteFile."""
    name = entry.get("name")
    fmt = (entry.get("format") or "").lower()
    if not fmt and isinstance(name, str) and "." in name:
        fmt = name.rsplit(".", 1)[-1].lower()
    mime = _FORMAT_MIME.get(fmt, "application/octet-stream")
    out = {"name": name, "format": fmt, "mime_type": mime, "size_bytes": 0}

    if not name:
        out["error"] = "missing name"
        return out

    path = os.path.join(work_dir, name)
    try:
        if not os.path.exists(path):
            out["error"] = "not produced"
            return out
        size = os.path.getsize(path)
        out["size_bytes"] = size
        if size > MAX_OUTPUT_BYTES:
            out["error"] = "exceeds max output size ({} bytes; cap is {} bytes)".format(
                size, MAX_OUTPUT_BYTES
            )
            return out
        if fmt in _TEXT_FORMATS:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    out["text"] = f.read()
            except UnicodeDecodeError:
                with open(path, "rb") as f:
                    out["data_base64"] = base64.b64encode(f.read()).decode()
                out["mime_type"] = "application/octet-stream"
        else:
            with open(path, "rb") as f:
                out["data_base64"] = base64.b64encode(f.read()).decode()
    except Exception as e:
        out["error"] = "read failed: {}".format(e)
    return out


def extract_outputs(outputs):
    """Iterate the declared outputs list, collecting file dicts."""
    if not outputs:
        return []
    results = []
    for entry in outputs:
        if not isinstance(entry, dict):
            continue
        results.append(_read_output_file(entry))
    return results


def execute_code(code):
    """Execute code with last-expression capture (Jupyter-style)."""
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    old_stdout, old_stderr, old_stdin = sys.stdout, sys.stderr, sys.stdin

    exception = None
    last_expr_repr = None
    last_expr_type = None
    images = []

    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        sys.stdin = _blocked_stdin

        # Parse the code into an AST.
        tree = ast.parse(code)

        # If the last statement is an expression, separate it for eval.
        last_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr_node = tree.body.pop()
            last_expr = ast.Expression(body=last_expr_node.value)
            ast.fix_missing_locations(last_expr)

        # Compile and exec the main body.
        if tree.body:
            compiled = compile(tree, "<code>", "exec")
            exec(compiled, namespace)

        # Eval the last expression.
        if last_expr is not None:
            compiled_expr = compile(last_expr, "<code>", "eval")
            result = eval(compiled_expr, namespace)
            if result is not None:
                # Check for special types.
                pil_img = detect_pil_image(result)
                if pil_img:
                    images.append(pil_img)
                    last_expr_repr = "<PIL Image size={}>".format(
                        getattr(result, "size", "?")
                    )
                    last_expr_type = "PIL.Image"
                else:
                    df_repr, df_type = detect_dataframe(result)
                    if df_repr:
                        last_expr_repr = df_repr
                        last_expr_type = df_type
                    else:
                        last_expr_repr = repr(result)
                        last_expr_type = type(result).__name__

                namespace["_"] = result

    except KeyboardInterrupt:
        exception = "KeyboardInterrupt: Execution was interrupted."
    except SystemExit as e:
        exception = "SystemExit: Code called sys.exit({}).".format(e.code)
    except Exception:
        exception = traceback.format_exc()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        sys.stdin = old_stdin

    # Capture matplotlib figures (runs after user code, even if it raised).
    images.extend(capture_matplotlib_figures())

    # Drain captured video frames from any FuncAnimation.save() calls.
    video_frames = [
        {"format": "png", "data_base64": base64.b64encode(frame).decode()}
        for frame in _animation_frames
    ]
    _animation_frames.clear()

    return {
        "stdout": stdout_capture.getvalue(),
        "stderr": stderr_capture.getvalue(),
        "exception": exception,
        "last_expr_repr": last_expr_repr,
        "last_expr_type": last_expr_type,
        "images": images,
        "video_frames": video_frames,
    }


def send(obj):
    """Write a JSON line to the real stdout and flush."""
    _real_stdout.write(json.dumps(obj) + "\n")
    _real_stdout.flush()


def send_error(msg):
    """Send an error response for protocol-level failures."""
    send(
        {
            "stdout": "",
            "stderr": "",
            "exception": msg,
            "last_expr_repr": None,
            "last_expr_type": None,
            "images": [],
            "execution_time_ms": 0,
        }
    )


# KeyboardInterrupt should propagate inside execute_code but not kill the main loop.
signal.signal(signal.SIGINT, signal.default_int_handler)

# Use iter(readline, "") so that output is not buffered on the read side.
# The script must be launched with python3 -u for fully unbuffered I/O.
for line in iter(sys.stdin.readline, ""):
    line = line.strip()
    if not line:
        continue

    try:
        msg = json.loads(line)
    except json.JSONDecodeError as e:
        send_error("JSONDecodeError: {}".format(e))
        continue

    msg_type = msg.get("type")

    if msg_type == "execute":
        start = time.time()
        outputs_decl = msg.get("outputs") or []
        try:
            result = execute_code(msg["code"])
        except KeyboardInterrupt:
            # KeyboardInterrupt arrived between calls or during AST parse
            result = {
                "stdout": "",
                "stderr": "",
                "exception": "KeyboardInterrupt: Execution was interrupted.",
                "last_expr_repr": None,
                "last_expr_type": None,
                "images": [],
                "video_frames": [],
            }
        # Read declared output files (runs even if user code raised, so
        # partial successes still surface).
        try:
            files = extract_outputs(outputs_decl)
        except Exception as e:
            files = [
                {
                    "name": (o or {}).get("name") if isinstance(o, dict) else None,
                    "format": ((o or {}).get("format") if isinstance(o, dict) else "")
                    or "",
                    "size_bytes": 0,
                    "error": "output scan failed: {}".format(e),
                }
                for o in outputs_decl
            ]
        # Mask SIGINT while building and sending the response so a
        # late-arriving signal cannot corrupt the protocol write.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        elapsed_ms = int((time.time() - start) * 1000)
        response = {
            **result,
            "files": files,
            "execution_time_ms": elapsed_ms,
        }
        send(response)
        signal.signal(signal.SIGINT, signal.default_int_handler)

    elif msg_type == "reset":
        namespace.clear()
        namespace["__builtins__"] = __builtins__
        namespace["__name__"] = "__main__"
        _captured_figures.clear()
        _animation_frames.clear()
        send({"success": True})

    elif msg_type == "shutdown":
        break

    else:
        send_error("Unknown message type: {}".format(msg_type))
