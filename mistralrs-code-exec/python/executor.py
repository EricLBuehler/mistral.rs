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

# ── Setup ──────────────────────────────────────────────────────────────────

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

# _render_frame is injected into the namespace after matplotlib setup (see below).

# Force matplotlib to non-interactive backend before any user code.
# Also install a savefig/show hook so that figures are captured even if the
# user calls plt.savefig() then plt.close().
_captured_figures = []

# When True, savefig captures go to _animation_frames (video) instead of
# _captured_figures (images). Set by the Animation.save hook.
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
        _animation_frames.clear()
        try:
            result = _orig_animation_save(self, *args, **kwargs)
        finally:
            _in_animation_save = False
        # Move captured frames into the namespace _video_frames list.
        if _animation_frames:
            vf = namespace.get("_video_frames")
            if not isinstance(vf, list):
                namespace["_video_frames"] = list(_animation_frames)
            else:
                vf.extend(_animation_frames)
            _animation_frames.clear()
        return result

    matplotlib.figure.Figure.savefig = _hooked_savefig
    _plt.show = _hooked_show
    _animation.Animation.save = _hooked_animation_save
except ImportError:
    _orig_savefig = None

_HAS_MATPLOTLIB = _orig_savefig is not None


def _render_frame(fig):
    """Render a matplotlib figure to PNG bytes without triggering the capture hook.

    Used by model code to build video frames via `_video_frames.append(_render_frame(fig))`.
    """
    if not _HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib is not available")
    buf = io.BytesIO()
    _orig_savefig(fig, buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return buf.read()


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
            return {"format": "png", "data_base64": base64.b64encode(buf.read()).decode()}
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

    # Capture video frames from the magic `_video_frames` variable.
    video_frames = []
    raw_frames = namespace.get("_video_frames")
    if isinstance(raw_frames, list) and raw_frames:
        for frame in raw_frames:
            if isinstance(frame, bytes):
                video_frames.append(
                    {"format": "png", "data_base64": base64.b64encode(frame).decode()}
                )
        # Clear so next execution starts fresh.
        namespace["_video_frames"] = []

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
    send({
        "stdout": "",
        "stderr": "",
        "exception": msg,
        "last_expr_repr": None,
        "last_expr_type": None,
        "images": [],
        "execution_time_ms": 0,
    })


# ── Main loop ──────────────────────────────────────────────────────────────

# Ensure KeyboardInterrupt propagates cleanly inside execute_code,
# but does not kill the main loop.
signal.signal(signal.SIGINT, signal.default_int_handler)

# Inject helpers into the namespace so model code can use them.
if _HAS_MATPLOTLIB:
    namespace["_render_frame"] = _render_frame

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
            }
        # Mask SIGINT while building and sending the response so a
        # late-arriving signal cannot corrupt the protocol write.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        elapsed_ms = int((time.time() - start) * 1000)
        response = {
            **result,
            "execution_time_ms": elapsed_ms,
        }
        send(response)
        signal.signal(signal.SIGINT, signal.default_int_handler)

    elif msg_type == "reset":
        namespace.clear()
        namespace["__builtins__"] = __builtins__
        namespace["__name__"] = "__main__"
        if _HAS_MATPLOTLIB:
            namespace["_render_frame"] = _render_frame
        _captured_figures.clear()
        send({"success": True})

    elif msg_type == "shutdown":
        break

    else:
        send_error("Unknown message type: {}".format(msg_type))
