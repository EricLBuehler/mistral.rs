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

work_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
os.chdir(work_dir)

namespace = {"__builtins__": __builtins__, "__name__": "__main__"}

# Force matplotlib to non-interactive backend before any user code.
# Also install a savefig/show hook so that figures are captured even if the
# user calls plt.savefig() then plt.close().
_captured_figures = []

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    _orig_savefig = matplotlib.figure.Figure.savefig
    _orig_show = _plt.show

    def _hooked_savefig(self, *args, **kwargs):
        """Intercept savefig to also capture the figure for the model."""
        _capture_single_figure(self)
        return _orig_savefig(self, *args, **kwargs)

    def _hooked_show(*args, **kwargs):
        """Intercept plt.show() to capture all open figures."""
        for fig_num in _plt.get_fignums():
            _capture_single_figure(_plt.figure(fig_num))
        return _orig_show(*args, **kwargs)

    matplotlib.figure.Figure.savefig = _hooked_savefig
    _plt.show = _hooked_show
except ImportError:
    pass


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


def capture_matplotlib_figures():
    """Return all captured figures and also grab any still-open ones."""
    images = list(_captured_figures)
    _captured_figures.clear()

    # Also capture any figures still open that weren't saved/shown.
    try:
        import matplotlib.pyplot as plt

        seen = {id(img) for img in images}
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            buf = io.BytesIO()
            _orig_savefig(fig, buf, format="png", bbox_inches="tight", dpi=150)
            buf.seek(0)
            img = {"format": "png", "data_base64": base64.b64encode(buf.read()).decode()}
            if id(img) not in seen:
                images.append(img)
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
    old_stdout, old_stderr = sys.stdout, sys.stderr

    exception = None
    last_expr_repr = None
    last_expr_type = None
    images = []

    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

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

    # Capture matplotlib figures (runs after user code, even if it raised).
    images.extend(capture_matplotlib_figures())

    return {
        "stdout": stdout_capture.getvalue(),
        "stderr": stderr_capture.getvalue(),
        "exception": exception,
        "last_expr_repr": last_expr_repr,
        "last_expr_type": last_expr_type,
        "images": images,
    }


def send(obj):
    """Write a JSON line to stdout and flush."""
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


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
        elapsed_ms = int((time.time() - start) * 1000)
        response = {
            **result,
            "execution_time_ms": elapsed_ms,
        }
        send(response)

    elif msg_type == "reset":
        namespace.clear()
        namespace["__builtins__"] = __builtins__
        namespace["__name__"] = "__main__"
        send({"success": True})

    elif msg_type == "shutdown":
        break

    else:
        send_error("Unknown message type: {}".format(msg_type))
