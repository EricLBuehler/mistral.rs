#!/usr/bin/env python3
"""
HTTP code-execution approval flow.

Start the server:
    mistralrs serve --agent -p 1234 -m Qwen/Qwen3-4B

Then run:
    python examples/server/code_execution_approval.py

The request uses agent_permission="ask", so the server streams an
agentic_tool_approval_required event before running an agent action. This client
shows Python code, asks for approval locally, and resolves it with the approval endpoint.
"""

import json
import sys
import urllib.request

BASE_URL = "http://localhost:1234/v1"


def post_json(path, payload):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def iter_sse_events(resp):
    event = "message"
    data_lines = []

    for raw_line in resp:
        line = raw_line.decode("utf-8").rstrip("\n")
        if line.endswith("\r"):
            line = line[:-1]

        if not line:
            if data_lines:
                yield event, "\n".join(data_lines)
            event = "message"
            data_lines = []
            continue

        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line.removeprefix("event:").strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())

    if data_lines:
        yield event, "\n".join(data_lines)


def approve_action(payload):
    print("\n\nApproval required")
    print(f"approval_id: {payload['approval_id']}")
    print(f"session_id: {payload['session_id']}")
    print(f"tool: {payload['tool']['label']} ({payload['tool']['kind']})")
    arguments = payload.get("arguments", {})
    if arguments.get("outputs"):
        print(f"outputs: {', '.join(arguments['outputs'])}")
    print("\nCode:")
    print(arguments.get("code", "<no code>"))

    while True:
        decision = (
            input("\nRun this Python code? [y]es / [n]o / [a]lways: ").strip().lower()
        )
        if decision in {"y", "yes"}:
            return "approve", False
        if decision in {"a", "always"}:
            return "approve", True
        if decision in {"", "n", "no"}:
            return "deny", False
        print("Please enter y, n, or a.")


def main():
    body = {
        "model": "default",
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": "Use Python to calculate and plot the first 20 Fibonacci numbers.",
            }
        ],
        "enable_code_execution": True,
        "agent_permission": "ask",
        "max_tool_rounds": 4,
        "session_id": "approval-demo",
    }

    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        for event, data in iter_sse_events(resp):
            if data == "[DONE]":
                print()
                return

            payload = json.loads(data)
            if event == "agentic_tool_approval_required":
                decision, remember = approve_action(payload)
                result = post_json(
                    f"/agent/approvals/{payload['approval_id']}",
                    {
                        "decision": decision,
                        "remember_for_session": remember,
                        "message": None
                        if decision == "approve"
                        else "The user denied this action.",
                    },
                )
                print(f"approval {result['status']}: {decision}\n")
                continue

            if event == "agentic_tool_call_progress":
                phase = payload.get("phase")
                tool_name = payload.get("tool_name")
                print(f"\n[{tool_name} {phase}]\n")
                continue

            if event == "file_produced":
                print(f"\n[file: {payload['name']} ({payload['format']})]\n")
                continue

            for choice in payload.get("choices", []):
                delta = choice.get("delta", {})
                text = delta.get("content")
                if text:
                    print(text, end="")
                    sys.stdout.flush()


if __name__ == "__main__":
    main()
