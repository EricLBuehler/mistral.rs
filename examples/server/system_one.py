"""Call mistral's System One API. Start a server with the CLM model first."""

import json
import os
from urllib.request import Request, urlopen

base_url = os.environ.get("SYSTEM_ONE_BASE_URL", "http://localhost:1234").rstrip("/")
if not base_url.endswith("/v1"):
    base_url += "/v1"
body = {
    "model": os.environ.get("SYSTEM_ONE_MODEL", "Contrastive-LM/CLM-v0.1-8B"),
    "state": "My invoice was charged twice and nobody answers the phone!",
    "questions": {
        "urgent": {"type": "noul", "instructions": "Is this urgent?"},
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "Charges, invoices, refunds",
                "technical": "Bugs and outages",
            },
        },
        "anger": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "criteria": ["Calm", "Frustrated", "Very angry"],
        },
    },
}
headers = {"Content-Type": "application/json"}
if key := os.environ.get("SYSTEM_ONE_API_KEY"):
    headers["Authorization"] = f"Bearer {key}"
request = Request(f"{base_url}/systemone", json.dumps(body).encode(), headers)
with urlopen(request, timeout=120) as response:
    print(json.dumps(json.load(response), indent=2))
