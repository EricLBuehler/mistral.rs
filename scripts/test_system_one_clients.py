"""Test a running mistral CLM server with the real Pydantic and TypeSafe clients.

pip install pydantic-ai-slim==2.54.0 typesafe-sdk==0.7.2
SYSTEM_ONE_BASE_URL=http://localhost:1234 SYSTEM_ONE_MODEL=default python scripts/test_system_one_clients.py
"""

import asyncio
import os
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.decision import (
    ChoiceQuestion,
    DecisionRequest,
    NoulQuestion,
    ScoreQuestion,
)
from pydantic_ai.models.system_one import SystemOneModel
from pydantic_ai.providers.system_one import SystemOneProvider
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

BASE_URL = os.environ.get("SYSTEM_ONE_BASE_URL", "http://localhost:1234")
MODEL = os.environ.get("SYSTEM_ONE_MODEL", "default")
API_KEY = os.environ.get("SYSTEM_ONE_API_KEY")
STATE = {"customer": "My invoice was charged twice.", "channel": "phone"}


class Triage(BaseModel):
    urgent: bool = Field(description="Is this urgent?")
    team: Literal["billing", "technical"] = Field(description="Which team?")


async def main():
    provider = SystemOneProvider(base_url=BASE_URL, api_key=API_KEY)
    model = SystemOneModel(MODEL, provider=provider)
    response = await model.decide(
        DecisionRequest(
            state=STATE,
            questions={
                "urgent": NoulQuestion(instructions="Is this urgent?"),
                "team": ChoiceQuestion(
                    instructions="Which team?",
                    criteria={"billing": None, "technical": None},
                ),
                "anger": ScoreQuestion(
                    instructions="How angry?",
                    criteria=["Calm", "Frustrated", "Very angry"],
                ),
                "bare": NoulQuestion(),
            },
        ),
        {"temperature": 1.0, "timeout": 120},
    )
    assert 0 <= response.answers["urgent"].noul <= 1
    assert response.answers["team"].choice in {"billing", "technical"}
    assert set(response.answers["anger"].probabilities) == {0, 1, 2}
    assert response.usage.output_tokens == 0
    result = await Agent(model, output_type=Triage).run("My invoice was charged twice.")
    assert isinstance(result.output, Triage)
    assert result.response.provider_details is not None
    with TypeSafeClient(
        base_url=BASE_URL.removesuffix("/v1"), api_key=API_KEY or "local", model=MODEL
    ) as client:
        result = client.system_one(
            state=STATE,
            questions={
                "urgent": Noul(instructions="Is this urgent?"),
                "team": Choice(
                    instructions="Which team?",
                    criteria={"billing": None, "technical": None},
                ),
                "anger": Score(
                    instructions="How angry?",
                    criteria=["Calm", "Frustrated", "Very angry"],
                ),
            },
        )
        assert 0 <= result.answers["urgent"].noul <= 1
        assert result.answers["team"].choice in {"billing", "technical"}
        assert set(result.answers["anger"].probabilities) == {0, 1, 2}
        assert client.models.list().models
    print("Pydantic SystemOneModel, Agent structured output, and TypeSafe SDK passed.")


if __name__ == "__main__":
    asyncio.run(main())
