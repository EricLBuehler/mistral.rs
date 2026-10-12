"""pip install pydantic-ai-slim; requires a release with SystemOneModel."""

import os
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.system_one import SystemOneModel
from pydantic_ai.profiles.decision import DecisionModelProfile
from pydantic_ai.providers.system_one import SystemOneProvider


class Triage(BaseModel):
    urgent: bool = Field(description="Does this need an urgent response?")
    team: Literal["billing", "technical", "account"] = Field(
        description="Which team should handle this request?"
    )
    urgency_probability: float = Field(ge=0, le=1, description="Is this urgent?")


model = SystemOneModel(
    os.environ.get("SYSTEM_ONE_MODEL", "Contrastive-LM/CLM-v0.1-8B"),
    provider=SystemOneProvider(
        base_url=os.environ.get("SYSTEM_ONE_BASE_URL", "http://localhost:1234"),
        api_key=os.environ.get("SYSTEM_ONE_API_KEY"),
    ),
    profile=DecisionModelProfile(
        context_window=2048,
        decision_max_choice_options=255,
        decision_max_score_levels=10,
    ),
)
agent = Agent(model, output_type=Triage, model_settings={"timeout": 120})
result = agent.run_sync("My invoice was charged twice and nobody answers the phone!")
print(result.output.model_dump_json(indent=2))
print(result.response.provider_details)
