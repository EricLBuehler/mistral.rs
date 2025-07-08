from enum import Enum
import json
from typing import List, Optional

from pydantic import BaseModel, Field

from mistralrs import Runner, Which, ChatCompletionRequest

runner = Runner(
    which=Which.Plain(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
    ),
)


class AirplaneType(str, Enum):
    commercial = "commercial"
    cargo = "cargo"
    private = "private"


class Airplane(BaseModel):
    id: int = Field(
        ..., title="Airplane ID", description="Unique identifier for the airplane"
    )
    model: str = Field(..., title="Model", min_length=1, max_length=100)
    manufacturer: str = Field(..., title="Manufacturer", min_length=1, max_length=100)
    type: AirplaneType = Field(default=AirplaneType.commercial, title="Airplane Type")
    capacity: Optional[int] = Field(None, title="Passenger Capacity", ge=1, le=1000)


class Fleet(BaseModel):
    fleet_name: str = Field(..., title="Fleet Name", min_length=1, max_length=50)
    airplanes: List[Airplane] = Field(..., title="Fleet Airplanes", min_length=1)


fleet_schema = Fleet.model_json_schema()

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[{"role": "user", "content": "Give me a sample address."}],
        max_tokens=256,
        temperature=0.1,
        grammar_type="json_schema",
        grammar=json.dumps(fleet_schema),
    )
)
print(res.choices[0].message.content)
print(res.usage)
