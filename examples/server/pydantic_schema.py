from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")


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

    class Config:
        extra = "forbid"  # Forbid extra fields


class Fleet(BaseModel):
    fleet_name: str = Field(..., title="Fleet Name", min_length=1, max_length=50)
    airplanes: List[Airplane] = Field(
        ..., title="Fleet Airplanes", min_length=1, max_length=3
    )

    class Config:
        extra = "forbid"  # Forbid extra fields


fleet_schema = Fleet.model_json_schema()

completion = client.chat.completions.create(
    model="mistral",
    messages=[
        {
            "role": "user",
            "content": "Can you please make me a fleet of airplanes?",
        }
    ],
    frequency_penalty=1.0,
    top_p=0.1,
    temperature=0,
    response_format={"type": "json_schema", "json_schema": fleet_schema},
)

print(completion.choices[0].message.content)
