from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


Name = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]

Identifier = Annotated[
    str,
    StringConstraints(
        pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$",
    ),
]


class RequestModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


class ObjectInventoryRequest(RequestModel):
    types: list[Name] = Field(min_length=1)
    colors: list[Name] = Field(min_length=1)


class CollectionSessionRequest(RequestModel):
    color: Name
    type: Name
    episodes: int = Field(gt=0)


class CollectionPlanRequest(RequestModel):
    task: Literal["pick_place"]
    dataset: Identifier
    seed: int
    target: Name
    distractor: bool
    positions: list[Name] = Field(min_length=1)
    objects: ObjectInventoryRequest
    sessions: list[CollectionSessionRequest] = Field(min_length=1)
    episode_time_s: float = Field(gt=0)
    reset_time_s: float = Field(ge=0)
    
class CollectionInputRequest(RequestModel):
    dataset: Identifier
    instruction: Name
    num_episodes: int = Field(gt=0)
    episode_time_s: float = Field(gt=0)
    reset_time_s: float = Field(ge=0)
    
class CollectionSessionInput(RequestModel):
    instruction: Name
    num_episodes: int = Field(gt=0)
    episode_time_s: float = Field(gt=0)
    reset_time_s: float = Field(ge=0)


class CollectionStartRequest(RequestModel):
    dataset: Identifier
    seed: int
    session_index: int = Field(gt=0)
    sessions: list[CollectionSessionInput] = Field(min_length=1)


class CollectionCommandRequest(RequestModel):
    action: Literal[
        "start_episode",
        "finish_episode",
        "retry_episode",
    ]
    recording_id: Name
    expected_phase: Literal[
        "preparing",
        "recording",
        "resetting",
    ]
    expected_episode: int = Field(gt=0)


class CollectionCommandResponse(BaseModel):
    command_id: str
    status: Literal["queued"]
