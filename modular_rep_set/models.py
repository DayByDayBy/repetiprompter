"""Pydantic models for repetiprompter configuration and output schemas (v0)."""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
import uuid


class TopologyMode(str, Enum):
    CHAIN = "chain"
    TREE = "tree"


class FramingStrategy(str, Enum):
    SIMPLE = "simple"
    LIAR_PARADOX = "liar_paradox"
    DISCUSSION = "discussion"
    REPHRASE = "rephrase"
    ECHO = "echo"
    CUSTOM = "custom"


class TemperatureRegimeType(str, Enum):
    STATIC = "static"
    RAMP = "ramp"
    SCHEDULE = "schedule"


class RunIdentity(BaseModel):
    run_id: Optional[str] = None
    created_at: Optional[datetime] = None
    git_commit: Optional[str] = None
    notes: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        if self.run_id is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = uuid.uuid4().hex[:4]
            self.run_id = f"{ts}_{suffix}"
        if self.created_at is None:
            self.created_at = datetime.now()


class ModelConfig(BaseModel):
    model_name: str = "stablelm2:zephyr"
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    repeat_penalty: float = Field(default=1.1, ge=0.0)
    seed: Optional[int] = None


class PromptingConfig(BaseModel):
    initial_prompt: str
    framing_strategy: FramingStrategy = FramingStrategy.SIMPLE
    custom_prefix: str = ""
    custom_suffix: str = ""


class ReminderConfig(BaseModel):
    enabled: bool = False
    probability: float = Field(default=0.3, ge=0.0, le=1.0)
    content: str = "original"


class ChainConfig(BaseModel):
    steps: int = Field(default=10, ge=1)


class TreeConfig(BaseModel):
    depth: int = Field(default=5, ge=1)
    branching_factor: int = Field(default=3, ge=1)
    branch_all_nodes: bool = True


class TopologyConfig(BaseModel):
    mode: TopologyMode = TopologyMode.CHAIN
    chain: ChainConfig = Field(default_factory=ChainConfig)
    tree: TreeConfig = Field(default_factory=TreeConfig)


class StaticTempConfig(BaseModel):
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)


class RampTempConfig(BaseModel):
    min_temperature: float = Field(default=0.5, ge=0.0, le=2.0)
    max_temperature: float = Field(default=1.2, ge=0.0, le=2.0)
    max_depth: int = Field(default=5, ge=1)


class ScheduleTempConfig(BaseModel):
    temperatures: Dict[int, float] = Field(default_factory=dict)
    default: float = Field(default=1.0, ge=0.0, le=2.0)


class TemperatureRegimeConfig(BaseModel):
    type: TemperatureRegimeType = TemperatureRegimeType.STATIC
    static: StaticTempConfig = Field(default_factory=StaticTempConfig)
    ramp: RampTempConfig = Field(default_factory=RampTempConfig)
    schedule: ScheduleTempConfig = Field(default_factory=ScheduleTempConfig)


class OutputConfig(BaseModel):
    output_dir: str = "./runs"
    format: str = "jsonl"
    flush_every: int = Field(default=10, ge=1)
    include_metadata: bool = True


class RunConfig(BaseModel):
    """Complete configuration for a repetiprompter run."""
    run_identity: RunIdentity = Field(default_factory=RunIdentity)
    model: ModelConfig = Field(default_factory=ModelConfig)
    prompting: PromptingConfig
    reminder: ReminderConfig = Field(default_factory=ReminderConfig)
    topology: TopologyConfig = Field(default_factory=TopologyConfig)
    temperature_regime: TemperatureRegimeConfig = Field(default_factory=TemperatureRegimeConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator('prompting', mode='before')
    @classmethod
    def validate_prompting(cls, v):
        if isinstance(v, dict) and 'initial_prompt' not in v:
            raise ValueError('initial_prompt is required in prompting config')
        return v


class NodeOutput(BaseModel):
    """Output schema for a single generated node (one line in JSONL)."""
    run_id: str
    node_id: str
    parent_id: Optional[str] = None
    depth: int = Field(ge=0)
    step_index: int = Field(ge=0)
    sibling_index: int = Field(ge=0)
    
    prompt: str
    prefix: str = ""
    suffix: str = ""
    reminder_fired: bool = False
    reminder_content: Optional[str] = None
    
    response: str
    
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None
    prompt_eval_duration_ms: Optional[int] = None
    eval_duration_ms: Optional[int] = None
    
    temperature: float
    model_name: str
    framing_strategy: str
    timestamp: datetime = Field(default_factory=datetime.now)

    def to_jsonl_dict(self) -> dict:
        """Convert to dict suitable for JSONL output."""
        d = self.model_dump()
        d['timestamp'] = self.timestamp.isoformat()
        return d
