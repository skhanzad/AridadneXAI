"""
Pydantic schemas for reasoning traces, interventions, and audit results.
"""
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ToolType(str, Enum):
    """Supported tool types for the agent."""
    SEARCH = "search"
    CALCULATOR = "calculator"
    NONE = "none"


class ReasoningStep(BaseModel):
    """A single reasoning step in the agent's chain of thought."""
    step_id: int = Field(..., description="Unique identifier for this reasoning step")
    thought: str = Field(..., description="The reasoning/thought process")
    tool_called: Optional[ToolType] = Field(None, description="Tool invoked in this step")
    tool_input: Optional[str] = Field(None, description="Input to the tool")
    tool_output: Optional[str] = Field(None, description="Output from the tool")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "step_id": 1,
                "thought": "I need to search for information about climate change",
                "tool_called": "search",
                "tool_input": "climate change effects",
                "tool_output": "Climate change leads to rising temperatures..."
            }
        }


class AgentResponse(BaseModel):
    """Complete agent response with reasoning trace and final answer."""
    query: str = Field(..., description="Original user query")
    reasoning_steps: List[ReasoningStep] = Field(..., description="Chain of reasoning steps")
    final_answer: str = Field(..., description="Final answer provided by the agent")
    execution_time: float = Field(..., description="Time taken in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the capital of France?",
                "reasoning_steps": [
                    {
                        "step_id": 1,
                        "thought": "I know this fact, but let me verify",
                        "tool_called": "search",
                        "tool_input": "capital of France",
                        "tool_output": "Paris is the capital..."
                    }
                ],
                "final_answer": "The capital of France is Paris.",
                "execution_time": 2.5
            }
        }


class InterventionType(str, Enum):
    """Types of causal interventions."""
    LOGIC_FLIP = "logic_flip"  # True -> False, Increase -> Decrease
    FACT_REVERSAL = "fact_reversal"  # Reverse factual claims
    PREMISE_NEGATION = "premise_negation"  # Negate premises
    CAUSAL_REVERSAL = "causal_reversal"  # Reverse causal relationships


class Intervention(BaseModel):
    """A causal intervention on a reasoning step."""
    intervention_id: str = Field(..., description="Unique identifier for this intervention")
    original_step: ReasoningStep = Field(..., description="The original reasoning step")
    intervention_type: InterventionType = Field(..., description="Type of intervention applied")
    intervened_thought: str = Field(..., description="The modified reasoning after intervention")
    intervention_rationale: str = Field(..., description="Why this intervention was made")
    step_index: int = Field(..., description="Index of the step in the original trace")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "intervention_id": "int_001",
                "original_step": {
                    "step_id": 1,
                    "thought": "The temperature is increasing",
                    "tool_called": None
                },
                "intervention_type": "logic_flip",
                "intervened_thought": "The temperature is decreasing",
                "intervention_rationale": "Flipped increase to decrease to test causal faithfulness",
                "step_index": 0
            }
        }


class AuditResult(BaseModel):
    """Result of a causal audit comparing original vs intervened responses."""
    audit_id: str = Field(..., description="Unique identifier for this audit")
    query: str = Field(..., description="Original query")
    original_response: AgentResponse = Field(..., description="Original agent response")
    intervention: Intervention = Field(..., description="Intervention applied")
    intervened_response: Optional[AgentResponse] = Field(None, description="Response after intervention")
    faithfulness_score: float = Field(..., description="Faithfulness score (0-1, higher = more faithful)")
    is_violation: bool = Field(..., description="Whether a faithfulness violation was detected")
    violation_reason: Optional[str] = Field(None, description="Reason for violation if detected")
    similarity_metrics: Dict[str, float] = Field(default_factory=dict, description="Additional similarity metrics")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "audit_id": "audit_001",
                "query": "What causes global warming?",
                "original_response": {
                    "query": "What causes global warming?",
                    "final_answer": "Greenhouse gases cause global warming",
                    "reasoning_steps": []
                },
                "intervention": {
                    "intervention_id": "int_001",
                    "intervened_thought": "Greenhouse gases do not cause global warming"
                },
                "faithfulness_score": 0.2,
                "is_violation": True,
                "violation_reason": "Answer remained identical despite contradictory reasoning"
            }
        }


class ReasoningTrace(BaseModel):
    """Structured reasoning trace containing steps and terminal answer."""
    steps: List[ReasoningStep] = Field(..., description="Sequence of reasoning steps s1, ..., sn")
    terminal_answer: str = Field(..., description="Final answer a")
    query: str = Field(..., description="Original query")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "steps": [step.dict() for step in self.steps],
            "terminal_answer": self.terminal_answer
        }
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the capital of France?",
                "steps": [
                    {
                        "step_id": 1,
                        "thought": "I need to recall the capital of France",
                        "tool_called": None
                    }
                ],
                "terminal_answer": "Paris"
            }
        }


class AuditSession(BaseModel):
    """A complete audit session with multiple interventions."""
    session_id: str = Field(..., description="Unique session identifier")
    query: str = Field(..., description="Query being audited")
    audit_results: List[AuditResult] = Field(default_factory=list, description="All audit results in this session")
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = Field(None, description="Session end time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session_001",
                "query": "What is the effect of CO2 on temperature?",
                "audit_results": []
            }
        }

