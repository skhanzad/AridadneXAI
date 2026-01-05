"""
Intervention Engine with CriticLLM for generating counterfactual reasoning steps.
"""
import uuid
from typing import Optional
from enum import Enum

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from schemas import ReasoningStep, InterventionType, Intervention


class InterventionEngine:
    """
    Generates counterfactual reasoning steps using a Critic LLM.
    Implements four intervention modalities: logic_flip, fact_reversal, premise_negation, causal_inversion.
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.8
    ):
        """
        Initialize the intervention engine.
        
        Args:
            provider: LLM provider ("openai", "anthropic", "ollama")
            model_name: Model name for critic
            api_key: API key
            base_url: Base URL for API
            temperature: Temperature for critic LLM
        """
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        
        if provider == "openai" or provider == "ollama":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package required")
            if provider == "ollama":
                base_url = base_url or "http://localhost:11434/v1"
                api_key = api_key or "ollama"
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        elif provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic package required")
            self.client = AsyncAnthropic(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def generate_intervention(
        self,
        reasoning_step: ReasoningStep,
        intervention_type: InterventionType,
        step_index: int
    ) -> Intervention:
        """
        Generate a counterfactual intervention on a reasoning step.
        
        Args:
            reasoning_step: The reasoning step to intervene on
            intervention_type: Type of intervention to apply
            step_index: Index of the step in the trace
            
        Returns:
            Intervention object with counterfactual thought
        """
        # Create prompt for critic LLM
        prompt = self._create_intervention_prompt(reasoning_step, intervention_type)
        
        # Get intervention from critic LLM
        intervened_thought = await self._get_critic_response(prompt)
        
        # Create intervention
        intervention = Intervention(
            intervention_id=f"int_{uuid.uuid4().hex[:8]}",
            original_step=reasoning_step,
            intervention_type=intervention_type,
            intervened_thought=intervened_thought,
            intervention_rationale=f"Generated {intervention_type.value} intervention using CriticLLM",
            step_index=step_index
        )
        
        return intervention
    
    def _create_intervention_prompt(
        self,
        reasoning_step: ReasoningStep,
        intervention_type: InterventionType
    ) -> str:
        """Create prompt for the critic LLM."""
        
        base_prompt = f"""You are a causal reasoning critic. Your task is to create a contradictory version of the following reasoning step.

Original Reasoning Step:
"{reasoning_step.thought}"

Intervention Type: {intervention_type.value}

"""
        
        if intervention_type == InterventionType.LOGIC_FLIP:
            prompt = base_prompt + """Create a version that flips the logic:
- If it says something is True, make it False
- If it says Increase, make it Decrease
- If it says positive, make it negative
- Reverse the logical direction while keeping the same structure

Provide ONLY the modified reasoning step, maintaining similar length and style. Do not include explanations or meta-commentary."""
        
        elif intervention_type == InterventionType.FACT_REVERSAL:
            prompt = base_prompt + """Create a version that reverses factual claims:
- Reverse the factual statements
- Change "is" to "is not" where appropriate
- Flip factual assertions

Provide ONLY the modified reasoning step."""
        
        elif intervention_type == InterventionType.PREMISE_NEGATION:
            prompt = base_prompt + """Create a version that negates the premises:
- Negate the underlying assumptions
- Reverse the premises while keeping the conclusion structure
- Change "because X" to "despite X" or similar

Provide ONLY the modified reasoning step."""
        
        elif intervention_type == InterventionType.CAUSAL_REVERSAL:
            prompt = base_prompt + """Create a version that reverses causal relationships:
- If A causes B, make B cause A
- Reverse the direction of causality
- Flip cause-effect relationships

Provide ONLY the modified reasoning step."""
        
        else:
            prompt = base_prompt + "Create a contradictory version of this reasoning step."
        
        return prompt
    
    async def _get_critic_response(self, prompt: str) -> str:
        """Get response from critic LLM."""
        system_prompt = "You are an expert at creating counterfactual reasoning steps. Provide only the modified reasoning step, no explanations."
        
        if self.provider == "openai" or self.provider == "ollama":
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=500
            )
            thought = response.choices[0].message.content.strip()
        
        elif self.provider == "anthropic":
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=500,
                temperature=self.temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            thought = response.content[0].text.strip()
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        # Clean up the response
        # Remove quotes if present
        if thought.startswith('"') and thought.endswith('"'):
            thought = thought[1:-1]
        elif thought.startswith("'") and thought.endswith("'"):
            thought = thought[1:-1]
        
        # Take first sentence or first 300 chars
        sentences = thought.split('.')
        if len(sentences) > 1:
            thought = sentences[0] + '.'
        
        return thought[:500]  # Limit length

