"""
Agent Runner for executing LLM agents with API support (GPT-4o, Claude, etc.).
"""
import time
import asyncio
from typing import Optional, Dict, Any, List
from enum import Enum
import json

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

from schemas import ReasoningStep, ReasoningTrace, ToolType


class AgentProvider(str, Enum):
    """Supported agent providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class AgentRunner:
    """
    Executes LLM agents and returns structured ReasoningTrace objects.
    Supports OpenAI, Anthropic, and Ollama.
    """
    
    def __init__(
        self,
        provider: AgentProvider = AgentProvider.OPENAI,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        """
        Initialize the agent runner.
        
        Args:
            provider: Agent provider (openai, anthropic, ollama)
            model_name: Model name (e.g., "gpt-4o", "claude-3-5-sonnet-20240620")
            api_key: API key (if None, uses environment variable)
            base_url: Base URL for API (for custom endpoints)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize client based on provider
        if provider == AgentProvider.OPENAI:
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package not installed. Install with: pip install openai")
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        elif provider == AgentProvider.ANTHROPIC:
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic package not installed. Install with: pip install anthropic")
            self.client = AsyncAnthropic(api_key=api_key, base_url=base_url)
        elif provider == AgentProvider.OLLAMA:
            # Ollama uses OpenAI-compatible API
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package not installed for Ollama compatibility")
            base_url = base_url or "http://localhost:11434/v1"
            self.client = AsyncOpenAI(api_key="ollama", base_url=base_url)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def run(self, query: str, context: Optional[List[ReasoningStep]] = None) -> ReasoningTrace:
        """
        Execute the agent on a query and return a ReasoningTrace.
        
        Args:
            query: User query
            context: Optional previous reasoning steps (for intervention reruns)
            
        Returns:
            ReasoningTrace with steps and terminal answer
        """
        start_time = time.time()
        
        # Build prompt with reasoning instruction
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(query, context)
        
        # Get response from LLM
        response_text = await self._get_llm_response(system_prompt, user_prompt)
        
        # Parse reasoning trace from response
        trace = self._parse_reasoning_trace(query, response_text)
        
        execution_time = time.time() - start_time
        
        return trace
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for reasoning."""
        return """You are a helpful AI assistant that reasons step by step.

When answering questions, you should:
1. Think through the problem step by step
2. Show your reasoning process clearly
3. Provide a final answer at the end

Format your response as follows:
Thought 1: [Your first reasoning step]
Thought 2: [Your second reasoning step]
...
Final Answer: [Your final answer]"""
    
    def _build_user_prompt(self, query: str, context: Optional[List[ReasoningStep]] = None) -> str:
        """Build the user prompt with optional context."""
        if context:
            context_text = "\n".join([
                f"Step {i+1}: {step.thought}" for i, step in enumerate(context)
            ])
            return f"""Previous reasoning steps:
{context_text}

Continue reasoning from the above steps and answer:
{query}"""
        else:
            return query
    
    async def _get_llm_response(self, system_prompt: str, user_prompt: str) -> str:
        """Get response from the LLM."""
        if self.provider == AgentProvider.OPENAI or self.provider == AgentProvider.OLLAMA:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        
        elif self.provider == AgentProvider.ANTHROPIC:
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.content[0].text
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _parse_reasoning_trace(self, query: str, response_text: str) -> ReasoningTrace:
        """
        Parse reasoning trace from LLM response.
        Extracts steps and final answer.
        """
        import re
        
        steps = []
        final_answer = ""
        
        # Try to extract structured thoughts
        thought_pattern = r'(?:Thought|Step)\s*(\d+)[:\-]\s*(.+?)(?=(?:Thought|Step|Final Answer)\s*\d+:|Final Answer:|$)'
        thoughts = re.findall(thought_pattern, response_text, re.IGNORECASE | re.DOTALL)
        
        for step_num, thought in thoughts:
            step_id = int(step_num)
            thought_text = thought.strip()
            steps.append(ReasoningStep(
                step_id=step_id,
                thought=thought_text,
                tool_called=None,
                tool_input=None,
                tool_output=None
            ))
        
        # Extract final answer
        final_answer_pattern = r'Final Answer[:\-]\s*(.+?)(?:\n\n|\Z)'
        final_match = re.search(final_answer_pattern, response_text, re.IGNORECASE | re.DOTALL)
        if final_match:
            final_answer = final_match.group(1).strip()
        else:
            # Fallback: use last paragraph or entire response
            paragraphs = response_text.split('\n\n')
            if paragraphs:
                final_answer = paragraphs[-1].strip()
            else:
                final_answer = response_text.strip()
        
        # If no structured thoughts found, create a single step
        if not steps:
            # Split response into sentences for steps
            sentences = re.split(r'[.!?]+\s+', response_text)
            for i, sentence in enumerate(sentences[:5], 1):  # Limit to 5 steps
                if sentence.strip():
                    steps.append(ReasoningStep(
                        step_id=i,
                        thought=sentence.strip(),
                        tool_called=None,
                        tool_input=None,
                        tool_output=None
                    ))
        
        # Ensure we have at least one step
        if not steps:
            steps.append(ReasoningStep(
                step_id=1,
                thought=response_text[:500],
                tool_called=None,
                tool_input=None,
                tool_output=None
            ))
        
        return ReasoningTrace(
            query=query,
            steps=steps,
            terminal_answer=final_answer
        )
    
    async def run_with_intervention(
        self,
        query: str,
        original_trace: ReasoningTrace,
        intervention_step_index: int,
        intervened_thought: str
    ) -> ReasoningTrace:
        """
        Run agent with do-calculus intervention: replace step k and regenerate subsequent steps.
        
        Args:
            query: Original query
            original_trace: Original reasoning trace
            intervention_step_index: Index of step to intervene on (0-based)
            intervened_thought: The intervened thought to replace step k
            
        Returns:
            New ReasoningTrace with intervention applied
        """
        # Build context up to intervention point
        context_steps = original_trace.steps[:intervention_step_index].copy()
        
        # Add intervened step
        intervened_step = ReasoningStep(
            step_id=intervention_step_index + 1,
            thought=intervened_thought,
            tool_called=original_trace.steps[intervention_step_index].tool_called if intervention_step_index < len(original_trace.steps) else None,
            tool_input=original_trace.steps[intervention_step_index].tool_input if intervention_step_index < len(original_trace.steps) else None,
            tool_output=original_trace.steps[intervention_step_index].tool_output if intervention_step_index < len(original_trace.steps) else None
        )
        context_steps.append(intervened_step)
        
        # Regenerate from intervention point
        new_trace = await self.run(query, context=context_steps)
        
        # Combine: context + new steps
        final_steps = context_steps + new_trace.steps
        
        return ReasoningTrace(
            query=query,
            steps=final_steps,
            terminal_answer=new_trace.terminal_answer
        )

