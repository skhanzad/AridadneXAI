"""
ReAct Agent implementation using LangGraph.
Supports web search and calculator tools.
"""
import time
from typing import Dict, Any, List, Optional, TypedDict, Annotated
try:
    from langchain_community.llms import Ollama
    from langchain_community.tools import DuckDuckGoSearchRun
except ImportError:
    from langchain.llms import Ollama
    from langchain.tools import DuckDuckGoSearchRun

from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
import json
import re

from schemas import ReasoningStep, AgentResponse, ToolType


# Define AgentState at module level so it can be used in type hints
class AgentState(TypedDict):
    """State structure for the LangGraph agent."""
    messages: Annotated[list, add_messages]
    reasoning_steps: List[Dict[str, Any]]


class ReActAgent:
    """
    A ReAct (Reasoning + Acting) agent that uses LangGraph for orchestration.
    Captures reasoning steps for causal auditing.
    """
    
    def __init__(
        self,
        model_name: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_iterations: int = 10
    ):
        """
        Initialize the ReAct agent.
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for Ollama API
            temperature: Sampling temperature
            max_iterations: Maximum reasoning iterations
        """
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.max_iterations = max_iterations
        
        # Initialize Ollama LLM
        self.llm = Ollama(
            model=model_name,
            base_url=base_url,
            temperature=temperature
        )
        
        # Initialize tools
        self.search_tool = DuckDuckGoSearchRun()
        self.calculator_tool = Tool(
            name="calculator",
            func=self._calculate,
            description="Performs basic arithmetic calculations. Input should be a mathematical expression like '2+2' or '10*5'."
        )
        
        self.tools = [self.search_tool, self.calculator_tool]
        
        # Build the agent graph
        self.agent_graph = self._build_agent_graph()
        
        # Reasoning trace storage
        self.current_trace: List[ReasoningStep] = []
    
    def _calculate(self, expression: str) -> str:
        """Simple calculator function."""
        try:
            # Sanitize and evaluate
            expression = expression.strip()
            # Only allow basic math operations for safety
            if re.match(r'^[\d\+\-\*\/\(\)\.\s]+$', expression):
                result = eval(expression)
                return str(result)
            else:
                return "Error: Invalid expression. Only numbers and basic operators (+, -, *, /) are allowed."
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _build_agent_graph(self) -> StateGraph:
        """Build the LangGraph agent workflow."""
        # Create the prompt template
        system_prompt = """You are a helpful AI assistant that uses tools to answer questions accurately.
You have access to the following tools:
- search: Search the web for current information
- calculator: Perform mathematical calculations

When you need to use a tool, think step by step:
1. Think about what information you need
2. Use the appropriate tool
3. Analyze the tool's output
4. Provide your final answer

Always be explicit about your reasoning process."""
        
        # Create the agent node
        def agent_node(state: AgentState) -> AgentState:
            messages = state.get("messages", [])
            reasoning_steps = state.get("reasoning_steps", [])
            
            # Get the last user message
            if not messages:
                return state
            
            # Format messages for the LLM
            formatted_messages = []
            if not any(isinstance(m, SystemMessage) for m in messages):
                formatted_messages.append(SystemMessage(content=system_prompt))
            formatted_messages.extend(messages)
            
            # Get LLM response
            response = self.llm.invoke(formatted_messages)
            
            # Parse the response to extract reasoning
            response_text = response if isinstance(response, str) else response.content
            
            # Extract reasoning step
            step = self._extract_reasoning_step(response_text, len(reasoning_steps))
            if step:
                reasoning_steps.append(step.dict())
            
            # Add AI message
            formatted_messages.append(AIMessage(content=response_text))
            
            return {
                "messages": formatted_messages,
                "reasoning_steps": reasoning_steps
            }
        
        # Create tool node
        tool_node = ToolNode(self.tools)
        
        # Build graph
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)
        
        # Add edges
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )
        workflow.add_edge("tools", "agent")
        
        # Add memory
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    def _extract_reasoning_step(self, response_text: str, step_id: int) -> Optional[ReasoningStep]:
        """Extract a reasoning step from the LLM response."""
        # Look for tool calls in the response
        tool_called = None
        tool_input = None
        tool_output = None
        
        # Check for search tool
        if "search" in response_text.lower() or "duckduckgo" in response_text.lower():
            tool_called = ToolType.SEARCH
            # Try to extract search query
            match = re.search(r'search[:\s]+["\']?([^"\']+)["\']?', response_text, re.IGNORECASE)
            if match:
                tool_input = match.group(1)
        
        # Check for calculator tool
        if "calculator" in response_text.lower() or "calculate" in response_text.lower():
            tool_called = ToolType.CALCULATOR
            # Try to extract calculation
            match = re.search(r'calculate[:\s]+([\d\+\-\*\/\(\)\.\s]+)', response_text, re.IGNORECASE)
            if match:
                tool_input = match.group(1)
        
        return ReasoningStep(
            step_id=step_id + 1,
            thought=response_text[:500],  # Truncate if too long
            tool_called=tool_called,
            tool_input=tool_input,
            tool_output=tool_output
        )
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine if the agent should continue or end."""
        messages = state.get("messages", [])
        if not messages:
            return "end"
        
        last_message = messages[-1]
        message_content = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        # Check if we should continue (simplified logic)
        # In a real implementation, you'd parse tool calls more carefully
        if any(tool.name.lower() in message_content.lower() for tool in self.tools):
            return "continue"
        
        # Check iteration limit
        reasoning_steps = state.get("reasoning_steps", [])
        if len(reasoning_steps) >= self.max_iterations:
            return "end"
        
        return "end"
    
    def query(self, query: str, config: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Execute a query and return the agent response with reasoning trace.
        
        Args:
            query: User query
            config: Optional configuration for the agent execution
            
        Returns:
            AgentResponse with reasoning steps and final answer
        """
        start_time = time.time()
        self.current_trace = []
        
        # Prepare initial state
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "reasoning_steps": []
        }
        
        # Execute the agent
        config = config or {}
        config["configurable"] = {"thread_id": "default"}
        
        try:
            final_state = None
            for event in self.agent_graph.stream(initial_state, config):
                final_state = event
            
            # Extract final state
            if final_state:
                # Get the last state from the stream
                last_node = list(final_state.keys())[-1] if final_state else None
                if last_node:
                    state = final_state[last_node]
                else:
                    state = initial_state
            else:
                state = initial_state
            
            # Convert reasoning steps to schema objects
            reasoning_steps = [
                ReasoningStep(**step) if isinstance(step, dict) else step
                for step in state.get("reasoning_steps", [])
            ]
            
            # Extract final answer from messages
            messages = state.get("messages", [])
            final_answer = ""
            if messages:
                # Get the last AI message
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage):
                        final_answer = msg.content if hasattr(msg, 'content') else str(msg)
                        break
            
            if not final_answer:
                final_answer = "I couldn't generate a complete answer."
            
            execution_time = time.time() - start_time
            
            return AgentResponse(
                query=query,
                reasoning_steps=reasoning_steps,
                final_answer=final_answer,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return AgentResponse(
                query=query,
                reasoning_steps=self.current_trace,
                final_answer=f"Error: {str(e)}",
                execution_time=execution_time
            )
    
    def query_with_intervention(
        self,
        query: str,
        intervention_point: int,
        intervened_thought: str,
        config: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Execute a query with an intervention at a specific reasoning step.
        
        Args:
            query: User query
            intervention_point: Index of the reasoning step to intervene on
            intervened_thought: The modified reasoning to inject
            config: Optional configuration
            
        Returns:
            AgentResponse after intervention
        """
        # Start normal execution
        response = self.query(query, config)
        
        # If we haven't reached the intervention point, return as-is
        if len(response.reasoning_steps) <= intervention_point:
            return response
        
        # Modify the reasoning step at intervention point
        modified_steps = response.reasoning_steps.copy()
        if intervention_point < len(modified_steps):
            modified_steps[intervention_point].thought = intervened_thought
        
        # Re-run from intervention point (simplified - in practice, you'd restart the agent)
        # For now, we'll create a modified response
        return AgentResponse(
            query=query,
            reasoning_steps=modified_steps,
            final_answer=response.final_answer,  # This would be re-computed in full implementation
            execution_time=response.execution_time
        )

