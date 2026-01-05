# Project Ariadne: Causal Audit for Agentic Reasoning

A research-grade framework for evaluating the **Causal Faithfulness** of LLM Agents through systematic causal interventions.

## Overview

Project Ariadne implements a comprehensive causal auditing system that:

1. **Captures Reasoning Traces**: Intercepts and records all reasoning steps from a ReAct agent
2. **Performs Causal Interventions**: Uses a Critic LLM to flip logic, reverse facts, or negate premises
3. **Reruns with Interventions**: Relaunches the agent from intervention points with modified reasoning
4. **Scores Faithfulness**: Compares original vs. intervened answers to detect faithfulness violations
5. **Exports Results**: Logs audit results to CSV format suitable for research paper tables

## Architecture

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ReAct Agent    │──► Reasoning Steps ──► Final Answer
│  (LangGraph)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Causal Auditor  │──► Intervention ──► Rerun Agent
│   (Ariadne)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Faithfulness     │──► Score & Violation Detection
│   Scorer        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  CSV Logger     │──► Research Paper Tables
└─────────────────┘
```

## Installation

### Prerequisites

- Python 3.9+
- NVIDIA GPU (recommended: 16GB+ VRAM for local models)
- [Ollama](https://ollama.ai/) installed and running locally

### Setup

1. **Install Ollama** and pull a model:
   ```bash
   ollama pull llama3
   # or
   ollama pull mistral
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Ollama is running**:
   ```bash
   curl http://localhost:11434/api/tags
   ```

## Usage

### Basic Usage

Run a single query audit:
```bash
python main.py --query "What causes global warming?"
```

### Batch Processing

Process multiple queries from a file:
```bash
python main.py --queries-file queries.txt
```

### Advanced Options

```bash
python main.py \
  --query "What is the capital of France?" \
  --model llama3 \
  --ollama-url http://localhost:11434 \
  --intervention-type logic_flip \
  --intervention-step 0 \
  --output-dir audit_results \
  --max-iterations 10
```

### Intervention Types

- `logic_flip`: Flips True→False, Increase→Decrease, etc.
- `fact_reversal`: Reverses factual claims
- `premise_negation`: Negates underlying premises
- `causal_reversal`: Reverses cause-effect relationships

## Project Structure

```
Aridadne-XAI/
├── main.py              # Main entry point
├── schemas.py           # Pydantic models for traces, interventions, results
├── agent.py             # ReAct agent implementation (LangGraph)
├── auditor.py           # Causal auditor (Ariadne module)
├── scorer.py            # Faithfulness scorer
├── logger.py            # CSV logging system
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── audit_results/      # CSV output directory (created automatically)
```

## Key Components

### 1. ReAct Agent (`agent.py`)

- Implements a ReAct (Reasoning + Acting) agent using LangGraph
- Supports web search (DuckDuckGo) and calculator tools
- Captures reasoning steps in structured format
- Uses Ollama for local LLM inference

### 2. Causal Auditor (`auditor.py`)

- **Step Capture**: Intercepts reasoning steps from agent execution
- **Causal Intervention**: Uses a Critic LLM to generate contradictory reasoning
- **Rerunning Logic**: Relaunches agent from intervention point with modified context

### 3. Faithfulness Scorer (`scorer.py`)

- Compares original vs. intervened final answers
- Calculates multiple similarity metrics (Jaccard, semantic, character-level)
- Detects violations when answers remain identical despite contradictory reasoning
- Uses LLM-based semantic similarity assessment

### 4. CSV Logger (`logger.py`)

- Exports audit results to CSV format
- Suitable for generating tables in research papers
- Includes all metrics, violation flags, and reasoning traces

## Example Output

The CSV output includes:
- `audit_id`: Unique identifier
- `query`: Original query
- `intervention_type`: Type of intervention applied
- `original_final_answer`: Answer before intervention
- `intervened_final_answer`: Answer after intervention
- `faithfulness_score`: Score (0-1, lower = more violations)
- `is_violation`: Whether a violation was detected
- `semantic_similarity`: LLM-based similarity score
- Additional similarity metrics (Jaccard, character, etc.)

## Research Use Cases

This framework is designed for:

1. **Evaluating Agent Faithfulness**: Detecting when agents ignore reasoning steps
2. **Causal Analysis**: Understanding how reasoning changes affect outputs
3. **Benchmarking**: Comparing different agent architectures
4. **Paper Tables**: CSV output ready for LaTeX/Excel import

## Configuration

### Model Selection

Supported models via Ollama:
- `llama3` (recommended)
- `mistral`
- `llama2`
- Any other Ollama-compatible model

### Hardware Optimization

For NVIDIA 4080 SUPER (16GB VRAM):
- Use `llama3-8b` or `mistral-7b` models
- Consider quantization if memory is constrained
- Adjust `max_iterations` to control reasoning depth

## Troubleshooting

### Ollama Connection Issues

If you see connection errors:
```bash
# Check Ollama is running
ollama serve

# Test connection
curl http://localhost:11434/api/tags
```

### Memory Issues

If running out of VRAM:
- Use smaller models (e.g., `llama3:8b` instead of `llama3:70b`)
- Reduce `max_iterations`
- Process queries one at a time

### Import Errors

Ensure all dependencies are installed:
```bash
pip install --upgrade -r requirements.txt
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{ariadne2024,
  title={Project Ariadne: Causal Audit for Agentic Reasoning},
  author={Sourena Khanzadeh},
  year={2024},
  url={https://github.com/skhanzad/ariadne-xai}
}
```

## License

[Specify your license here]

## Contributing

[Contributing guidelines]

## Acknowledgments

Built with:
- [LangGraph](https://github.com/langchain-ai/langgraph) for agent orchestration
- [Ollama](https://ollama.ai/) for local LLM inference
- [LangChain](https://www.langchain.com/) for agent framework
- [Pydantic](https://docs.pydantic.dev/) for data validation

