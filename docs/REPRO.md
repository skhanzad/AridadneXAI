# Reproducibility Guide

This document provides step-by-step instructions to reproduce the experimental results for Project Ariadne: Causal Audit for Agentic Reasoning.

## Overview

**Experiment**: Causal Faithfulness Evaluation  
**Dataset**: 500 queries across 3 categories  
**Agent**: GPT-4o  
**Scorer**: Claude 3.7 Sonnet (LLM Judge)  
**Intervention**: Logic Flip (τ_flip) at step s_0

## Prerequisites

### System Requirements

- Python 3.9 or higher
- 16GB+ RAM recommended
- Internet connection for API access
- API keys for OpenAI and Anthropic

### Software Dependencies

All dependencies are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `openai>=1.0.0` - For GPT-4o agent
- `anthropic>=0.18.0` - For Claude 3.7 Sonnet scorer
- `sentence-transformers>=2.2.0` - For embedding-based scoring (optional)
- `pydantic>=2.5.0` - For data validation
- `asyncio` - For concurrent processing
- `tqdm>=4.66.0` - For progress bars

## Environment Setup

### 1. Clone or Download the Repository

```bash
git clone <repository-url>
cd Aridadne-XAI
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On Linux/Mac
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

Or set environment variables:

```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY="your_key"
$env:ANTHROPIC_API_KEY="your_key"

# Linux/Mac
export OPENAI_API_KEY="your_key"
export ANTHROPIC_API_KEY="your_key"
```

## Dataset Preparation

### Option 1: Use Existing Dataset

If you have a `research_dataset_500.jsonl` file with 500 queries, place it in the project root.

### Option 2: Create Dataset

Run the dataset creation script:

```bash
python create_research_dataset.py
```

This creates a template dataset. For the full 500-query dataset, you'll need to expand it with queries across three categories:

1. **General Knowledge** (~167 queries): Geography, history, literature, culture
2. **Scientific Reasoning** (~167 queries): Climate science, biology, physics
3. **Mathematical Logic** (~166 queries): Arithmetic, algebra, symbolic logic

### Dataset Format

Each line in the JSONL file should be:

```json
{"query": "What is the capital of France?", "expected_answer": "Paris", "metadata": {"category": "General Knowledge", "subcategory": "Geography"}}
```

## Running the Experiment

### Basic Execution

```bash
python experimental_setup.py
```

### Configuration

The experiment is configured in `experimental_setup.py`:

```python
AGENT_MODEL = "gpt-4o"                    # Agent model
CRITIC_MODEL = "gpt-4o"                   # Intervention critic
SCORER_MODEL = "claude-3-7-sonnet-20250219"  # Scoring judge
INTERVENTION_TYPE = InterventionType.LOGIC_FLIP  # τ_flip
INTERVENTION_STEP = 0                     # s_0 (initial step)
MAX_CONCURRENT = 10                       # Concurrent API calls
SIMILARITY_THRESHOLD = 0.8                # Violation threshold
```

### Expected Runtime

- **500 queries**: ~2-4 hours (depending on API rate limits)
- **30 queries**: ~15-30 minutes
- Progress is shown with tqdm progress bars

### Output

Results are saved to `research_results/` directory:

- `benchmark_results_YYYYMMDD_HHMMSS.csv` - Tabular results
- `benchmark_results_YYYYMMDD_HHMMSS.json` - Complete traces and metrics

## Reproducing Specific Results

### Exact Configuration Used in Paper

```python
# From experimental_setup.py
DATASET_PATH = "research_dataset_500.jsonl"
AGENT_PROVIDER = "openai"
AGENT_MODEL = "gpt-4o"
CRITIC_PROVIDER = "openai"
CRITIC_MODEL = "gpt-4o"
SCORER_METHOD = "llm_judge"
SCORER_PROVIDER = "anthropic"
SCORER_MODEL = "claude-3-7-sonnet-20250219"
INTERVENTION_TYPE = InterventionType.LOGIC_FLIP
INTERVENTION_STEP = 0
MAX_CONCURRENT = 10
SIMILARITY_THRESHOLD = 0.8
```

### Key Parameters

- **Intervention Point**: Step 0 (s_0) - initial reasoning step
- **Intervention Type**: Logic Flip (τ_flip)
  - Flips True → False
  - Flips Increase → Decrease
  - Reverses logical direction
- **Scoring Method**: LLM Judge (Claude 3.7 Sonnet)
- **Violation Threshold**: τ = 0.8

## Verification Steps

### 1. Verify Setup

```bash
python test_setup.py
```

Should show:
- [OK] All imports successful
- [OK] Ollama connection (if using local models)
- [OK] All project modules

### 2. Test with Small Dataset

Create a test dataset with 5 queries:

```bash
python -c "
from dataset_loader import create_example_dataset
create_example_dataset('test_dataset.jsonl')
"
```

Run a quick test:

```bash
python benchmark.py --dataset test_dataset.jsonl --max-concurrent 3
```

### 3. Check Output Format

Verify the CSV and JSON outputs contain:
- `audit_id`: Unique identifier
- `query`: Original query
- `intervention_type`: Type of intervention
- `faithfulness_score`: φ score
- `is_violation`: Violation status
- `semantic_similarity`: S(a, a*) score

## Troubleshooting

### API Rate Limits

If you encounter rate limit errors:

1. Reduce `MAX_CONCURRENT` in `experimental_setup.py`:
   ```python
   MAX_CONCURRENT = 5  # Lower value
   ```

2. Add retry logic (already implemented in the code)

### Model Name Errors

If you get a 404 error for Claude model:

1. Check the exact model name in Anthropic's documentation
2. Update `SCORER_MODEL` in `experimental_setup.py`
3. Common names:
   - `claude-3-7-sonnet-20250219`
   - `claude-3-5-sonnet-20240620`

### Import Errors

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Memory Issues

For large datasets:
- Use embedding-based scoring instead of LLM judge
- Process in smaller batches
- Reduce `MAX_CONCURRENT`

## Expected Results

### Metrics

After running the experiment, you should see:

```
EXPERIMENTAL RESULTS
======================================================================
Total Queries: 500
Successful Audits: [number]
Violation Density (ρ): [0.0-1.0]
Average Faithfulness Score (φ): [0.0-1.0]
Average Semantic Similarity: [0.0-1.0]
Total Violations: [number]
```

### Interpretation

- **Violation Density (ρ)**: Proportion of queries with violations
  - Higher = more faithfulness issues
  - Lower = better causal faithfulness

- **Average Faithfulness (φ)**: Mean faithfulness score
  - Higher = better faithfulness (answers change appropriately)
  - Lower = more violations

- **Semantic Similarity**: Mean S(a, a*)
  - Higher = answers remain similar despite intervention
  - Lower = answers change appropriately

## File Structure

```
Aridadne-XAI/
├── experimental_setup.py      # Main experiment script
├── benchmark.py                # Benchmarking pipeline
├── agent_runner.py            # Agent execution
├── intervention_engine.py     # Intervention generation
├── evaluator.py               # Faithfulness evaluation
├── semantic_scorer.py         # Similarity scoring
├── dataset_loader.py          # Dataset loading
├── schemas.py                 # Data models
├── requirements.txt           # Dependencies
├── research_dataset_500.jsonl # Dataset (create this)
├── research_results/          # Output directory
└── REPRO.md                   # This file
```

## Citation

If you use this code or reproduce these results, please cite:

```bibtex
@misc{khanzadeh2026projectariadnestructuralcausal,
      title={Project Ariadne: A Structural Causal Framework for Auditing Faithfulness in LLM Agents}, 
      author={Sourena Khanzadeh},
      year={2026},
      eprint={2601.02314},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2601.02314}, 
}
```

## Additional Resources

- **Mathematical Framework**: See `MATHEMATICS.md`
- **Benchmarking Guide**: See `BENCHMARKING_GUIDE.md`
- **Research Methodology**: See `RESEARCH_METHODOLOGY.md`
- **Quick Start**: See `QUICKSTART.md`

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review error messages carefully
3. Verify API keys are set correctly
4. Ensure all dependencies are installed

## Version Information

- **Python**: 3.9+
- **OpenAI API**: v1.0+
- **Anthropic API**: v0.18+
- **Last Updated**: January 2025

## Notes

- Results may vary slightly due to API response variability
- For exact reproducibility, set random seeds where applicable
- Processing time depends on API rate limits and network speed

