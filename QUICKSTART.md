# Quick Start Guide

## 1. Prerequisites

```bash
# Install Ollama (if not already installed)
# Visit https://ollama.ai/ for installation instructions

# Pull a model
ollama pull llama3
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Test Setup

```bash
python test_setup.py
```

This will verify:
- All Python packages are installed
- Ollama is running and accessible
- Project modules can be imported

## 4. Run Your First Audit

```bash
# Single query
python main.py --query "What is the capital of France?"

# Batch processing
python main.py --queries-file example_queries.txt
```

## 5. View Results

Results are saved to `audit_results/` directory as CSV files, ready for:
- Excel/Google Sheets import
- LaTeX table generation
- Statistical analysis

## Common Issues

### Ollama Not Running
```bash
# Start Ollama
ollama serve

# In another terminal, verify it's running
curl http://localhost:11434/api/tags
```

### Model Not Found
```bash
# List available models
ollama list

# Pull a model if needed
ollama pull llama3
```

### Import Errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

## Next Steps

1. **Customize Queries**: Edit `example_queries.txt` with your own queries
2. **Experiment with Interventions**: Try different `--intervention-type` values
3. **Analyze Results**: Open CSV files in Excel/Python for analysis
4. **Extend Framework**: Modify `auditor.py` to add new intervention types

## Example Workflow

```bash
# 1. Test setup
python test_setup.py

# 2. Run audit on single query
python main.py --query "What causes climate change?" --intervention-type logic_flip

# 3. Run batch audit
python main.py --queries-file example_queries.txt --output-dir my_results

# 4. Analyze results
# Open audit_results/*.csv in your preferred tool
```

