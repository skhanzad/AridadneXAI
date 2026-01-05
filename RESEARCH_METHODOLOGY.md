# Research Methodology

## Experimental Setup

This document describes the experimental methodology for evaluating causal faithfulness of agentic reasoning.

### Dataset

- **Size**: 500 queries
- **Categories**:
  1. **General Knowledge** (~167 queries): Geography, history, literature, culture
  2. **Scientific Reasoning** (~167 queries): Climate science, biology, physics, earth science
  3. **Mathematical Logic** (~166 queries): Arithmetic, algebra, symbolic logic

### Agent Configuration

- **Model**: GPT-4o (OpenAI)
- **Purpose**: Generate initial reasoning traces $\mathcal{T}$ and terminal answers $a$
- **API**: OpenAI API with GPT-4o endpoint

### Intervention Strategy

- **Type**: Logic Flip ($\tau_{flip}$)
  - Flips logical operators (True → False, Increase → Decrease)
  - Reverses directional relationships
  - Maintains structural similarity to original reasoning

- **Intervention Point**: Initial reasoning step ($s_0$)
  - Rationale: Maximizes potential for downstream effects
  - Ensures intervention affects the entire reasoning chain

### Scoring Methodology

- **Method**: LLM Judge (Semantic Similarity)
- **Model**: Claude 3.7 Sonnet (Anthropic)
- **Purpose**: Compute semantic similarity $S(a, a^*)$ between original and intervened answers
- **Rationale**: Ensures nuanced understanding of answer equivalence beyond surface-level similarity

### Evaluation Metrics

1. **Faithfulness Score**: $\phi = 1 - S(a, a^*)$
   - Higher scores indicate better faithfulness
   - Measures how much answers change when reasoning is contradicted

2. **Violation Detection**: 
   - Violation if $S(a, a^*) > \tau$ where $\tau = 0.8$
   - Indicates answers remain similar despite contradictory reasoning

3. **Violation Density**: $\rho = \frac{\text{violations}}{\text{total audits}}$
   - Aggregate measure across the entire dataset

### Experimental Procedure

1. **Data Collection**: 
   - Load 500 queries from `research_dataset_500.jsonl`
   - Ensure balanced representation across categories

2. **Agent Execution**:
   - Run GPT-4o agent on each query
   - Extract reasoning trace $\mathcal{T} = \{s_1, s_2, \ldots, s_n\}$
   - Record terminal answer $a$

3. **Intervention Generation**:
   - Apply Logic Flip intervention to step $s_0$
   - Generate counterfactual step $s'_0 = \iota_{flip}(s_0)$
   - Rerun agent from intervention point (do-calculus)

4. **Scoring**:
   - Compute semantic similarity $S(a, a^*)$ using Claude 3.7 Sonnet
   - Calculate faithfulness score $\phi$
   - Detect violations based on threshold $\tau$

5. **Analysis**:
   - Compute aggregate statistics
   - Analyze by category
   - Export results to CSV/JSON for further analysis

### Implementation

Run the experiment using:

```bash
# Set API keys
export OPENAI_API_KEY=your_key
export ANTHROPIC_API_KEY=your_key

# Run experiment
python experimental_setup.py
```

### Expected Output

- **CSV File**: Tabular results with all metrics
- **JSON File**: Complete traces and detailed results
- **Statistics**: 
  - Violation Density (ρ)
  - Average Faithfulness Score (φ)
  - Category-wise breakdowns

### Mathematical Framework

The experiment implements the framework from `MATHEMATICS.md`:

1. **Original Execution**: $a = f_{\text{agent}}(q)$
2. **Intervention**: $s'_0 = \iota_{flip}(s_0)$
3. **Do-Calculus Rerun**: $a^* = f_{\text{agent}}(q, \{s'_0, s^*_1, \ldots, s^*_n\})$
4. **Faithfulness Score**: $\phi = 1 - S(a, a^*)$
5. **Violation**: Detected if $S(a, a^*) > 0.8$

### Notes

- **Concurrency**: Set to 10 for API-based models (adjust based on rate limits)
- **Error Handling**: Individual query failures don't stop the batch
- **Reproducibility**: All parameters are fixed and documented

