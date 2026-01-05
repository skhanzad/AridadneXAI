"""
Experimental Setup for Research Paper
Matches the methodology described:
- 500 queries across 3 categories
- GPT-4o agent
- Logic Flip interventions at step s_0
- Claude 3.7 Sonnet as scoring judge
"""
import asyncio
from benchmark import BenchmarkPipeline
from agent_runner import AgentRunner, AgentProvider
from intervention_engine import InterventionEngine
from evaluator import Evaluator
from semantic_scorer import SemanticScorer
from schemas import InterventionType
from dotenv import load_dotenv
load_dotenv()



async def run_research_experiment():
    """
    Run the experimental setup described in the paper.
    
    Methodology:
    - Dataset: 500 queries (General Knowledge, Scientific Reasoning, Mathematical Logic)
    - Agent: GPT-4o
    - Intervention: Logic Flip (τ_flip) at initial step (s_0)
    - Scorer: Claude 3.7 Sonnet as LLM judge
    """
    
    # Configuration matching paper methodology
    DATASET_PATH = "research_dataset_500.jsonl"  # 500 queries dataset
    AGENT_PROVIDER = "openai"
    AGENT_MODEL = "gpt-4o"
    CRITIC_PROVIDER = "openai"  # Same provider for intervention generation
    CRITIC_MODEL = "gpt-4o"  # Can use GPT-4o or Claude for critic
    SCORER_METHOD = "llm_judge"
    SCORER_PROVIDER = "anthropic"
    SCORER_MODEL = "claude-3-7-sonnet-20250219"  # Claude 3.7 Sonnet (released Feb 2025)
    INTERVENTION_TYPE = InterventionType.LOGIC_FLIP  # τ_flip
    INTERVENTION_STEP = 0  # s_0 (initial reasoning step)
    MAX_CONCURRENT = 10  # Higher for API-based models
    SIMILARITY_THRESHOLD = 0.8
    
    # Get API keys from environment
    import os
    agent_api_key = os.getenv("OPENAI_API_KEY")
    critic_api_key = os.getenv("OPENAI_API_KEY")
    scorer_api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not agent_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    if not scorer_api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    
    print("="*70)
    print("Research Experiment: Causal Faithfulness Evaluation")
    print("="*70)
    print(f"Dataset: {DATASET_PATH} (500 queries)")
    print(f"Categories: General Knowledge, Scientific Reasoning, Mathematical Logic")
    print(f"Agent: {AGENT_PROVIDER}/{AGENT_MODEL}")
    print(f"Critic: {CRITIC_PROVIDER}/{CRITIC_MODEL}")
    print(f"Scorer: {SCORER_PROVIDER}/{SCORER_MODEL} (LLM Judge)")
    print(f"Intervention: {INTERVENTION_TYPE.value} at step {INTERVENTION_STEP} (s_0)")
    print(f"Max Concurrent: {MAX_CONCURRENT}")
    print("="*70)
    print()
    
    # Initialize components
    print("Initializing components...")
    
    agent_runner = AgentRunner(
        provider=AgentProvider(AGENT_PROVIDER),
        model_name=AGENT_MODEL,
        api_key=agent_api_key
    )
    
    intervention_engine = InterventionEngine(
        provider=CRITIC_PROVIDER,
        model_name=CRITIC_MODEL,
        api_key=critic_api_key
    )
    
    scorer = SemanticScorer(
        method=SCORER_METHOD,
        model_name=SCORER_MODEL,
        provider=SCORER_PROVIDER,
        api_key=scorer_api_key
    )
    
    evaluator = Evaluator(
        scorer=scorer,
        similarity_threshold=SIMILARITY_THRESHOLD
    )
    
    # Create pipeline
    pipeline = BenchmarkPipeline(
        agent_runner=agent_runner,
        intervention_engine=intervention_engine,
        evaluator=evaluator,
        max_concurrent=MAX_CONCURRENT,
        intervention_step_index=INTERVENTION_STEP  # Force intervention at s_0
    )
    
    # Run benchmark
    print("Starting research experiment...")
    print("This will process 500 queries - may take significant time.")
    print()
    
    results = await pipeline.run_benchmark(
        dataset_path=DATASET_PATH,
        intervention_type=INTERVENTION_TYPE,
        output_dir="research_results"
    )
    
    # Print detailed statistics
    stats = results["statistics"]
    print("\n" + "="*70)
    print("EXPERIMENTAL RESULTS")
    print("="*70)
    print(f"Total Queries: 500")
    print(f"Successful Audits: {stats.get('total_audits', 0)}")
    print(f"Violation Density (ρ): {stats.get('violation_density', 0.0):.4f}")
    print(f"Average Faithfulness Score (φ): {stats.get('average_faithfulness', 0.0):.4f}")
    print(f"Average Semantic Similarity: {stats.get('average_semantic_similarity', 0.0):.4f}")
    print(f"Total Violations: {stats.get('total_violations', 0)}")
    print(f"\nResults exported to:")
    print(f"  CSV: {results['csv_path']}")
    print(f"  JSON: {results['json_path']}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    asyncio.run(run_research_experiment())

