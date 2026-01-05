"""
Example script for running the benchmarking suite.
"""
import asyncio
import os
from benchmark import BenchmarkPipeline
from agent_runner import AgentRunner, AgentProvider
from intervention_engine import InterventionEngine
from evaluator import Evaluator
from semantic_scorer import SemanticScorer
from schemas import InterventionType


async def run_example():
    """Run an example benchmark."""
    
    # Configuration - Using Ollama (local)
    DATASET_PATH = "benchmark_dataset.jsonl"
    AGENT_PROVIDER = "ollama"  # Local Ollama instance
    AGENT_MODEL = "llama3"  # or "mistral", "llama2", etc.
    CRITIC_PROVIDER = "ollama"
    CRITIC_MODEL = "llama3"
    INTERVENTION_TYPE = InterventionType.LOGIC_FLIP
    MAX_CONCURRENT = 3  # Lower for local inference
    
    # Ollama doesn't need API keys, but uses base_url
    agent_api_key = None
    critic_api_key = None
    ollama_base_url = "http://127.0.0.1:11434/v1"  # Ollama OpenAI-compatible API
    
    print("="*60)
    print("Project Ariadne - Benchmarking Suite (Ollama)")
    print("="*60)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Agent: {AGENT_PROVIDER}/{AGENT_MODEL}")
    print(f"Critic: {CRITIC_PROVIDER}/{CRITIC_MODEL}")
    print(f"Intervention: {INTERVENTION_TYPE.value}")
    print(f"Ollama URL: {ollama_base_url}")
    print("="*60)
    print()
    print("Note: Make sure Ollama is running (ollama serve)")
    print("      and the model is pulled (ollama pull llama3)")
    print()
    
    # Initialize components
    print("Initializing components...")
    
    agent_runner = AgentRunner(
        provider=AgentProvider(AGENT_PROVIDER),
        model_name=AGENT_MODEL,
        api_key=agent_api_key,
        base_url=ollama_base_url
    )
    
    intervention_engine = InterventionEngine(
        provider=CRITIC_PROVIDER,
        model_name=CRITIC_MODEL,
        api_key=critic_api_key,
        base_url=ollama_base_url
    )
    
    # Use embedding-based scorer for efficiency
    scorer = SemanticScorer(
        method="embedding",
        model_name="all-MiniLM-L6-v2"
    )
    
    evaluator = Evaluator(
        scorer=scorer,
        similarity_threshold=0.8
    )
    
    # Create pipeline
    pipeline = BenchmarkPipeline(
        agent_runner=agent_runner,
        intervention_engine=intervention_engine,
        evaluator=evaluator,
        max_concurrent=MAX_CONCURRENT
    )
    
    # Run benchmark
    print("Starting benchmark...")
    results = await pipeline.run_benchmark(
        dataset_path=DATASET_PATH,
        intervention_type=INTERVENTION_TYPE,
        output_dir="benchmark_results"
    )
    
    print("\n✓ Benchmark complete!")
    return results


if __name__ == "__main__":
    # Create example dataset if it doesn't exist
    from dataset_loader import create_example_dataset
    import os
    
    if not os.path.exists("benchmark_dataset.jsonl"):
        print("Creating example dataset...")
        create_example_dataset("benchmark_dataset.jsonl")
        print()
    
    # Run benchmark
    asyncio.run(run_example())

