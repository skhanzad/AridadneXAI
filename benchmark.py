"""
Main benchmarking pipeline with async processing, progress bars, and error handling.
"""
import asyncio
import json
import csv
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
try:
    from tqdm.asyncio import tqdm as atqdm
except ImportError:
    # Fallback for older tqdm versions
    import asyncio
    class atqdm:
        @staticmethod
        def as_completed(coros, total=None, desc=None):
            if desc:
                print(desc)
            return asyncio.as_completed(coros)

from dataset_loader import DatasetLoader, QueryExample
from agent_runner import AgentRunner, AgentProvider
from intervention_engine import InterventionEngine
from evaluator import Evaluator
from semantic_scorer import SemanticScorer
from schemas import InterventionType, AuditResult, ReasoningTrace


class BenchmarkPipeline:
    """
    Comprehensive benchmarking pipeline for causal faithfulness evaluation.
    """
    
    def __init__(
        self,
        agent_runner: AgentRunner,
        intervention_engine: InterventionEngine,
        evaluator: Evaluator,
        max_concurrent: int = 5,
        intervention_step_index: Optional[int] = None
    ):
        """
        Initialize the benchmark pipeline.
        
        Args:
            agent_runner: AgentRunner instance
            intervention_engine: InterventionEngine instance
            evaluator: Evaluator instance
            max_concurrent: Maximum concurrent API calls
            intervention_step_index: Step index to intervene on (None = auto-select)
        """
        self.agent_runner = agent_runner
        self.intervention_engine = intervention_engine
        self.evaluator = evaluator
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.intervention_step_index = intervention_step_index
    
    async def process_single_query(
        self,
        query_example: QueryExample,
        intervention_type: InterventionType
    ) -> Optional[AuditResult]:
        """
        Process a single query through the full pipeline.
        
        Args:
            query_example: Query example from dataset
            intervention_type: Type of intervention to apply
            
        Returns:
            AuditResult or None if error occurred
        """
        async with self.semaphore:
            try:
                # Step 1: Run agent to get original trace
                original_trace = await self.agent_runner.run(query_example.query)
                
                if not original_trace.steps:
                    print(f"Warning: No reasoning steps for query: {query_example.query}")
                    return None
                
                # Step 2: Select intervention step
                intervention_step_idx = self._select_intervention_step(
                    original_trace,
                    self.intervention_step_index
                )
                
                if intervention_step_idx is None:
                    print(f"Warning: No suitable step for intervention: {query_example.query}")
                    return None
                
                target_step = original_trace.steps[intervention_step_idx]
                
                # Step 3: Generate intervention
                intervention = await self.intervention_engine.generate_intervention(
                    target_step,
                    intervention_type,
                    intervention_step_idx
                )
                
                # Step 4: Rerun with intervention (do-calculus: regenerate subsequent steps)
                intervened_trace = await self.agent_runner.run_with_intervention(
                    query_example.query,
                    original_trace,
                    intervention_step_idx,
                    intervention.intervened_thought
                )
                
                # Step 5: Evaluate faithfulness
                audit_result = await self.evaluator.evaluate(
                    query_example.query,
                    original_trace,
                    intervened_trace,
                    intervention
                )
                
                return audit_result
                
            except Exception as e:
                print(f"Error processing query '{query_example.query}': {e}")
                import traceback
                traceback.print_exc()
                return None
    
    def _select_intervention_step(
        self,
        trace: ReasoningTrace,
        preferred_index: Optional[int] = None
    ) -> Optional[int]:
        """
        Select the step to intervene on.
        
        Args:
            trace: Reasoning trace
            preferred_index: Preferred step index (if provided)
            
        Returns:
            Step index or None if no suitable step
        """
        if preferred_index is not None:
            if 0 <= preferred_index < len(trace.steps):
                return preferred_index
            else:
                print(f"Warning: Preferred index {preferred_index} out of range")
        
        # Auto-select: first substantive step
        for i, step in enumerate(trace.steps):
            if step.thought and len(step.thought.strip()) > 10:
                return i
        
        # Fallback: first step
        if trace.steps:
            return 0
        
        return None
    
    async def run_benchmark(
        self,
        dataset_path: str,
        intervention_type: InterventionType = InterventionType.LOGIC_FLIP,
        output_dir: str = "benchmark_results"
    ) -> Dict[str, Any]:
        """
        Run the full benchmark on a dataset.
        
        Args:
            dataset_path: Path to JSONL dataset
            intervention_type: Type of intervention to apply
            output_dir: Directory for output files
            
        Returns:
            Dictionary with results and statistics
        """
        # Load dataset
        print(f"Loading dataset: {dataset_path}")
        loader = DatasetLoader(dataset_path)
        examples = loader.load()
        print(f"Loaded {len(examples)} examples")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Process all queries
        print(f"Processing {len(examples)} queries with {intervention_type.value} intervention...")
        
        tasks = [
            self.process_single_query(example, intervention_type)
            for example in examples
        ]
        
        # Run with progress bar
        audit_results = []
        for coro in atqdm.as_completed(tasks, total=len(tasks), desc="Processing queries"):
            result = await coro
            if result:
                audit_results.append(result)
        
        # Compute aggregate statistics
        stats = self.evaluator.compute_aggregate_stats(audit_results)
        
        # Export results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = output_path / f"benchmark_results_{timestamp}.csv"
        json_path = output_path / f"benchmark_results_{timestamp}.json"
        
        self._export_csv(audit_results, csv_path)
        self._export_json(audit_results, stats, json_path)
        
        # Print summary
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        print(f"Total Queries: {len(examples)}")
        print(f"Successful Audits: {len(audit_results)}")
        print(f"Violation Density (ρ): {stats.get('violation_density', 0.0):.4f}")
        print(f"Average Faithfulness (φ): {stats.get('average_faithfulness', 0.0):.4f}")
        print(f"Total Violations: {stats.get('total_violations', 0)}")
        print(f"\nResults exported to:")
        print(f"  CSV: {csv_path}")
        print(f"  JSON: {json_path}")
        print("="*60)
        
        return {
            "audit_results": audit_results,
            "statistics": stats,
            "csv_path": str(csv_path),
            "json_path": str(json_path)
        }
    
    def _export_csv(self, audit_results: List[AuditResult], output_path: Path):
        """Export results to CSV."""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Headers
            headers = [
                "audit_id",
                "timestamp",
                "query",
                "intervention_type",
                "intervention_step_index",
                "original_trace_steps",
                "intervened_trace_steps",
                "original_answer",
                "intervened_answer",
                "faithfulness_score",
                "is_violation",
                "violation_reason",
                "semantic_similarity",
                "jaccard_similarity",
                "character_similarity",
                "exact_match",
                "length_ratio"
            ]
            writer.writerow(headers)
            
            # Data rows
            for result in audit_results:
                row = [
                    result.audit_id,
                    result.timestamp.isoformat(),
                    result.query,
                    result.intervention.intervention_type.value,
                    result.intervention.step_index,
                    len(result.original_response.reasoning_steps),
                    len(result.intervened_response.reasoning_steps) if result.intervened_response else 0,
                    result.original_response.final_answer[:500],
                    result.intervened_response.final_answer[:500] if result.intervened_response else "",
                    f"{result.faithfulness_score:.4f}",
                    "Yes" if result.is_violation else "No",
                    result.violation_reason[:200] if result.violation_reason else "",
                    f"{result.similarity_metrics.get('semantic_similarity', 0.0):.4f}",
                    f"{result.similarity_metrics.get('jaccard_similarity', 0.0):.4f}",
                    f"{result.similarity_metrics.get('character_similarity', 0.0):.4f}",
                    "Yes" if result.similarity_metrics.get('exact_match', 0.0) > 0.5 else "No",
                    f"{result.similarity_metrics.get('length_ratio', 0.0):.4f}"
                ]
                writer.writerow(row)
    
    def _export_json(self, audit_results: List[AuditResult], stats: Dict[str, Any], output_path: Path):
        """Export results to JSON."""
        def serialize_pydantic(obj):
            """Serialize Pydantic model with datetime handling."""
            if hasattr(obj, 'model_dump'):
                # Use model_dump with json mode for Pydantic v2
                return obj.model_dump(mode='json')
            elif hasattr(obj, 'dict'):
                data = obj.dict()
                # Convert datetime objects to ISO strings recursively
                def convert_datetime(value):
                    if hasattr(value, 'isoformat'):
                        return value.isoformat()
                    elif isinstance(value, list):
                        return [convert_datetime(item) for item in value]
                    elif isinstance(value, dict):
                        return {k: convert_datetime(v) for k, v in value.items()}
                    return value
                return convert_datetime(data)
            return obj
        
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "statistics": stats,
            "audit_results": [
                {
                    "audit_id": r.audit_id,
                    "query": r.query,
                    "intervention_type": r.intervention.intervention_type.value,
                    "intervention_step_index": r.intervention.step_index,
                    "original_trace": serialize_pydantic(r.original_response),
                    "intervened_trace": serialize_pydantic(r.intervened_response) if r.intervened_response else None,
                    "faithfulness_score": r.faithfulness_score,
                    "is_violation": r.is_violation,
                    "violation_reason": r.violation_reason,
                    "similarity_metrics": r.similarity_metrics
                }
                for r in audit_results
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)


async def main():
    """Main entry point for benchmarking."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Project Ariadne Benchmarking Suite")
    parser.add_argument("--dataset", type=str, required=True, help="Path to JSONL dataset")
    parser.add_argument("--agent-provider", type=str, default="ollama", choices=["openai", "anthropic", "ollama"])
    parser.add_argument("--agent-model", type=str, default="llama3", help="Agent model name")
    parser.add_argument("--agent-api-key", type=str, default=None, help="Agent API key")
    parser.add_argument("--agent-base-url", type=str, default="http://localhost:11434/v1", help="Agent base URL (for Ollama)")
    parser.add_argument("--critic-provider", type=str, default="ollama", choices=["openai", "anthropic", "ollama"])
    parser.add_argument("--critic-model", type=str, default="llama3", help="Critic model name")
    parser.add_argument("--critic-api-key", type=str, default=None, help="Critic API key")
    parser.add_argument("--critic-base-url", type=str, default="http://localhost:11434/v1", help="Critic base URL (for Ollama)")
    parser.add_argument("--scorer-method", type=str, default="embedding", choices=["embedding", "llm_judge"])
    parser.add_argument("--scorer-model", type=str, default="all-MiniLM-L6-v2", help="Scorer model name")
    parser.add_argument("--intervention-type", type=str, default="logic_flip", 
                       choices=["logic_flip", "fact_reversal", "premise_negation", "causal_reversal"])
    parser.add_argument("--intervention-step", type=int, default=None, help="Step index to intervene on")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Max concurrent API calls")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Output directory")
    parser.add_argument("--similarity-threshold", type=float, default=0.8, help="Similarity threshold for violations")
    
    args = parser.parse_args()
    
    # Initialize components
    print("Initializing components...")
    
    agent_runner = AgentRunner(
        provider=AgentProvider(args.agent_provider),
        model_name=args.agent_model,
        api_key=args.agent_api_key,
        base_url=args.agent_base_url if args.agent_provider == "ollama" else None
    )
    
    intervention_engine = InterventionEngine(
        provider=args.critic_provider,
        model_name=args.critic_model,
        api_key=args.critic_api_key,
        base_url=args.critic_base_url if args.critic_provider == "ollama" else None
    )
    
    scorer = SemanticScorer(
        method=args.scorer_method,
        model_name=args.scorer_model,
        provider=args.agent_provider if args.scorer_method == "llm_judge" else None,
        api_key=args.agent_api_key if args.scorer_method == "llm_judge" else None
    )
    
    evaluator = Evaluator(
        scorer=scorer,
        similarity_threshold=args.similarity_threshold
    )
    
    # Create pipeline
    pipeline = BenchmarkPipeline(
        agent_runner=agent_runner,
        intervention_engine=intervention_engine,
        evaluator=evaluator,
        max_concurrent=args.max_concurrent,
        intervention_step_index=args.intervention_step
    )
    
    # Run benchmark
    intervention_type = InterventionType(args.intervention_type)
    results = await pipeline.run_benchmark(
        dataset_path=args.dataset,
        intervention_type=intervention_type,
        output_dir=args.output_dir
    )
    
    print("\nBenchmark complete!")


if __name__ == "__main__":
    asyncio.run(main())

