"""
Main entry point for Project Ariadne: Causal Audit for Agentic Reasoning

This script orchestrates the agent, auditor, and scorer to perform
causal faithfulness evaluations.
"""
import argparse
import sys
from typing import List, Optional
from schemas import InterventionType, AuditResult, AuditSession
from agent import ReActAgent
from auditor import CausalAuditor
from scorer import FaithfulnessScorer
from logger import AuditLogger


def run_single_audit(
    query: str,
    agent: ReActAgent,
    auditor: CausalAuditor,
    scorer: FaithfulnessScorer,
    intervention_type: InterventionType = InterventionType.LOGIC_FLIP,
    intervention_step: Optional[int] = None
) -> AuditResult:
    """
    Run a single causal audit on a query.
    
    Args:
        query: Query to audit
        agent: ReAct agent instance
        auditor: Causal auditor instance
        scorer: Faithfulness scorer instance
        intervention_type: Type of intervention to apply
        intervention_step: Specific step to intervene on (None = auto-select)
        
    Returns:
        AuditResult with scores and violation status
    """
    print(f"\n{'='*60}")
    print(f"Auditing Query: {query}")
    print(f"{'='*60}\n")
    
    # Perform audit
    print("Step 1: Performing causal audit...")
    audit_result = auditor.audit(query, intervention_type, intervention_step)
    
    # Score faithfulness
    print("Step 2: Scoring faithfulness...")
    audit_result = scorer.score(audit_result)
    
    # Display results
    print(f"\n{'─'*60}")
    print("AUDIT RESULTS")
    print(f"{'─'*60}")
    print(f"Query: {query}")
    print(f"Intervention Type: {intervention_type.value}")
    print(f"Intervention Step: {audit_result.intervention.step_index}")
    print(f"\nOriginal Answer:")
    print(f"  {audit_result.original_response.final_answer[:200]}...")
    print(f"\nIntervened Answer:")
    if audit_result.intervened_response:
        print(f"  {audit_result.intervened_response.final_answer[:200]}...")
    else:
        print("  (No intervened response)")
    print(f"\nFaithfulness Score: {audit_result.faithfulness_score:.4f}")
    print(f"Violation Detected: {'YES' if audit_result.is_violation else 'NO'}")
    if audit_result.violation_reason:
        print(f"Violation Reason: {audit_result.violation_reason[:200]}...")
    print(f"\nSimilarity Metrics:")
    for metric, value in audit_result.similarity_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print(f"{'─'*60}\n")
    
    return audit_result


def run_batch_audit(
    queries: List[str],
    agent: ReActAgent,
    auditor: CausalAuditor,
    scorer: FaithfulnessScorer,
    intervention_type: InterventionType = InterventionType.LOGIC_FLIP,
    logger: Optional[AuditLogger] = None
) -> List[AuditResult]:
    """
    Run causal audits on multiple queries.
    
    Args:
        queries: List of queries to audit
        agent: ReAct agent instance
        auditor: Causal auditor instance
        scorer: Faithfulness scorer instance
        intervention_type: Type of intervention to apply
        logger: Optional logger for CSV export
        
    Returns:
        List of AuditResults
    """
    results = []
    
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] Processing query...")
        try:
            result = run_single_audit(
                query,
                agent,
                auditor,
                scorer,
                intervention_type
            )
            results.append(result)
            
            # Log if logger provided
            if logger:
                logger.log_audit_result(result)
                
        except Exception as e:
            print(f"Error processing query '{query}': {e}")
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print("BATCH AUDIT SUMMARY")
    print(f"{'='*60}")
    print(f"Total Queries: {len(queries)}")
    print(f"Successful Audits: {len(results)}")
    violations = sum(1 for r in results if r.is_violation)
    print(f"Violations Detected: {violations}")
    if results:
        avg_faithfulness = sum(r.faithfulness_score for r in results) / len(results)
        print(f"Average Faithfulness Score: {avg_faithfulness:.4f}")
    print(f"{'='*60}\n")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Project Ariadne: Causal Audit for Agentic Reasoning"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to audit"
    )
    parser.add_argument(
        "--queries-file",
        type=str,
        help="Path to file with queries (one per line)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3",
        help="Ollama model name (default: llama3)"
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--intervention-type",
        type=str,
        default="logic_flip",
        choices=["logic_flip", "fact_reversal", "premise_negation", "causal_reversal"],
        help="Type of intervention to apply (default: logic_flip)"
    )
    parser.add_argument(
        "--intervention-step",
        type=int,
        default=None,
        help="Specific reasoning step to intervene on (default: auto-select)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="audit_results",
        help="Directory for CSV output (default: audit_results)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum agent reasoning iterations (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Determine queries
    queries = []
    if args.query:
        queries = [args.query]
    elif args.queries_file:
        try:
            with open(args.queries_file, 'r', encoding='utf-8') as f:
                queries = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Error: File '{args.queries_file}' not found.")
            sys.exit(1)
    else:
        # Default: run example query
        queries = [
            "What is the capital of France?",
            "What causes global warming?",
            "Calculate 25 * 4 + 10"
        ]
        print("No query specified. Running example queries...")
    
    if not queries:
        print("Error: No queries to process.")
        sys.exit(1)
    
    # Initialize components
    print("Initializing components...")
    print(f"Model: {args.model}")
    print(f"Ollama URL: {args.ollama_url}")
    
    try:
        agent = ReActAgent(
            model_name=args.model,
            base_url=args.ollama_url,
            max_iterations=args.max_iterations
        )
        
        auditor = CausalAuditor(
            agent=agent,
            critic_model_name=args.model,
            critic_base_url=args.ollama_url
        )
        
        scorer = FaithfulnessScorer(
            scorer_model_name=args.model,
            scorer_base_url=args.ollama_url
        )
        
        logger = AuditLogger(output_dir=args.output_dir)
        
        # Map intervention type
        intervention_type_map = {
            "logic_flip": InterventionType.LOGIC_FLIP,
            "fact_reversal": InterventionType.FACT_REVERSAL,
            "premise_negation": InterventionType.PREMISE_NEGATION,
            "causal_reversal": InterventionType.CAUSAL_REVERSAL
        }
        intervention_type = intervention_type_map[args.intervention_type]
        
        # Run audits
        if len(queries) == 1:
            result = run_single_audit(
                queries[0],
                agent,
                auditor,
                scorer,
                intervention_type,
                args.intervention_step
            )
            logger.log_audit_result(result)
            print(f"\nResults saved to: {logger.output_dir}/")
        else:
            results = run_batch_audit(
                queries,
                agent,
                auditor,
                scorer,
                intervention_type,
                logger
            )
            # Also save batch summary
            batch_file = logger.log_batch(results)
            print(f"\nBatch results saved to: {batch_file}")
        
        print("\nAudit complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

