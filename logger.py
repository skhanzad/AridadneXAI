"""
CSV Logging System for exporting audit results to research paper format.
"""
import csv
import os
from typing import List, Optional
from datetime import datetime
from schemas import AuditResult, AuditSession


class AuditLogger:
    """
    Logs audit results to CSV format suitable for research paper tables.
    """
    
    def __init__(self, output_dir: str = "audit_results"):
        """
        Initialize the logger.
        
        Args:
            output_dir: Directory to save CSV files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def log_audit_result(self, audit_result: AuditResult, filename: Optional[str] = None):
        """
        Log a single audit result to CSV.
        
        Args:
            audit_result: The audit result to log
            filename: Optional custom filename (default: audit_results_YYYYMMDD_HHMMSS.csv)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audit_results_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Check if file exists to determine if we need headers
        file_exists = os.path.exists(filepath)
        
        with open(filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write headers if new file
            if not file_exists:
                headers = [
                    "audit_id",
                    "timestamp",
                    "query",
                    "intervention_type",
                    "intervention_step_index",
                    "original_reasoning_steps_count",
                    "intervened_reasoning_steps_count",
                    "original_final_answer",
                    "intervened_final_answer",
                    "faithfulness_score",
                    "is_violation",
                    "violation_reason",
                    "jaccard_similarity",
                    "character_similarity",
                    "semantic_similarity",
                    "exact_match",
                    "length_ratio",
                    "original_execution_time",
                    "intervened_execution_time"
                ]
                writer.writerow(headers)
            
            # Write data row
            row = [
                audit_result.audit_id,
                audit_result.timestamp.isoformat(),
                audit_result.query,
                audit_result.intervention.intervention_type.value,
                audit_result.intervention.step_index,
                len(audit_result.original_response.reasoning_steps),
                len(audit_result.intervened_response.reasoning_steps) if audit_result.intervened_response else 0,
                audit_result.original_response.final_answer[:500],  # Truncate long answers
                audit_result.intervened_response.final_answer[:500] if audit_result.intervened_response else "",
                f"{audit_result.faithfulness_score:.4f}",
                "Yes" if audit_result.is_violation else "No",
                audit_result.violation_reason[:200] if audit_result.violation_reason else "",
                f"{audit_result.similarity_metrics.get('jaccard_similarity', 0.0):.4f}",
                f"{audit_result.similarity_metrics.get('character_similarity', 0.0):.4f}",
                f"{audit_result.similarity_metrics.get('semantic_similarity', 0.0):.4f}",
                "Yes" if audit_result.similarity_metrics.get('exact_match', 0.0) > 0.5 else "No",
                f"{audit_result.similarity_metrics.get('length_ratio', 0.0):.4f}",
                f"{audit_result.original_response.execution_time:.2f}",
                f"{audit_result.intervened_response.execution_time:.2f}" if audit_result.intervened_response else "0.00"
            ]
            writer.writerow(row)
    
    def log_audit_session(self, session: AuditSession, filename: Optional[str] = None):
        """
        Log an entire audit session to CSV.
        
        Args:
            session: The audit session to log
            filename: Optional custom filename
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audit_session_{session.session_id}_{timestamp}.csv"
        
        # Log each audit result in the session
        for result in session.audit_results:
            self.log_audit_result(result, filename)
    
    def log_batch(self, audit_results: List[AuditResult], filename: Optional[str] = None):
        """
        Log multiple audit results to a single CSV file.
        
        Args:
            audit_results: List of audit results to log
            filename: Optional custom filename
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audit_batch_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write headers
            headers = [
                "audit_id",
                "timestamp",
                "query",
                "intervention_type",
                "intervention_step_index",
                "original_reasoning_steps_count",
                "intervened_reasoning_steps_count",
                "original_final_answer",
                "intervened_final_answer",
                "faithfulness_score",
                "is_violation",
                "violation_reason",
                "jaccard_similarity",
                "character_similarity",
                "semantic_similarity",
                "exact_match",
                "length_ratio",
                "original_execution_time",
                "intervened_execution_time"
            ]
            writer.writerow(headers)
            
            # Write data rows
            for audit_result in audit_results:
                row = [
                    audit_result.audit_id,
                    audit_result.timestamp.isoformat(),
                    audit_result.query,
                    audit_result.intervention.intervention_type.value,
                    audit_result.intervention.step_index,
                    len(audit_result.original_response.reasoning_steps),
                    len(audit_result.intervened_response.reasoning_steps) if audit_result.intervened_response else 0,
                    audit_result.original_response.final_answer[:500],
                    audit_result.intervened_response.final_answer[:500] if audit_result.intervened_response else "",
                    f"{audit_result.faithfulness_score:.4f}",
                    "Yes" if audit_result.is_violation else "No",
                    audit_result.violation_reason[:200] if audit_result.violation_reason else "",
                    f"{audit_result.similarity_metrics.get('jaccard_similarity', 0.0):.4f}",
                    f"{audit_result.similarity_metrics.get('character_similarity', 0.0):.4f}",
                    f"{audit_result.similarity_metrics.get('semantic_similarity', 0.0):.4f}",
                    "Yes" if audit_result.similarity_metrics.get('exact_match', 0.0) > 0.5 else "No",
                    f"{audit_result.similarity_metrics.get('length_ratio', 0.0):.4f}",
                    f"{audit_result.original_response.execution_time:.2f}",
                    f"{audit_result.intervened_response.execution_time:.2f}" if audit_result.intervened_response else "0.00"
                ]
                writer.writerow(row)
        
        return filepath

