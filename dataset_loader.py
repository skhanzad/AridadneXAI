"""
JSONL Dataset Loader for benchmarking queries.
"""
import json
from typing import List, Dict, Any, Iterator, Optional
from pathlib import Path
from pydantic import BaseModel, Field


class QueryExample(BaseModel):
    """A single query example from the dataset."""
    query: str = Field(..., description="The query to be answered")
    expected_answer: Optional[str] = Field(None, description="Expected answer (optional)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the capital of France?",
                "expected_answer": "Paris",
                "metadata": {"category": "geography", "difficulty": "easy"}
            }
        }


class DatasetLoader:
    """Loads and manages JSONL datasets for benchmarking."""
    
    def __init__(self, dataset_path: str):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_path: Path to JSONL file
        """
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    def load(self) -> List[QueryExample]:
        """
        Load all examples from the JSONL file.
        
        Returns:
            List of QueryExample objects
        """
        examples = []
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    example = QueryExample(**data)
                    examples.append(example)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Skipping line {line_num} due to error: {e}")
                    continue
        
        return examples
    
    def iter_examples(self) -> Iterator[QueryExample]:
        """
        Iterate over examples without loading all into memory.
        
        Yields:
            QueryExample objects
        """
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    yield QueryExample(**data)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Skipping line {line_num} due to error: {e}")
                    continue
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        count = 0
        for _ in self.iter_examples():
            count += 1
        return count


def create_example_dataset(output_path: str = "benchmark_dataset.jsonl"):
    """Create an example JSONL dataset for testing."""
    examples = [
        {
            "query": "What is the capital of France?",
            "expected_answer": "Paris",
            "metadata": {"category": "geography", "difficulty": "easy"}
        },
        {
            "query": "What causes global warming?",
            "expected_answer": "Greenhouse gases trap heat in the atmosphere",
            "metadata": {"category": "science", "difficulty": "medium"}
        },
        {
            "query": "Calculate 25 * 4 + 10",
            "expected_answer": "110",
            "metadata": {"category": "math", "difficulty": "easy"}
        },
        {
            "query": "Who wrote Romeo and Juliet?",
            "expected_answer": "William Shakespeare",
            "metadata": {"category": "literature", "difficulty": "easy"}
        },
        {
            "query": "What is the largest planet in our solar system?",
            "expected_answer": "Jupiter",
            "metadata": {"category": "astronomy", "difficulty": "easy"}
        }
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Created example dataset: {output_path}")

