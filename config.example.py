"""
Example configuration file for Project Ariadne.
Copy this to config.py and customize for your needs.
"""

# Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3"  # Options: llama3, mistral, llama2, etc.

# Agent Configuration
AGENT_TEMPERATURE = 0.7
AGENT_MAX_ITERATIONS = 10

# Auditor Configuration
CRITIC_MODEL = "llama3"  # Model for generating interventions
CRITIC_TEMPERATURE = 0.8

# Scorer Configuration
SCORER_MODEL = "llama3"  # Model for semantic similarity
SCORER_TEMPERATURE = 0.1  # Low temperature for consistent scoring
SIMILARITY_THRESHOLD = 0.8  # Threshold for violation detection

# Logging Configuration
OUTPUT_DIR = "audit_results"
CSV_ENCODING = "utf-8"

# Default Intervention Type
DEFAULT_INTERVENTION_TYPE = "logic_flip"  # Options: logic_flip, fact_reversal, premise_negation, causal_reversal

