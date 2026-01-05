"""
Simple test script to verify the setup is working correctly.
"""
import sys

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    
    try:
        import torch
        version = getattr(torch, '__version__', 'installed')
        print(f"[OK] torch {version}")
    except ImportError as e:
        print(f"✗ torch: {e}")
        return False
    
    try:
        import transformers
        version = getattr(transformers, '__version__', 'installed')
        print(f"[OK] transformers {version}")
    except ImportError as e:
        print(f"[FAIL] transformers: {e}")
        return False
    
    try:
        import langgraph
        version = getattr(langgraph, '__version__', 'installed')
        print(f"[OK] langgraph {version}")
    except ImportError as e:
        print(f"[FAIL] langgraph: {e}")
        return False
    
    try:
        import langchain
        version = getattr(langchain, '__version__', 'installed')
        print(f"[OK] langchain {version}")
    except ImportError as e:
        print(f"[FAIL] langchain: {e}")
        return False
    
    try:
        from duckduckgo_search import DDGS
        print("[OK] duckduckgo-search")
    except ImportError as e:
        print(f"[FAIL] duckduckgo-search: {e}")
        return False
    
    try:
        import pydantic
        version = getattr(pydantic, '__version__', 'installed')
        print(f"[OK] pydantic {version}")
    except ImportError as e:
        print(f"[FAIL] pydantic: {e}")
        return False
    
    try:
        import pandas
        version = getattr(pandas, '__version__', 'installed')
        print(f"[OK] pandas {version}")
    except ImportError as e:
        print(f"[FAIL] pandas: {e}")
        return False
    
    try:
        import numpy
        version = getattr(numpy, '__version__', 'installed')
        print(f"[OK] numpy {version}")
    except ImportError as e:
        print(f"✗ numpy: {e}")
        return False
    
    return True


def test_ollama():
    """Test Ollama connection."""
    print("\nTesting Ollama connection...")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"[OK] Ollama is running")
            if models:
                print(f"  Available models: {[m.get('name', 'unknown') for m in models]}")
            else:
                print("  Warning: No models found. Run 'ollama pull llama3'")
            return True
        else:
            print(f"[FAIL] Ollama returned status {response.status_code}")
            return False
    except ImportError:
        print("[WARN] requests not installed (optional for this test)")
        return True  # Not critical
    except Exception as e:
        print(f"[FAIL] Ollama connection failed: {e}")
        print("  Make sure Ollama is running: ollama serve")
        return False


def test_project_modules():
    """Test that project modules can be imported."""
    print("\nTesting project modules...")
    
    try:
        from schemas import ReasoningStep, AgentResponse, Intervention, AuditResult
        print("[OK] schemas")
    except Exception as e:
        print(f"[FAIL] schemas: {e}")
        return False
    
    try:
        from agent import ReActAgent
        print("[OK] agent")
    except Exception as e:
        print(f"[FAIL] agent: {e}")
        return False
    
    try:
        from auditor import CausalAuditor
        print("[OK] auditor")
    except Exception as e:
        print(f"[FAIL] auditor: {e}")
        return False
    
    try:
        from scorer import FaithfulnessScorer
        print("[OK] scorer")
    except Exception as e:
        print(f"[FAIL] scorer: {e}")
        return False
    
    try:
        from logger import AuditLogger
        print("[OK] logger")
    except Exception as e:
        print(f"[FAIL] logger: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Project Ariadne - Setup Test")
    print("=" * 60)
    
    all_passed = True
    
    all_passed = test_imports() and all_passed
    all_passed = test_ollama() and all_passed
    all_passed = test_project_modules() and all_passed
    
    print("\n" + "=" * 60)
    if all_passed:
        print("[SUCCESS] All tests passed! Setup looks good.")
        return 0
    else:
        print("[FAILURE] Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

