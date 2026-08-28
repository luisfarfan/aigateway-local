import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

try:
    from src.modules.providers.orchestrator.provider import CrewAIOrchestratorProvider
    print("Orchestrator import SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Orchestrator import FAILED: {e}")
