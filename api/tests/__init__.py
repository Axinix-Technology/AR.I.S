import os
import sys

# Ensure workspace root is in sys.path when api.tests is imported
_workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)
