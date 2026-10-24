# pyright: ignore
r"""
AR.I.S — Phase 1: Dynamic Capability Affordance Perception Test

Proves:
  1. Capabilities are discovered dynamically from ArisCapabilityRegistry.
  2. Multiple candidate capabilities are produced before any execution.
  3. Goal information (action, target, instrument, path) is preserved.
  4. Execution count is strictly ZERO (No tool execution in Phase 1).
"""

import os
import sys
import unittest

_workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
try:
    import django
    django.setup()
except Exception:
    pass

from api.cognitive.brain import ArisBrain
from api.semantic.executable_tools import ArisCapabilityRegistry


class Phase1CapabilityAffordancePerceptionTests(unittest.TestCase):
    """
    Validates Phase 1 Dynamic Capability Affordance Perception.
    """

    def test_git_path_query_produces_multiple_candidates_with_zero_execution(self):
        r"""
        Query: 'What is the main branch status of E:\Loigmax\tracker-v2?'
        Must perceive both command_prompt and browser_access, identify parameters & deficits,
        and execute ZERO capabilities.
        """
        query = r"What is the main branch status of E:\Loigmax\tracker-v2?"
        
        result = ArisBrain.perceive_capability_affordances(query)

        # 1. Zero execution invariant
        self.assertEqual(result.get("execution_count"), 0, "Phase 1 must execute ZERO capabilities.")

        # 2. Path preservation
        goal_info = result.get("goal", {})
        self.assertEqual(
            goal_info.get("path"),
            r"E:\Loigmax\tracker-v2",
            f"Target filesystem path must be preserved. Got: {goal_info.get('path')}",
        )

        # 3. Dynamic Registry Discovery
        reg_caps = result.get("registered_capabilities", [])
        self.assertIn("command_prompt", reg_caps, "command_prompt must be discovered from registry.")
        self.assertIn("browser_access", reg_caps, "browser_access must be discovered from registry.")

        # 4. Multiple Candidates Produced
        candidates = result.get("candidates", {})
        self.assertIn("command_prompt", candidates, "command_prompt must be in candidate evaluations.")
        self.assertIn("browser_access", candidates, "browser_access must be in candidate evaluations.")

        # 5. command_prompt candidate inspection: relevant=yes, working_dir bound, command missing
        cp_cand = candidates["command_prompt"]
        self.assertEqual(cp_cand.get("relevant"), "yes", "command_prompt must be relevant for path query.")
        self.assertEqual(
            cp_cand.get("parameters", {}).get("working_dir"),
            r"E:\Loigmax\tracker-v2",
            "working_dir must be bound to the target repository path.",
        )
        self.assertIn("command", cp_cand.get("missing", []), "command parameter must be reported as missing.")

        # 6. browser_access candidate inspection: relevant=possible, query bound, missing none
        ba_cand = candidates["browser_access"]
        self.assertEqual(ba_cand.get("relevant"), "possible", "browser_access must be evaluated as possible.")
        self.assertTrue(bool(ba_cand.get("parameters", {}).get("query")), "browser_access query must be bound.")
        self.assertEqual(ba_cand.get("missing"), [], "browser_access must report missing: none.")

    def test_dynamic_runtime_capability_discovery(self):
        """
        Proves capability discovery is dynamic and introspects ArisCapabilityRegistry at runtime.
        """
        class DynamicInspectionTool:
            CAPABILITY_NAME = "dynamic_inspector"
            DESCRIPTION = "Dynamic test tool added at runtime."
            PARAMETERS_SCHEMA = {"inspection_target": "Target to inspect"}
            ACCEPTS = {"domain": "custom_inspection", "requires": ["inspection_target"]}

        # Dynamically register new capability
        ArisCapabilityRegistry._CAPABILITIES["dynamic_inspector"] = DynamicInspectionTool
        try:
            res = ArisBrain.perceive_capability_affordances("Inspect running services on machine")
            self.assertIn("dynamic_inspector", res.get("registered_capabilities", []))
            self.assertIn("dynamic_inspector", res.get("candidates", {}))
            self.assertEqual(res.get("execution_count"), 0)
        finally:
            # Clean up dynamic capability
            ArisCapabilityRegistry._CAPABILITIES.pop("dynamic_inspector", None)


if __name__ == "__main__":
    unittest.main()
