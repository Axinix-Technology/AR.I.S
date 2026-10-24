# pyright: ignore
r"""
AR.I.S — Command Prompt & Local Shell Utterance Execution Test

Real-world execution test with ZERO mocks, ZERO fake browser calls, and ZERO synthetic patches.
Tests whether AR.I.S accepts a genuine user utterance to execute a command prompt operation,
correctly identifies the capability, and dispatches real execution.
"""

import os
import sys
import unittest
import subprocess

# Ensure workspace root is in python path
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


class CommandPromptUtteranceExecutionTests(unittest.TestCase):
    """
    Tests real execution of command prompt utterance without any mock data or fake calls.
    """

    def tearDown(self):
        # Clean up any opened notepad instances if launched
        subprocess.run("taskkill /f /im notepad.exe", shell=True, capture_output=True)

    def test_open_command_prompt_and_open_notepad(self):
        """
        Utterance: 'Open command prompt and open notepad'
        Must genuinely invoke command_prompt to execute the requested action.
        """
        utterance = "Open command prompt and open notepad"
        print(f"\n[REAL TEST] Input Utterance: '{utterance}'")

        # Pure real execution call - no mocks, no patches, no fake data
        result = ArisBrain.deliberate_reasoning(question_text=utterance, allow_external=True)

        print("\n" + "=" * 78)
        print("[REAL EXECUTION RESULT]")
        print("  Status:", result.get("status"))
        print("  Final Answer:", result.get("final_answer"))
        trace = result.get("thinking_trace", {})
        executed_steps = trace.get("executed_steps", [])
        print(f"  Executed Steps Count: {len(executed_steps)}")
        for idx, step in enumerate(executed_steps):
            print(f"    Step {idx + 1}: capability={step.get('capability')}, status={step.get('status')}, result={step.get('result')}")
        print("=" * 78 + "\n")

        # Verify real execution: command_prompt capability must have been executed
        executed_capabilities = [step.get("capability") for step in executed_steps if step.get("capability")]
        self.assertTrue(
            any("command_prompt" in str(cap) for cap in executed_capabilities),
            f"AR.I.S did not execute 'command_prompt' for utterance '{utterance}'. Steps: {executed_steps}",
        )


if __name__ == "__main__":
    unittest.main()
