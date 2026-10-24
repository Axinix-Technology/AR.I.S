# pyright: ignore
r"""
AR.I.S — Autonomous Multi-Tool Git Execution & Planner Test

Tests autonomous multi-tool composition without hardcoded orchestration pipelines:
  Step 1: [Tool: browser_access] -> Research git command syntax via web search.
  Step 2: [Tool: command_prompt] -> Validate and target repository path (E:\Loigmax\tracker-v2).
  Step 3: [Tool: command_prompt] -> Execute 'git status' and extract branch state.
  Planner Transparency: Prints the decomposed multi-step plan and every step's execution capability.
"""

import unittest
from unittest.mock import patch, MagicMock
from api.cognitive.brain import ArisBrain
from api.semantic.executable_tools import ArisCapabilityRegistry


class MultiToolGitExecutionTests(unittest.TestCase):
    """
    Validates autonomous multi-tool planning, browser research, and command prompt execution.
    """

    def test_autonomous_3_step_git_status_planner(self):
        """
        Prompt: 'What is the main branch status of this folder git: E:\\Loigmax\\E-commerce?'
        AR.I.S must decompose into 3 steps, print the plan, and execute using browser_access + command_prompt.
        """
        target_repo = r"E:\Loigmax\E-commerce"
        query = f"What is the main branch status of this folder git: {target_repo}?"

        mock_web_html = (
            '<a class="result__snippet">Use git status to view current working tree status and branch status. '
            'Use git branch to view all branches including main.</a>'
        )

        with patch("httpx.Client.post") as mock_http:
            mock_http.return_value = MagicMock(status_code=200, text=mock_web_html)

            # Execute deliberative reasoning
            result = ArisBrain.deliberate_reasoning(question_text=query, allow_external=True)

            # 1. Verify Trace & Plan
            trace = result.get("thinking_trace", {})
            plan = trace.get("plan", [])
            executed_steps = trace.get("executed_steps", [])

            print("\n" + "=" * 78)
            print("[TEST VERIFICATION] Trace Summary:")
            print(f"  Plan Length: {len(plan)} step(s)")
            print(f"  Executed Steps: {len(executed_steps)}")
            print(f"  Final Answer Preview:\n    {str(result.get('final_answer'))[:200]}")
            print("=" * 78 + "\n")

            # 2. Plan length verification
            self.assertEqual(len(plan), 3, "Planner must decompose task into exactly 3 execution steps.")
            self.assertEqual(len(executed_steps), 3, "All 3 steps must execute successfully.")

            # 3. Capability assignments verification
            self.assertEqual(
                plan[0].get("capability_name"),
                "browser_access",
                "Step 1 must be browser_access to research git command syntax.",
            )
            self.assertEqual(
                plan[1].get("capability_name"),
                "command_prompt",
                "Step 2 must be command_prompt to navigate / verify target repository path.",
            )
            self.assertEqual(
                plan[2].get("capability_name"),
                "command_prompt",
                "Step 3 must be command_prompt to execute 'git status'.",
            )

            # 4. Step 3 Execution Result Verification
            step_3_result = executed_steps[2].get("result", "")
            self.assertIn(
                "branch",
                step_3_result.lower(),
                "Step 3 command_prompt output must contain real git branch information from E-commerce.",
            )
            self.assertTrue(
                "bug-fix" in step_3_result or "main" in step_3_result or "On branch" in step_3_result,
                f"Step 3 output must capture repository state. Got:\n{step_3_result}",
            )


if __name__ == "__main__":
    unittest.main()
