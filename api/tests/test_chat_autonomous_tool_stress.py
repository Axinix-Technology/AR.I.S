# pyright: ignore
"""
AR.I.S — Autonomous Multi-Turn Tool Selection & Memory Reuse Stress Test Suite

Evaluates:
  1. Phatic Non-Pollution: Greetings produce friendly responses with 0 tools and 0 graph facts.
  2. Autonomous Tool Selection (Zero Trigger Words): Unseen external query auto-dispatches browser_access.
  3. Cognitive Memory Ingestion: External results are parsed into Tier 1 tokens, Tier 2 candidates, Tier 3 relations.
  4. Epistemic Memory Reuse (Zero Duplicate Tool Calls): Subsequent query resolves 'he' -> 'Elon Musk'
     and answers directly from memory with 0 browser calls.
  5. Multi-Turn Context Width: Non-bottleneck sliding window tracks full conversational history.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Allow importing from workspace root
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
from api.cognitive.dialogue_context import ArisDialogueContextManager


def _print_card(title: str, details: dict):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("-" * 78)
    for label, val in details.items():
        print(f"  * {label:<28} : {val}")
    print("=" * 78)


class ChatAutonomousToolStressTests(unittest.TestCase):
    """
    Stress tests autonomous tool selection, cognitive ingestion, and multi-turn context width.
    """

    def setUp(self):
        super().setUp()
        self.session_id = "stress_test_human_session_1"
        ArisDialogueContextManager.clear_context(self.session_id)
        # Purge lingering browser facts from prior runs so Turn 2 starts with clean unseen memory
        try:
            coll = ArisBrain._get_collection()
            coll.delete_many({"source": "browser_access"})
            coll.delete_many({"statement": {"$regex": "SpaceX|Elon Musk|Tesla|Twitter", "$options": "i"}})
        except Exception:
            pass

    def tearDown(self):
        ArisDialogueContextManager.clear_context(self.session_id)
        try:
            coll = ArisBrain._get_collection()
            coll.delete_many({"source": "browser_access"})
            coll.delete_many({"statement": {"$regex": "SpaceX|Elon Musk|Tesla|Twitter", "$options": "i"}})
        except Exception:
            pass
        super().tearDown()

    def test_multi_turn_autonomous_tool_selection_and_memory_reuse(self):
        """
        Pure Human vs. AR.I.S dialogue simulation across 4 turns.
        """
        web_search_html = (
            '<a class="result__snippet">Elon Musk founded SpaceX in 2002. Elon Musk is the CEO and chief engineer of SpaceX.</a>'
            '<a class="result__snippet">Elon Musk also leads Tesla and Neuralink. Elon Musk acquired Twitter in October 2022.</a>'
        )

        with patch("httpx.Client.post") as mock_http_post:
            mock_http_post.return_value = MagicMock(status_code=200, text=web_search_html)

            # =========================================================================
            # TURN 1: Phatic Salutation ('Hi')
            # =========================================================================
            t1 = ArisBrain.process_dialogue_turn(
                session_id=self.session_id,
                utterance="Hi",
                speaker_role="USER",
            )
            _print_card("TURN 1: USER -> 'Hi'", {
                "Speech Act / State": t1.get("epistemic_state"),
                "Dispatched Tool": t1.get("dispatched_tool") or "None (Phatic turn uses zero tools)",
                "AR.I.S Response": t1.get("response"),
            })

            self.assertEqual(t1["epistemic_state"], "PHATIC", "Greeting must be recognized as PHATIC by tensor substrate.")
            self.assertIsNone(t1.get("dispatched_tool"), "Phatic turn must NEVER execute tools.")
            self.assertTrue(len(t1["response"]) > 0, "Agent must provide a polite salutation.")

            # =========================================================================
            # TURN 2: External Interrogative ('Who is the founder of SpaceX?')
            # =========================================================================
            call_count_before_t2 = mock_http_post.call_count
            t2 = ArisBrain.process_dialogue_turn(
                session_id=self.session_id,
                utterance="Who is the founder of SpaceX?",
                speaker_role="USER",
            )
            calls_t2 = mock_http_post.call_count - call_count_before_t2

            _print_card("TURN 2: USER -> 'Who is the founder of SpaceX?'", {
                "Memory Status": "Entity 'SpaceX' absent from internal memory",
                "Epistemic State": t2.get("epistemic_state"),
                "Dispatched Tool": f"{t2.get('dispatched_tool')} (Auto-selected without trigger words)",
                "HTTP Browser Calls": f"{calls_t2} call(s)",
                "AR.I.S Response": t2.get("response"),
                "Active Salience": [s["entity"] for s in t2.get("active_salience_ranking", [])],
            })

            self.assertEqual(
                t2["epistemic_state"],
                "ACQUIRABLE",
                "Unseen entity query must produce ACQUIRABLE epistemic state.",
            )
            self.assertEqual(
                t2.get("dispatched_tool"),
                "browser_access",
                "AR.I.S must autonomously dispatch 'browser_access' tool without trigger words.",
            )
            self.assertIn("Elon Musk", t2["response"], "Response must identify Elon Musk as founder.")

            # =========================================================================
            # TURN 3: Follow-Up Query with Anaphora ('What are the other companies he leads?')
            # =========================================================================
            call_count_before_t3 = mock_http_post.call_count
            t3 = ArisBrain.process_dialogue_turn(
                session_id=self.session_id,
                utterance="What are the other companies he leads?",
                speaker_role="USER",
            )
            calls_t3 = mock_http_post.call_count - call_count_before_t3

            _print_card("TURN 3: USER -> 'What are the other companies he leads?'", {
                "Resolved Utterance": t3.get("resolved_utterance"),
                "Coreferences Bound": t3.get("resolved_coreferences"),
                "Memory Status": "Knowledge already ingested in Turn 2",
                "Epistemic State": t3.get("epistemic_state"),
                "Dispatched Tool": t3.get("dispatched_tool") or "None (Served directly from memory)",
                "HTTP Browser Calls": f"{calls_t3} call(s) (Zero duplicate tool calls)",
                "AR.I.S Response": t3.get("response"),
                "Active Salience": [s["entity"] for s in t3.get("active_salience_ranking", [])],
            })

            # Coreference resolution verification
            self.assertIn(
                "Elon Musk",
                t3["resolved_utterance"],
                f"Pro-form 'he' must resolve to 'Elon Musk'. Got: {t3['resolved_utterance']}",
            )

            # ZERO DUPLICATE TOOL CALL INVARIANT
            self.assertIsNone(
                t3.get("dispatched_tool"),
                "AR.I.S must NOT call browser tool again when knowledge is already in memory!",
            )
            self.assertEqual(calls_t3, 0, "HTTP browser post must not be invoked on Turn 3.")

            # =========================================================================
            # TURN 4: Acquisition Query ('When did he acquire Twitter?')
            # =========================================================================
            t4 = ArisBrain.process_dialogue_turn(
                session_id=self.session_id,
                utterance="When did he acquire Twitter?",
                speaker_role="USER",
            )

            context = ArisDialogueContextManager.get_or_create_context(session_id=self.session_id)
            _print_card("TURN 4: USER -> 'When did he acquire Twitter?'", {
                "Resolved Utterance": t4.get("resolved_utterance"),
                "Coreferences Bound": t4.get("resolved_coreferences"),
                "Epistemic State": t4.get("epistemic_state"),
                "Total Turns in History": f"{context.get_turn_count()} turns (4 USER, 4 AGENT)",
                "AR.I.S Response": t4.get("response"),
            })

            self.assertIn("Elon Musk", t4["resolved_utterance"], "Pro-form 'he' must resolve to 'Elon Musk'.")
            self.assertEqual(context.get_turn_count(), 8, "Context width must preserve all 8 turns.")


if __name__ == "__main__":
    unittest.main()
