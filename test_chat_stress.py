# pyright: ignore
"""
AR.I.S — Clean Standalone Multi-Turn Dialogue & Autonomous Tool Selection Inspector

Run directly in console:
    python test_chat_stress.py
"""

import os
import sys
from unittest.mock import patch, MagicMock

# Allow importing from workspace root
_workspace_root = os.path.abspath(os.path.dirname(__file__))
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


def print_card(turn_num: int, user_input: str, result: dict, calls: int, total_calls: int):
    print("\n" + "=" * 78)
    print(f"  TURN {turn_num}: USER -> \"{user_input}\"")
    print("-" * 78)
    print(f"  * Resolved Utterance   : {result.get('resolved_utterance')}")
    if result.get("resolved_coreferences"):
        print(f"  * Coreferences Bound   : {result.get('resolved_coreferences')}")
    print(f"  * Epistemic State      : {result.get('epistemic_state')}")
    print(f"  * Dispatched Tool      : {result.get('dispatched_tool') or 'None (Zero tool calls)'}")
    print(f"  * HTTP Browser Calls   : {calls} in this turn (Cumulative: {total_calls})")
    print(f"  * AR.I.S Response      : {result.get('response')}")
    salient = [s.get("entity") for s in result.get("active_salience_ranking", [])]
    print(f"  * Active Salience      : {salient}")
    print("=" * 78)


def run_clean_inspector():
    session_id = "user_inspect_session_1"

    # Reset session context and clean lingering browser facts
    ArisDialogueContextManager.clear_context(session_id)
    try:
        coll = ArisBrain._get_collection()
        coll.delete_many({"source": "browser_access"})
        coll.delete_many({"statement": {"$regex": "SpaceX|Elon Musk|Tesla|Twitter", "$options": "i"}})
    except Exception:
        pass

    web_search_html = (
        '<a class="result__snippet">Elon Musk founded SpaceX in 2002. Elon Musk is the CEO and chief engineer of SpaceX.</a>'
        '<a class="result__snippet">Elon Musk also leads Tesla and Neuralink. Elon Musk acquired Twitter in October 2022.</a>'
    )

    turns = [
        "Hi",
        "Who is the founder of SpaceX?",
        "What are the other companies he owned?",
        "When did he acquire Twitter?",
    ]

    print("\n" + "#" * 78)
    print("  AR.I.S MULTI-TURN AUTONOMOUS TOOL SELECTION & MEMORY REUSE INSPECTOR")
    print("#" * 78)

    with patch("httpx.Client.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200, text=web_search_html)

        for idx, user_msg in enumerate(turns, 1):
            call_count_before = mock_post.call_count

            res = ArisBrain.process_dialogue_turn(
                session_id=session_id,
                utterance=user_msg,
                speaker_role="USER",
            )

            calls_in_this_turn = mock_post.call_count - call_count_before
            print_card(idx, user_msg, res, calls_in_this_turn, mock_post.call_count)

    ctx = ArisDialogueContextManager.get_or_create_context(session_id=session_id)
    print("\n" + "#" * 78)
    print(f"  DISCOURSE HISTORY — {ctx.get_turn_count()} TURNS RECORDED")
    print("#" * 78)
    for t in ctx.turns:
        print(f"  Turn {t.turn_index:02d} [{t.speaker_role:<5}]: {t.raw_text[:65]}... (State: {t.epistemic_state})")
    print("#" * 78 + "\n")


if __name__ == "__main__":
    run_clean_inspector()
