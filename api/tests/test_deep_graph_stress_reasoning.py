import os
import sys
import unittest

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

from api.linguistic.pipeline import ArisLinguisticPipeline
from api.linguistic.tokenizer import ArisTokenizer
from api.cognitive.cognitive_learner import ArisCognitiveReasoner
from api.cognitive.epistemic_state import ArisGoalFormer
from api.relational.symbolic_graph import ArisSymbolicGraph
from api.relational.gnn_reasoner import ArisGNNReasoner


class DeepGraphStressReasoningTests(unittest.TestCase):
    """
    AR.I.S Deep Relational Graph Stress Test Suite.

    Evaluates:
      - 6 to 8 hops deep graph exploration.
      - 5 distinct branching paths.
      - 2 dead-end distractor branches.
      - Question-conditioned path discrimination (avoiding overshoot, undershoot, and branch switching).
      - Evidence-backed epistemic rejection on unsupported relational goals.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Define knowledge statements purely as raw natural language sentences without semantic separation.
        cls.sentences = [
            # Primary 8-Hop Path: Elena -> CryoCore -> AeroSpace -> Toulouse -> South France -> TransEuropean -> Brussels Commission -> Belgian Tribunal -> Henri Laurent
            "Dr. Elena Rostova designed CryoCore Propulsion.",
            "CryoCore Propulsion integrated into AeroSpace Dynamics.",
            "AeroSpace Dynamics operates out of Toulouse Advanced Hangar.",
            "Toulouse Advanced Hangar receives power from South France Grid.",
            "South France Grid governed by TransEuropean Power Authority.",
            "TransEuropean Power Authority audited by Brussels Regulatory Commission.",
            "Brussels Regulatory Commission appeals to Belgian Federal Tribunal.",
            "Belgian Federal Tribunal presided over by Chief Justice Henri Laurent.",

            # Branch 2: AeroSpace Dynamics -> Valence Microelectronics -> Lyon Data Center -> Geneva Quantum Lab -> Dr. Martin Keller (6 hops from Elena)
            "AeroSpace Dynamics procures components from Valence Microelectronics.",
            "Valence Microelectronics hosts servers at Lyon Data Center.",
            "Lyon Data Center interconnected with Geneva Quantum Lab.",
            "Geneva Quantum Lab managed by Dr. Martin Keller.",

            # Branch 3: TransEuropean Power Authority -> Brussels Central District -> Belgium (7 hops from Elena, Target Type: GPE_COUNTRY)
            "TransEuropean Power Authority headquartered in Brussels Central District.",
            "Brussels Central District situated in Belgium.",

            # Wrong Path 1 / Dead-End: CryoCore Propulsion -> Bordeaux Wind Tunnel
            "CryoCore Propulsion aerodynamically tested at Bordeaux Wind Tunnel.",

            # Wrong Path 2 / Dead-End: Brussels Regulatory Commission -> EU Environmental Registry
            "Brussels Regulatory Commission publishes directives to EU Environmental Registry.",
        ]

        from api.relational.relational import ArisRelationalBinder
        cls.facts = []
        for idx, stmt in enumerate(cls.sentences):
            toks = ArisTokenizer.tokenize(stmt)
            cands = ArisTokenizer.extract_candidates(stmt, toks)
            rels = ArisRelationalBinder.bind_relations(stmt, toks, cands)
            cls.facts.append({
                "id": f"stress_fact_{idx+1}",
                "statement": stmt,
                "tokens": toks,
                "candidates": cands,
                "relations": rels,
                "confidence": 1.0,
            })

    def test_01_8_hop_backbone_connectivity_and_path_discovery(self):
        """Verify symbolic graph can traverse 8 full hops from Elena Rostova to Chief Justice Henri Laurent."""
        graph = ArisSymbolicGraph.build_from_facts(self.facts)
        paths = graph.find_paths("Dr. Elena Rostova", "Chief Justice Henri Laurent", max_depth=8)
        self.assertTrue(len(paths) >= 1, "Graph must discover the 8-hop path between Dr. Elena Rostova and Chief Justice Henri Laurent.")
        self.assertEqual(len(paths[0].edges), 8, f"Expected exactly 8 hops, got {len(paths[0].edges)}")

    def test_02_8_hop_deep_relational_reasoning(self):
        """Stress test: 8-hop goal-conditioned query traversing through 7 intermediate organizations to Chief Justice."""
        query = "Who presides over the Belgian Federal Tribunal that hears appeals from the Brussels Regulatory Commission auditing the authority powering the Toulouse Advanced Hangar of AeroSpace Dynamics using CryoCore Propulsion designed by Dr. Elena Rostova?"
        res = ArisCognitiveReasoner.reason(query, facts=self.facts, allow_external=False)

        self.assertEqual(res.get("status"), "plausible_hypothesis")
        self.assertGreaterEqual(res.get("matches_count", 0), 1)
        top_ans = res["answers"][0]
        self.assertEqual(top_ans.get("answer"), "Chief Justice Henri Laurent")
        self.assertIn("Chief Justice Henri Laurent", top_ans.get("path_summary", ""))

    def test_03_6_hop_alternative_branch_discrimination(self):
        """Stress test: Reasoner must branch off at hop 2 into Branch 2 and identify Dr. Martin Keller (6 hops)."""
        query = "Who is the manager of Geneva Quantum Lab interconnected with the servers of the component supplier for AeroSpace Dynamics?"
        res = ArisCognitiveReasoner.reason(query, facts=self.facts, allow_external=False)

        self.assertEqual(res.get("status"), "plausible_hypothesis")
        self.assertGreaterEqual(res.get("matches_count", 0), 1)
        top_ans = res["answers"][0]
        self.assertEqual(top_ans.get("answer"), "Dr. Martin Keller")

    def test_04_intermediate_node_stopping_without_overshoot(self):
        """Stress test: Query asks for 3-hop intermediate facility. Must NOT overshoot to 8-hop leaf."""
        query = "Which facility does AeroSpace Dynamics operate out of?"
        res = ArisCognitiveReasoner.reason(query, facts=self.facts, allow_external=False)

        self.assertEqual(res.get("status"), "plausible_hypothesis")
        self.assertGreaterEqual(res.get("matches_count", 0), 1)
        top_ans = res["answers"][0]
        self.assertEqual(top_ans.get("answer"), "Toulouse Advanced Hangar")

    def test_05_target_type_discrimination_country_vs_person(self):
        """Stress test: Query asks for Country. Must select Belgium (hop 7) rather than Henri Laurent (hop 8, Person)."""
        query = "Which country is the headquarters of TransEuropean Power Authority situated in?"
        res = ArisCognitiveReasoner.reason(query, facts=self.facts, allow_external=False)

        self.assertEqual(res.get("status"), "plausible_hypothesis")
        self.assertGreaterEqual(res.get("matches_count", 0), 1)
        top_ans = res["answers"][0]
        self.assertEqual(top_ans.get("answer"), "Belgium")

    def test_06_dead_end_distractor_branch_rejection(self):
        """Stress test: Dead-end wind tunnel distractor at hop 2 must not pollute propulsion consumer query."""
        query = "Which company integrated CryoCore Propulsion designed by Dr. Elena Rostova?"
        res = ArisCognitiveReasoner.reason(query, facts=self.facts, allow_external=False)

        self.assertEqual(res.get("status"), "plausible_hypothesis")
        self.assertGreaterEqual(res.get("matches_count", 0), 1)
        top_ans = res["answers"][0]
        self.assertEqual(top_ans.get("answer"), "AeroSpace Dynamics")
        self.assertNotEqual(top_ans.get("answer"), "Bordeaux Wind Tunnel")

    def test_07_unsupported_query_epistemic_rejection(self):
        """Stress test: Epistemic rejection. Reaching deep nodes must not trigger false positives for absent relations."""
        query = "Who is the maritime fleet admiral of CryoCore Propulsion?"
        res = ArisCognitiveReasoner.reason(query, facts=self.facts, allow_external=False)

        # Absent relation must yield 0 answers and NOT hallucinate a connected node
        self.assertEqual(res.get("matches_count", 0), 0)
        self.assertEqual(len(res.get("answers", [])), 0)


if __name__ == "__main__":
    unittest.main()
