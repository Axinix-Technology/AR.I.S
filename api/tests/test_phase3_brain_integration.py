"""
AR.I.S — Phase 3 Unified Brain Integration Unit Tests

Verifies that the 5 existing trained AR.I.S cognitive tensor banks:
1. aris_ud_tensor_bank.pt (Universal Dependencies & continuous R^64 linguistic representations)
2. aris_pragmatic_tensor_bank.pt (Speech acts, deictic roles, interrogative intentions in R^64)
3. aris_goal_reasoning_tensor_bank_v1.pt (Predicate clusters, entity target types, whole-path compatibility net in R^192)
4. aris_affordance_tensor_bank.pt (Epistemic state classification & capability affordance heads in R^64)
5. aris_frame_classifier.pt (Structural frame parameter variable classification)

operate together as ONE unified cognitive system through ONE canonical path:
Raw Language -> ArisCanonicalTokenizer -> TokenizedSequence -> Phase 0 Interface -> Trained Substrates -> Relational Cognition.

Pure console logs only (no artificial marketing banners or orchestrated narratives).
"""

import os
import math
import pytest
import torch

from api.linguistic.canonical_tokenizer import (
    Vocabulary,
    ArisCanonicalTokenizer,
    TokenizedSequence,
)
from api.linguistic.phase0_substrate import ArisPhase0Model
from api.linguistic.ud_tensor_engine import ArisUniversalLinguisticEncoder
from api.cognitive.pragmatic_tensor_engine import ArisPragmaticTensorEngine
from api.cognitive.goal_reasoning_tensor_engine import ArisGoalReasoningTensorEngine
from api.cognitive.affordance_tensor_engine import ArisAffordanceTensorEngine
from api.linguistic.neural_frame_classifier import ArisNeuralFrameLearner
from api.relational.relational import ArisRelationalBinder
from api.relational.relation_abstraction import ArisRelationAbstractionEngine
from api.relational.symbolic_graph import ArisSymbolicGraph
from api.relational.retrieval import ArisRetriever, ArisRelationalQuery


class TestPhase3BrainIntegration:
    """Verifies end-to-end integration of all 5 trained cognitive tensor banks through the canonical boundary."""

    @pytest.fixture
    def setup_brain(self):
        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "models",
        )
        vocab_path = os.path.join(models_dir, "aris_phase2_vocabulary.json")
        tokenizer = ArisCanonicalTokenizer.from_vocabulary_file(vocab_path)
        vocab = tokenizer.vocab

        # Phase 0 neural substrate in R^64
        substrate = ArisPhase0Model(
            vocab_size=len(vocab),
            embed_dim=64,
            max_seq_len=24,
            num_heads=2,
            num_layers=2,
        )
        substrate.eval()

        # Ensure all existing trained substrates are loaded via existing loaders
        ArisUniversalLinguisticEncoder._ensure_loaded()
        ArisPragmaticTensorEngine._ensure_loaded()
        ArisGoalReasoningTensorEngine._ensure_loaded()
        ArisAffordanceTensorEngine.initialize()

        return {
            "tokenizer": tokenizer,
            "substrate": substrate,
            "ud": ArisUniversalLinguisticEncoder,
            "pragmatic": ArisPragmaticTensorEngine,
            "goal": ArisGoalReasoningTensorEngine,
            "affordance": ArisAffordanceTensorEngine,
            "frame": ArisNeuralFrameLearner,
        }

    def test_1_canonical_serialization_entrypoint(self, setup_brain):
        tokenizer = setup_brain["tokenizer"]
        text = "Where is Axinix located?"
        seq = tokenizer.serialize(text, source="user_query", add_special_tokens=True)

        assert isinstance(seq, TokenizedSequence)
        assert seq.sequence_length > 0
        assert seq.input_ids.shape[-1] == seq.sequence_length
        assert seq.attention_mask.shape[-1] == seq.sequence_length
        assert seq.source == "user_query"

        print("\n--- TEST 1: Canonical Serialization Entrypoint ---")
        print(f"Input text:        '{text}'")
        print(f"Sequence Tokens:   {seq.tokens}")
        print(f"Input IDs:         {seq.input_ids.tolist()[0]}")
        print(f"Attention Mask:    {seq.attention_mask.tolist()[0]}")

    def test_2_ud_linguistic_substrate(self, setup_brain):
        tokenizer = setup_brain["tokenizer"]
        substrate = setup_brain["substrate"]
        ud = setup_brain["ud"]

        text = "Arun works at Axinix"
        seq = tokenizer.serialize(text, source="statement", add_special_tokens=True)
        ud_res = ud.encode_tokenized_sequence(seq, substrate_model=substrate)

        hs = ud_res["hidden_states"]
        assert hs.shape == (1, seq.sequence_length, 64)
        assert len(ud_res["upos"]) == seq.sequence_length
        assert len(ud_res["dependencies"]) > 0

        print("\n--- TEST 2: UD Linguistic Substrate (aris_ud_tensor_bank.pt) ---")
        print(f"Text:              '{text}'")
        print(f"Hidden States:     {tuple(hs.shape)}")
        print(f"Learned UPOS:      {ud_res['upos']}")
        print(f"Dependencies:      {[(d['head'], d['relation'], d['dependent']) for d in ud_res['dependencies']]}")

    def test_3_pragmatic_speech_act_classification(self, setup_brain):
        tokenizer = setup_brain["tokenizer"]
        ud = setup_brain["ud"]
        pragmatic = setup_brain["pragmatic"]

        # 1. Statement -> DECLARATIVE
        stmt_seq = tokenizer.serialize("Arun is brother of Ravi", source="statement")
        stmt_ud = ud.encode_tokenized_sequence(stmt_seq)
        stmt_act = pragmatic.classify_speech_act_sequence(stmt_seq, hidden_states=stmt_ud["hidden_states"])

        assert stmt_act["dominant_act"] == "DECLARATIVE"
        assert stmt_act["confidence"] > 0.0

        # 2. Question -> INTERROGATIVE
        q_seq = tokenizer.serialize("Where is Axinix located", source="question")
        q_ud = ud.encode_tokenized_sequence(q_seq)
        q_act = pragmatic.classify_speech_act_sequence(q_seq, hidden_states=q_ud["hidden_states"])

        assert q_act["dominant_act"] == "INTERROGATIVE"

        print("\n--- TEST 3: Pragmatic Speech Act Substrate (aris_pragmatic_tensor_bank.pt) ---")
        print(f"Statement Result:  {stmt_act['dominant_act']} (confidence: {stmt_act['confidence']})")
        print(f"Question Result:   {q_act['dominant_act']} (confidence: {q_act['confidence']})")

    def test_4_question_intent_projection(self, setup_brain):
        tokenizer = setup_brain["tokenizer"]
        ud = setup_brain["ud"]
        pragmatic = setup_brain["pragmatic"]

        q_seq = tokenizer.serialize("Where", source="operator", add_special_tokens=False)
        q_ud = ud.encode_tokenized_sequence(q_seq)
        role, conf = pragmatic.infer_interrogative_role_sequence(q_seq, hidden_states=q_ud["hidden_states"])

        assert role == "location"
        assert conf > 0.0

        print("\n--- TEST 4: Question Intent Projection (aris_pragmatic_tensor_bank.pt) ---")
        print(f"Operator Token:    'Where'")
        print(f"Projected Role:    '{role}' (confidence: {round(conf, 4)})")

    def test_5_relational_binding_and_neural_pooling(self, setup_brain):
        tokenizer = setup_brain["tokenizer"]
        ud = setup_brain["ud"]

        text = "Arun works at Axinix"
        seq = tokenizer.serialize(text, source="statement", add_special_tokens=True)
        ud_res = ud.encode_tokenized_sequence(seq)
        bindings = ArisRelationalBinder.bind_tokenized_sequence(seq, hidden_states=ud_res["hidden_states"])

        assert len(bindings) == 1
        b = bindings[0]
        assert b["subject"]["text"] == "Arun"
        assert b["relation"]["text"] == "works at"
        assert b["object"]["text"] == "Axinix"
        assert len(b["neural_representation"]) == 64

        norm = math.sqrt(sum(x * x for x in b["neural_representation"]))
        assert abs(norm - 1.0) < 1e-4

        print("\n--- TEST 5: Relational Binding & Neural Pooling ---")
        print(f"Subject:           '{b['subject']['text']}'")
        print(f"Relation Span:     '{b['relation']['text']}' (indices: {b['relation']['token_indices']})")
        print(f"Object:            '{b['object']['text']}'")
        print(f"Neural Vector Dim: {len(b['neural_representation'])} (L2-norm: {round(norm, 4)})")

    def test_6_structural_frame_classification(self, setup_brain):
        frame_classifier = setup_brain["frame"]

        # 1. Forward query frame: "? works at Axinix" -> Variable is subject (?)
        b_fwd = {
            "subject": {"text": "?", "token_indices": [0]},
            "relation": {"text": "works at", "token_indices": [1, 2]},
            "object": {"text": "Axinix", "token_indices": [3]},
        }
        res_fwd = frame_classifier.predict_from_relational_binding(b_fwd)
        assert res_fwd["status"] == "ok"
        assert res_fwd["variable_slot"] == "subject"
        assert res_fwd["anchor_slot"] == "object"
        assert res_fwd["confidence"] == 1.0

        # 2. Reverse query frame: "Arun works at ?" -> Variable is object (?)
        b_rev = {
            "subject": {"text": "Arun", "token_indices": [0]},
            "relation": {"text": "works at", "token_indices": [1, 2]},
            "object": {"text": "?", "token_indices": [3]},
        }
        res_rev = frame_classifier.predict_from_relational_binding(b_rev)
        assert res_rev["status"] == "ok"
        assert res_rev["variable_slot"] == "object"
        assert res_rev["anchor_slot"] == "subject"
        assert res_rev["confidence"] == 1.0

        print("\n--- TEST 6: Structural Frame Classifier (aris_frame_classifier.pt) ---")
        print(f"Forward Query '? works at Axinix':  variable={res_fwd['variable_slot']}, anchor={res_fwd['anchor_slot']} (conf={res_fwd['confidence']})")
        print(f"Reverse Query 'Arun works at ?':   variable={res_rev['variable_slot']}, anchor={res_rev['anchor_slot']} (conf={res_rev['confidence']})")

    def test_7_predicate_alignment_via_goal_substrate(self, setup_brain):
        tokenizer = setup_brain["tokenizer"]
        goal = setup_brain["goal"]

        seq_a = tokenizer.serialize("founder", source="predicate_a", add_special_tokens=False)
        seq_b = tokenizer.serialize("founded", source="predicate_b", add_special_tokens=False)
        seq_c = tokenizer.serialize("brother of", source="predicate_c", add_special_tokens=False)

        sim_ab = goal.align_predicates_sequence(seq_a, seq_b)
        sim_ac = goal.align_predicates_sequence(seq_a, seq_c)

        # 'founder' aligns significantly higher with 'founded' than with 'brother of'
        assert sim_ab > sim_ac
        assert sim_ab >= 0.50

        print("\n--- TEST 7: Predicate Alignment via Goal Substrate (aris_goal_reasoning_tensor_bank_v1.pt) ---")
        print(f"align('founder', 'founded'):     {sim_ab}")
        print(f"align('founder', 'brother of'):  {sim_ac}")

    def test_8_target_type_prediction_via_goal_substrate(self, setup_brain):
        tokenizer = setup_brain["tokenizer"]
        goal = setup_brain["goal"]

        seq = tokenizer.serialize("Which company does Arun work for", source="question")
        pred = goal.predict_target_type_sequence(seq)

        assert "best_type" in pred
        assert pred["confidence"] > 0.0
        # "company" query targets ORGANIZATION
        assert pred["best_type"] == "ORGANIZATION"

        print("\n--- TEST 8: Target Type Prediction via Goal Substrate ---")
        print(f"Question:          'Which company does Arun work for'")
        print(f"Predicted Target:  '{pred['best_type']}' (confidence: {pred['confidence']})")

    def test_9_capability_affordance_perception(self, setup_brain):
        tokenizer = setup_brain["tokenizer"]
        affordance = setup_brain["affordance"]

        # Terminal command query -> command_prompt affordance
        cmd_seq = tokenizer.serialize("git status in current directory", source="command_query")
        cmd_res = affordance.predict_epistemic_and_affordance_sequence(cmd_seq)

        assert "affordance_vector" in cmd_res
        assert len(cmd_res["affordance_vector"]) == 5
        # Index 1 is command_prompt
        cmd_score = cmd_res["affordance_vector"][1]
        assert cmd_score > 0.0

        print("\n--- TEST 9: Capability Affordance Perception (aris_affordance_tensor_bank.pt) ---")
        print(f"Query:             'git status in current directory'")
        print(f"Affordance Vector: {[round(x, 4) for x in cmd_res['affordance_vector']]}")
        print(f"Command Prompt Affordance Score: {round(cmd_score, 4)}")

    def test_10_epistemic_state_determination(self, setup_brain):
        tokenizer = setup_brain["tokenizer"]
        affordance = setup_brain["affordance"]

        seq = tokenizer.serialize("Calculate 12 + 5", source="math_query")
        res = affordance.predict_epistemic_and_affordance_sequence(seq)

        assert "epistemic_state" in res
        assert res["epistemic_state"] in affordance.EPISTEMIC_NAMES
        assert res["epistemic_confidence"] > 0.0

        print("\n--- TEST 10: Epistemic State Determination ---")
        print(f"Query:             'Calculate 12 + 5'")
        print(f"Epistemic State:   '{res['epistemic_state']}' (confidence: {round(res['epistemic_confidence'], 4)})")

    def test_11_end_to_end_connected_brain_flow(self, setup_brain):
        tokenizer = setup_brain["tokenizer"]
        substrate = setup_brain["substrate"]
        ud = setup_brain["ud"]
        pragmatic = setup_brain["pragmatic"]
        goal = setup_brain["goal"]

        # Step 1: Ingest knowledge through canonical sequence
        statement = "Elena works at NeuroVanguard"
        stmt_seq = tokenizer.serialize(statement, source="statement", add_special_tokens=True)

        # Step 2: UD brain encoding
        ud_res = ud.encode_tokenized_sequence(stmt_seq)
        hs = ud_res["hidden_states"]

        # Step 3: Pragmatic verification of fact assertion
        act = pragmatic.classify_speech_act_sequence(stmt_seq, hidden_states=hs)
        assert act["dominant_act"] == "DECLARATIVE"

        # Step 4: Relational binding with continuous pooling
        bindings = ArisRelationalBinder.bind_tokenized_sequence(stmt_seq, hidden_states=hs)
        assert len(bindings) == 1
        b = bindings[0]
        subj, rel_span, obj = b["subject"]["text"], b["relation"]["text"], b["object"]["text"]

        # Step 5: Relation abstraction via trained prototype centroids
        engine = ArisRelationAbstractionEngine.from_trained_ud_substrate(
            tokenizer=tokenizer,
            ud_encoder=ud,
            canonical_relations=["works_at", "located_in", "brother_of"],
        )
        engine.train_exemplars([("works at", "works_at")])
        abs_res = engine.abstract_relation(rel_span)
        canonical_rel = abs_res["predicted_relation"]
        assert canonical_rel == "works_at"

        # Step 6: Ground into Symbolic Graph
        b["relation"]["normalized"] = canonical_rel
        graph = ArisSymbolicGraph()
        graph.add_relational_binding(
            subject=subj,
            relation=canonical_rel,
            obj=obj,
            statement=statement,
        )

        # Step 7: Retrieve answer via ArisRetriever
        facts = [{"statement": statement, "relations": bindings}]
        query = ArisRelationalQuery(subject=subj, relation=canonical_rel, target_variable="object")
        ret_res = ArisRetriever.execute_query(query, facts=facts)

        assert ret_res["matches_count"] == 1
        answer = ret_res["answers"][0]["answer"]
        assert answer == "NeuroVanguard"

        # Step 8: Verify target type consistency of the answer
        cand_seq = tokenizer.serialize(answer, source="candidate_entity", add_special_tokens=False)
        target_type_pred = goal.predict_target_type_sequence(cand_seq)
        assert "best_type" in target_type_pred

        print("\n--- TEST 11: End-to-End Connected Brain Flow ---")
        print(f"Ingested Fact:     '{statement}'")
        print(f"Speech Act:        '{act['dominant_act']}'")
        print(f"Extracted Triple:  ({subj}, '{rel_span}', {obj})")
        print(f"Abstracted Rel:    '{canonical_rel}' (confidence: {abs_res['confidence']})")
        print(f"Query:             Elena --[{canonical_rel}]--> ?")
        print(f"Retrieved Answer:  '{answer}'")
        print(f"Candidate Type:    '{target_type_pred['best_type']}'")

    def test_12_boundary_guard_no_string_bypass(self, setup_brain):
        raw_string = "Elena works at NeuroVanguard"

        # Relational binder strictly requires TokenizedSequence
        with pytest.raises(AttributeError):
            ArisRelationalBinder.bind_tokenized_sequence(raw_string)

        # UD encoder strictly requires TokenizedSequence
        with pytest.raises(AttributeError):
            ArisUniversalLinguisticEncoder.encode_tokenized_sequence(raw_string)

        print("\n--- TEST 12: Boundary Guard Verification ---")
        print("Verified: Passing raw string bypass directly to sequence interfaces raises AttributeError as mandated.")
