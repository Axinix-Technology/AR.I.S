"""
AR.I.S — Phase 2 Relational Integration Unit Tests

Verifies the unified cognitive path connecting:
1. Phase 1 canonical language boundary (ArisCanonicalTokenizer -> TokenizedSequence)
2. Phase 0 PyTorch neural substrate interface (ArisPhase0Model, embed_dim=64)
3. Existing trained AR.I.S Universal Dependencies Brain (ArisUniversalLinguisticEncoder with aris_ud_tensor_bank.pt)
4. Learned linguistic representations (UPOS, dependency relations, continuous 64-dim tensors)
5. Existing relational cognition (ArisRelationalBinder, ArisRelationAbstractionEngine, ArisSymbolicGraph, ArisRetriever)

Key Invariants:
- ONE canonical path: Raw Text -> Canonical Tokenizer -> TokenizedSequence -> Phase 0 Substrate -> Trained UD Brain -> Relational Cognition.
- Zero duplicate brains: Phase 0 is the neural interface, UD Brain carries learned linguistic knowledge, Relational System consumes it.
- Pure console logs only (no artificial marketing banners or orchestrated narratives).
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
from api.relational.relational import ArisRelationalBinder
from api.relational.relation_abstraction import ArisRelationAbstractionEngine
from api.relational.symbolic_graph import ArisSymbolicGraph
from api.relational.retrieval import ArisRetriever, ArisRelationalQuery


class TestPhase2RelationalIntegration:
    """Verifies Phase 2 integration across the unified canonical cognitive path."""

    @pytest.fixture
    def setup_environment(self):
        # Load the already trained and persisted Phase 2 canonical vocabulary
        vocab_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "models",
            "aris_phase2_vocabulary.json",
        )
        tokenizer = ArisCanonicalTokenizer.from_vocabulary_file(vocab_path)
        vocab = tokenizer.vocab

        # Phase 0 neural substrate interface in R^64
        model = ArisPhase0Model(
            vocab_size=len(vocab),
            embed_dim=64,
            max_seq_len=24,
            num_heads=2,
            num_layers=2,
        )
        model.eval()

        # Existing trained UD Brain (models/aris_ud_tensor_bank.pt)
        ud_encoder = ArisUniversalLinguisticEncoder
        ud_encoder._ensure_loaded()

        return {
            "vocab": vocab,
            "tokenizer": tokenizer,
            "model": model,
            "ud_encoder": ud_encoder,
        }

    def test_1_canonical_tokenized_sequence_reaches_relational_binder(self, setup_environment):
        tokenizer = setup_environment["tokenizer"]
        text = "Arun is brother of Ravi"
        seq = tokenizer.serialize(text, source="statement", add_special_tokens=True)

        # Directly invoke bind_tokenized_sequence
        bindings = ArisRelationalBinder.bind_tokenized_sequence(seq)

        assert len(bindings) == 1
        b = bindings[0]
        assert b["subject"]["text"] == "Arun"
        assert b["relation"]["text"] == "is brother of"
        assert b["object"]["text"] == "Ravi"
        assert b["relation"]["token_indices"] == [2, 3, 4]

        print("\n--- TEST 1: TokenizedSequence -> Relational Binder ---")
        print(f"Input text:       '{text}'")
        print(f"Tokens:           {seq.tokens}")
        print(f"Input IDs:        {seq.input_ids.tolist()[0]}")
        print(f"Subject:          '{b['subject']['text']}' (token_indices: {b['subject']['token_indices']})")
        print(f"Relation:         '{b['relation']['text']}' (token_indices: {b['relation']['token_indices']})")
        print(f"Object:           '{b['object']['text']}' (token_indices: {b['object']['token_indices']})")

    def test_2_trained_ud_brain_encodes_canonical_sequence_into_learned_linguistic_tensors(self, setup_environment):
        tokenizer = setup_environment["tokenizer"]
        model = setup_environment["model"]
        ud_encoder = setup_environment["ud_encoder"]

        text = "Arun is brother of Ravi"
        seq = tokenizer.serialize(text, source="statement", add_special_tokens=True)

        # Route through trained UD Brain backed by Phase 0 PyTorch substrate interface
        ud_res = ud_encoder.encode_tokenized_sequence(seq, substrate_model=model)

        assert "hidden_states" in ud_res
        hs = ud_res["hidden_states"]
        assert hs.shape == (1, 7, 64)
        assert len(ud_res["upos"]) == 7
        assert len(ud_res["dependencies"]) > 0

        # Check UPOS classifications assigned by the trained UD tensor bank
        token_tag_map = {tok: tag for tok, (tag, conf) in zip(seq.tokens, ud_res["upos"])}
        assert token_tag_map["is"] == "AUX"
        assert token_tag_map["brother"] == "NOUN"
        assert token_tag_map["of"] == "ADP"

        print("\n--- TEST 2: Canonical Sequence -> Trained UD Brain (models/aris_ud_tensor_bank.pt) ---")
        print(f"Sequence Tokens:          {seq.tokens}")
        print(f"Trained UD Hidden States: {tuple(hs.shape)}")
        print(f"Learned UPOS Tags:        {ud_res['upos']}")
        print(f"Syntactic Dependencies:   {[(d['head'], '--[' + d['relation'] + ']-->', d['dependent']) for d in ud_res['dependencies']]}")

    def test_3_neural_hidden_states_bind_to_relation_representation(self, setup_environment):
        tokenizer = setup_environment["tokenizer"]
        model = setup_environment["model"]
        ud_encoder = setup_environment["ud_encoder"]

        text = "Arun is brother of Ravi"
        seq = tokenizer.serialize(text, source="statement", add_special_tokens=True)
        ud_res = ud_encoder.encode_tokenized_sequence(seq, substrate_model=model)
        hidden_states = ud_res["hidden_states"]

        bindings = ArisRelationalBinder.bind_tokenized_sequence(seq, hidden_states=hidden_states)

        assert len(bindings) == 1
        b = bindings[0]
        assert "neural_representation" in b
        vec = b["neural_representation"]
        assert isinstance(vec, list)
        assert len(vec) == 64
        assert b["tensor_shape"] == [64]

        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-4

        print("\n--- TEST 3: Trained UD Linguistic Tensors -> Relational Representation ---")
        print(f"Trained hidden_states shape: {tuple(hidden_states.shape)}")
        print(f"Relation span indices:       {b['relation']['token_indices']}")
        print(f"Pooled neural vector dim:    {len(vec)} (matches aris_ud_tensor_bank.pt R^64)")
        print(f"Vector L2-norm:              {round(norm, 6)}")
        print(f"Vector sample:               {[round(x, 4) for x in vec[:6]]}")

    def test_4_relation_abstraction_engine_from_trained_ud_substrate(self, setup_environment):
        tokenizer = setup_environment["tokenizer"]
        model = setup_environment["model"]
        ud_encoder = setup_environment["ud_encoder"]

        engine = ArisRelationAbstractionEngine.from_trained_ud_substrate(
            tokenizer=tokenizer,
            ud_encoder=ud_encoder,
            substrate_model=model,
            canonical_relations=["brother_of", "works_at", "located_in"],
        )

        assert engine.dimension == 64
        assert len(engine.prototypes) == 3
        for rel in ["brother_of", "works_at", "located_in"]:
            proto = engine.prototypes[rel]
            assert len(proto) == 64
            norm = math.sqrt(sum(x * x for x in proto))
            assert abs(norm - 1.0) < 1e-4

        print("\n--- TEST 4: Prototype Centroids Built from Trained UD Substrate ---")
        print(f"Engine embedding dim:    {engine.dimension}")
        print(f"Canonical relations:     {list(engine.prototypes.keys())}")
        for r, p in engine.prototypes.items():
            p_norm = math.sqrt(sum(x * x for x in p))
            print(f"  Prototype '{r}': length={len(p)}, L2-norm={round(p_norm, 4)}, head={[round(x, 4) for x in p[:4]]}")

    def test_5_exemplar_training_updates_neural_prototype_centroids(self, setup_environment):
        tokenizer = setup_environment["tokenizer"]
        model = setup_environment["model"]
        ud_encoder = setup_environment["ud_encoder"]

        engine = ArisRelationAbstractionEngine.from_trained_ud_substrate(
            tokenizer=tokenizer,
            ud_encoder=ud_encoder,
            substrate_model=model,
            canonical_relations=["brother_of"],
        )

        init_proto = list(engine.prototypes["brother_of"])

        exemplars = [
            ("brother of", "brother_of"),
            ("is brother of", "brother_of"),
        ]
        engine.train_exemplars(exemplars, learning_rate=1.0)

        updated_proto = engine.prototypes["brother_of"]
        delta = math.sqrt(sum((a - b) ** 2 for a, b in zip(init_proto, updated_proto)))
        norm = math.sqrt(sum(x * x for x in updated_proto))

        assert delta > 0.0
        assert abs(norm - 1.0) < 1e-4
        assert engine.prototype_counts["brother_of"] == 3

        print("\n--- TEST 5: Exemplar Training Updates Centroids in R^64 ---")
        print(f"Initial centroid head:   {[round(x, 4) for x in init_proto[:5]]}")
        print(f"Updated centroid head:   {[round(x, 4) for x in updated_proto[:5]]}")
        print(f"Centroid shift ||delta||:{round(delta, 6)}")
        print(f"Post-training count:     {engine.prototype_counts['brother_of']}")

    def test_6_abstract_relation_inference_via_neural_representation(self, setup_environment):
        tokenizer = setup_environment["tokenizer"]
        model = setup_environment["model"]
        ud_encoder = setup_environment["ud_encoder"]

        engine = ArisRelationAbstractionEngine.from_trained_ud_substrate(
            tokenizer=tokenizer,
            ud_encoder=ud_encoder,
            substrate_model=model,
            canonical_relations=["brother_of", "works_at", "located_in"],
        )
        engine.train_exemplars([("brother of", "brother_of")], learning_rate=1.0)

        query_phrase = "is brother of"
        res = engine.abstract_relation(query_phrase)

        assert "predicted_relation" in res
        assert "confidence" in res
        assert res["predicted_relation"] == "brother_of"
        assert res["confidence"] > 0.0

        print("\n--- TEST 6: Query Relation Phrase Abstraction via Trained Substrate ---")
        print(f"Query phrase:            '{query_phrase}'")
        print(f"Query result:            {res}")

    def test_7_symbolic_graph_and_retriever_consume_neural_relation_bindings(self, setup_environment):
        tokenizer = setup_environment["tokenizer"]
        model = setup_environment["model"]
        ud_encoder = setup_environment["ud_encoder"]

        # 1. Bind sentence through canonical sequence -> trained UD brain
        text = "Arun is brother of Ravi"
        seq = tokenizer.serialize(text, source="statement")
        ud_res = ud_encoder.encode_tokenized_sequence(seq, substrate_model=model)
        bindings = ArisRelationalBinder.bind_tokenized_sequence(seq, hidden_states=ud_res["hidden_states"])
        assert len(bindings) == 1
        b = bindings[0]
        extracted_rel_span = b["relation"]["text"]

        # 2. Derive canonical relation dynamically from trained UD substrate
        engine = ArisRelationAbstractionEngine.from_trained_ud_substrate(
            tokenizer=tokenizer,
            ud_encoder=ud_encoder,
            substrate_model=model,
            canonical_relations=["brother_of", "works_at", "located_in"],
        )
        engine.train_exemplars([("brother of", "brother_of")], learning_rate=1.0)
        abs_res = engine.abstract_relation(extracted_rel_span)
        dynamically_predicted_relation = abs_res["predicted_relation"]
        assert dynamically_predicted_relation == "brother_of"

        b["relation"]["normalized"] = dynamically_predicted_relation

        # 3. Ingest into Symbolic Graph
        graph = ArisSymbolicGraph()
        graph.add_relational_binding(
            subject=b["subject"]["text"],
            relation=dynamically_predicted_relation,
            obj=b["object"]["text"],
            statement=text,
        )

        # 4. Query via ArisRetriever
        facts = [{"statement": text, "relations": bindings}]
        q = ArisRelationalQuery(
            subject=b["subject"]["text"],
            relation=dynamically_predicted_relation,
            target_variable="object",
        )
        res = ArisRetriever.execute_query(q, facts=facts)

        assert res["matches_count"] == 1
        assert res["answers"][0]["answer"] == "Ravi"

        print("\n--- TEST 7: End-to-End Symbolic Graph & Retriever Query Execution ---")
        print(f"Statement:               '{text}'")
        print(f"Extracted relation span: '{extracted_rel_span}'")
        print(f"Engine abstraction:      {abs_res}")
        print(f"Graph edges for 'arun':  {[(e.relation, e.to_entity) for e in graph.adjacency['arun']]}")
        print(f"Query:                   {q}")
        print(f"Retriever result:        {res}")

    def test_8_backward_compatibility_preserved(self):
        text = "Arun is brother of Ravi."
        from api.linguistic.tokenizer import ArisTokenizer
        toks = ArisTokenizer.tokenize(text)
        cands = ArisTokenizer.extract_candidates(text, toks)
        bindings = ArisRelationalBinder.bind_relations(text, toks, cands)

        assert len(bindings) == 1
        assert bindings[0]["subject"]["text"] == "Arun"
        assert bindings[0]["relation"]["text"] == "is brother of"
        assert bindings[0]["object"]["text"] == "Ravi"

        legacy_engine = ArisRelationAbstractionEngine(canonical_relations=["works_at"])
        assert legacy_engine.dimension == 64
        assert len(legacy_engine.prototypes["works_at"]) == 64

        print("\n--- TEST 8: Backward Compatibility Check ---")
        print(f"Legacy text:             '{text}'")
        print(f"Legacy binding:          Subject='{bindings[0]['subject']['text']}', Rel='{bindings[0]['relation']['text']}', Obj='{bindings[0]['object']['text']}'")
        print(f"Legacy engine dimension: {legacy_engine.dimension}")

    def test_9_no_raw_string_bypass_at_relational_boundary(self):
        with pytest.raises(AttributeError):
            ArisRelationalBinder.bind_tokenized_sequence("raw string bypassing TokenizedSequence")

        print("\n--- TEST 9: Boundary Guard Check ---")
        print("Verified: Passing raw string to bind_tokenized_sequence raises AttributeError")

    def test_10_seen_relation_with_unseen_entities(self, setup_environment):
        tokenizer = setup_environment["tokenizer"]
        model = setup_environment["model"]
        ud_encoder = setup_environment["ud_encoder"]

        # Sentence contains completely unseen entities (Elena, NeuroVanguard)
        text = "Elena works at NeuroVanguard"
        seq = tokenizer.serialize(text, source="statement")
        ud_res = ud_encoder.encode_tokenized_sequence(seq, substrate_model=model)

        bindings = ArisRelationalBinder.bind_tokenized_sequence(seq, hidden_states=ud_res["hidden_states"])
        assert len(bindings) == 1
        b = bindings[0]
        assert b["subject"]["text"] == "Elena"
        assert b["relation"]["text"] == "works at"
        assert b["object"]["text"] == "NeuroVanguard"

        engine = ArisRelationAbstractionEngine.from_trained_ud_substrate(
            tokenizer=tokenizer,
            ud_encoder=ud_encoder,
            substrate_model=model,
            canonical_relations=["works_at", "located_in", "brother_of"],
        )
        engine.train_exemplars([("works at", "works_at")], learning_rate=1.0)
        abs_res = engine.abstract_relation(b["relation"]["text"])
        rel_label = abs_res["predicted_relation"]
        assert rel_label == "works_at"

        b["relation"]["normalized"] = rel_label

        graph = ArisSymbolicGraph()
        graph.add_relational_binding(
            subject=b["subject"]["text"],
            relation=rel_label,
            obj=b["object"]["text"],
            statement=text,
        )
        facts = [{"statement": text, "relations": bindings}]
        q = ArisRelationalQuery(subject="Elena", relation=rel_label, target_variable="object")
        res = ArisRetriever.execute_query(q, facts=facts)

        assert res["matches_count"] == 1
        assert res["answers"][0]["answer"] == "NeuroVanguard"

        print("\n--- TEST 10: Unseen Entities Query Execution via Trained UD Substrate ---")
        print(f"Statement:               '{text}'")
        print(f"Discovered Subject:      '{b['subject']['text']}'")
        print(f"Discovered Relation:     '{b['relation']['text']}' -> Predicted: '{rel_label}'")
        print(f"Discovered Object:       '{b['object']['text']}'")
        print(f"Graph edges for 'elena': {[(e.relation, e.to_entity) for e in graph.adjacency['elena']]}")
        print(f"Query:                   {q}")
        print(f"Retriever result:        {res}")

    def test_11_held_out_paraphrase_generalization(self, setup_environment):
        tokenizer = setup_environment["tokenizer"]
        model = setup_environment["model"]
        ud_encoder = setup_environment["ud_encoder"]

        engine = ArisRelationAbstractionEngine.from_trained_ud_substrate(
            tokenizer=tokenizer,
            ud_encoder=ud_encoder,
            substrate_model=model,
            canonical_relations=["works_at", "located_in", "brother_of"],
        )
        engine.train_exemplars([
            ("works at", "works_at"),
            ("located in", "located_in"),
            ("brother of", "brother_of"),
        ])

        # Query unobserved surface realization: "works for" (held-out prepositional variant)
        query_phrase = "works for"
        res = engine.abstract_relation(query_phrase)

        assert res["predicted_relation"] == "works_at", (
            f"Expected 'works_at' for '{query_phrase}', got '{res['predicted_relation']}' with scores {res['candidate_relations']}"
        )
        assert res["confidence"] > 0.0

        print("\n--- TEST 11: Held-Out Prepositional Query Abstraction (Trained UD Substrate) ---")
        print(f"Trained exemplars:       {[('works at', 'works_at'), ('located in', 'located_in'), ('brother of', 'brother_of')]}")
        print(f"Query phrase:            '{query_phrase}'")
        print(f"Query result:            {res}")

    def test_12_unseen_structural_combination(self, setup_environment):
        tokenizer = setup_environment["tokenizer"]
        model = setup_environment["model"]
        ud_encoder = setup_environment["ud_encoder"]

        text = "Axinix was founded by Arunbharathi"
        seq = tokenizer.serialize(text, source="statement")
        ud_res = ud_encoder.encode_tokenized_sequence(seq, substrate_model=model)

        bindings = ArisRelationalBinder.bind_tokenized_sequence(seq, hidden_states=ud_res["hidden_states"])
        assert len(bindings) == 1
        b = bindings[0]
        assert b["subject"]["text"] == "Axinix"
        assert b["relation"]["text"] == "was founded by"
        assert b["object"]["text"] == "Arunbharathi"

        print("\n--- TEST 12: Passive Voice Frame Structural Extraction (Trained UD Substrate) ---")
        print(f"Statement:               '{text}'")
        print(f"Trained UPOS Tags:       {ud_res['upos']}")
        print(f"Subject:                 '{b['subject']['text']}'")
        print(f"Relation Span:           '{b['relation']['text']}' (token_indices: {b['relation']['token_indices']})")
        print(f"Object:                  '{b['object']['text']}'")
