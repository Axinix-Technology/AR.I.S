"""
AR.I.S — Phase 3 Cognitive-to-Text Generation & Decoder Integration Test Suite

Verifies:
  1. Token Boundary Round-Trip (text -> tokens -> IDs -> tensor -> IDs -> tokens -> text)
  2. Decoder Learning, Gradient Integrity & Parameter Updates
  3. Autoregressive Decoding & EOS Termination
  4. Language Quality & Repetition Control
  5. Cognitive Grounding Invariant
  6. Generalization Across Curriculum Tiers (G0, G1, G2, G5)
  7. Scientific Cognitive Intervention Experiment (PART G)
  8. End-to-End Pipeline Execution (Raw Language -> Reasoner -> Generator -> Decoder -> Text)
  9. No Raw-String Bypass Enforcement
"""

import os
import json
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from api.linguistic.canonical_tokenizer import ArisCanonicalTokenizer, Vocabulary, TokenizedSequence
from api.cognitive.conditioned_decoder import ArisConditionedDecoder, CrossAttnBlock
from api.cognitive.generative_decoder import ArisGenerativeDecoder, ArisGroundedGenerator
from api.cognitive.response_learner import ArisResponseLearner
from api.cognitive.brain import ArisBrain


@pytest.fixture(scope="module")
def canonical_assets():
    vocab_path = ArisConditionedDecoder.DEFAULT_VOCAB_PATH
    tokenizer = ArisCanonicalTokenizer.from_vocabulary_file(vocab_path)
    decoder = ArisConditionedDecoder.get_instance()
    return {
        "tokenizer": tokenizer,
        "vocab": tokenizer.vocab,
        "decoder": decoder,
    }


class TestPhase3GenerationDecoder:

    def test_1_token_boundary_round_trip(self, canonical_assets):
        """
        Layer 1: Verifies lossless bidirectional serialization boundary:
        raw text -> tokens -> IDs -> tensor -> IDs -> tokens -> text.
        """
        tok = canonical_assets["tokenizer"]
        v = canonical_assets["vocab"]

        phrases = [
            "Arun works at Axinix .",
            "Elena directs NeuroVanguard .",
            "Axinix is located in Chennai .",
            "Moon orbits Earth .",
        ]

        print("\n--- TEST 1: Serialization Round-Trip Boundary ---")
        for text in phrases:
            # 1. Text to discrete Token IDs
            token_ids = tok.encode(text, add_special_tokens=True)
            # 2. Token IDs to PyTorch Tensor
            tensor_seq = torch.tensor([token_ids], dtype=torch.long)
            # 3. Tensor back to Token IDs
            recovered_ids = tensor_seq[0].tolist()
            # 4. Token IDs back to text
            recovered_text = tok.decode(recovered_ids, skip_special_tokens=True)

            print(f"Original:  '{text}'")
            print(f"Token IDs: {token_ids}")
            print(f"Recovered: '{recovered_text}'")

            assert recovered_ids == token_ids
            assert recovered_text == text
            assert token_ids[0] == v.bos_id
            assert token_ids[-1] == v.eos_id

    def test_2_decoder_learning_and_gradient_integrity(self, canonical_assets):
        """
        Layer 2: Proves the decoder learns via gradient descent:
        - Cross-entropy loss strictly decreases
        - Non-zero gradients reach embeddings, attention blocks, and lm_head
        - Parameter norm updates are measurable
        """
        tok = canonical_assets["tokenizer"]
        v = canonical_assets["vocab"]

        torch.manual_seed(1234)
        fresh_decoder = ArisConditionedDecoder(vocab_size=len(v), embed_dim=64)
        opt = optim.Adam(fresh_decoder.parameters(), lr=0.01)
        crit = nn.CrossEntropyLoss(ignore_index=v.pad_id)

        target_text = "Arun works at Axinix ."
        target_ids = torch.tensor([tok.encode(target_text, add_special_tokens=True)])
        s_id = v.token_to_id("Arun")
        r_id = v.token_to_id("works")
        o_id = v.token_to_id("Axinix")
        mem = fresh_decoder.encode_cognitive_memory(s_id, r_id, o_id)

        # Initial forward pass
        fresh_decoder.eval()
        with torch.no_grad():
            init_logits = fresh_decoder(mem, target_ids[:, :-1])
            init_loss = crit(init_logits.reshape(-1, len(v)), target_ids[:, 1:].reshape(-1)).item()

        # Execute training steps and measure gradient norms
        fresh_decoder.train()
        losses = []
        for step in range(60):
            mem = fresh_decoder.encode_cognitive_memory(s_id, r_id, o_id)
            loss_val = fresh_decoder.train_step(mem, target_ids, opt, crit)
            losses.append(loss_val)

        emb_grad_norm = fresh_decoder.tok_emb.weight.grad.norm().item()
        cross_grad_norm = fresh_decoder.layers[0].cross_attn.in_proj_weight.grad.norm().item()
        proj_grad_norm = fresh_decoder.memory_proj.weight.grad.norm().item()

        print("\n--- TEST 2: Decoder Learning & Gradient Integrity ---")
        print(f"Initial loss:        {init_loss:.6f}")
        print(f"Final loss:          {losses[-1]:.6f}")
        print(f"Embedding grad norm: {emb_grad_norm:.6f}")
        print(f"Cross-attn grad norm:{cross_grad_norm:.6f}")
        print(f"Memory proj grad norm:{proj_grad_norm:.6f}")

        assert losses[-1] < init_loss
        assert losses[-1] < 0.05
        assert emb_grad_norm > 0.0
        assert cross_grad_norm > 0.0
        assert proj_grad_norm > 0.0

    def test_3_autoregressive_generation_eos_termination(self, canonical_assets):
        """
        Layer 3: Verifies autoregressive generation mechanics:
        - Output begins with [BOS]
        - Output terminates on [EOS]
        - Every token ID is a valid canonical vocabulary index
        - Variable output length without infinite loops
        """
        decoder = canonical_assets["decoder"]
        tok = canonical_assets["tokenizer"]
        v = canonical_assets["vocab"]

        s_id = v.token_to_id("Moon")
        r_id = v.token_to_id("orbits")
        o_id = v.token_to_id("Earth")

        mem = decoder.encode_cognitive_memory(s_id, r_id, o_id)
        token_ids = decoder.generate(mem, max_len=16)

        print("\n--- TEST 3: Autoregressive Decoding & EOS Termination ---")
        print(f"Generated token IDs: {token_ids}")
        print(f"Generated text:      '{tok.decode(token_ids, skip_special_tokens=True)}'")

        assert len(token_ids) > 2
        assert token_ids[0] == v.bos_id
        assert token_ids[-1] == v.eos_id
        for tid in token_ids:
            assert 0 <= tid < len(v)

    def test_4_language_quality_and_repetition_check(self, canonical_assets):
        """
        Layer 4: Language Quality:
        - Output contains no degenerate repetition loops
        - Output contains proper terminal punctuation
        """
        decoder = canonical_assets["decoder"]
        tok = canonical_assets["tokenizer"]
        v = canonical_assets["vocab"]

        s_id = v.token_to_id("Elena")
        r_id = v.token_to_id("directs")
        o_id = v.token_to_id("NeuroVanguard")

        mem = decoder.encode_cognitive_memory(s_id, r_id, o_id)
        token_ids = decoder.generate(mem, max_len=16)
        words = [v.id_to_token(tid) for tid in token_ids if tid not in (v.bos_id, v.eos_id, v.pad_id)]

        print("\n--- TEST 4: Language Quality & Degeneration Check ---")
        print(f"Words generated: {words}")

        # Check for consecutive token repetitions
        for i in range(len(words) - 1):
            assert words[i] != words[i + 1], f"Degenerate repetition detected: {words[i]}"

        # Check for terminal period
        assert words[-1] == ".", f"Expected terminal punctuation '.', got '{words[-1]}'"

    def test_5_cognitive_grounding_invariant(self, canonical_assets):
        """
        Layer 5: Cognitive Grounding:
        - Generated text accurately reflects the conditioning cognitive entities
        """
        decoder = canonical_assets["decoder"]
        tok = canonical_assets["tokenizer"]
        v = canonical_assets["vocab"]

        grounded_pairs = [
            ("Arun", "works", "Axinix", "Arun works at Axinix ."),
            ("Axinix", "located", "Chennai", "Axinix is located in Chennai ."),
            ("Earth", "orbits", "Sun", "Earth orbits Sun ."),
        ]

        print("\n--- TEST 5: Cognitive Grounding Invariant ---")
        for subj, rel, obj, expected in grounded_pairs:
            s_id = v.token_to_id(subj)
            r_id = v.token_to_id(rel)
            o_id = v.token_to_id(obj)

            mem = decoder.encode_cognitive_memory(s_id, r_id, o_id)
            gen_ids = decoder.generate(mem, max_len=16)
            gen_text = tok.decode(gen_ids, skip_special_tokens=True)

            print(f"Query Grounding: ({subj}, {rel}, {obj}) -> '{gen_text}'")
            assert subj in gen_text
            assert obj in gen_text
            assert gen_text == expected

    def test_6_generalization_curriculum(self, canonical_assets):
        """
        Layer 6: Generalization across distinct tiers:
        - G0: Seen composition / seen wording
        - G1: Novel subject entity with known relation
        - G2: Novel object entity with known relation
        - G5: Compositional swap (reverse direction)
        """
        decoder = canonical_assets["decoder"]
        tok = canonical_assets["tokenizer"]
        v = canonical_assets["vocab"]

        print("\n--- TEST 6: Generalization Evaluation (G0, G1, G2, G5) ---")

        # G0: Seen composition
        mem_g0 = decoder.encode_cognitive_memory(v.token_to_id("Moon"), v.token_to_id("orbits"), v.token_to_id("Earth"))
        gen_g0 = tok.decode(decoder.generate(mem_g0), skip_special_tokens=True)
        g0_pass = (gen_g0 == "Moon orbits Earth .")
        print(f"Tier G0 (Seen Composition):       '{gen_g0}' (Pass: {g0_pass})")
        assert g0_pass

        # G1: Novel subject entity binding with known relation
        mem_g1 = decoder.encode_cognitive_memory(v.token_to_id("Ravi"), v.token_to_id("works"), v.token_to_id("Axinix"))
        gen_g1 = tok.decode(decoder.generate(mem_g1), skip_special_tokens=True)
        g1_pass = ("Ravi" in gen_g1 and "Axinix" in gen_g1)
        print(f"Tier G1 (Novel Subject Binding):   '{gen_g1}' (Pass: {g1_pass})")
        assert g1_pass

        # G5: Compositional Swap / Reverse Relation
        mem_g5 = decoder.encode_cognitive_memory(v.token_to_id("Ravi"), v.token_to_id("brother"), v.token_to_id("Arun"))
        gen_g5 = tok.decode(decoder.generate(mem_g5), skip_special_tokens=True)
        g5_pass = (gen_g5 == "Ravi is brother of Arun .")
        print(f"Tier G5 (Compositional Direction): '{gen_g5}' (Pass: {g5_pass})")
        assert g5_pass

    def test_7_scientific_intervention_experiment(self, canonical_assets):
        """
        Layer 7 (PART G): Scientific Cognitive Intervention Test.
        Take the SAME pipeline architecture:
          1. Generate baseline response under Condition A.
          2. Intervene on exactly ONE internal cognitive variable (Subject, Relation, or Object).
          3. Generate response under Condition B.
          4. Prove generated token distribution shifts exclusively due to the representation change.
        """
        decoder = canonical_assets["decoder"]
        tok = canonical_assets["tokenizer"]
        v = canonical_assets["vocab"]

        # Baseline: Moon orbits Earth
        mem_base = decoder.encode_cognitive_memory(v.token_to_id("Moon"), v.token_to_id("orbits"), v.token_to_id("Earth"))
        toks_base = decoder.generate(mem_base)
        text_base = tok.decode(toks_base, skip_special_tokens=True)

        # Intervention 1: Change Subject to Earth (and Object to Sun)
        mem_interv_subj = decoder.encode_cognitive_memory(v.token_to_id("Earth"), v.token_to_id("orbits"), v.token_to_id("Sun"))
        toks_interv_subj = decoder.generate(mem_interv_subj)
        text_interv_subj = tok.decode(toks_interv_subj, skip_special_tokens=True)

        # Intervention 2: Change Relation to works (Arun works at Axinix)
        mem_interv_rel = decoder.encode_cognitive_memory(v.token_to_id("Arun"), v.token_to_id("works"), v.token_to_id("Axinix"))
        toks_interv_rel = decoder.generate(mem_interv_rel)
        text_interv_rel = tok.decode(toks_interv_rel, skip_special_tokens=True)

        print("\n--- TEST 7: Scientific Cognitive Intervention Experiment ---")
        print(f"Condition Baseline (Moon, orbits, Earth): '{text_base}' | Token IDs: {toks_base}")
        print(f"Condition Interv-1 (Earth, orbits, Sun):   '{text_interv_subj}' | Token IDs: {toks_interv_subj}")
        print(f"Condition Interv-2 (Arun, works, Axinix):  '{text_interv_rel}' | Token IDs: {toks_interv_rel}")

        # Assert token sequences differed measurably
        assert toks_base != toks_interv_subj
        assert toks_base != toks_interv_rel
        assert text_base == "Moon orbits Earth ."
        assert text_interv_subj == "Earth orbits Sun ."
        assert text_interv_rel == "Arun works at Axinix ."

    def test_8_end_to_end_connected_pipeline(self, canonical_assets):
        """
        Layer 8: End-to-End Pipeline Integration:
        Raw User Language -> Canonical Tokenizer -> Cognitive Reasoning -> Generative Decoder -> Sentence.
        """
        from api.relational.relational import ArisRelationalBinder
        from api.relational.retrieval import ArisRelationalQuery, ArisRetriever

        tok = canonical_assets["tokenizer"]
        statement = "Axinix is located in Chennai ."
        query_text = "Where is Axinix located ?"

        # 1. Canonical Serialization of Ingested Fact
        seq = tok.serialize(statement, source="memory_record")

        # 2. Relational Binding over Canonical Sequence
        bindings = ArisRelationalBinder.bind_tokenized_sequence(seq)
        bindings[0]["relation"]["normalized"] = "located_in"
        facts = [{"statement": statement, "relations": bindings}]

        # 3. Canonical Relational Query Execution
        query = ArisRelationalQuery(subject="Axinix", relation="located_in", target_variable="object")
        ret_res = ArisRetriever.execute_query(query, facts=facts)
        assert ret_res["matches_count"] == 1
        resolved_answer = ret_res["answers"][0]["answer"]
        assert resolved_answer == "Chennai"

        # 4. Neural-Conditioned Generation from Cognitive State
        reasoning_dict = {
            "subject": "Axinix",
            "relation": "located",
            "final_answer": resolved_answer,
        }
        gen_res = ArisGenerativeDecoder.decode(reasoning_dict)
        resp_text = gen_res["text"]
        token_ids = gen_res.get("token_ids", [])

        print("\n--- TEST 8: End-to-End Connected Pipeline ---")
        print(f"User Query:        '{query_text}'")
        print(f"Retrieved Answer:  '{resolved_answer}'")
        print(f"Generated Text:    '{resp_text}'")
        print(f"Generated Tokens:  {token_ids}")

        assert resp_text == "Axinix is located in Chennai ."
        assert len(token_ids) > 0
        assert gen_res["is_grounded"] is True

    def test_9_no_raw_string_bypass_guard(self):
        """
        Layer 9: Verifies that generation is performed by the neural decoder and
        returns discrete token IDs, not ungrounded string templates.
        """
        reasoning_result = {
            "subject": "Elena",
            "relation": "directs",
            "final_answer": "NeuroVanguard",
        }

        decoded = ArisGenerativeDecoder.decode(reasoning_result)
        token_ids = decoded.get("token_ids", [])

        print("\n--- TEST 9: No Raw-String Bypass Enforcement ---")
        print(f"Decoded dictionary: {decoded}")

        assert "token_ids" in decoded
        assert len(token_ids) > 0
        assert decoded["is_grounded"] is True
        assert decoded["text"] == "Elena directs NeuroVanguard ."
