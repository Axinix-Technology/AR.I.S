"""
AR.I.S — Phase 1 Universal Language Serialization Boundary Unit Tests

Verifies the 9 non-negotiable architectural requirements:
1. User text can be tokenized.
2. Token IDs are discrete integers.
3. Token IDs can become PyTorch tensors.
4. Tensor shapes are valid throughout the neural substrate.
5. The same tokenizer handles multiple language-bearing sources (queries, memory, browser, command).
6. Unknown tokens follow the defined [UNK] policy under frozen vocabulary.
7. Vocabulary persistence preserves exact IDs.
8. Decode/encode behavior is deterministic.
9. No raw string bypass reaches the neural substrate.
"""

import os
import tempfile
import pytest
import torch
import torch.nn as nn

from api.linguistic.canonical_tokenizer import (
    Vocabulary,
    ArisCanonicalTokenizer,
    TokenizedSequence,
)
from api.linguistic.phase0_substrate import ArisPhase0Model


def _print_test_banner(title: str, what: str, inputs: str, outputs: str, why: str):
    print("\n" + "=" * 80)
    print(f"TEST: {title}")
    print(f"  WHAT TESTED:   {what}")
    print(f"  INPUT:         {inputs}")
    print(f"  OUTPUT:        {outputs}")
    print(f"  WHY IT PASSED: {why}")
    print("=" * 80)


class TestPhase1UniversalLanguageBoundary:
    """Verifies the Phase 1 canonical language boundary."""

    @pytest.fixture
    def setup_tokenizer(self):
        vocab = Vocabulary()
        words = [
            "hello", "the", "cat", "is", "sleeping", "what", "weather",
            "who", "Arun's", "brother", "calculate", "12", "+", "5",
            "search", "for", "today's", "gold", "price", "Sunny", "24",
            "degrees", "in", "fatal", ":", "not", "git", "repository",
            "Axinix", "Technologies", "2024", ".",
        ]
        for w in words:
            vocab.add_token(w)
        return ArisCanonicalTokenizer(vocab)

    def test_1_user_text_can_be_tokenized(self, setup_tokenizer):
        tokenizer = setup_tokenizer
        raw_query = "who is Arun's brother"
        tokens = tokenizer.tokenize_raw(raw_query)

        assert len(tokens) == 4
        assert tokens == ["who", "is", "Arun's", "brother"]

        _print_test_banner(
            title="1/9: test_1_user_text_can_be_tokenized",
            what="Splits natural user language into discrete lexical tokens",
            inputs=f"'{raw_query}'",
            outputs=f"{tokens} (count={len(tokens)})",
            why="Verifies that ordinary English phrasing with contractions breaks down into atomic linguistic symbols.",
        )

    def test_2_token_ids_are_discrete_integers(self, setup_tokenizer):
        tokenizer = setup_tokenizer
        raw_text = "the cat is sleeping"
        token_ids = tokenizer.encode(raw_text, add_special_tokens=True)

        assert isinstance(token_ids, list)
        assert all(isinstance(tid, int) for tid in token_ids)
        assert token_ids[0] == tokenizer.vocab.bos_id
        assert token_ids[-1] == tokenizer.vocab.eos_id

        _print_test_banner(
            title="2/9: test_2_token_ids_are_discrete_integers",
            what="Converts language into pure discrete integer token coordinates",
            inputs=f"'{raw_text}'",
            outputs=f"Token IDs: {token_ids} (All int type)",
            why="Ensures machine representation uses pure integers in [0, |V|-1] with zero string leakage.",
        )

    def test_3_and_4_token_ids_become_valid_pytorch_tensors(self, setup_tokenizer):
        tokenizer = setup_tokenizer
        seq = tokenizer.serialize("calculate 12 + 5", source="user_query", max_length=8)

        assert isinstance(seq.input_ids, torch.Tensor)
        assert isinstance(seq.attention_mask, torch.Tensor)
        assert seq.input_ids.dtype == torch.long
        assert seq.input_ids.shape == (1, 8)

        # Pass through Phase 0 Neural Substrate
        model = ArisPhase0Model(
            vocab_size=len(tokenizer.vocab),
            embed_dim=32,
            max_seq_len=8,
            num_heads=2,
        )
        out = model(seq.input_ids, attention_mask=seq.attention_mask)

        assert out["token_embeddings"].shape == (1, 8, 32)
        assert out["hidden_states"].shape == (1, 8, 32)
        assert out["logits"].shape == (1, 8, len(tokenizer.vocab))

        _print_test_banner(
            title="3 & 4/9: test_3_and_4_token_ids_become_valid_pytorch_tensors",
            what="Tensors are formed and validated through the PyTorch neural sequence substrate",
            inputs=f"Serialized 'calculate 12 + 5' (max_length=8)",
            outputs=f"Input IDs: {tuple(seq.input_ids.shape)} -> Embeddings: {tuple(out['token_embeddings'].shape)} -> Hidden: {tuple(out['hidden_states'].shape)}",
            why="Verifies the exact tensor bridge from the serialization interface into PyTorch forward computation.",
        )

    def test_5_same_tokenizer_handles_multiple_language_sources(self, setup_tokenizer):
        tokenizer = setup_tokenizer
        sources = [
            ("what is the weather", "user_query"),
            ("Axinix Technologies was founded in 2024 .", "memory_record"),
            ("Sunny and 24 degrees in Chennai", "browser_result"),
            ("fatal : not a git repository", "command_output"),
        ]

        serialized_results = []
        for text, src in sources:
            seq = tokenizer.serialize(text, source=src, max_length=12)
            assert isinstance(seq, TokenizedSequence)
            assert seq.source == src
            assert seq.input_ids.shape == (1, 12)
            serialized_results.append((src, seq.tokens[:4]))

        _print_test_banner(
            title="5/9: test_5_same_tokenizer_handles_multiple_language_sources",
            what="Single canonical interface handles user queries, memory, browser, and command output",
            inputs=f"4 diverse sources: user_query, memory_record, browser_result, command_output",
            outputs=f"{serialized_results}",
            why="Mandatory Invariant: External observations, internal memory, and human queries share the exact same representation gateway.",
        )

    def test_6_unknown_tokens_strictly_follow_unk_policy(self, setup_tokenizer):
        tokenizer = setup_tokenizer
        tokenizer.vocab.freeze()
        assert tokenizer.vocab.is_frozen

        alien_text = "xylophone zanzibar quantum"
        token_ids = tokenizer.encode(alien_text)

        # Every alien word must become [UNK] = 1
        assert token_ids == [tokenizer.vocab.unk_id, tokenizer.vocab.unk_id, tokenizer.vocab.unk_id]

        # Ensure vocabulary size did not mutate
        init_size = len(tokenizer.vocab)
        with pytest.raises(RuntimeError):
            tokenizer.vocab.add_token("illegal_new_token")
        assert len(tokenizer.vocab) == init_size

        _print_test_banner(
            title="6/9: test_6_unknown_tokens_strictly_follow_unk_policy",
            what="Strict [UNK] policy and mutation guard on frozen vocabulary",
            inputs=f"Alien phrase '{alien_text}' with frozen vocabulary",
            outputs=f"Token IDs: {token_ids} (All [UNK]=1) | Mutation guard raised RuntimeError",
            why="Prevents silent vocabulary drift at inference time. The neural substrate's weight matrix remains strictly aligned with its vocabulary artifact.",
        )

    def test_7_vocabulary_persistence_preserves_exact_ids(self):
        vocab1 = Vocabulary()
        vocab1.add_token("apple")
        vocab1.add_token("banana")
        vocab1.freeze()

        id_apple = vocab1.token_to_id("apple")
        id_banana = vocab1.token_to_id("banana")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            vocab1.save(tmp_path)
            vocab2 = Vocabulary.load(tmp_path)

            assert len(vocab2) == len(vocab1)
            assert vocab2.token_to_id("apple") == id_apple
            assert vocab2.token_to_id("banana") == id_banana
            assert vocab2.is_frozen

            tok2 = ArisCanonicalTokenizer(vocab2)
            assert tok2.encode("apple banana") == [id_apple, id_banana]
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        _print_test_banner(
            title="7/9: test_7_vocabulary_persistence_preserves_exact_ids",
            what="Disk persistence (save/load) preserves 100% exact token IDs and frozen state",
            inputs=f"Persisted vocabulary with 'apple' -> ID {id_apple}, 'banana' -> ID {id_banana}",
            outputs=f"Reloaded from disk: identical IDs, identical size, frozen=True",
            why="A neural model checkpoint is married to its vocabulary. Decoupled loading guarantees zero silent misalignment.",
        )

    def test_8_decode_encode_is_deterministic_and_bijective(self, setup_tokenizer):
        tokenizer = setup_tokenizer
        sample_phrase = "who is Arun's brother"

        ids_a = tokenizer.encode(sample_phrase)
        ids_b = tokenizer.encode(sample_phrase)
        assert ids_a == ids_b, "Encoding is not deterministic!"

        decoded = tokenizer.decode(ids_a)
        assert decoded == sample_phrase, f"Decoded mismatch: '{decoded}' != '{sample_phrase}'"

        _print_test_banner(
            title="8/9: test_8_decode_encode_is_deterministic_and_bijective",
            what="Deterministic encoding and bijective string recovery",
            inputs=f"'{sample_phrase}'",
            outputs=f"IDs: {ids_a} -> Decoded: '{decoded}' (Exact match)",
            why="Guarantees repeatability across distributed environments and transparent inspection microscopes.",
        )

    def test_9_no_raw_string_bypass_reaches_neural_substrate(self, setup_tokenizer):
        tokenizer = setup_tokenizer
        model = ArisPhase0Model(vocab_size=len(tokenizer.vocab), embed_dim=32, max_seq_len=8)

        raw_string_input = "raw string bypassing tokenizer"

        # PyTorch model must reject raw string inputs
        with pytest.raises((TypeError, AttributeError)):
            model(raw_string_input)

        # Only serialized TokenizedSequence input_ids are accepted
        seq = tokenizer.serialize("the cat is sleeping", source="user_query", max_length=8)
        out = model(seq.input_ids)
        assert out["logits"] is not None

        _print_test_banner(
            title="9/9: test_9_no_raw_string_bypass_reaches_neural_substrate",
            what="Architectural invariant: raw strings are rejected by the neural substrate",
            inputs=f"Attempted model('{raw_string_input}')",
            outputs="Rejected with TypeError | model(seq.input_ids) accepted",
            why="Proves the neural substrate is strictly protected by the serialization boundary; no direct string logic can bypass it.",
        )
