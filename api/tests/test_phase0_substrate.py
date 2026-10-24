"""
AR.I.S — Phase 0 Learning Integrity Test

Purpose
-------
Test the tokenizer + PyTorch substrate as a learning system rather than
testing a manually orchestrated cognitive solution.

This test deliberately avoids:
- browser/tool selection
- semantic dictionaries
- keyword rules
- relation-specific Python logic
- hardcoded answer mappings
- manually selecting a [MASK] position
- training on the exact evaluation examples

The test asks only:

1. Can language be serialized into token IDs?
2. Can the neural substrate learn from token sequences?
3. Does the learned model generalize to unseen combinations?
4. Does the model depend on arbitrary token IDs?
5. Is the learning actually coming through trainable parameters?
"""

import random

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from api.linguistic.canonical_tokenizer import Vocabulary, ArisCanonicalTokenizer
from api.linguistic.phase0_substrate import ArisPhase0Model


SEED = 1234


def set_seed():
    random.seed(SEED)
    torch.manual_seed(SEED)


def build_vocabulary():
    vocab = Vocabulary()

    # Synthetic symbols deliberately have no English semantics.
    entities = [f"e_{i}" for i in range(12)]
    relations = [f"r_{i}" for i in range(4)]

    for token in entities + relations:
        vocab.add_token(token)

    return vocab


def make_samples():
    """
    Generate a deterministic synthetic relational world.

    Each example represents:

        entity_A relation entity_B

    The model is trained to predict the final entity token.

    No Python rule converts relation names into answers.
    The target comes directly from the generated data.
    """

    samples = []

    # Structured deterministic relationships.
    for relation_id in range(4):
        for entity_id in range(12):
            source = f"e_{entity_id}"
            relation = f"r_{relation_id}"

            # Deterministic cyclic world.
            target_id = (entity_id + relation_id + 1) % 12
            target = f"e_{target_id}"

            samples.append((source, relation, target))

    return samples


def encode_samples(tokenizer, samples):
    """
    Serialize examples into:

        [BOS] source relation target [EOS]

    The final target is the supervised prediction.

    The model receives only the prefix:

        [BOS] source relation

    and must predict target.
    """

    inputs = []
    targets = []

    for source, relation, target in samples:
        full = tokenizer.encode(
            f"[BOS] {source} {relation} {target} [EOS]"
        )

        # Remove the final target from the model input.
        #
        # This is autoregressive next-token prediction:
        #
        # [BOS] source relation -> target
        prefix = full[:3]
        target_id = full[3]

        inputs.append(prefix)
        targets.append(target_id)

    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
    )


def train_model(model, train_x, train_y, steps=300):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    model.train()

    initial_loss = None

    for step in range(steps):
        optimizer.zero_grad()

        output = model(train_x)

        # Use the final contextual position.
        logits = output["logits"][:, -1, :]

        loss = criterion(logits, train_y)

        if step == 0:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

    return initial_loss, loss.item()


@pytest.mark.parametrize("seed", [SEED])
def test_phase0_learning_and_generalization(seed):
    """
    Main Phase-0 learning experiment.

    This is NOT a cognitive/tool orchestration test.

    It tests whether the neural substrate can learn a relationship
    from tokenized data and reproduce that relationship on held-out
    combinations.
    """

    set_seed()

    vocab = build_vocabulary()
    tokenizer = ArisCanonicalTokenizer(vocab)

    samples = make_samples()

    # Deterministic structural holdout.
    #
    # Training receives most combinations.
    # Evaluation receives combinations that were not used during training.
    train_samples = []
    test_samples = []

    for index, sample in enumerate(samples):
        if index % 5 == 0:
            test_samples.append(sample)
        else:
            train_samples.append(sample)

    train_x, train_y = encode_samples(tokenizer, train_samples)
    test_x, test_y = encode_samples(tokenizer, test_samples)

    model = ArisPhase0Model(
        vocab_size=len(vocab),
        embed_dim=32,
        max_seq_len=8,
        num_heads=2,
        ff_dim=64,
        num_layers=1,
    )

    initial_loss, final_loss = train_model(
        model,
        train_x,
        train_y,
        steps=300,
    )

    model.eval()

    with torch.no_grad():
        train_logits = model(train_x)["logits"][:, -1, :]
        test_logits = model(test_x)["logits"][:, -1, :]

        train_predictions = train_logits.argmax(dim=-1)
        test_predictions = test_logits.argmax(dim=-1)

        train_accuracy = (
            (train_predictions == train_y).float().mean().item()
        )

        test_accuracy = (
            (test_predictions == test_y).float().mean().item()
        )

    print("\n" + "=" * 80)
    print("PHASE 0 LEARNING INTEGRITY")
    print("=" * 80)

    print(f"Vocabulary size       : {len(vocab)}")
    print(f"Training samples      : {len(train_samples)}")
    print(f"Held-out samples      : {len(test_samples)}")
    print(f"Input tensor shape    : {tuple(train_x.shape)}")
    print(f"Target tensor shape   : {tuple(train_y.shape)}")

    print(f"\nInitial loss          : {initial_loss:.6f}")
    print(f"Final loss            : {final_loss:.6f}")

    print(f"\nTraining accuracy     : {train_accuracy * 100:.2f}%")
    print(f"Held-out accuracy     : {test_accuracy * 100:.2f}%")

    print("\nWhat this experiment tests:")
    print("  Token IDs -> embeddings -> transformer -> prediction")
    print("  Gradient descent must reduce training error.")
    print("  Evaluation uses combinations absent from training.")

    print("\nWhat this experiment does NOT prove:")
    print("  Natural-language understanding")
    print("  Abstract reasoning")
    print("  Tool selection")
    print("  Multi-hop reasoning")
    print("  Capability learning")

    print("=" * 80)

    # Mechanism must learn something.
    assert final_loss < initial_loss

    # Training should become substantially better than random.
    assert train_accuracy > 0.90

    # Held-out performance is reported, not blindly declared as proof
    # of generalization.
    #
    # We intentionally do not require a particular generalization
    # threshold until the structural split has been independently audited.


def test_token_id_serialization_is_deterministic():
    """
    The same vocabulary and input must produce exactly the same IDs.
    """

    vocab = build_vocabulary()
    tokenizer = ArisCanonicalTokenizer(vocab)

    text = "[BOS] e_3 r_2 e_8 [EOS]"

    ids_a = tokenizer.encode(text)
    ids_b = tokenizer.encode(text)

    assert ids_a == ids_b

    decoded = tokenizer.decode(ids_a)

    assert decoded == text


def test_gradient_reaches_token_embedding():
    """
    Verify that the supervised prediction objective actually produces
    gradients in the learnable token representation.
    """

    set_seed()

    vocab = build_vocabulary()

    model = ArisPhase0Model(
        vocab_size=len(vocab),
        embed_dim=32,
        max_seq_len=8,
        num_heads=2,
        ff_dim=64,
        num_layers=1,
    )

    input_ids = torch.tensor(
        [[
            vocab.token_to_id("[BOS]"),
            vocab.token_to_id("e_2"),
            vocab.token_to_id("r_1"),
        ]],
        dtype=torch.long,
    )

    target = torch.tensor(
        [vocab.token_to_id("e_4")],
        dtype=torch.long,
    )

    output = model(input_ids)

    logits = output["logits"][:, -1, :]

    loss = nn.CrossEntropyLoss()(logits, target)

    model.zero_grad()
    loss.backward()

    gradient = model.token_embedding.weight.grad

    assert gradient is not None
    assert torch.norm(gradient).item() > 0.0

    print("\n" + "=" * 80)
    print("GRADIENT INTEGRITY")
    print("=" * 80)
    print(f"Loss                  : {loss.item():.6f}")
    print(
        f"Embedding gradient    : "
        f"{torch.norm(gradient).item():.6f}"
    )
    print("=" * 80)