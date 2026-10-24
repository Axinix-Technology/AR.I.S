# Cognitive Test Integrity Rule

## Problem Statement

A test that mocks cognitive methods (tokenization, candidate extraction, relational binding) and
then asserts on those mocked results proves only that **Python orchestration works**.
It does NOT prove that **AR.I.S Brain works**.

This was discovered during Phase 2.5 audit when `test_cli_inspect_microscope_two_tiers` was found
to pre-compute tokens and candidates outside the brain, inject them via mock, then verify the CLI
formatted them correctly. The brain was completely bypassed. All assertions would pass even if
ArisTokenizer was deleted.

---

## The Rule

> **Only MongoDB I/O (`_get_collection`) may be mocked in cognitive tests.**
> Cognitive methods — `ArisTokenizer.tokenize`, `ArisTokenizer.extract_candidates`,
> `ArisRelationalBinder.bind_relations` — must **always** be called on **real text**.

---

## Correct Test Pattern

```python
@patch('api.brain.ArisBrain._get_collection')
def test_brain_produces_real_output(self, mock_get_coll):
    # Wire mock for MongoDB I/O isolation only
    stored_doc = {}

    def fake_insert(doc):
        stored_doc.update(doc)
        stored_doc["_id"] = "aabbccddeeff001122334455"  # valid 24-char hex ObjectId
        return MagicMock(inserted_id="aabbccddeeff001122334455")

    def fake_find_one(query):
        return dict(stored_doc)

    mock_collection = MagicMock()
    mock_collection.insert_one.side_effect = fake_insert
    mock_collection.find_one.side_effect = fake_find_one
    mock_get_coll.return_value = mock_collection

    # Real cognitive chain — brain runs on actual text
    ArisBrain.store_fact(statement="Divya is sister of Akila.")
    fact = ArisBrain.inspect_fact("aabbccddeeff001122334455")

    # Assert structural invariants on real brain output
    self.assertGreater(fact["candidate_count"], 0)
    cand_texts = [c["text"] for c in fact["candidates"]]
    self.assertIn("Divya", cand_texts)
    self.assertIn("Akila", cand_texts)
```

---

## Forbidden Pattern

```python
# ❌ WRONG: cognitive output is fabricated — brain is bypassed
mock_inspect.return_value = {
    "tokens": pre_computed_tokens,     # brain never ran
    "candidates": pre_computed_cands   # brain never ran
}
```

---

## Why This Matters

AR.I.S is an **Artificial Intelligent System**, not a Python script orchestrator.
If the brain is mocked in tests, the tests prove nothing about cognitive capability.
Tests must exercise the brain's own reasoning on unseen input, and assertions must
be based purely on structural invariants that hold for any valid input.

---

## Technical Note: Fake ObjectId Requirement

`ArisBrain.get_fact(fact_id)` calls `bson.ObjectId(fact_id)`, which requires a valid
24-character hexadecimal string. Non-hex fake IDs (e.g. `"fake-id-abc"`) raise `InvalidId`
and cause `get_fact` to return `None`. Always use a valid hex string:

```python
FAKE_OID = "aabbccddeeff001122334455"  # valid 24-char hex
```

---

## Status
- Discovered: Phase 2.5 Tier 2 audit session
- Fixed: `test_cli_inspect_microscope_two_tiers` replaced with `test_inspect_fact_produces_real_brain_output`
- All 16 tests pass on real cognitive output
