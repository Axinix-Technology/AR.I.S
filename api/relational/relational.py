"""
AR.I.S — Relational Binding Primitive (Tier 3)

Architectural Invariants:
  - No hardcoded relation dictionaries (Zero Hard Patches — Rule 1 & Rule 4).
  - No string matching on relation text.
  - Relation spans are discovered STRUCTURALLY as the intervening token
    sequence between two entity candidates within the same sentence.
  - Direction is determined purely by sequence order:
      Subject (E_i) --[intervening tokens]--> Object (E_j), where i < j.
  - Sentence-local: bindings never cross sentence boundaries.
  - Exact character span preservation invariant:
      source[rel['char_start']:rel['char_end']] == rel['text']

This is a REPRESENTATION PRIMITIVE, not a semantic reasoner.
It records what the structure of a sentence says — not what it means.
Meaning is the job of Tier 4.
"""

from collections import defaultdict
from typing import Optional, List, Dict, Any, Tuple


class ArisRelationalBinder:
    """
    Tier 3 Relational Binding Primitive.

    Discovers directional subject → object relations as the structural
    intervening span between entity_candidate pairs within a sentence.
    """

    @classmethod
    def bind(cls, text: str) -> list[dict]:
        """
        Convenience wrapper extracting relational bindings directly from text.
        """
        if not text or not isinstance(text, str) or not text.strip():
            return []
        from ..linguistic.tokenizer import ArisTokenizer
        tokens = ArisTokenizer.tokenize(text.strip())
        candidates = ArisTokenizer.extract_candidates(text.strip(), tokens)
        return cls.bind_relations(text.strip(), tokens, candidates)

    @classmethod
    def bind_relations(
        cls,
        text: str,
        tokens: list[dict],
        candidates: list[dict],
    ) -> list[dict]:
        """
        Extracts directional relational bindings from a tokenised text.

        Algorithm (purely structural — no dictionaries):
          1. Partition entity_candidate spans by sentence_index.
          2. Within each sentence, take ordered pairs (E_i, E_{i+1}).
          3. The relation span is the contiguous character slice between
             E_i.char_end and E_{i+1}.char_start — stripped of leading/trailing
             whitespace, with exact source-aligned char indices.
          4. Emit a binding dict for each pair.

        Args:
            text:       The original source text (never modified).
            tokens:     Tier 1 token list produced by ArisTokenizer.tokenize.
            candidates: Tier 2 candidate list produced by ArisTokenizer.extract_candidates.

        Returns:
            List of relational binding dicts, sorted by sentence_index then
            char_start of the subject.
        """
        if not text or not tokens:
            return []
        candidates = candidates or []

        # Named candidates (entity_candidate + ambiguous_capitalized) participate in binding.
        #
        # Reasoning: ambiguous_capitalized means "capitalization is structurally ambiguous" —
        # it does NOT mean "this is not a named referent". Sentence-initial words like "Arun"
        # in "Arun is brother of Bala" are ambiguous at Tier 2 because we cannot assert
        # entityhood from capitalization alone. But at Tier 3 we are discovering a STRUCTURAL
        # relation between positioned spans — not asserting semantic entity type. Including them
        # as anchors is correct.
        #
        # compound_word is excluded: it describes a lexical concept ("co-founders"),
        # not a discrete named referent that can anchor a directional relation.
        existing_spans = {(c["char_start"], c["char_end"]) for c in candidates}
        pron_anchors = []
        for t in tokens:
            pos = t.get("position", {})
            span = (pos.get("char_start", 0), pos.get("char_end", 0))
            if t.get("upos") == "PRON" and span not in existing_spans:
                pron_anchors.append({
                    "text": t.get("raw", ""),
                    "normalized": t.get("token", "").lower(),
                    "candidate_type": "pronoun_anchor",
                    "token_indices": [t.get("sequence_index", 0)],
                    "char_start": span[0],
                    "char_end": span[1],
                    "sentence_index": pos.get("sentence_index", 0),
                })

        ANCHOR_TYPES = {"entity_candidate", "ambiguous_capitalized", "pronoun_anchor"}
        all_candidates = candidates + pron_anchors
        entity_candidates = [
            c for c in all_candidates if c["candidate_type"] in ANCHOR_TYPES
        ]

        if not entity_candidates:
            return []

        # Partition entity candidates by sentence, preserving sequence order.
        by_sentence: dict[int, list[dict]] = defaultdict(list)
        for cand in entity_candidates:
            by_sentence[cand["sentence_index"]].append(cand)

        # Sort within each sentence by char_start (candidates are already
        # sorted globally, but partitioning may disrupt local order).
        for sent_idx in by_sentence:
            by_sentence[sent_idx].sort(key=lambda c: c["char_start"])

        bindings: list[dict] = []
        binding_index = 0

        for sent_idx in sorted(by_sentence.keys()):
            sent_candidates = by_sentence[sent_idx]

            # Single-entity sentence: resolve predicate and complement
            if len(sent_candidates) == 1:
                cand = sent_candidates[0]
                pred_toks = [
                    t for t in tokens
                    if t.get("position", {}).get("sentence_index", 0) == sent_idx
                    and t.get("sequence_index", 0) not in cand.get("token_indices", [])
                    and t.get("upos") not in ("PUNCT", "SYM")
                ]
                if pred_toks:
                    cand_end = cand["char_end"]
                    after_toks = [t for t in pred_toks if t["position"]["char_start"] >= cand_end]
                    verbal_after = [t for t in after_toks if t.get("upos") in ("VERB", "AUX")]
                    nominal_after = [
                        t for t in after_toks
                        if t.get("upos") in ("PROPN", "NOUN", "X")
                        and (not verbal_after or t["position"]["char_start"] >= verbal_after[-1]["position"]["char_end"])
                    ]

                    if verbal_after and nominal_after:
                        # Declarative copular/transitive assertion (e.g. 'I am Arun'): verbal is relation, nominal is object
                        v_start = min(t["position"]["char_start"] for t in verbal_after)
                        v_end = max(t["position"]["char_end"] for t in verbal_after)
                        o_start = min(t["position"]["char_start"] for t in nominal_after)
                        o_end = max(t["position"]["char_end"] for t in nominal_after)
                        v_text = text[v_start:v_end]
                        o_text = text[o_start:o_end]

                        bindings.append({
                            "binding_index": binding_index,
                            "sentence_index": sent_idx,
                            "subject": cand,
                            "object": {
                                "text": o_text,
                                "normalized": o_text.lower(),
                                "candidate_type": "entity_candidate",
                                "char_start": o_start,
                                "char_end": o_end,
                                "token_indices": [t["sequence_index"] for t in nominal_after],
                            },
                            "relation": {
                                "text": v_text,
                                "normalized": v_text.lower(),
                                "char_start": v_start,
                                "char_end": v_end,
                                "token_indices": [t["sequence_index"] for t in verbal_after],
                            },
                            "char_start": min(cand["char_start"], v_start),
                            "char_end": max(cand["char_end"], o_end),
                        })
                        binding_index += 1
                    else:
                        # Open unary relation binding — open target role inferred by cognitive neural model
                        p_start = min(t["position"]["char_start"] for t in pred_toks)
                        p_end = max(t["position"]["char_end"] for t in pred_toks)
                        rel_text = text[p_start:p_end]
                        bindings.append({
                            "binding_index": binding_index,
                            "sentence_index": sent_idx,
                            "subject": cand,
                            "object": None,
                            "relation": {
                                "text": rel_text,
                                "normalized": rel_text.lower(),
                                "char_start": p_start,
                                "char_end": p_end,
                                "token_indices": [t["sequence_index"] for t in pred_toks],
                            },
                            "char_start": min(cand["char_start"], p_start),
                            "char_end": max(cand["char_end"], p_end),
                        })
                        binding_index += 1
                continue

            # Slide a window over consecutive pairs: (E0,E1), (E1,E2), ...
            for i in range(len(sent_candidates) - 1):
                subject = sent_candidates[i]
                obj = sent_candidates[i + 1]

                # Raw intervening slice from the source text
                raw_between = text[subject["char_end"]:obj["char_start"]]

                # Strip leading/trailing whitespace and compute exact span
                stripped = raw_between.strip()
                if not stripped:
                    # No tokens between the two entities — skip this pair.
                    # (e.g. comma-separated list items with no relation words)
                    continue

                # Find exact character positions of stripped relation span
                leading_ws = len(raw_between) - len(raw_between.lstrip())
                rel_char_start = subject["char_end"] + leading_ws
                rel_char_end = rel_char_start + len(stripped)

                # Exact span preservation invariant
                assert text[rel_char_start:rel_char_end] == stripped, (
                    f"Relation span invariant violated: "
                    f"'{text[rel_char_start:rel_char_end]}' != '{stripped}'"
                )

                # Collect token indices that fall within the relation span
                relation_token_indices = [
                    t["sequence_index"]
                    for t in tokens
                    if (
                        t["position"]["char_start"] >= rel_char_start
                        and t["position"]["char_end"] <= rel_char_end
                    )
                ]

                bindings.append({
                    "index": binding_index,
                    "sentence_index": sent_idx,
                    "subject": {
                        "candidate_index": subject.get("index", -1),
                        "text": subject["text"],
                        "token_indices": subject["token_indices"],
                        "char_start": subject["char_start"],
                        "char_end": subject["char_end"],
                    },
                    "relation": {
                        "text": stripped,
                        "normalized": stripped.lower(),
                        "token_indices": relation_token_indices,
                        "char_start": rel_char_start,
                        "char_end": rel_char_end,
                    },
                    "object": {
                        "candidate_index": obj.get("index", -1),
                        "text": obj["text"],
                        "token_indices": obj["token_indices"],
                        "char_start": obj["char_start"],
                        "char_end": obj["char_end"],
                    },
                })
                binding_index += 1

        return bindings

    @classmethod
    def bind_tokenized_sequence(
        cls,
        sequence: Any,
        hidden_states: Optional[Any] = None,
        candidates: Optional[list[dict]] = None,
    ) -> list[dict]:
        """
        Phase 2: Binds directional relational spans directly from a canonical TokenizedSequence
        and optionally pools PyTorch sequence hidden states into a neural relation representation.

        Args:
            sequence: TokenizedSequence emitted by ArisCanonicalTokenizer.
            hidden_states: Optional PyTorch Tensor of shape (1, L, D) or (Batch, L, D).
            candidates: Optional pre-extracted candidate list. If None, structural candidates
                        are extracted directly from sequence.tokens and metadata['token_spans'].

        Returns:
            List of relational binding dicts, with optional 'neural_representation' (List[float])
            and 'tensor_shape' ([D]).
        """
        if not sequence or not sequence.tokens:
            return []

        tokens = sequence.tokens
        spans = sequence.metadata.get("token_spans", [])
        special_tokens = {"[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]"}

        # If candidates not provided, extract structurally from sequence tokens
        if candidates is None:
            extracted_cands = []
            cand_idx = 0
            i = 0
            n = len(tokens)
            while i < n:
                tok = tokens[i]
                if tok in special_tokens:
                    i += 1
                    continue

                span = spans[i] if i < len(spans) else (0, 0)
                if tok[0].isupper() and any(c.isalpha() for c in tok):
                    seq_start = i
                    while (
                        i + 1 < n
                        and tokens[i + 1] not in special_tokens
                        and tokens[i + 1][0].isupper()
                        and any(c.isalpha() for c in tokens[i + 1])
                    ):
                        i += 1
                    seq_end = i
                    cand_tokens = tokens[seq_start : seq_end + 1]
                    cand_text = " ".join(cand_tokens)
                    cand_start = spans[seq_start][0] if seq_start < len(spans) else 0
                    cand_end = spans[seq_end][1] if seq_end < len(spans) else 0

                    extracted_cands.append({
                        "index": cand_idx,
                        "candidate_index": cand_idx,
                        "text": cand_text,
                        "token_indices": list(range(seq_start, seq_end + 1)),
                        "char_start": cand_start,
                        "char_end": cand_end,
                        "candidate_type": "entity_candidate",
                        "sentence_index": 0,
                    })
                    cand_idx += 1
                i += 1
            candidates = extracted_cands

        if len(candidates) < 2:
            return []

        # Partition candidates by sentence_index
        by_sentence: dict[int, list[dict]] = defaultdict(list)
        for cand in candidates:
            by_sentence[cand.get("sentence_index", 0)].append(cand)

        for sent_idx in by_sentence:
            by_sentence[sent_idx].sort(key=lambda c: c["char_start"])

        bindings = []
        binding_index = 0

        for sent_idx in sorted(by_sentence.keys()):
            sent_candidates = by_sentence[sent_idx]
            for i in range(len(sent_candidates) - 1):
                subject = sent_candidates[i]
                obj = sent_candidates[i + 1]

                subj_last_tok = max(subject["token_indices"]) if subject.get("token_indices") else 0
                obj_first_tok = min(obj["token_indices"]) if obj.get("token_indices") else len(tokens)

                # Intervening token indices
                rel_token_indices = [
                    idx for idx in range(subj_last_tok + 1, obj_first_tok)
                    if idx < len(tokens) and tokens[idx] not in special_tokens
                ]

                if not rel_token_indices:
                    continue

                rel_tokens = [tokens[idx] for idx in rel_token_indices]
                rel_text = " ".join(rel_tokens)
                rel_start = spans[rel_token_indices[0]][0] if spans and rel_token_indices[0] < len(spans) else subject["char_end"]
                rel_end = spans[rel_token_indices[-1]][1] if spans and rel_token_indices[-1] < len(spans) else obj["char_start"]

                binding = {
                    "index": binding_index,
                    "sentence_index": sent_idx,
                    "subject": {
                        "candidate_index": subject.get("candidate_index", subject.get("index", -1)),
                        "text": subject["text"],
                        "token_indices": subject["token_indices"],
                        "char_start": subject["char_start"],
                        "char_end": subject["char_end"],
                    },
                    "relation": {
                        "text": rel_text,
                        "normalized": rel_text.lower(),
                        "token_indices": rel_token_indices,
                        "char_start": rel_start,
                        "char_end": rel_end,
                    },
                    "object": {
                        "candidate_index": obj.get("candidate_index", obj.get("index", -1)),
                        "text": obj["text"],
                        "token_indices": obj["token_indices"],
                        "char_start": obj["char_start"],
                        "char_end": obj["char_end"],
                    },
                }

                # If hidden_states provided, pool the relation token representations
                if hidden_states is not None:
                    import torch
                    hs = hidden_states[0] if hidden_states.dim() == 3 else hidden_states
                    valid_indices = [idx for idx in rel_token_indices if idx < hs.shape[0]]
                    if valid_indices:
                        rel_tensor = hs[valid_indices]
                        h_rel = rel_tensor.mean(dim=0)
                        norm = torch.norm(h_rel, p=2)
                        if norm > 1e-8:
                            h_rel = h_rel / norm
                        binding["neural_representation"] = h_rel.detach().cpu().tolist()
                        binding["tensor_shape"] = list(h_rel.shape)
                        binding["relation"]["tensor_shape"] = list(h_rel.shape)

                bindings.append(binding)
                binding_index += 1

        return bindings

    @classmethod
    def canonicalize_to_known(cls, span: str, known_spans: set) -> str:
        """
        Data-driven relation span canonicalization.

        Finds the SHORTEST suffix of 'span' that exists in the stored memory
        corpus ('known_spans'). Scanning from the rightmost suffix ensures the
        system converges to the minimal semantic core of the relation —
        discarding vacuous leading words (e.g. 'is', 'does') that are present
        in some phrasings but absent in stored facts.

        Zero hardcoded word lists — the corpus itself teaches which prefixes
        are structurally transparent:
          - If corpus contains 'brother of' AND 'is brother of',
            the shortest suffix is 'brother of' → canonical form.
          - If corpus only contains 'is brother of',
            no shorter suffix matches → original span returned unchanged.

        Example (corpus has both 'is brother of' and 'brother of'):
            canonicalize_to_known('is brother of', {'is brother of', 'brother of'})
            → shortest match: 'brother of'
        """
        words = span.strip().lower().split()
        shortest_match = None
        for i in range(len(words)):
            candidate = " ".join(words[i:])
            if candidate in known_spans:
                shortest_match = candidate   # keep updating — last match is shortest
        return shortest_match if shortest_match is not None else span
