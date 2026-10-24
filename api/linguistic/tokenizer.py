"""
AR.I.S — Deterministic Token + Positional Representation & Candidate Extraction Primitive

Architectural Invariants:
Tier 1: Atomic Tokens & Exact Positional Coordinates
Tier 2: Concept & Entity Candidates (compounds, multi-token sequences, structural entity candidates)
Tier 3: Future Relational Binding & Learning Ground

Strict Rules:
- NO hardcoded dictionaries or common-word lists (Zero Hard Patches).
- Pure structural invariants:
  - Hyphenated compounds: [word][-][word] -> compound_word
  - Multi-word capitalization: always entity_candidate (grammar never forces 2+ capitalized words)
  - Single capitalized word inside sentence: entity_candidate
  - Single sentence-initial capitalized word: ambiguous_capitalized (not blindly asserted as entity)
- Exact Lexical Preservation: source[char_start:char_end] == span_text for all tiers.
"""

class ArisTokenizer:
    """
    Deterministic cognitive tokenizer, positional encoder, and concept candidate extractor.
    """

    SENTENCE_TERMINATORS = {".", "!", "?"}

    @classmethod
    def tokenize(cls, text: str) -> list[dict]:
        """
        Tier 1: Tokenizes input text into a deterministic list of tokens with exact positional coordinates.
        """
        if not text or not isinstance(text, str):
            return []

        from .canonical_tokenizer import ArisCanonicalTokenizer
        canonical_spans = ArisCanonicalTokenizer().tokenize_with_spans(text)
        total_tokens = len(canonical_spans)

        tokens = []
        current_sentence = 0
        token_in_sentence = 0

        for seq_idx, (raw, char_start, char_end) in enumerate(canonical_spans):
            # Exact lexical preservation invariant
            assert text[char_start:char_end] == raw, (
                f"Lexical preservation invariant violated: '{raw}' != '{text[char_start:char_end]}'"
            )

            if raw.isdigit():
                token_type = "number"
                normalized = raw
            elif (len(raw) >= 3 and raw[1:3] == ":\\" and raw[0].isalpha()) or raw.startswith("/"):
                # Absolute filesystem path — preserve exactly as written
                token_type = "path"
                normalized = raw
            elif any(c.isalnum() for c in raw):
                token_type = "word"
                normalized = raw.lower()
            else:
                token_type = "symbol"
                normalized = raw

            if total_tokens > 1:
                rel_pos = round(seq_idx / (total_tokens - 1), 4)
            else:
                rel_pos = 0.0

            token_doc = {
                "sequence_index": seq_idx,
                "token": normalized,
                "raw": raw,
                "type": token_type,
                "position": {
                    "sequence_index": seq_idx,
                    "sentence_index": current_sentence,
                    "token_in_sentence": token_in_sentence,
                    "char_start": char_start,
                    "char_end": char_end,
                    "relative_pos": rel_pos,
                }
            }
            tokens.append(token_doc)

            # A sentence terminator must either be followed by whitespace or be at the end of input.
            # Intra-token or intra-name punctuation without trailing space (e.g. "MS.Dhoni", "3.14")
            # does not advance the sentence index.
            is_at_end = (char_end >= len(text))
            followed_by_space = (not is_at_end and text[char_end].isspace())
            if raw in cls.SENTENCE_TERMINATORS and (is_at_end or followed_by_space):
                current_sentence += 1
                token_in_sentence = 0
            else:
                token_in_sentence += 1

        return tokens

    @classmethod
    def enrich_tokens_with_syntax(cls, tokens: list[dict]) -> list[dict]:
        """
        Enriches Tier 1 tokens with Universal Dependencies (UD v2.18) UPOS tags
        using the in-memory continuous PyTorch tensor substrate.
        """
        if not tokens:
            return []

        from .ud_tensor_engine import ArisUniversalLinguisticEncoder

        n = len(tokens)
        for i, tok in enumerate(tokens):
            if tok.get("type") == "word":
                left_ctx = tokens[i - 1]["raw"] if i > 0 else ""
                right_ctx = tokens[i + 1]["raw"] if i + 1 < n else ""
                tag, conf = ArisUniversalLinguisticEncoder.predict_upos_contextual(
                    token=tok.get("raw", ""),
                    left_context=left_ctx,
                    right_context=right_ctx,
                )
                tok["upos"] = tag
                tok["upos_confidence"] = conf
            elif tok.get("type") == "symbol":
                tok["upos"] = "PUNCT"
                tok["upos_confidence"] = 1.0
            elif tok.get("type") == "number":
                tok["upos"] = "NUM"
                tok["upos_confidence"] = 1.0
            else:
                tok["upos"] = "X"
                tok["upos_confidence"] = 0.5

        # Recalibrate sentence boundaries using UD tensor classifications:
        # A PUNCT symbol between two PROPN tokens (e.g. title/honorific dot in "Dr. Martin")
        # is an intra-name constituent separator, not a sentence boundary.
        curr_sent = 0
        tok_in_sent = 0
        for i, tok in enumerate(tokens):
            tok["position"]["sentence_index"] = curr_sent
            tok["position"]["token_in_sentence"] = tok_in_sent

            if tok.get("type") == "symbol" and tok.get("raw") in cls.SENTENCE_TERMINATORS:
                is_intra_propn = (
                    i > 0
                    and i + 1 < n
                    and tokens[i - 1].get("upos") == "PROPN"
                    and tokens[i + 1].get("upos") == "PROPN"
                )
                if not is_intra_propn:
                    curr_sent += 1
                    tok_in_sent = 0
                    continue

            tok_in_sent += 1

        return tokens

    @classmethod
    def distance(cls, token_a: dict, token_b: dict) -> int:
        """
        Computes deterministic sequence distance primitive: abs(sequence_index_a - sequence_index_b).
        """
        idx_a = token_a.get("sequence_index", token_a.get("position", {}).get("sequence_index", 0))
        idx_b = token_b.get("sequence_index", token_b.get("position", {}).get("sequence_index", 0))
        return abs(idx_a - idx_b)

    @classmethod
    def extract_candidates(cls, text: str, tokens: list[dict]) -> list[dict]:
        """
        Tier 2: Extracts concept and entity candidate spans without destroying Tier 1 tokens.
        Preserves complete mapping back to token indices and character spans.

        Enforces zero hard patches: No hidden English dictionaries.
        Uses pure structural and orthographic invariants:
        1. Hyphenated compounds: [word][-][word] -> compound_word
        2. Multi-token capitalization: entity_candidate
        3. Non-sentence-initial capitalized single words: entity_candidate
        4. Sentence-initial single capitalized word: ambiguous_capitalized
        """
        if not text or not tokens:
            return []

        # Lazy UPOS enrichment: the PROPN-bridge in the accumulation loop below
        # requires upos tags produced by the trained UD tensor engine. When
        # extract_candidates is called directly (test setup, symbolic_graph
        # retrocompute) rather than through ArisLinguisticPipeline, tokens arrive
        # without upos. Enrich in-place so the model decision is always available.
        if tokens and "upos" not in tokens[0]:
            cls.enrich_tokens_with_syntax(tokens)

        candidates = []
        n = len(tokens)
        consumed_compound_indices = set()

        # Step 1: Scan for hyphenated compound concepts: [word] [-] [word]
        i = 0
        while i < n - 2:
            t0, t1, t2 = tokens[i], tokens[i + 1], tokens[i + 2]
            pos0, pos1, pos2 = t0["position"], t1["position"], t2["position"]

            # Structural check: word + '-' + word with contiguous character spans
            if (
                t0["type"] == "word"
                and t1["raw"] == "-"
                and t2["type"] == "word"
                and pos0["char_end"] == pos1["char_start"]
                and pos1["char_end"] == pos2["char_start"]
            ):
                start_char = pos0["char_start"]
                end_char = pos2["char_end"]
                compound_text = text[start_char:end_char]
                compound_indices = [t0["sequence_index"], t1["sequence_index"], t2["sequence_index"]]

                candidates.append({
                    "text": compound_text,
                    "normalized": compound_text.lower(),
                    "candidate_type": "compound_word",
                    "token_indices": compound_indices,
                    "char_start": start_char,
                    "char_end": end_char,
                    "sentence_index": pos0["sentence_index"]
                })
                consumed_compound_indices.update(compound_indices)
                i += 3
                continue
            i += 1

        # Step 2: Scan for capitalized sequences (proper-noun candidates by sentence)
        # Multi-word capitalization (e.g. "Axinix Technologies") forms entity_candidate compounds.
        i = 0
        while i < n:
            if i in consumed_compound_indices:
                i += 1
                continue

            t = tokens[i]
            is_start_cap = t["type"] == "word" and bool(t.get("raw")) and t["raw"][0].isupper()
            if is_start_cap:
                seq_start = i
                current_sent = t["position"]["sentence_index"]

                # Accumulate contiguous uppercase words in the same sentence (e.g. 'Axinix Technologies').
                # Extension: when the current token is followed by an intra-name separator symbol
                # with a capitalized word beyond it (e.g. 'Dr. Martin'), bridge across it.
                while i + 1 < n:
                    nxt = tokens[i + 1]

                    # Standard case: next token is an uppercase word in the same sentence
                    is_nxt_cap = nxt["type"] == "word" and bool(nxt.get("raw")) and nxt["raw"][0].isupper()
                    if (
                        (i + 1) not in consumed_compound_indices
                        and is_nxt_cap
                        and nxt["position"]["sentence_index"] == current_sent
                    ):
                        i += 1
                        continue

                    # Intra-symbol bridge: current token is followed by a punctuation symbol
                    # with an uppercase word beyond it, where either:
                    # 1) The symbol is contiguous with no whitespace (e.g. "MS.Dhoni")
                    # 2) Both words are PROPN according to the trained UD tensor engine
                    if (
                        nxt["type"] == "symbol"
                        and i + 2 < n
                        and (i + 2) not in consumed_compound_indices
                        and tokens[i + 2]["type"] == "word"
                        and tokens[i + 2]["raw"][0].isupper()
                        and tokens[i + 2]["position"]["sentence_index"] == current_sent
                        and (
                            nxt["position"]["char_start"] == tokens[i]["position"]["char_end"]
                            or (tokens[i].get("upos") == "PROPN" and tokens[i + 2].get("upos") == "PROPN")
                        )
                    ):
                        i += 2  # skip the symbol, land on the next uppercase word
                        continue

                    break

                seq_end = i
                candidate_tokens = tokens[seq_start:seq_end + 1]
                token_indices = [tok["sequence_index"] for tok in candidate_tokens]
                start_char = candidate_tokens[0]["position"]["char_start"]
                end_char = candidate_tokens[-1]["position"]["char_end"]
                span_text = text[start_char:end_char]

                assert text[start_char:end_char] == span_text

                if len(candidate_tokens) >= 2:
                    cand_type = "entity_candidate"
                else:
                    is_sentence_initial = (candidate_tokens[0]["position"]["token_in_sentence"] == 0)
                    is_clause_initial = (
                        is_sentence_initial
                        or (seq_start > 0 and tokens[seq_start - 1]["raw"] in {",", ";", ":", "-", "--"})
                    )
                    followed_by_lowercase = (
                        seq_end + 1 < n
                        and tokens[seq_end + 1]["type"] == "word"
                        and tokens[seq_end + 1]["raw"].islower()
                    )

                    if is_sentence_initial or (is_clause_initial and followed_by_lowercase):
                        cand_type = "ambiguous_capitalized"
                    else:
                        cand_type = "entity_candidate"

                candidates.append({
                    "text": span_text,
                    "normalized": span_text.lower(),
                    "candidate_type": cand_type,
                    "token_indices": token_indices,
                    "char_start": start_char,
                    "char_end": end_char,
                    "sentence_index": current_sent
                })
                consumed_compound_indices.update(token_indices)

            i += 1

        # Sort candidates deterministically by char_start
        candidates.sort(key=lambda c: c["char_start"])

        # Add zero-based index to candidates
        for idx, c in enumerate(candidates):
            c["index"] = idx

        return candidates
