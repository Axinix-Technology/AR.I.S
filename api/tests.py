# pyright: ignore
"""
AR.I.S â€” API, Brain, Tokenizer & Concept Candidate Audit Verification Tests

Strict Audit Suite covering:
1. Tier 1: Exact Lexical Preservation, Deterministic Indexing, Sentence Locality, Distance.
2. Tier 2: Concept & Entity Candidates (Compounds, Multi-Token Names, Sentence-Initial Ambiguity).
3. Adversarial Structural Suite (Zero Hard Patches / No Hidden Dictionaries).
4. Cognitive Brain Invariants (Real Brain Output â€” no cognitive method mocking).
5. Strict Backward Compatibility.

AUDIT RULE: Only MongoDB I/O (_get_collection) may be mocked.
Never mock ArisTokenizer.tokenize, extract_candidates, or any cognitive method.
A test that mocks cognitive output does NOT prove AR.I.S Brain works â€” it proves Python orchestration works.
"""

import io
import os
import sys
from unittest.mock import patch, MagicMock
from django.urls import reverse
from django.core.management import call_command
from rest_framework import status
from rest_framework.test import APITestCase
from api.cognitive.brain import ArisBrain
from api.linguistic.tokenizer import ArisTokenizer

# Enable submodule package resolution for the api/tests/ directory
_tests_dir = os.path.join(os.path.dirname(__file__), "tests")
if os.path.isdir(_tests_dir):
    __path__ = [_tests_dir]


class TokenizerTier1InvariantTests(APITestCase):
    """Phase 2 Acceptance Invariants for Deterministic Token + Positional Primitive."""

    def test_exact_lexical_preservation_invariant(self):
        source = "Arunbharathi founded Axinix Technologies in Chennai. It drives AI systems!"
        tokens = ArisTokenizer.tokenize(source)

        self.assertGreater(len(tokens), 0)
        for t in tokens:
            pos = t["position"]
            span_text = source[pos["char_start"]:pos["char_end"]]
            self.assertEqual(t["raw"], span_text)

    def test_deterministic_indexing_invariant(self):
        text = "First token, second token, third token."
        tokens = ArisTokenizer.tokenize(text)

        expected_indices = list(range(len(tokens)))
        actual_indices = [t["sequence_index"] for t in tokens]
        self.assertEqual(actual_indices, expected_indices)
        self.assertEqual(tokens, ArisTokenizer.tokenize(text))

    def test_sentence_locality_invariant(self):
        text = "Hello world. How are you? I am well! Yes."
        tokens = ArisTokenizer.tokenize(text)
        sentence_indices = [t["position"]["sentence_index"] for t in tokens]
        self.assertEqual(sentence_indices[0], 0)
        self.assertEqual(sentence_indices[-1], 3)

    def test_relative_position_normalization_invariant(self):
        text = "One two three four five."
        tokens = ArisTokenizer.tokenize(text)
        self.assertEqual(tokens[0]["position"]["relative_pos"], 0.0)
        self.assertEqual(tokens[-1]["position"]["relative_pos"], 1.0)

    def test_deterministic_distance_primitive(self):
        text = "Zero one two three four five."
        tokens = ArisTokenizer.tokenize(text)
        self.assertEqual(ArisTokenizer.distance(tokens[2], tokens[5]), 3)


class ConceptCandidateTier2Tests(APITestCase):
    """Phase 2.5 Tests for Compound Words and Entity Candidates."""

    def test_compound_word_extraction(self):
        text = "Axinix Technologies has co-founders and well-known engineers."
        tokens = ArisTokenizer.tokenize(text)
        candidates = ArisTokenizer.extract_candidates(text, tokens)

        compounds = [c for c in candidates if c["candidate_type"] == "compound_word"]
        compound_texts = [c["text"] for c in compounds]

        self.assertIn("co-founders", compound_texts)
        self.assertIn("well-known", compound_texts)

        for c in compounds:
            self.assertEqual(text[c["char_start"]:c["char_end"]], c["text"])

    def test_multi_word_proper_name_candidate(self):
        text = "Arunbharathi founded Axinix Technologies. Guru Prasath joined later."
        tokens = ArisTokenizer.tokenize(text)
        candidates = ArisTokenizer.extract_candidates(text, tokens)

        entity_cands = [c for c in candidates if c["candidate_type"] == "entity_candidate"]
        cand_texts = [c["text"] for c in entity_cands]

        self.assertIn("Axinix Technologies", cand_texts)
        self.assertIn("Guru Prasath", cand_texts)

    def test_structural_sentence_initial_ambiguity_no_dictionary(self):
        """
        Structural Rule: Sentence-initial single capitalized words are ambiguous.
        Non-initial capitalized words are entity_candidate.
        """
        text = "The founder is Arun. Apple is a company."
        tokens = ArisTokenizer.tokenize(text)
        candidates = ArisTokenizer.extract_candidates(text, tokens)

        cand_map = {c["text"]: c["candidate_type"] for c in candidates}

        # Sentence-initial single word "The" -> ambiguous_capitalized
        self.assertEqual(cand_map.get("The"), "ambiguous_capitalized")
        # Non-initial "Arun" -> entity_candidate
        self.assertEqual(cand_map.get("Arun"), "entity_candidate")
        # Sentence-initial single word "Apple" -> ambiguous_capitalized
        self.assertEqual(cand_map.get("Apple"), "ambiguous_capitalized")

    def test_co_founders_statement_the_ambiguity_regression(self):
        """
        Regression Test: In 'Axinix Technologies has other co-founders, The names are Ajay...',
        'The' must be classified as 'ambiguous_capitalized' (not entity_candidate),
        while 'Axinix Technologies', 'Ajay', 'Boopalan', 'Guru Prasath', 'Siva', 'Karuppasamy'
        must remain 'entity_candidate', and 'co-founders' must be 'compound_word'.
        """
        text = "Axinix Technologies has other co-founders, The names are Ajay, Boopalan, Guru Prasath, Siva, Karuppasamy"
        tokens = ArisTokenizer.tokenize(text)
        candidates = ArisTokenizer.extract_candidates(text, tokens)

        cand_map = {c["text"]: c["candidate_type"] for c in candidates}

        self.assertEqual(cand_map.get("Axinix Technologies"), "entity_candidate")
        self.assertEqual(cand_map.get("co-founders"), "compound_word")
        self.assertEqual(cand_map.get("The"), "ambiguous_capitalized")
        self.assertEqual(cand_map.get("Ajay"), "entity_candidate")
        self.assertEqual(cand_map.get("Boopalan"), "entity_candidate")
        self.assertEqual(cand_map.get("Guru Prasath"), "entity_candidate")
        self.assertEqual(cand_map.get("Siva"), "entity_candidate")
        self.assertEqual(cand_map.get("Karuppasamy"), "entity_candidate")


class AdversarialStructuralSuiteTests(APITestCase):
    """
    Adversarial structural test suite validating transparency and zero hardcoded dictionaries.
    """

    def test_quantum_dynamics_statement(self):
        text = "Quantum Dynamics was co-founded by Maya Lin in Berlin."
        tokens = ArisTokenizer.tokenize(text)
        candidates = ArisTokenizer.extract_candidates(text, tokens)

        cand_map = {c["text"]: c["candidate_type"] for c in candidates}

        # Multi-word sentence-initial capitalized -> entity_candidate
        self.assertEqual(cand_map.get("Quantum Dynamics"), "entity_candidate")
        # Hyphenated -> compound_word
        self.assertEqual(cand_map.get("co-founded"), "compound_word")
        # Multi-word inside sentence -> entity_candidate
        self.assertEqual(cand_map.get("Maya Lin"), "entity_candidate")
        # Single capitalized inside sentence -> entity_candidate
        self.assertEqual(cand_map.get("Berlin"), "entity_candidate")

    def test_quick_brown_fox(self):
        text = "The quick Brown fox jumps over the lazy Dog."
        tokens = ArisTokenizer.tokenize(text)
        candidates = ArisTokenizer.extract_candidates(text, tokens)

        cand_map = {c["text"]: c["candidate_type"] for c in candidates}
        self.assertEqual(cand_map.get("The"), "ambiguous_capitalized")
        self.assertEqual(cand_map.get("Brown"), "entity_candidate")
        self.assertEqual(cand_map.get("Dog"), "entity_candidate")

    def test_john_smith_mary_jane(self):
        text = "John Smith met Mary Jane."
        tokens = ArisTokenizer.tokenize(text)
        candidates = ArisTokenizer.extract_candidates(text, tokens)

        cand_map = {c["text"]: c["candidate_type"] for c in candidates}
        self.assertEqual(cand_map.get("John Smith"), "entity_candidate")
        self.assertEqual(cand_map.get("Mary Jane"), "entity_candidate")

    def test_well_known_technology_company(self):
        text = "A well-known technology company was established."
        tokens = ArisTokenizer.tokenize(text)
        candidates = ArisTokenizer.extract_candidates(text, tokens)

        compounds = [c["text"] for c in candidates if c["candidate_type"] == "compound_word"]
        self.assertIn("well-known", compounds)


class BrainCognitiveInvariantTests(APITestCase):
    """
    Tests that exercise the actual cognitive chain through ArisBrain.

    RULE: Only MongoDB I/O (_get_collection) is mocked for isolation.
    No cognitive method (tokenize, extract_candidates) is mocked.
    If a test mocks the cognitive output, it proves Python orchestration â€” not AR.I.S Brain.
    """

    @patch('api.cognitive.brain.ArisBrain._get_collection')
    def test_store_fact_attaches_candidates(self, mock_get_coll):
        """
        store_fact must call the real tokenizer and candidate extractor on the given statement.
        The candidates returned must reflect the actual structural content of the statement.
        """
        mock_collection = MagicMock()
        mock_collection.insert_one.return_value = MagicMock(inserted_id="507f1f77bcf86cd799439011")
        mock_get_coll.return_value = mock_collection

        statement = "Axinix Technologies has co-founders Ajay and Guru Prasath."
        fact = ArisBrain.store_fact(statement=statement)

        # Brain must populate structural fields from real tokenization
        self.assertIn("candidates", fact)
        self.assertGreater(fact["candidate_count"], 0)
        self.assertGreater(fact["token_count"], 0)

        # Structural invariant: entity candidates must reflect the real statement content
        cand_texts = [c["text"] for c in fact["candidates"]]
        self.assertIn("Axinix Technologies", cand_texts)
        self.assertIn("co-founders", cand_texts)
        self.assertIn("Guru Prasath", cand_texts)

        # Structural invariant: every candidate span must match the source text exactly
        for c in fact["candidates"]:
            self.assertEqual(
                statement[c["char_start"]:c["char_end"]],
                c["text"],
                msg=f"Span mismatch for candidate '{c['text']}'"
            )

    @patch('api.cognitive.brain.ArisBrain._get_collection')
    def test_backward_compatibility(self, mock_get_coll):
        legacy_doc = {
            "_id": "legacy-123",
            "statement": "Legacy fact from Phase 1 without tokens or candidates.",
            "topic": "general",
            "tags": [],
            "source": "legacy",
            "confidence": 1.0,
            "created_at": "2026-09-01T00:00:00Z",
            "updated_at": "2026-09-01T00:00:00Z"
        }
        mock_collection = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = mock_cursor
        mock_cursor.limit.return_value = [legacy_doc]
        mock_collection.find.return_value = mock_cursor
        mock_get_coll.return_value = mock_collection

        facts = ArisBrain.retrieve_facts()
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["tokens"], [])
        self.assertEqual(facts[0]["candidates"], [])


    @patch('api.cognitive.brain.ArisBrain._get_collection')
    def test_inspect_fact_produces_real_brain_output(self, mock_get_coll):
        """
        End-to-end structural invariant.

        Flow:
          1. ArisBrain.store_fact(statement) â€” real text, real cognitive chain.
          2. ArisBrain.inspect_fact(fact_id) â€” brain retrieves and returns real tokens/candidates.

        No cognitive method is mocked.

        Assertions prove ArisBrain produced structurally correct output ON ITS OWN
        from raw text â€” not because Python orchestrated a pre-built answer.

        This test must hold for ANY entity names â€” not just "Divya/Akila".
        """
        statement = "Divya is sister of Akila."

        # Must be a valid 24-char hex ObjectId so get_fact's ObjectId() call succeeds
        FAKE_OID = "aabbccddeeff001122334455"

        stored_doc = {}

        def fake_insert(doc):
            stored_doc.update(doc)
            stored_doc["_id"] = FAKE_OID
            return MagicMock(inserted_id=FAKE_OID)

        def fake_find_one(query):
            return dict(stored_doc)

        mock_collection = MagicMock()
        mock_collection.insert_one.side_effect = fake_insert
        mock_collection.find_one.side_effect = fake_find_one
        mock_get_coll.return_value = mock_collection

        # Real store â€” full cognitive chain runs on actual text
        ArisBrain.store_fact(statement=statement)

        # Real inspect â€” brain must return cognitively computed output
        fact = ArisBrain.inspect_fact(FAKE_OID)

        self.assertIsNotNone(fact, "inspect_fact must return a result for a stored fact")
        self.assertGreater(fact["token_count"], 0, "Brain must produce tokens from real text")
        self.assertGreater(fact["candidate_count"], 0, "Brain must produce candidates from real text")

        # Structural invariant: entity names must be present as candidates
        cand_texts = [c["text"] for c in fact["candidates"]]
        self.assertIn("Divya", cand_texts, "Brain must recognise 'Divya' as an entity candidate")
        self.assertIn("Akila", cand_texts, "Brain must recognise 'Akila' as an entity candidate")

        # Structural invariant: all token spans must be exact
        for t in fact["tokens"]:
            pos = t["position"]
            self.assertEqual(
                statement[pos["char_start"]:pos["char_end"]],
                t["raw"],
                msg=f"Token span mismatch for '{t['raw']}'"
            )


class RelationalBindingTier3Tests(APITestCase):
    """
    Tier 3 Relational Binding Primitive â€” Structural Invariant Suite.

    AUDIT RULE: No cognitive method is mocked. Only MongoDB I/O (_get_collection) is mocked
    in brain-level tests. ArisRelationalBinder is always called on real text.
    Tests prove structural properties that hold for ANY valid relational statement.
    """

    def test_exact_span_preservation_invariant(self):
        """
        For every binding, source[rel['char_start']:rel['char_end']] must equal rel['text'].
        This invariant must hold for any input â€” not just specific sentences.
        """
        from api.relational.relational import ArisRelationalBinder

        text = "Maya Lin joined Quantum Dynamics in Berlin."
        tokens = ArisTokenizer.tokenize(text)
        candidates = ArisTokenizer.extract_candidates(text, tokens)
        bindings = ArisRelationalBinder.bind_relations(text, tokens, candidates)

        self.assertGreater(len(bindings), 0, "Must produce at least one binding")

        for b in bindings:
            rel = b["relation"]
            self.assertEqual(
                text[rel["char_start"]:rel["char_end"]],
                rel["text"],
                msg=f"Relation span invariant violated for '{rel['text']}'"
            )
            subj = b["subject"]
            self.assertEqual(
                text[subj["char_start"]:subj["char_end"]],
                subj["text"],
                msg=f"Subject span invariant violated for '{subj['text']}'"
            )
            obj = b["object"]
            self.assertEqual(
                text[obj["char_start"]:obj["char_end"]],
                obj["text"],
                msg=f"Object span invariant violated for '{obj['text']}'"
            )

    def test_directional_asymmetry_invariant(self):
        """
        Two statements with subject/object swapped must produce structurally reversed bindings.
        S1.subject.text == S2.object.text and S1.object.text == S2.subject.text.
        Works for any pair of entity names â€” no string comparison on relation text.
        """
        from api.relational.relational import ArisRelationalBinder

        s1 = "Arun is brother of Bala."
        s2 = "Bala is brother of Arun."

        def get_first_binding(text):
            tokens = ArisTokenizer.tokenize(text)
            candidates = ArisTokenizer.extract_candidates(text, tokens)
            bindings = ArisRelationalBinder.bind_relations(text, tokens, candidates)
            self.assertGreater(len(bindings), 0, f"Expected binding for: {text}")
            return bindings[0]

        b1 = get_first_binding(s1)
        b2 = get_first_binding(s2)

        # Asymmetry: the subject of S1 must be the object of S2 (and vice versa)
        self.assertEqual(b1["subject"]["text"], b2["object"]["text"])
        self.assertEqual(b1["object"]["text"], b2["subject"]["text"])

        # The two bindings must NOT be equal (strict asymmetry)
        self.assertNotEqual(
            (b1["subject"]["text"], b1["object"]["text"]),
            (b2["subject"]["text"], b2["object"]["text"]),
        )

    def test_multi_proposition_locality_invariant(self):
        """
        Multi-sentence input must produce sentence-partitioned bindings.
        Bindings from sentence 0 must not bleed into sentence 1.
        """
        from api.relational.relational import ArisRelationalBinder

        text = "Arun is brother of Bala. Bala is father of Kumar."
        tokens = ArisTokenizer.tokenize(text)
        candidates = ArisTokenizer.extract_candidates(text, tokens)
        bindings = ArisRelationalBinder.bind_relations(text, tokens, candidates)

        # Must produce exactly 2 bindings â€” one per sentence
        self.assertEqual(len(bindings), 2, "Multi-sentence input must yield one binding per sentence")

        # Sentence locality: binding 0 comes from sentence 0, binding 1 from sentence 1
        self.assertEqual(bindings[0]["sentence_index"], 0)
        self.assertEqual(bindings[1]["sentence_index"], 1)

        # Subjects must differ â€” they come from different sentences
        self.assertNotEqual(
            bindings[0]["subject"]["text"],
            bindings[1]["subject"]["text"],
        )

    def test_zero_relation_dictionary_invariant(self):
        """
        A completely novel, unseen relation phrase produces a valid structural binding
        without any dictionary lookup or string matching.
        The relation span text is irrelevant to whether a binding is produced.
        """
        from api.relational.relational import ArisRelationalBinder

        # Novel relation: "orchestrates the cognitive core of" â€” never seen before
        text = "Divya orchestrates the cognitive core of Akila."
        tokens = ArisTokenizer.tokenize(text)
        candidates = ArisTokenizer.extract_candidates(text, tokens)
        bindings = ArisRelationalBinder.bind_relations(text, tokens, candidates)

        self.assertGreater(len(bindings), 0, "Novel relation must produce a binding structurally")
        b = bindings[0]

        self.assertEqual(b["subject"]["text"], "Divya")
        self.assertEqual(b["object"]["text"], "Akila")

        # Relation span must be the raw intervening text â€” no dictionary shaping it
        self.assertIn("orchestrates", b["relation"]["text"])

        # Span preservation
        self.assertEqual(
            text[b["relation"]["char_start"]:b["relation"]["char_end"]],
            b["relation"]["text"]
        )

    def test_no_binding_for_single_entity(self):
        """
        A sentence with only one entity candidate must produce zero bindings.
        Relation requires at least two anchors.
        """
        from api.relational.relational import ArisRelationalBinder

        text = "The sky is blue today."
        tokens = ArisTokenizer.tokenize(text)
        candidates = ArisTokenizer.extract_candidates(text, tokens)
        bindings = ArisRelationalBinder.bind_relations(text, tokens, candidates)

        # "The" is ambiguous_capitalized â€” only one ambiguous candidate, not entity_candidate
        entity_cands = [c for c in candidates if c["candidate_type"] == "entity_candidate"]
        if len(entity_cands) < 2:
            self.assertEqual(len(bindings), 0, "Single entity must produce zero bindings")

    @patch('api.cognitive.brain.ArisBrain._get_collection')
    def test_brain_stores_and_returns_relations(self, mock_get_coll):
        """
        End-to-end: store_fact on a relational statement must persist relations.
        The brain itself runs Tier 3 â€” no cognitive method is mocked.
        """
        stored_doc = {}
        FAKE_OID = "ccddee112233445566778899"

        def fake_insert(doc):
            stored_doc.update(doc)
            stored_doc["_id"] = FAKE_OID
            return MagicMock(inserted_id=FAKE_OID)

        def fake_find_one(query):
            return dict(stored_doc)

        mock_collection = MagicMock()
        mock_collection.insert_one.side_effect = fake_insert
        mock_collection.find_one.side_effect = fake_find_one
        mock_get_coll.return_value = mock_collection

        statement = "Divya is sister of Akila."
        fact = ArisBrain.store_fact(statement=statement)

        # Brain must have computed and stored relations
        self.assertIn("relations", fact)
        self.assertGreater(fact["relation_count"], 0)

        b = fact["relations"][0]
        self.assertEqual(b["subject"]["text"], "Divya")
        self.assertEqual(b["object"]["text"], "Akila")

        # Span preservation on stored relation
        self.assertEqual(
            statement[b["relation"]["char_start"]:b["relation"]["char_end"]],
            b["relation"]["text"]
        )

    @patch('api.cognitive.brain.ArisBrain._get_collection')
    def test_legacy_fact_backward_compatibility_with_relations(self, mock_get_coll):
        """
        Legacy facts (no tokens, candidates, or relations) must still pass inspect_fact
        and receive retrocomputed relations from their statement text.
        """
        legacy_doc = {
            "_id": "aabbccddeeff001122334400",
            "statement": "Arun is brother of Bala.",
            "topic": "general",
            "tags": [],
            "source": "legacy",
            "confidence": 1.0,
            "created_at": "2026-09-01T00:00:00Z",
            "updated_at": "2026-09-01T00:00:00Z"
        }
        mock_collection = MagicMock()
        mock_collection.find_one.return_value = dict(legacy_doc)
        mock_get_coll.return_value = mock_collection

        fact = ArisBrain.inspect_fact("aabbccddeeff001122334400")

        self.assertIsNotNone(fact)
        # inspect_fact must retrocompute relations from statement
        self.assertGreater(fact["relation_count"], 0)
        self.assertEqual(fact["relations"][0]["subject"]["text"], "Arun")
        self.assertEqual(fact["relations"][0]["object"]["text"], "Bala")


class RelationalRetrievalTier4Tests(APITestCase):
    """
    Tier 4 Acceptance Invariant Tests for Deterministic 1-Hop Relational Retrieval.
    Contract: Intelligence Level = ZERO. Exact structural constraint matching only.
    Cognitive Test Integrity: Only MongoDB I/O is mocked.
    """

    def setUp(self):
        self.stored_docs = []
        self._next_id = 1

    def _setup_mock_collection(self, mock_get_coll):
        def fake_insert(doc):
            doc_copy = dict(doc)
            hex_id = f"{self._next_id:024x}"
            self._next_id += 1
            doc_copy["_id"] = hex_id
            self.stored_docs.append(doc_copy)
            mock_res = MagicMock()
            mock_res.inserted_id = hex_id
            return mock_res

        def fake_find(filters=None):
            # Return list of copies with sort support
            filtered = list(self.stored_docs)
            mock_cursor = MagicMock()
            mock_cursor.sort.return_value = filtered
            mock_cursor.__iter__.return_value = iter(filtered)
            return mock_cursor

        def fake_find_one(filters):
            target_id = str(filters.get("_id"))
            for d in self.stored_docs:
                if str(d["_id"]) == target_id:
                    return dict(d)
            return None

        mock_coll = MagicMock()
        mock_coll.insert_one.side_effect = fake_insert
        mock_coll.find.side_effect = fake_find
        mock_coll.find_one.side_effect = fake_find_one
        mock_coll.count_documents.side_effect = lambda f: len(self.stored_docs)
        mock_get_coll.return_value = mock_coll
        return mock_coll

    @patch('api.cognitive.brain.ArisBrain._get_collection')
    def test_forward_query_retrieval(self, mock_coll):
        """
        Forward Query: (Ajay, leads, ?) -> Axinix Technologies Development Team
        """
        self._setup_mock_collection(mock_coll)
        ArisBrain.store_fact(statement="Ajay leads Axinix Technologies Development Team.")

        result = ArisBrain.query_relations(
            subject="Ajay",
            relation="leads",
        )

        self.assertEqual(result["matches_count"], 1)
        ans = result["answers"][0]
        self.assertEqual(ans["answer"], "Axinix Technologies Development Team")
        self.assertEqual(ans["target_role"], "object")
        self.assertEqual(result["query"]["direction"], "forward")

    @patch('api.cognitive.brain.ArisBrain._get_collection')
    def test_reverse_query_retrieval(self, mock_coll):
        """
        Reverse Query: (?, leads, Axinix Technologies Development Team) -> Ajay
        """
        self._setup_mock_collection(mock_coll)
        ArisBrain.store_fact(statement="Ajay leads Axinix Technologies Development Team.")

        result = ArisBrain.query_relations(
            relation="leads",
            obj="Axinix Technologies Development Team",
        )

        self.assertEqual(result["matches_count"], 1)
        ans = result["answers"][0]
        self.assertEqual(ans["answer"], "Ajay")
        self.assertEqual(ans["target_role"], "subject")
        self.assertEqual(result["query"]["direction"], "reverse")

    @patch('api.cognitive.brain.ArisBrain._get_collection')
    def test_reverse_brother_relationship(self, mock_coll):
        """
        Reverse Query on family relation:
        Stored: Arun is brother of Bharathikannan.
        Query: (?, is brother of, Bharathikannan) -> Arun
        """
        self._setup_mock_collection(mock_coll)
        ArisBrain.store_fact(statement="Arun is brother of Bharathikannan.")

        result = ArisBrain.query_relations(
            relation="is brother of",
            obj="Bharathikannan",
        )

        self.assertEqual(result["matches_count"], 1)
        ans = result["answers"][0]
        self.assertEqual(ans["answer"], "Arun")

    @patch('api.cognitive.brain.ArisBrain._get_collection')
    def test_link_discovery(self, mock_coll):
        """
        Link Discovery Query: (Ajay, ?, Axinix Technologies Development Team) -> leads
        """
        self._setup_mock_collection(mock_coll)
        ArisBrain.store_fact(statement="Ajay leads Axinix Technologies Development Team.")

        result = ArisBrain.query_relations(
            subject="Ajay",
            obj="Axinix Technologies Development Team",
        )

        self.assertEqual(result["matches_count"], 1)
        ans = result["answers"][0]
        self.assertEqual(ans["answer"], "leads")
        self.assertEqual(ans["target_role"], "relation")
        self.assertEqual(result["query"]["direction"], "link")

    @patch('api.cognitive.brain.ArisBrain._get_collection')
    def test_no_match_returns_empty_on_constraint_mismatch(self, mock_coll):
        """
        Complete Constraint Integrity Test:
        Stored: Ajay leads Axinix Technologies Development Team.
        1. Query with wrong relation: (Ajay, manages, ?) -> 0 matches.
        2. Query with wrong object:   (Ajay, leads, Testing Team) -> 0 matches.
        Proves Tier 4 doesn't just match keywords; it matches complete structural constraints.
        """
        self._setup_mock_collection(mock_coll)
        ArisBrain.store_fact(statement="Ajay leads Axinix Technologies Development Team.")

        # 1. Unmatched relation
        res_rel = ArisBrain.query_relations(
            subject="Ajay",
            relation="manages",
        )
        self.assertEqual(res_rel["matches_count"], 0)
        self.assertEqual(len(res_rel["answers"]), 0)

        # 2. Unmatched object constraint in verification
        res_obj = ArisBrain.query_relations(
            subject="Ajay",
            relation="leads",
            obj="Testing Team",
        )
        self.assertEqual(res_obj["matches_count"], 0)
        self.assertEqual(len(res_obj["answers"]), 0)

    @patch('api.cognitive.brain.ArisBrain._get_collection')
    def test_natural_query_sentence_parsing(self, mock_coll):
        """
        Natural Query: 'Who leads Axinix Technologies Development Team?'
        Structural parser maps: (?, leads, Axinix Technologies Development Team) -> Ajay
        """
        self._setup_mock_collection(mock_coll)
        ArisBrain.store_fact(statement="Ajay leads Axinix Technologies Development Team.")

        result = ArisBrain.query_relations(
            query_text="Who leads Axinix Technologies Development Team?"
        )

        self.assertEqual(result["matches_count"], 1)
        ans = result["answers"][0]
        self.assertEqual(ans["answer"], "Ajay")
        self.assertEqual(ans["target_role"], "subject")

    @patch('api.cognitive.brain.ArisBrain._get_collection')
    def test_case_insensitive_entity_retrieval(self, mock_coll):
        """
        Entities match cleanly regardless of casing: 'ajay' matches 'Ajay'.
        """
        self._setup_mock_collection(mock_coll)
        ArisBrain.store_fact(statement="Ajay leads Axinix Technologies Development Team.")

        result = ArisBrain.query_relations(
            subject="ajay",
            relation="leads",
        )

        self.assertEqual(result["matches_count"], 1)
        self.assertEqual(result["answers"][0]["answer"], "Axinix Technologies Development Team")

    @patch('api.cognitive.brain.ArisBrain._get_collection')
    def test_multi_match_aggregation(self, mock_coll):
        """
        When two facts share the same relation:
        1. 'Arun founded Axinix Technologies.'
        2. 'Boopalan founded Axinix Technologies.'
        Query: (?, founded, Axinix Technologies) returns both Arun and Boopalan.
        """
        self._setup_mock_collection(mock_coll)
        ArisBrain.store_fact(statement="Arun founded Axinix Technologies.")
        ArisBrain.store_fact(statement="Boopalan founded Axinix Technologies.")

        result = ArisBrain.query_relations(
            relation="founded",
            obj="Axinix Technologies",
        )

        self.assertEqual(result["matches_count"], 2)
        answers = [item["answer"] for item in result["answers"]]
        self.assertIn("Arun", answers)
        self.assertIn("Boopalan", answers)

    @patch('api.cognitive.brain.ArisBrain._get_collection')
    def test_rest_api_relational_query_endpoint(self, mock_coll):
        """
        REST API: GET /api/brain/query/?subject=Ajay&relation=leads
        """
        self._setup_mock_collection(mock_coll)
        ArisBrain.store_fact(statement="Ajay leads Axinix Technologies Development Team.")

        url = reverse('brain-relational-query')
        response = self.client.get(url, {'subject': 'Ajay', 'relation': 'leads'})

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data["matches_count"], 1)
        self.assertEqual(data["answers"][0]["answer"], "Axinix Technologies Development Team")


class CognitiveOperationGeneralizationTier5Tests(APITestCase):
    """
    Tier 5 Invariant Tests: Generalization to Structurally Valid but Lexically Unseen Situations.
    Strict Invariant: Zero Spoon-Feeding, Zero Word Dictionaries, Zero Entity Memorization.
    """

    def setUp(self):
        self.stored_docs = []
        self._next_id = 1
        self.in_memory_templates = []

    def _setup_mock_collection(self, mock_get_coll):
        def fake_insert(doc):
            doc_copy = dict(doc)
            hex_id = f"{self._next_id:024x}"
            self._next_id += 1
            doc_copy["_id"] = hex_id
            self.stored_docs.append(doc_copy)
            mock_res = MagicMock()
            mock_res.inserted_id = hex_id
            return mock_res

        def fake_find(filters=None):
            filtered = list(self.stored_docs)
            mock_cursor = MagicMock()
            mock_cursor.sort.return_value = filtered
            mock_cursor.__iter__.return_value = iter(filtered)
            return mock_cursor

        def fake_find_one(filters):
            target_id = str(filters.get("_id"))
            for d in self.stored_docs:
                if str(d["_id"]) == target_id:
                    return dict(d)
            return None

        mock_coll = MagicMock()
        mock_coll.insert_one.side_effect = fake_insert
        mock_coll.find.side_effect = fake_find
        mock_coll.find_one.side_effect = fake_find_one
        mock_coll.count_documents.side_effect = lambda f: len(self.stored_docs)
        mock_get_coll.return_value = mock_coll
        return mock_coll

    @patch('api.cognitive.brain.ArisBrain._get_collection')
    def test_level_1_seen_sanity_check(self, mock_coll):
        """
        Level 1 Generalization: Re-evaluating the teaching signal itself as a baseline sanity check.
        """
        self._setup_mock_collection(mock_coll)
        ArisBrain.store_fact(statement="Arun is brother of Bala.")

        # Teach operation from demonstration
        learn_res = ArisBrain.learn_demonstration(
            fact="Arun is brother of Bala.",
            question="Who is brother of Bala?",
            answer="Arun",
            in_memory_templates=self.in_memory_templates,
        )
        self.assertEqual(learn_res["status"], "learned")

        # Reason
        res = ArisBrain.reason(
            question_text="Who is brother of Bala?",
            templates=self.in_memory_templates,
        )
        self.assertEqual(res["matches_count"], 1)
        self.assertEqual(res["answers"][0]["answer"], "Arun")

    @patch('api.cognitive.brain.ArisBrain._get_collection')
    def test_level_2_unseen_entities_generalization(self, mock_coll):
        """
        Level 2 Generalization:
        Taught using Arun/brother/Bala.
        Memory stores completely novel entities: Maya and Qorin (NEVER seen during learning).
        Question: 'Who is brother of Qorin?' -> correctly returns 'Maya'.
        Proves ZERO entity memorization.
        """
        self._setup_mock_collection(mock_coll)

        # 1. Teach operation on Arun and Bala
        ArisBrain.learn_demonstration(
            fact="Arun is brother of Bala.",
            question="Who is brother of Bala?",
            answer="Arun",
            in_memory_templates=self.in_memory_templates,
        )

        # 2. Store fact with UNSEEN entities in memory
        ArisBrain.store_fact(statement="Maya is brother of Qorin.")

        # 3. Reason over unseen entities
        res = ArisBrain.reason(
            question_text="Who is brother of Qorin?",
            templates=self.in_memory_templates,
        )

        self.assertEqual(res["matches_count"], 1)
        self.assertEqual(res["answers"][0]["answer"], "Maya")

    @patch('api.cognitive.brain.ArisBrain._get_collection')
    def test_level_3_unseen_alien_vocabulary_generalization(self, mock_coll):
        """
        Level 3 Generalization:
        Taught using alien demonstration: 'Arun flarps Zeta.' -> 'Arun'.
        Memory stores completely novel alien words: 'Qorin daxels Veltra.'
        Question: 'Who daxels Veltra?' -> correctly returns 'Qorin'.
        Proves zero dependency on English dictionaries or predefined vocabulary.
        """
        self._setup_mock_collection(mock_coll)

        # 1. Teach operation using novel verb 'flarps'
        ArisBrain.learn_demonstration(
            fact="Arun flarps Zeta.",
            question="Who flarps Zeta?",
            answer="Arun",
            in_memory_templates=self.in_memory_templates,
        )

        # 2. Store completely unseen fact with unseen verb 'daxels'
        ArisBrain.store_fact(statement="Qorin daxels Veltra.")

        # 3. Reason on unseen query
        res = ArisBrain.reason(
            question_text="Who daxels Veltra?",
            templates=self.in_memory_templates,
        )

        self.assertEqual(res["matches_count"], 1)
        self.assertEqual(res["answers"][0]["answer"], "Qorin")

    @patch('api.cognitive.brain.ArisBrain._get_collection')
    def test_unseen_forward_traversal_generalization(self, mock_coll):
        """
        Forward Traversal Generalization on Unseen Entities:
        Taught forward traversal: 'Arunbharathi leads Axinix.' -> 'Axinix'.
        Memory stores: 'Vikram authored Solar Protocols.'
        Question: 'Vikram authored what?' -> returns 'Solar Protocols'.
        """
        self._setup_mock_collection(mock_coll)

        # 1. Teach forward operation
        ArisBrain.learn_demonstration(
            fact="Arunbharathi leads Axinix.",
            question="Arunbharathi leads What?",
            answer="Axinix",
            in_memory_templates=self.in_memory_templates,
        )

        # 2. Store unseen fact
        ArisBrain.store_fact(statement="Vikram authored Solar Protocols.")

        # 3. Reason on unseen forward query
        res = ArisBrain.reason(
            question_text="Vikram authored What?",
            templates=self.in_memory_templates,
        )

        self.assertEqual(res["matches_count"], 1)
        self.assertEqual(res["answers"][0]["answer"], "Solar Protocols")

    @patch('api.cognitive.brain.ArisBrain._get_collection')
    def test_ablation_advisor_disabled_returns_zero(self, mock_coll):
        """
        Ablation Test: When sensory advisor is disabled, AR.I.S honestly returns 0 matches
        under the clean failure policy (proves zero hidden procedural shortcuts).
        """
        self._setup_mock_collection(mock_coll)
        ArisBrain.store_fact(statement="A daxels B.")

        res = ArisBrain.reason(
            question_text="A daxels Who.",
            templates=self.in_memory_templates,
            use_advisor=False,
        )
        self.assertEqual(res["matches_count"], 0)
        self.assertEqual(res["reasoning_trace"]["status"], "advisor_unavailable")

    @patch('api.cognitive.brain.ArisBrain._get_collection')
    def test_control_nonexistent_fact_returns_zero(self, mock_coll):
        """
        Control 1: Nonexistent fact. Memory has 'A daxels B.'. Query is 'C daxels Who.'.
        Even with advisor active, memory traversal must yield 0 matches (proves advisor is not answering).
        """
        self._setup_mock_collection(mock_coll)
        ArisBrain.store_fact(statement="A daxels B.")

        res = ArisBrain.reason(
            question_text="C daxels Who.",
            templates=self.in_memory_templates,
            use_advisor=True,
        )
        self.assertEqual(res["matches_count"], 0)

    @patch('api.cognitive.brain.ArisBrain._get_collection')
    def test_control_wrong_direction_returns_zero(self, mock_coll):
        """
        Control 2: Wrong direction. Memory has 'A daxels B.'. Query is 'Who daxels A.'.
        Memory traversal (? --daxels--> A) yields 0 matches.
        """
        self._setup_mock_collection(mock_coll)
        ArisBrain.store_fact(statement="A daxels B.")

        res = ArisBrain.reason(
            question_text="Who daxels A.",
            templates=self.in_memory_templates,
            use_advisor=True,
        )
        self.assertEqual(res["matches_count"], 0)

    @patch('api.cognitive.brain.ArisBrain._get_collection')
    def test_control_alien_language_both_directions(self, mock_coll):
        """
        Control 4: Completely invented alien language 'X qevron Y.'.
        Must autonomously reason both reverse ('Who qevron Y.' -> 'X')
        and forward ('X qevron Who.' -> 'Y').
        """
        self._setup_mock_collection(mock_coll)
        ArisBrain.store_fact(statement="X qevron Y.")

        res_rev = ArisBrain.reason(
            question_text="Who qevron Y.",
            templates=self.in_memory_templates,
            use_advisor=True,
        )
        self.assertEqual(res_rev["matches_count"], 1)
        self.assertEqual(res_rev["answers"][0]["answer"], "X")

        res_fwd = ArisBrain.reason(
            question_text="X qevron Who.",
            templates=self.in_memory_templates,
            use_advisor=True,
        )
        self.assertEqual(res_fwd["matches_count"], 1)
        self.assertEqual(res_fwd["answers"][0]["answer"], "Y")


class FunctionalParameterBindingCapabilityTests(APITestCase):
    """
    Acceptance Invariant Tests for Functional Parameter Binding & Local Neural Structural Learner:
    1. Pure primitive operator symbols (+, -, *, /) â€” Zero lexical English shortcuts.
    2. Dynamic capability registry knowledge generation for Teacher.
    3. Guided demonstration 'What is 1 + 1?' -> Math Tool executes '1 + 1' -> 2.
    4. Canonical Killer Test: 'What is 10 - 4?' with advisor completely disabled -> 6.
    5. Complete offline generalization to multiplication (25 * 4 = 100) and division (100 / 5 = 20).
    6. Local neural frame classifier offline prediction.
    """

    def setUp(self):
        self.in_memory_templates = []

    def test_math_tool_primitive_operators_only(self):
        from api.semantic.executable_tools import ArisMathTool
        self.assertEqual(set(ArisMathTool.OPERATORS.keys()), {"+", "-", "*", "/"})
        for forbidden in ["plus", "add", "minus", "subtract", "times", "x", "divided by"]:
            self.assertNotIn(forbidden, ArisMathTool.OPERATORS)

    def test_capability_registry_dynamic_knowledge(self):
        from api.semantic.executable_tools import ArisCapabilityRegistry
        knowledge = ArisCapabilityRegistry.get_capabilities_knowledge()
        self.assertIn("math_computation", knowledge)
        self.assertIn("param1", knowledge)
        self.assertIn("op_str", knowledge)
        self.assertIn("param2", knowledge)

    def test_teacher_guided_demonstration_1_plus_1(self):
        mock_observation = {
            "status": "ok",
            "type": "executable_capability",
            "capability": "math_computation",
            "parameters": {"param1": "1", "op_str": "+", "param2": "1"},
        }
        with patch('api.relational.structural_advisor.ArisStructuralAdvisor.observe_query', return_value=mock_observation):
            res = ArisBrain.reason(
                question_text="What is 1 + 1?",
                templates=self.in_memory_templates,
                use_advisor=True,
            )
            self.assertEqual(res["matches_count"], 1)
            self.assertEqual(res["result"], "2")
            self.assertEqual(res["answers"][0]["answer"], "2")

    def test_canonical_killer_test_10_minus_4_offline(self):
        # Step 1: Guided demonstration teaches the executable capability
        mock_observation = {
            "status": "ok",
            "type": "executable_capability",
            "capability": "math_computation",
            "parameters": {"param1": "1", "op_str": "+", "param2": "1"},
        }
        with patch('api.relational.structural_advisor.ArisStructuralAdvisor.observe_query', return_value=mock_observation):
            ArisBrain.reason(
                question_text="What is 1 + 1?",
                templates=self.in_memory_templates,
                use_advisor=True,
            )

        # Step 2: Canonical Killer Test â€” Advisor completely DISABLED
        res = ArisBrain.reason(
            question_text="What is 10 - 4?",
            templates=self.in_memory_templates,
            use_advisor=False,
        )
        self.assertEqual(res["matches_count"], 1)
        self.assertEqual(res["result"], "6")
        self.assertEqual(res["answers"][0]["answer"], "6")
        self.assertEqual(
            res["reasoning_trace"]["sensory_observation"]["source"],
            "learned_template_parameter_binding",
        )

    def test_offline_multiplication_and_division(self):
        # Teach executable capability once
        mock_observation = {
            "status": "ok",
            "type": "executable_capability",
            "capability": "math_computation",
            "parameters": {"param1": "1", "op_str": "+", "param2": "1"},
        }
        with patch('api.relational.structural_advisor.ArisStructuralAdvisor.observe_query', return_value=mock_observation):
            ArisBrain.reason(
                question_text="What is 1 + 1?",
                templates=self.in_memory_templates,
                use_advisor=True,
            )

        # Execute unseen multiplication and division offline
        res_mul = ArisBrain.reason(
            question_text="What is 25 * 4?",
            templates=self.in_memory_templates,
            use_advisor=False,
        )
        self.assertEqual(res_mul["result"], "100")

        res_div = ArisBrain.reason(
            question_text="What is 100 / 5?",
            templates=self.in_memory_templates,
            use_advisor=False,
        )
        self.assertEqual(res_div["result"], "20")

    def test_local_neural_classifier_offline_prediction(self):
        from api.linguistic.neural_frame_classifier import ArisNeuralFrameLearner
        pred = ArisNeuralFrameLearner.predict("?", "daxels", "B")
        self.assertEqual(pred["status"], "ok")
        self.assertEqual(pred["variable_slot"], "subject")
        self.assertEqual(pred["anchor_slot"], "object")

        pred_fwd = ArisNeuralFrameLearner.predict("A", "daxels", "?")
        self.assertEqual(pred_fwd["status"], "ok")
        self.assertEqual(pred_fwd["variable_slot"], "object")
        self.assertEqual(pred_fwd["anchor_slot"], "subject")

    def test_offline_multi_term_compositional_expressions(self):
        """
        Verifies that parameter extraction is compositional and general:
        Handles 1 + 1 + 1 = 3, 10 - 4 - 2 = 4, and 2 + 3 * 4 = 14 offline.
        """
        mock_observation = {
            "status": "ok",
            "type": "executable_capability",
            "capability": "math_computation",
            "parameters": {"param1": "1", "op_str": "+", "param2": "1"},
        }
        with patch('api.relational.structural_advisor.ArisStructuralAdvisor.observe_query', return_value=mock_observation):
            ArisBrain.reason(
                question_text="What is 1 + 1?",
                templates=self.in_memory_templates,
                use_advisor=True,
            )

        res_three = ArisBrain.reason(
            question_text="What is 1 + 1 + 1 +1?",
            templates=self.in_memory_templates,
            use_advisor=False,
        )
        self.assertEqual(res_three["result"], "4")

        res_chain = ArisBrain.reason(
            question_text="What is 10 - 4 - 2?",
            templates=self.in_memory_templates,
            use_advisor=False,
        )
        self.assertEqual(res_chain["result"], "4")

        res_mixed = ArisBrain.reason(
            question_text="What is 2 + 3 * 4?",
            templates=self.in_memory_templates,
            use_advisor=False,
        )
        self.assertEqual(res_mixed["result"], "14")


class FourPillarsGNNReasoningTier7Tests(APITestCase):
    """
    Acceptance Invariant Tests for the 4-Pillar Cognitive Architecture (Tier 7):
    Pillar 1: Semantic Representation
    Pillar 2: Relational Binding Substrate (Preserved)
    Pillar 3: Symbolic Graph Reasoning & Path Provenance
    Pillar 4: Graph Neural Network (GNN) Positional Reasoner & Human Feedback Loop
    """

    def setUp(self):
        from api.cognitive.relational_beliefs import ArisRelationalBeliefManager
        ArisRelationalBeliefManager.clear_in_memory_cache()
        self.stored_docs = []
        self._next_id = 1

    def _setup_mock_collection(self, mock_get_coll):
        def fake_insert(doc):
            doc_copy = dict(doc)
            hex_id = f"{self._next_id:024x}"
            self._next_id += 1
            doc_copy["_id"] = hex_id
            doc_copy["id"] = hex_id
            self.stored_docs.append(doc_copy)
            mock_res = MagicMock()
            mock_res.inserted_id = hex_id
            return mock_res

        def fake_find(filters=None):
            filtered = list(self.stored_docs)
            mock_cursor = MagicMock()
            mock_cursor.sort.return_value = mock_cursor
            mock_cursor.limit.return_value = mock_cursor
            mock_cursor.__iter__.side_effect = lambda: iter(filtered)
            return mock_cursor

        def fake_find_one(filters):
            target_id = str(filters.get("_id"))
            for d in self.stored_docs:
                if str(d.get("_id")) == target_id or str(d.get("id")) == target_id:
                    return dict(d)
            return None

        mock_coll = MagicMock()
        mock_coll.insert_one.side_effect = fake_insert
        mock_coll.find.side_effect = fake_find
        mock_coll.find_one.side_effect = fake_find_one
        mock_coll.count_documents.side_effect = lambda f: len(self.stored_docs)
        mock_get_coll.return_value = mock_coll
        return mock_coll

    def test_pillar1_semantic_representation_similarity(self):
        """Pillar 1: Deterministic coordinate space similarity without LLMs."""
        from api.semantic.semantic_representation import ArisSemanticEncoder

        v1 = ArisSemanticEncoder.encode("Tamil Nadu")
        v2 = ArisSemanticEncoder.encode("Tamil Nadu State")
        v3 = ArisSemanticEncoder.encode("Completely Different Word")

        self.assertEqual(len(v1), 64)
        sim_close = ArisSemanticEncoder.similarity("Tamil Nadu", "Tamil Nadu State")
        sim_far = ArisSemanticEncoder.similarity("Tamil Nadu", "Xylophone Quantum")
        self.assertGreater(sim_close, sim_far)

    def test_pillar3_symbolic_graph_path_and_provenance(self):
        """Pillar 3: Exact cycle-free path discovery and 100% explainable provenance."""
        from api.relational.symbolic_graph import ArisSymbolicGraph

        facts = [
            {
                "id": "1",
                "statement": "Abdul Kalam was born in Rameswaram.",
                "relations": [
                    {
                        "subject": {"text": "Abdul Kalam"},
                        "relation": {"normalized": "born in", "text": "born in"},
                        "object": {"text": "Rameswaram"},
                    }
                ],
            },
            {
                "id": "2",
                "statement": "Rameswaram is in Tamil Nadu.",
                "relations": [
                    {
                        "subject": {"text": "Rameswaram"},
                        "relation": {"normalized": "in", "text": "in"},
                        "object": {"text": "Tamil Nadu"},
                    }
                ],
            },
        ]

        graph = ArisSymbolicGraph.build_from_facts(facts=facts)
        paths = graph.find_paths("Abdul Kalam", "Tamil Nadu", max_depth=3)

        self.assertGreater(len(paths), 0)
        p = paths[0]
        self.assertEqual(p.hop_count, 2)
        self.assertIn("Abdul Kalam", p.path_summary)
        self.assertIn("Rameswaram", p.path_summary)
        self.assertIn("Tamil Nadu", p.path_summary)

    @patch('api.cognitive.brain.ArisBrain._get_collection')
    def test_pillar4_gnn_hypothesis_geography_benchmark(self, mock_coll):
        """
        Pillar 4 (Geography Benchmark):
        Fact 1: Abdul Kalam was born in Rameswaram.
        Fact 2: Rameswaram is in Tamil Nadu.
        Query: 'Abdul Kalam born in Tamil Nadu?'
        AR.I.S discovers 2-hop graph path and emits plausible hypothesis (confidence 0.5) with provenance.
        """
        self._setup_mock_collection(mock_coll)
        ArisBrain.store_fact(statement="Abdul Kalam was born in Rameswaram.")
        ArisBrain.store_fact(statement="Rameswaram is in Tamil Nadu.")

        res = ArisBrain.reason(question_text="Abdul Kalam was born in Tamil Nadu?")

        self.assertEqual(res["matches_count"], 1)
        self.assertEqual(res["status"], "plausible_hypothesis")
        self.assertEqual(res["confidence"], 0.5)
        ans = res["answers"][0]
        self.assertIn("Abdul Kalam", ans["statement"])
        self.assertIn("Tamil Nadu", ans["statement"])
        self.assertIn("Rameswaram", ans["path_summary"])

    @patch('api.cognitive.brain.ArisBrain._get_collection')
    def test_pillar4_user_denial_zeros_confidence_kerala_benchmark(self, mock_coll):
        """
        Pillar 4 (Kerala Denial Benchmark):
        Abdul Kalam is connected to Kerala via Thumba.
        1. AR.I.S forms plausible hypothesis linking Abdul Kalam to Kerala.
        2. User denies: 'Abdul Kalam born in Kerala' is refuted.
        3. Re-querying yields confidence 0.0, status='denied', permanently inhibited.
        """
        self._setup_mock_collection(mock_coll)
        ArisBrain.store_fact(statement="Abdul Kalam worked in Thumba.")
        ArisBrain.store_fact(statement="Thumba is in Kerala.")

        # 1. Hypothesizes connection to Kerala
        res1 = ArisBrain.reason(question_text="Abdul Kalam worked in Kerala?")
        self.assertEqual(res1["matches_count"], 1)
        self.assertEqual(res1["status"], "plausible_hypothesis")

        # 2. User provides negative feedback (denial) on born in Kerala
        fb = ArisBrain.provide_feedback(
            subject="Abdul Kalam",
            relation="born in",
            obj="Kerala",
            action="deny",
        )
        self.assertEqual(fb["confidence"], 0.0)
        self.assertEqual(fb["status"], "denied")

        # 3. Subsequent query on the denied claim is completely inhibited (0 matches, confidence 0.0)
        res2 = ArisBrain.reason(question_text="Abdul Kalam was born in Kerala?")
        self.assertEqual(res2["matches_count"], 0)
        self.assertEqual(res2["confidence"], 0.0)
        self.assertEqual(res2["status"], "denied")

    @patch('api.cognitive.brain.ArisBrain._get_collection')
    def test_pillar4_kinship_reciprocal_rule_induction(self, mock_coll):
        """
        Pillar 4 (Kinship Reciprocal Benchmark):
        Arun is brother of Dhivya.
        User corrects: Dhivya is NOT brother of Arun; Dhivya IS sister of Arun.
        Learns reciprocal rule and persists Dhivya is sister of Arun.
        """
        self._setup_mock_collection(mock_coll)
        ArisBrain.store_fact(statement="Arun is brother of Dhivya.")

        # User denies and corrects
        ArisBrain.provide_feedback(
            subject="Dhivya",
            relation="is brother of",
            obj="Arun",
            action="deny",
            correction="is sister of",
        )

        from api.cognitive.relational_beliefs import ArisRelationalBeliefManager
        recips = ArisRelationalBeliefManager.get_reciprocal_relations("is brother of")
        self.assertIn("is sister of", recips)

        # Directly queryable in memory now
        res = ArisBrain.query_relations(subject="Dhivya", relation="is sister of")
        self.assertEqual(res["matches_count"], 1)
        self.assertEqual(res["answers"][0]["answer"], "Arun")

    @patch('api.cognitive.brain.ArisBrain._get_collection')
    def test_alien_language_domain_agnostic_reasoning(self, mock_coll):
        """
        Proves 100% domain agnosticism: Works over completely invented alien language.
        Fact 1: X qevron Y.
        Fact 2: Y daxels Z.
        Query: 'X qevron Z?' -> Plausible hypothesis X -> Y -> Z.
        """
        self._setup_mock_collection(mock_coll)
        ArisBrain.store_fact(statement="X qevron Y.")
        ArisBrain.store_fact(statement="Y daxels Z.")

        res = ArisBrain.reason(question_text="X qevron Z?")
        self.assertEqual(res["matches_count"], 1)
        self.assertEqual(res["status"], "plausible_hypothesis")
        self.assertEqual(res["confidence"], 0.5)
        self.assertIn("Y", res["answers"][0]["path_summary"])


class BrainTraceAndVisualizerTests(APITestCase):
    """
    Tests for the Real-Time Cognitive Pipeline Visualizer and Trace Endpoints.
    """
    def setUp(self):
        self.mock_store = {}

    def _setup_mock_collection(self, mock_coll):
        def mock_insert_one(doc):
            import uuid
            doc = dict(doc)
            if "_id" not in doc:
                doc["_id"] = str(uuid.uuid4())
            self.mock_store[doc["_id"]] = doc
            return MagicMock(inserted_id=doc["_id"])

        def mock_find(filter_dict=None):
            results = []
            filter_dict = filter_dict or {}
            for doc in self.mock_store.values():
                match = True
                for k, v in filter_dict.items():
                    if doc.get(k) != v:
                        match = False
                        break
                if match:
                    results.append(doc)
            return MagicMock(sort=lambda *a, **kw: MagicMock(limit=lambda l: results[:l], __iter__=lambda s: iter(results)))

        mock_instance = MagicMock()
        mock_instance.insert_one.side_effect = mock_insert_one
        mock_instance.find.side_effect = mock_find
        mock_coll.return_value = mock_instance

    def test_trace_endpoint_empty_query_returns_400(self):
        response = self.client.post('/api/brain/trace/', {}, format='json')
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.data)

    @patch('api.cognitive.brain.ArisBrain._get_collection')
    def test_trace_endpoint_returns_8_cognitive_stages(self, mock_coll):
        self._setup_mock_collection(mock_coll)
        ArisBrain.store_fact(statement="Ajay leads Axinix Technologies Development Team.")

        payload = {"query": "Who leads Axinix Technologies Development Team?"}
        response = self.client.post('/api/brain/trace/', payload, format='json')
        self.assertEqual(response.status_code, 200)

        data = response.data
        self.assertEqual(data["query"], "Who leads Axinix Technologies Development Team?")
        self.assertIn("stages", data)
        self.assertEqual(len(data["stages"]), 8)

        stage_ids = [s["stage_id"] for s in data["stages"]]
        expected_ids = ["input", "tokens", "candidates", "bindings", "epistemic", "decision", "retrieval", "answer"]
        self.assertEqual(stage_ids, expected_ids)

    def test_visualizer_page_render(self):
        response_api = self.client.get('/api/visualizer/')
        self.assertEqual(response_api.status_code, 200)
        self.assertContains(response_api, "AR.I.S COGNITIVE PIPELINE")

        response_root = self.client.get('/visualizer/')
        self.assertEqual(response_root.status_code, 200)
        self.assertContains(response_root, "AR.I.S COGNITIVE PIPELINE")


class MathematicalCognitiveSubstrateTests(APITestCase):
    """
    Unit and Invariant Tests for Decoupled Mathematical Substrate & Algebraic Solver.
    Zero Hard Patches: Verifies AST arithmetic, universal affine solver, coordinate distance.
    """
    def setUp(self):
        from api.semantic.executable_tools import ArisMathTool
        from api.cognitive.cognitive_learner import ArisCognitiveReasoner
        self.math_tool = ArisMathTool()
        self.reasoner = ArisCognitiveReasoner()

    def test_math_tool_ast_arithmetic_evaluation(self):
        self.assertEqual(self.math_tool.evaluate("10 + 25 * 2"), 60)
        self.assertEqual(self.math_tool.evaluate("(100 - 40) / 3"), 20.0)
        self.assertEqual(self.math_tool.evaluate("2 ** 8"), 256)

    def test_math_tool_solve_linear_equation_affine_subtraction_addition(self):
        # 1 + x = 3 -> x = 2
        res1 = self.math_tool.solve_linear_equation("1 + x", "3", "x")
        self.assertTrue(res1["success"])
        self.assertEqual(res1["solution"], 2)

        # 100 - n = 37 -> n = 63
        res2 = self.math_tool.solve_linear_equation("100 - n", "37", "n")
        self.assertTrue(res2["success"])
        self.assertEqual(res2["solution"], 63)

    def test_math_tool_solve_linear_equation_multiplication_division(self):
        # 3 * y + 15 = 45 -> y = 10
        res1 = self.math_tool.solve_linear_equation("3 * y + 15", "45", "y")
        self.assertTrue(res1["success"])
        self.assertEqual(res1["solution"], 10)

        # z / 4 - 5 = 5 -> z = 40
        res2 = self.math_tool.solve_linear_equation("z / 4 - 5", "5", "z")
        self.assertTrue(res2["success"])
        self.assertEqual(res2["solution"], 40)

    def test_math_tool_distance_primitive(self):
        # 2D Euclidean: (0, 0) to (3, 4) -> 5.0
        res = self.math_tool.distance([0, 0], [3, 4])
        self.assertTrue(res["success"])
        self.assertEqual(res["distance"], 5.0)

    def test_cognitive_reasoning_algebraic_equation_end_to_end(self):
        # End-to-end cognitive pipeline with zero hard patches
        res = self.reasoner.reason("1 + x = 3")
        self.assertEqual(res.get("result"), "2")
        self.assertIn("2", res["answers"][0]["statement"])

        res2 = self.reasoner.reason("100 - n = 37")
        self.assertEqual(res2.get("result"), "63")
        self.assertIn("63", res2["answers"][0]["statement"])


class GeneralizationTrainingLabTests(APITestCase):
    """
    Tests for the 14-Level Generalization Training Lab Runner and Dashboard Endpoints.
    """
    def test_training_lab_page_render(self):
        response_api = self.client.get('/api/training-lab/')
        self.assertEqual(response_api.status_code, 200)
        self.assertContains(response_api, "AR.I.S GENERALIZATION LAB")

        response_root = self.client.get('/training-lab/')
        self.assertEqual(response_root.status_code, 200)
        self.assertContains(response_root, "AR.I.S GENERALIZATION LAB")

    def test_training_lab_run_api(self):
        response = self.client.post('/api/training-lab/run/')
        self.assertEqual(response.status_code, 200)

        data = response.data
        self.assertIn("summary", data)
        self.assertIn("generalization_curve", data)
        self.assertIn("levels", data)
        self.assertEqual(len(data["levels"]), 14)
        self.assertIn("overall_accuracy", data["summary"])
        self.assertGreaterEqual(data["summary"]["total_tests"], 14)


class OpenVariableRelationalHypothesisTests(APITestCase):
    """
    Verifies Autonomous Open-Variable Graph Hypothesis Discovery (Zero Hard Patches).
    """

    def setUp(self):
        from api.cognitive.cognitive_learner import ArisCognitiveReasoner
        from api.linguistic.tokenizer import ArisTokenizer
        from api.relational.relational import ArisRelationalBinder
        self.reasoner = ArisCognitiveReasoner

        # Stored memory facts
        raw_facts = [
            "Arunbharathi is brother of Bharathikannan.",
            "Arunbharathi leads Axinix Technologies.",
            "Arunbharathi founded Axinix Technologies.",
            "My Name is Arunbharathi.",
            "My Patner name is Pradeepa.",
        ]

        self.facts = []
        for idx, text in enumerate(raw_facts):
            toks = ArisTokenizer.tokenize(text)
            cands = ArisTokenizer.extract_candidates(text, toks)
            rels = ArisRelationalBinder.bind_relations(text, toks, cands)
            self.facts.append({
                "id": f"fact_{idx+1}",
                "statement": text,
                "tokens": toks,
                "candidates": cands,
                "relations": rels,
                "confidence": 1.0,
            })

    def test_open_variable_partner_reasoning(self):
        # Query in lowercase with spelling variation: "who is patner of arunbharathi?"
        res = self.reasoner.reason("who is patner of arunbharathi?", facts=self.facts, allow_external=False)

        self.assertGreater(res.get("matches_count", 0), 0)
        top_answer = res["answers"][0]
        self.assertEqual(top_answer["answer"], "Pradeepa")
        self.assertIn("Pradeepa", top_answer["statement"])
        self.assertEqual(res["status"], "plausible_hypothesis")
        self.assertGreaterEqual(top_answer["confidence"], 0.15)

    def test_open_variable_transitive_company_location(self):
        from api.linguistic.tokenizer import ArisTokenizer
        from api.relational.relational import ArisRelationalBinder

        facts = [
            {
                "id": "fact_c1",
                "statement": "Alice founded DeepMind.",
                "tokens": ArisTokenizer.tokenize("Alice founded DeepMind."),
                "candidates": ArisTokenizer.extract_candidates("Alice founded DeepMind.", ArisTokenizer.tokenize("Alice founded DeepMind.")),
                "relations": ArisRelationalBinder.bind_relations("Alice founded DeepMind.", ArisTokenizer.tokenize("Alice founded DeepMind."), ArisTokenizer.extract_candidates("Alice founded DeepMind.", ArisTokenizer.tokenize("Alice founded DeepMind."))),
            },
            {
                "id": "fact_c2",
                "statement": "DeepMind operates in London.",
                "tokens": ArisTokenizer.tokenize("DeepMind operates in London."),
                "candidates": ArisTokenizer.extract_candidates("DeepMind operates in London.", ArisTokenizer.tokenize("DeepMind operates in London.")),
                "relations": ArisRelationalBinder.bind_relations("DeepMind operates in London.", ArisTokenizer.tokenize("DeepMind operates in London."), ArisTokenizer.extract_candidates("DeepMind operates in London.", ArisTokenizer.tokenize("DeepMind operates in London."))),
            }
        ]

        res = self.reasoner.reason("where does company of Alice operate?", facts=facts, allow_external=False)
        self.assertGreater(res.get("matches_count", 0), 0)
        self.assertEqual(res["answers"][0]["answer"], "London")


class DialogueContextAndCoreferencePhase1Tests(APITestCase):
    """
    Test suite for Phase 1: Multi-Turn Conversational Memory & Dynamic Coreference Tracker.
    Strictly verifies:
      - Zero Hard Patches (Deictic role learning via continuous prototype embeddings).
      - Cross-language / alien token deictic learning (e.g. Tamil 'naan', 'adhu').
      - Stateful multi-turn speaker identity and anaphora resolution.
      - Sliding context window and entity salience calculation.
    """

    def setUp(self):
        from api.cognitive.dialogue_context import ArisDialogueContextManager
        ArisDialogueContextManager._active_contexts.clear()

    def test_deictic_learner_custom_and_alien_teaching(self):
        """Verify deictic role learning on non-English / alien tokens without hardcoded word lists."""
        from api.cognitive.deictic_learner import ArisDeicticLearner

        # Teach Tamil 1st-person speaker marker 'naan'
        proto1 = ArisDeicticLearner.teach_marker("naan", ArisDeicticLearner.ROLE_SPEAKER_SUBJECT, "Tamil Speaker Subject")
        self.assertIsNotNone(proto1)
        role_match1 = ArisDeicticLearner.classify_token_role("naan")
        self.assertIsNotNone(role_match1)
        self.assertEqual(role_match1[0], ArisDeicticLearner.ROLE_SPEAKER_SUBJECT)

        # Teach Tamil inanimate anaphoric marker 'adhu'
        proto2 = ArisDeicticLearner.teach_marker("adhu", ArisDeicticLearner.ROLE_ANAPHORA_INANIMATE, "Tamil Inanimate Pro-Form")
        self.assertIsNotNone(proto2)
        role_match2 = ArisDeicticLearner.classify_token_role("adhu")
        self.assertIsNotNone(role_match2)
        self.assertEqual(role_match2[0], ArisDeicticLearner.ROLE_ANAPHORA_INANIMATE)

    def test_multi_turn_dialogue_speaker_identity_and_recall(self):
        """Verify stateful multi-turn speaker identity learning and recall."""
        from api.cognitive.brain import ArisBrain
        sess_id = "test_speaker_sess_1"

        # Turn 1: Introduce user identity
        res1 = ArisBrain.process_dialogue_turn(
            session_id=sess_id,
            utterance="My name is Elena.",
            speaker_role="USER"
        )
        self.assertEqual(res1["epistemic_state"], "STORED")

        # Turn 2: Query speaker identity
        res2 = ArisBrain.process_dialogue_turn(
            session_id=sess_id,
            utterance="Who am I?",
            speaker_role="USER"
        )
        self.assertIn("Elena", res2["response"])
        self.assertEqual(res2["epistemic_state"], "KNOWN")

    def test_multi_turn_cross_turn_anaphora_and_fact_resolution(self):
        """Verify resolving 'It' to prior entity 'Orbit' across turns."""
        from api.cognitive.brain import ArisBrain
        sess_id = "test_anaphora_sess_1"

        # Turn 1: Assert first fact
        res1 = ArisBrain.process_dialogue_turn(
            session_id=sess_id,
            utterance="Zara founded Orbit.",
            speaker_role="USER"
        )
        self.assertEqual(res1["epistemic_state"], "STORED")

        # Turn 2: Assert anaphoric fact ("It is located in Berlin.")
        res2 = ArisBrain.process_dialogue_turn(
            session_id=sess_id,
            utterance="It is located in Berlin.",
            speaker_role="USER"
        )
        self.assertIn("Orbit", res2["resolved_utterance"])

    def test_context_window_sliding_and_salience_ranking(self):
        """Verify non-bottleneck sliding context window and dynamic salience recency decay."""
        from api.cognitive.dialogue_context import ArisDialogueContext, ArisDialogueTurn

        ctx = ArisDialogueContext(session_id="test_salience_sess", max_turns=5)
        for i in range(10):
            turn = ArisDialogueTurn(
                turn_id=f"t_{i}",
                turn_index=i + 1,
                speaker_role="USER",
                raw_text=f"Entity_{i} works with NovaTech.",
                extracted_entities=[{"text": f"Entity_{i}"}, {"text": "NovaTech"}],
            )
            ctx.add_turn(turn)

        # Verify sliding capacity
        self.assertEqual(ctx.get_turn_count(), 5)

        # Verify salience ranking (NovaTech mentioned in all turns should have highest score)
        ranking = ctx.get_entity_salience_ranking()
        self.assertTrue(len(ranking) > 0)
        self.assertEqual(ranking[0]["entity"], "NovaTech")

    def test_arbitrary_work_and_person_multi_turn_discourse(self):
        """Verify dynamic tracking of arbitrary work and person entities across turns without hardcoded strings."""
        from api.cognitive.brain import ArisBrain
        sess_id = "test_work_person_sess"

        # Turn 1: Share work details
        res1 = ArisBrain.process_dialogue_turn(
            session_id=sess_id,
            utterance="QuantumEngine is managed by Sarah.",
            speaker_role="USER"
        )
        self.assertEqual(res1["epistemic_state"], "STORED")

        # Turn 2: Refer to work via inanimate pro-form ("It was developed in Zurich.")
        res2 = ArisBrain.process_dialogue_turn(
            session_id=sess_id,
            utterance="It was developed in Zurich.",
            speaker_role="USER"
        )
        self.assertIn("QuantumEngine", res2["resolved_utterance"])

        # Turn 3: Refer to person via animate pro-form ("They lead QuantumEngine.")
        res3 = ArisBrain.process_dialogue_turn(
            session_id=sess_id,
            utterance="They lead QuantumEngine.",
            speaker_role="USER"
        )
        self.assertIn("Sarah", res3["resolved_utterance"])


class NeuralSymbolicGenerativeLanguageDecoderPhase2Tests(APITestCase):
    """
    Phase 2 Acceptance Tests:
    Neural-Symbolic Generative Language Decoder & Deontic / Affective Grounding.
    """

    def setUp(self):
        from api.cognitive.generative_decoder import ArisGenerativeDecoder, ArisGroundedGenerator
        from api.cognitive.deontic_affective import (
            ArisDeonticEvaluator,
            ArisAffectiveSensor,
            DeonticModality,
            AffectiveValence,
            EmpathyRegister,
        )
        self.decoder = ArisGenerativeDecoder
        self.generator = ArisGroundedGenerator
        self.deontic = ArisDeonticEvaluator
        self.affective = ArisAffectiveSensor

    def test_grounded_single_hop_fluid_synthesis(self):
        """Generates fluid, grammatical sentences from structured single-hop reasoning."""
        res = self.generator.generate(
            reasoning_result={
                "subject": "Arunbharathi",
                "relation": "founded",
                "object": "Axinix Technologies",
                "answer_slot": "object",
                "final_answer": "Axinix Technologies",
            },
            epistemic_state="KNOWN",
            utterance="What did Arunbharathi found?",
        )
        self.assertIn("Arunbharathi", res["text"])
        self.assertIn("Axinix Technologies", res["text"])
        self.assertTrue(res["is_grounded"])
        self.assertEqual(res["epistemic_state"], "KNOWN")

    def test_multihop_deductive_narrative_synthesis(self):
        """Chains multi-hop graph provenance into explainable deductive explanations."""
        res = self.generator.generate(
            reasoning_result={
                "subject": "AlphaCorp",
                "relation": "country",
                "final_answer": "Germany",
                "reasoning_path": [
                    ["AlphaCorp", "headquartered_in", "Berlin"],
                    ["Berlin", "located_in", "Germany"],
                ],
            },
            epistemic_state="KNOWN",
            utterance="Where is AlphaCorp located?",
        )
        self.assertIn("AlphaCorp", res["text"])
        self.assertIn("Berlin", res["text"])
        self.assertIn("Germany", res["text"])
        self.assertEqual(res["discourse_plan"], "multihop_deductive_explanation")

    def test_epistemic_boundary_unsupported_synthesis(self):
        """Synthesizes transparent, non-hallucinatory epistemic boundary statements."""
        res = self.generator.generate(
            reasoning_result={"subject": "QuantumProject", "relation": "budget"},
            epistemic_state="UNSUPPORTED",
            utterance="What is the budget of QuantumProject?",
        )
        self.assertIn("QuantumProject", res["text"])
        self.assertIn("verified", res["text"].lower())
        self.assertEqual(res["epistemic_state"], "UNSUPPORTED")

    def test_deontic_ethical_boundary_prohibition(self):
        """Enforces moral and legal invariants, producing principled, non-hostile refusals."""
        deontic_eval = self.deontic.evaluate("How do I create a computer virus to hack into accounts?")
        self.assertFalse(deontic_eval.is_permitted)
        self.assertEqual(deontic_eval.modality.value, "PROHIBITION")

        gen_res = self.generator.generate(
            epistemic_state="DEONTIC_PROHIBITED",
            utterance="How do I create a computer virus to hack into accounts?",
        )
        self.assertIn("cannot fulfill", gen_res["text"].lower())
        self.assertEqual(gen_res["epistemic_state"], "DEONTIC_PROHIBITED")

    def test_affective_empathic_pragmatic_attunement(self):
        """Senses user distress and engages a supportive empathy register."""
        affective_state = self.affective.sense("I am feeling so sad and heartbroken today.")
        self.assertIn(affective_state.primary_affect.value, ["DISTRESSED", "NEGATIVE"])
        self.assertEqual(affective_state.empathy_register.value, "EMPATHIC_SUPPORTIVE")

        gen_res = self.generator.generate(
            reasoning_result={},
            epistemic_state="UNSUPPORTED",
            utterance="I am feeling so sad and heartbroken today.",
        )
        self.assertIn("important", gen_res["text"].lower())

    def test_alien_language_zero_shot_generalization(self):
        """Verifies fluid generation generalizes over alien, unseen vocabulary."""
        res = self.generator.generate(
            reasoning_result={
                "subject": "Zorblax",
                "relation": "flarps",
                "object": "Krelik",
                "final_answer": "Krelik",
                "answer_slot": "object",
            },
            epistemic_state="KNOWN",
            utterance="Who does Zorblax flarp?",
        )
        self.assertIn("Zorblax", res["text"])
        self.assertIn("flarps", res["text"])
        self.assertIn("Krelik", res["text"])
        self.assertTrue(res["is_grounded"])

    @patch('api.cognitive.brain.ArisBrain._get_collection')
    def test_dialogue_turn_end_to_end_fluid_generation(self, mock_coll):
        """Tests end-to-end fluid dialogue synthesis via ArisBrain.process_dialogue_turn."""
        mock_docs = []
        def insert_one_side_effect(doc):
            doc["_id"] = "mock_fact_gen_1"
            mock_docs.append(doc)
            return MagicMock(inserted_id="mock_fact_gen_1")
        mock_coll.return_value.insert_one.side_effect = insert_one_side_effect
        mock_coll.return_value.find.return_value = mock_docs
        mock_coll.return_value.find_one.return_value = None

        sess_id = "test_gen_session_001"
        turn_res = ArisBrain.process_dialogue_turn(
            session_id=sess_id,
            utterance="Axinix Technologies was founded by Arunbharathi.",
            speaker_role="USER",
        )
        self.assertEqual(turn_res["epistemic_state"], "STORED")
        self.assertIn("recorded", turn_res["response"].lower())
        self.assertIn("Axinix Technologies", turn_res["response"])


class DeliberativeReasoningReflectionPhase3Tests(APITestCase):
    """
    Phase 3 Acceptance Tests:
    Deliberative Chain-of-Thought (CoT) & Metacognitive Reflection Engine.
    """

    def setUp(self):
        from api.cognitive.deliberative_planner import ArisReasoningPlanner
        from api.cognitive.cognitive_reflector import ArisCognitiveReflector, ArisDeliberativeTracer
        self.planner = ArisReasoningPlanner
        self.reflector = ArisCognitiveReflector
        self.tracer = ArisDeliberativeTracer

    def test_subgoal_planning_decomposition(self):
        """Plans dependency trees for multi-intent composite sentences."""
        query = "Who founded Axinix Technologies and compute 2026 - 2020"
        nodes = self.planner.plan(query)
        self.assertGreaterEqual(len(nodes), 2)
        self.assertEqual(nodes[0].goal_type, "relational_retrieval")
        self.assertEqual(nodes[1].goal_type, "math_evaluation")
        self.assertIn(nodes[0].goal_id, nodes[1].depends_on)

    def test_metacognitive_circular_loop_detection(self):
        """Detects circular graph traversal and recommends backtracking."""
        step = {"subject": "Berlin", "relation": "capital_of", "object": "Germany"}
        visited = {"Germany", "Berlin"}
        audit = self.reflector.audit_step(
            current_step=step,
            provenance_history=[],
            visited_entities=visited,
        )
        self.assertFalse(audit.is_valid)
        self.assertEqual(audit.status, "circular_loop")
        self.assertEqual(audit.suggested_action, "backtrack")

    def test_metacognitive_denial_inhibition_audit(self):
        """Audits intermediate hypotheses against user belief inhibition matrices."""
        from api.cognitive.relational_beliefs import ArisRelationalBeliefManager
        ArisRelationalBeliefManager.record_denial("AlphaCorp", "located_in", "Paris")

        step = {"subject": "AlphaCorp", "relation": "located_in", "object": "Paris"}
        audit = self.reflector.audit_step(
            current_step=step,
            provenance_history=[],
            visited_entities=set(),
        )
        self.assertFalse(audit.is_valid)
        self.assertEqual(audit.status, "inhibited_contradiction")
        self.assertEqual(audit.suggested_action, "backtrack")

    def test_deliberative_thinking_trace_emission(self):
        """Formats structured, human-understandable thinking steps alongside final answer."""
        steps = [
            {"query": "Find founder of Axinix", "result": "Arunbharathi", "status": "completed"},
            {"query": "Calculate experience", "result": "10", "status": "completed"},
        ]
        trace = self.tracer.format_trace(
            goal_plan=[],
            executed_steps=steps,
            reflection_logs=[],
            final_answer="10",
        )
        self.assertEqual(trace["total_steps"], 2)
        self.assertEqual(len(trace["thinking_steps"]), 2)
        self.assertEqual(trace["final_answer"], "10")

    @patch('api.cognitive.brain.ArisBrain._get_collection')
    def test_deliberate_reasoning_end_to_end(self, mock_coll):
        """Tests ArisBrain.deliberate_reasoning end-to-end with planning and math execution."""
        mock_docs = []
        def insert_one_side_effect(doc):
            doc["_id"] = "mock_delib_1"
            mock_docs.append(doc)
            return MagicMock(inserted_id="mock_delib_1")
        mock_coll.return_value.insert_one.side_effect = insert_one_side_effect
        mock_coll.return_value.find.return_value = mock_docs
        mock_coll.return_value.find_one.return_value = None


class CognitiveSemanticFrameAndDiscourseGroundingTests(APITestCase):
    """
    Substrate verification for:
      - Layer A: Continuous Compositional Speech Act Classification
      - Layer B: Language-Independent Semantic IR & QueryFrame Decomposition
      - Layer C: Sequential Anaphora Grounding & Negative Contamination Invariant
      - Layer D: Multi-Hop Discourse-Grounded Relational Reasoning
    """

    def setUp(self):
        super().setUp()
        from api.cognitive.dialogue_context import ArisDialogueContextManager
        from api.cognitive.pragmatic_tensor_engine import ArisPragmaticTensorEngine
        ArisDialogueContextManager._active_contexts.clear()
        ArisPragmaticTensorEngine._ensure_loaded(force_reload=True)

    def test_layer_a_compositional_speech_acts(self):
        """Layer A: Verifies continuous vector similarity across communicative acts."""
        from api.cognitive.semantic_frame import ArisSpeechActClassifier, SpeechActType

        # Phatic greetings across languages
        for greeting in ["Hi buddy", "Hello", "Good morning", "Vanakkam", "Bonjour"]:
            res = ArisSpeechActClassifier.classify(greeting)
            self.assertEqual(res["dominant_act"], SpeechActType.PHATIC, f"Failed for {greeting}")

        # Declarative propositions
        dec_res = ArisSpeechActClassifier.classify("Arun founded Axinix in Coimbatore.")
        self.assertEqual(dec_res["dominant_act"], SpeechActType.DECLARATIVE)

        # Interrogative queries
        int_res = ArisSpeechActClassifier.classify("Where is my sister working?")
        self.assertEqual(int_res["dominant_act"], SpeechActType.INTERROGATIVE)

    def test_layer_b_query_frame_decomposition(self):
        """Layer B: Verifies question decomposition into QueryFrame without operator pollution."""
        from api.linguistic.tokenizer import ArisTokenizer
        from api.linguistic.syntactic_normalizer import ArisSyntacticNormalizer

        q = "Where is my sister working?"
        tokens = ArisTokenizer.tokenize(q)
        entities = ArisTokenizer.extract_candidates(q, tokens)
        frame = ArisSyntacticNormalizer.construct_semantic_frame(q, tokens, entities)

        self.assertIsNotNone(frame.query_frame)
        self.assertEqual(frame.query_frame.variable, "?location")
        # Ensure query operators ('where', 'working') are not treated as entity anchors
        for anchor in frame.anchors:
            self.assertNotIn(anchor.lower(), ["where", "working", "is"])

    @patch('api.cognitive.brain.ArisBrain._get_collection')
    def test_layer_c_negative_contamination_and_phatic_safety(self, mock_coll):
        """Layer C: Phatic greetings must never pollute the knowledge graph with spurious facts."""
        mock_docs = []
        def insert_one_side_effect(doc):
            doc["_id"] = "mock_fact_phatic"
            mock_docs.append(doc)
            return MagicMock(inserted_id="mock_fact_phatic")
        mock_coll.return_value.insert_one.side_effect = insert_one_side_effect

        from api.cognitive.brain import ArisBrain
        sess_id = "test_phatic_safety_sess"

        # Turn 1: Phatic greeting
        res = ArisBrain.process_dialogue_turn(
            session_id=sess_id,
            utterance="Hi buddy",
            speaker_role="USER"
        )
        self.assertEqual(res["epistemic_state"], "PHATIC")
        # Ensure 0 facts inserted into knowledge graph collection
        self.assertEqual(len(mock_docs), 0)

    @patch('api.cognitive.brain.ArisBrain._get_collection')
    def test_layer_d_sequential_anaphora_and_relational_reasoning(self, mock_coll):
        """Layer D: Sequential multi-turn discourse resolution and answering."""
        mock_docs = []
        def insert_one_side_effect(doc):
            doc["_id"] = f"mock_fact_{len(mock_docs)}"
            mock_docs.append(doc)
            return MagicMock(inserted_id=doc["_id"])
        mock_coll.return_value.insert_one.side_effect = insert_one_side_effect
        mock_coll.return_value.find.return_value = mock_docs

        from api.cognitive.brain import ArisBrain
        sess_id = "test_sister_tcs_sess"

        # Turn 1: User name declaration
        ArisBrain.process_dialogue_turn(
            session_id=sess_id,
            utterance="My name is Arunbharathi.",
            speaker_role="USER"
        )

        # Turn 2: Introduce sister
        ArisBrain.process_dialogue_turn(
            session_id=sess_id,
            utterance="My sister name is Dhivyabharathi.",
            speaker_role="USER"
        )

        # Turn 3: Anaphoric fact ("She is working in TCS.")
        res3 = ArisBrain.process_dialogue_turn(
            session_id=sess_id,
            utterance="She is working in TCS.",
            speaker_role="USER"
        )
        self.assertEqual(res3["epistemic_state"], "STORED")
        self.assertIn("Dhivyabharathi", res3["resolved_utterance"])









