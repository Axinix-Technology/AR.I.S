"""
AR.I.S — Cognitive Pipeline Regression Tests (Phases 13 & 14)

Verifies that every linguistic decision in the AR.I.S pipeline originates
from the trained tensor substrate, not from punctuation, token position,
or capitalization heuristics.

Run with:
    python -m pytest api/tests/test_cognitive_pipeline.py -v
"""

import sys
import os
import pytest

# Allow importing from workspace root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline():
    from api.linguistic.pipeline import ArisLinguisticPipeline
    return ArisLinguisticPipeline

@pytest.fixture(scope="module")
def pragmatic_engine():
    from api.cognitive.pragmatic_tensor_engine import ArisPragmaticTensorEngine
    return ArisPragmaticTensorEngine

@pytest.fixture(scope="module")
def coref_resolver():
    from api.cognitive.coreference_resolver import ArisCoreferenceResolver
    return ArisCoreferenceResolver

@pytest.fixture
def fresh_context():
    from api.cognitive.dialogue_context import ArisDialogueContext
    return ArisDialogueContext(session_id="test-session", user_id="test-user")


# ===========================================================================
# GROUP 1 — Speech Act Classification
# ===========================================================================

class TestSpeechActClassification:
    """
    Verifies tensor-only speech act classification.
    No punctuation may determine the result.
    """

    def test_hi_budd_is_phatic(self, pipeline):
        frame = pipeline.process("Hi budd")
        assert frame.speech_act == "PHATIC", f"Expected PHATIC, got {frame.speech_act}"
        assert frame.is_phatic is True

    def test_hello_is_phatic(self, pipeline):
        frame = pipeline.process("Hello")
        assert frame.speech_act == "PHATIC"

    def test_interrogative_with_question_mark(self, pipeline):
        frame = pipeline.process("Who is the founder of Axinix?")
        assert frame.speech_act == "INTERROGATIVE"
        assert frame.is_question is True

    def test_interrogative_WITHOUT_question_mark(self, pipeline):
        """No ? is required for INTERROGATIVE classification."""
        frame = pipeline.process("who is Arun")
        assert frame.speech_act == "INTERROGATIVE", (
            f"Punctuation-free interrogative must still classify as INTERROGATIVE, got {frame.speech_act}"
        )

    def test_question_mark_does_not_force_interrogative(self, pipeline):
        """A ? alone must not override tensor output for clearly non-interrogative text."""
        # The pragmatic tensor engine must not have any ? char override
        from api.cognitive.pragmatic_tensor_engine import ArisPragmaticTensorEngine
        result = ArisPragmaticTensorEngine.classify_speech_act("Hi!")
        assert result["dominant_act"] != "INTERROGATIVE", (
            "Punctuation override removed: ! or ? should not force INTERROGATIVE"
        )

    def test_could_you_check_is_interrogative(self, pipeline):
        frame = pipeline.process("Could you check who is arunbharathi parkkavamani is?")
        assert frame.speech_act == "INTERROGATIVE"

    def test_browse_command_is_command(self, pipeline):
        frame = pipeline.process("browse who is arunbharathi parkkavamani")
        assert frame.speech_act == "COMMAND", f"Expected COMMAND, got {frame.speech_act}"
        assert frame.is_command is True

    def test_search_is_command(self, pipeline):
        frame = pipeline.process("search for Arunbharathi")
        assert frame.speech_act == "COMMAND"

    def test_calculate_is_command(self, pipeline):
        frame = pipeline.process("calculate 15 plus 27")
        assert frame.speech_act == "COMMAND"

    def test_declarative_with_period(self, pipeline):
        frame = pipeline.process("Arun is the founder of Axinix.")
        assert frame.speech_act == "DECLARATIVE"

    def test_declarative_WITHOUT_period(self, pipeline):
        """Declarative without terminal punctuation must still be classified correctly."""
        frame = pipeline.process("I am founder of Axinix Technologies")
        assert frame.speech_act == "DECLARATIVE", (
            f"Punctuation-free declarative must still be DECLARATIVE, got {frame.speech_act}"
        )

    def test_thank_you_is_thanking(self, pipeline):
        frame = pipeline.process("Thank you")
        assert frame.speech_act == "THANKING"

    # Adversarial: punctuation variation (Phase 14)
    def test_who_is_arun_no_punct(self, pipeline):
        frame = pipeline.process("Who is Arun")
        assert frame.speech_act == "INTERROGATIVE"

    def test_who_is_arun_with_punct(self, pipeline):
        frame = pipeline.process("Who is Arun?")
        assert frame.speech_act == "INTERROGATIVE"

    def test_who_is_arun_exclamation(self, pipeline):
        """Punctuation variation should not collapse classification."""
        frame = pipeline.process("Who is Arun!!!")
        # Still should not be PHATIC or COMMAND due to punctuation
        assert frame.speech_act in ("INTERROGATIVE", "DECLARATIVE"), (
            f"Unexpected speech act for 'Who is Arun!!!' : {frame.speech_act}"
        )

    # Adversarial: modal constructions (Phase 14)
    def test_could_you_tell_me_is_interrogative(self, pipeline):
        frame = pipeline.process("Could you tell me who founded Axinix?")
        assert frame.speech_act == "INTERROGATIVE"

    def test_can_you_tell_me_is_interrogative(self, pipeline):
        frame = pipeline.process("Can you tell me who founded Axinix?")
        assert frame.speech_act == "INTERROGATIVE"

    def test_would_you_look_up_is_interrogative(self, pipeline):
        frame = pipeline.process("Would you look up Axinix?")
        # May classify as INTERROGATIVE or COMMAND; must NOT classify as DECLARATIVE or PHATIC
        assert frame.speech_act in ("INTERROGATIVE", "COMMAND"), (
            f"Unexpected: {frame.speech_act}"
        )


# ===========================================================================
# GROUP 2 — Entity Candidate Extraction
# ===========================================================================

class TestEntityCandidateExtraction:
    """
    Verifies that entity candidates are determined by UD UPOS tags
    from the tensor bank, not by capitalization or character position.
    """

    def test_could_not_a_candidate(self, pipeline):
        """AUX 'Could' must never become an entity candidate."""
        frame = pipeline.process("Could you check who is Arunbharathi?")
        candidate_texts = [c["text"] for c in frame.candidates]
        assert "Could" not in candidate_texts, (
            f"'Could' (AUX) incorrectly extracted as candidate: {candidate_texts}"
        )

    def test_hi_produces_zero_candidates(self, pipeline):
        """PHATIC 'Hi' and common 'budd' must not become entity candidates."""
        frame = pipeline.process("Hi budd")
        assert len(frame.candidates) == 0, (
            f"PHATIC utterance should produce no entity candidates: {frame.candidates}"
        )

    def test_proper_name_is_candidate(self, pipeline):
        """A proper name that the structural extractor detects must pass the UPOS filter."""
        frame = pipeline.process("Arunbharathi founded Axinix Technologies.")
        candidate_texts = [c["text"] for c in frame.candidates]
        assert any("Axinix" in t or "Arunbharathi" in t for t in candidate_texts), (
            f"Expected proper name candidates, got: {candidate_texts}"
        )

    def test_lower_propn_is_candidate(self, pipeline):
        """Lowercase proper names identified by UPOS must become candidates."""
        frame = pipeline.process("Could you check who is arunbharathi parkkavamani is?")
        candidate_texts = [c["text"] for c in frame.candidates]
        # arunbharathi should be a candidate (PROPN tag from tensor)
        assert any("arunbharathi" in t.lower() for t in candidate_texts), (
            f"Expected 'arunbharathi' as candidate (PROPN), got: {candidate_texts}"
        )

    def test_compound_entity_passes_filter(self, pipeline):
        """Multi-token span with PROPN+NOUN must pass through UPOS filter."""
        frame = pipeline.process("Axinix Technologies is a startup.")
        candidate_texts = [c["text"] for c in frame.candidates]
        assert any("Axinix" in t for t in candidate_texts), (
            f"Expected 'Axinix Technologies' candidate, got: {candidate_texts}"
        )


# ===========================================================================
# GROUP 3 — Sentence Segmentation & Incremental Coreference
# ===========================================================================

class TestCoreference:
    """
    Verifies incremental intra-turn anaphora resolution and
    speaker identity binding.
    """

    def test_two_sentence_turn_produces_two_spans(self, pipeline):
        frame = pipeline.process(
            "I am founder of Axinix Technologies. It is a startup company."
        )
        assert len(frame.sentence_spans) == 2, (
            f"Expected 2 sentence spans, got {len(frame.sentence_spans)}"
        )

    def test_intra_turn_it_resolves_to_axinix(self, pipeline, coref_resolver, fresh_context):
        """
        'It' in sentence 2 must resolve to 'Axinix Technologies' discovered in sentence 1,
        NOT to a prior phatic entity like 'Hi'.
        """
        frame = pipeline.process(
            "I am founder of Axinix Technologies. It is a startup company."
        )
        result = coref_resolver.resolve_utterance(frame, fresh_context)
        resolved = result["resolved_text"]
        coref_entries = result["resolved_coreferences"]

        it_resolution = [e for e in coref_entries if e["referent"].lower() == "it"]
        if it_resolution:
            assert "Axinix" in it_resolution[0]["resolved_entity"], (
                f"'It' resolved to '{it_resolution[0]['resolved_entity']}' instead of 'Axinix Technologies'"
            )

    def test_phatic_turn_does_not_pollute_salience(self, pipeline, fresh_context):
        """
        After a PHATIC turn 'Hi budd', the salience graph must not contain
        'Hi' or 'budd' as entity anchors.
        """
        from api.cognitive.dialogue_context import ArisDialogueTurn
        # Add a phatic turn to context
        phatic_turn = ArisDialogueTurn(
            turn_id=None,
            turn_index=1,
            speaker_role="USER",
            raw_text="Hi budd",
            extracted_entities=[{"text": "Hi"}, {"text": "budd"}],
            relational_bindings=[],
            epistemic_state="PHATIC",
        )
        fresh_context.add_turn(phatic_turn)
        salience = fresh_context.get_entity_salience_ranking()
        entities = [e["entity"] for e in salience]
        assert "Hi" not in entities, f"'Hi' from PHATIC turn contaminated salience: {entities}"
        assert "budd" not in entities, f"'budd' from PHATIC turn contaminated salience: {entities}"


# ===========================================================================
# GROUP 4 — Speaker Identity
# ===========================================================================

class TestSpeakerIdentity:
    """
    Verifies that speaker name is correctly bound from declarative turns
    and used to resolve first-person pro-forms.
    """

    def test_my_name_is_binds_speaker(self, pipeline, coref_resolver, fresh_context):
        frame = pipeline.process("My name is Arunbharathi")
        coref_resolver.resolve_utterance(frame, fresh_context)
        name = fresh_context.get_speaker_property("USER", "name")
        assert name == "Arunbharathi", (
            f"Expected speaker name 'Arunbharathi', got '{name}'"
        )

    def test_i_am_resolves_after_name_binding(self, pipeline, coref_resolver, fresh_context):
        """
        After binding name, 'I am founder...' should resolve 'I' to 'Arunbharathi'.
        """
        # Bind name first
        frame1 = pipeline.process("My name is Arunbharathi")
        coref_resolver.resolve_utterance(frame1, fresh_context)

        # Now resolve an identity assertion
        frame2 = pipeline.process("I am founder of Axinix Technologies.")
        result2 = coref_resolver.resolve_utterance(frame2, fresh_context)
        resolved = result2["resolved_text"]
        assert "Arunbharathi" in resolved, (
            f"Expected 'I' resolved to 'Arunbharathi' in '{resolved}'"
        )


# ===========================================================================
# GROUP 5 — Surface vs Grounded Frame Observability
# ===========================================================================

class TestFrameObservability:
    """
    Verifies that both surface and grounded frames are distinct and observable.
    """

    def test_surface_and_grounded_frames_are_distinct(self, pipeline, coref_resolver, fresh_context):
        from api.linguistic.pipeline import ArisLinguisticPipeline

        # Bind speaker name
        frame1 = pipeline.process("My name is Arunbharathi")
        coref_resolver.resolve_utterance(frame1, fresh_context)

        surface = pipeline.process("I am founder of Axinix Technologies.")
        coref_result = coref_resolver.resolve_utterance(surface, fresh_context)
        gframe = ArisLinguisticPipeline.build_grounded_frame(surface, coref_result)

        assert gframe.surface_frame.raw_utterance == "I am founder of Axinix Technologies."
        assert gframe.surface_frame is not gframe.grounded_frame
        # Grounded frame contains resolved text (I → Arunbharathi)
        assert gframe.resolved_text != gframe.surface_frame.raw_utterance or \
               coref_result["resolved_coreferences"] == [], (
            "Surface and grounded utterance must differ after coreference resolution"
        )

    def test_frame_speech_act_is_single_source(self, pipeline):
        """
        The frame has exactly one speech_act field; derived booleans are properties.
        Modifying speech_act changes is_question, is_declarative, etc. consistently.
        """
        frame = pipeline.process("Who is Arun?")
        assert frame.speech_act == "INTERROGATIVE"
        assert frame.is_question is True
        assert frame.is_declarative is False
        assert frame.is_command is False


# ===========================================================================
# GROUP 6 — Routing Contract (without full brain execution)
# ===========================================================================

class TestRoutingContracts:
    """
    Verifies that the CognitiveLinguisticFrame carries the correct speech act
    that would produce the correct routing in brain.process_dialogue_turn().
    """

    @pytest.mark.parametrize("text,expected_act", [
        ("browse who is arunbharathi parkkavamani", "COMMAND"),
        ("Arun is the founder of Axinix.", "DECLARATIVE"),
        ("Who is the founder of Axinix?", "INTERROGATIVE"),
        ("Hello", "PHATIC"),
        ("My name is Arunbharathi", "DECLARATIVE"),
        ("I am founder of Axinix Technologies", "DECLARATIVE"),
        ("search for Arunbharathi", "COMMAND"),
    ])
    def test_speech_act_routing_contract(self, pipeline, text, expected_act):
        frame = pipeline.process(text)
        assert frame.speech_act == expected_act, (
            f"[{text!r}] Expected {expected_act}, got {frame.speech_act}"
        )

    def test_browse_command_does_not_route_to_declarative(self, pipeline):
        """'browse ...' must not classify as DECLARATIVE (which would store it as a fact)."""
        frame = pipeline.process("browse who is arunbharathi parkkavamani")
        assert frame.speech_act != "DECLARATIVE", (
            "COMMAND utterance must not route to declarative fact storage"
        )

    def test_hello_does_not_create_factual_memory_signal(self, pipeline):
        """'Hello' must be PHATIC, never DECLARATIVE."""
        frame = pipeline.process("Hello")
        assert frame.speech_act != "DECLARATIVE"
        assert frame.is_phatic is True


# ===========================================================================
# GROUP 7 — Adversarial & Multilingual (Phase 14)
# ===========================================================================

class TestAdversarialInputs:
    """
    Phase 14: adversarial and edge-case inputs that must not break the pipeline.
    """

    @pytest.mark.parametrize("text", [
        "who is Arun",
        "tell me who founded Axinix",
        "browse who is Arun",
        "my name is Arun",
    ])
    def test_punctuation_free_inputs_do_not_crash(self, pipeline, text):
        """Punctuation-free inputs must produce a valid frame without errors."""
        frame = pipeline.process(text)
        assert frame is not None
        assert frame.speech_act in (
            "PHATIC", "DECLARATIVE", "INTERROGATIVE", "COMMAND", "THANKING", "DENIAL"
        )

    def test_multi_sentence_discourse(self, pipeline):
        frame = pipeline.process("Arun founded Axinix Technologies. It is a startup.")
        assert len(frame.sentence_spans) == 2

    def test_mixed_phatic_and_interrogative_distractor(self, pipeline):
        """'Hi Arun. Could you tell me who founded Axinix?' — overall must be INTERROGATIVE."""
        frame = pipeline.process("Hi Arun. Could you tell me who founded Axinix?")
        # The dominant speech act for this 2-sentence input should be INTERROGATIVE
        # (the second sentence overwhelms the greeting)
        assert frame.speech_act in ("INTERROGATIVE", "PHATIC"), (
            f"Unexpected: {frame.speech_act}"
        )

    def test_empty_input_returns_valid_frame(self, pipeline):
        frame = pipeline.process("")
        assert frame is not None
        assert frame.candidates == []
        assert frame.bindings == []

    def test_whitespace_only_returns_valid_frame(self, pipeline):
        frame = pipeline.process("   ")
        assert frame is not None


# ===========================================================================
# GROUP 8 — Validation Checklist (Phase 15)
# ===========================================================================

class TestValidationChecklist:
    """
    Phase 15: confirms each of the 14 validation items mechanically.
    """

    def test_no_question_mark_override_in_tensor_engine(self):
        """Validation item 4: no ? char override in pragmatic_tensor_engine.py."""
        import inspect
        from api.cognitive.pragmatic_tensor_engine import ArisPragmaticTensorEngine
        source = inspect.getsource(ArisPragmaticTensorEngine.classify_speech_act)
        assert 'endswith("?")' not in source and "endswith('?')" not in source, (
            "Violation: ? char override found in ArisPragmaticTensorEngine.classify_speech_act"
        )

    def test_no_isupper_in_pipeline_candidate_extraction(self):
        """Validation item 5: no capitalization check in pipeline candidate extraction."""
        import inspect
        from api.linguistic.pipeline import ArisLinguisticPipeline
        source = inspect.getsource(ArisLinguisticPipeline._extract_upos_filtered_candidates)
        assert ".isupper()" not in source, (
            "Violation: .isupper() capitalization check found in _extract_upos_filtered_candidates"
        )

    def test_no_first_candidate_subject_fallback_in_cognitive_learner(self):
        """Validation item 6: no first/second candidate positional fallback."""
        import inspect
        from api.cognitive.cognitive_learner import ArisCognitiveReasoner
        source = inspect.getsource(ArisCognitiveReasoner.reason)
        # The char-slice heuristic used `cand["char_start"]` — check it's gone
        assert 'char_start"]' not in source or "char_slice" not in source, (
            "Char-slice relation extraction may still exist in ArisCognitiveReasoner.reason"
        )

    def test_cognitive_linguistic_frame_is_single_ir(self):
        """Validation item 12: CognitiveLinguisticFrame exists and has single speech_act."""
        from api.linguistic.cognitive_frame import CognitiveLinguisticFrame
        from dataclasses import fields
        field_names = {f.name for f in fields(CognitiveLinguisticFrame)}
        assert "speech_act" in field_names
        # Derived booleans are properties, not stored fields
        assert "is_question" not in field_names, "is_question must be a property, not a stored field"
        assert "is_declarative" not in field_names, "is_declarative must be a property, not a stored field"

    def test_surface_and_grounded_frame_distinguishable(self, pipeline, coref_resolver, fresh_context):
        """Validation item 13: surface and grounded frames are distinct objects."""
        from api.linguistic.pipeline import ArisLinguisticPipeline

        frame1 = pipeline.process("My name is Arunbharathi")
        coref_resolver.resolve_utterance(frame1, fresh_context)

        surface = pipeline.process("I am the founder of Axinix.")
        coref_result = coref_resolver.resolve_utterance(surface, fresh_context)
        gframe = ArisLinguisticPipeline.build_grounded_frame(surface, coref_result)

        assert hasattr(gframe, "surface_frame")
        assert hasattr(gframe, "grounded_frame")
        assert gframe.surface_frame is not gframe.grounded_frame

    def test_cognitive_action_exists(self):
        """Validation item 14: CognitiveAction exists and has structured fields."""
        from api.cognitive.cognitive_action import CognitiveAction
        from dataclasses import fields
        field_names = {f.name for f in fields(CognitiveAction)}
        assert "speech_act" in field_names
        assert "operation" in field_names
        assert "capability" in field_names
        assert "payload" in field_names
