"""
Unit and regression tests for Universal Dependencies training observability,
progress reporting, ETA calculation, and stage-level diagnostics.
"""

import io
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from api.linguistic.ud_progress import UDTrainingProgressReporter
from api.linguistic.ud_ingestion_engine import ArisUDIngestionEngine, UDSentence, UDToken
from api.linguistic.ud_tensor_engine import ArisUniversalLinguisticEncoder


class TestUDTrainingObservability(unittest.TestCase):
    """Test suite for UDTrainingProgressReporter and its pipeline integration."""

    def test_duration_formatting(self):
        """Tests conversion of seconds into MM:SS and HH:MM:SS."""
        self.assertEqual(UDTrainingProgressReporter.format_duration(-5), "00:00")
        self.assertEqual(UDTrainingProgressReporter.format_duration(0), "00:00")
        self.assertEqual(UDTrainingProgressReporter.format_duration(59), "00:59")
        self.assertEqual(UDTrainingProgressReporter.format_duration(60), "01:00")
        self.assertEqual(UDTrainingProgressReporter.format_duration(3665), "01:01:05")

    def test_safe_str_encoding(self):
        """Tests console encoding fallback for non-UTF-8 encodings."""
        mock_stdout = MagicMock()
        mock_stdout.encoding = "ascii"
        reporter = UDTrainingProgressReporter(stdout=mock_stdout)

        # Standard text
        self.assertIn("Hello", reporter._safe_str("Hello"))

        # Test ASCII fallback when encoding cannot handle Unicode
        fallback = reporter._safe_str("✓ Complete ━━━━━━━━━")
        self.assertIn("[OK]", fallback)
        self.assertIn("=", fallback)

    def test_reporter_initialization(self):
        """Tests initialization state for bounded and unbounded training."""
        # Bounded
        rep_bounded = UDTrainingProgressReporter(
            languages=["English", "Tamil"],
            max_sentences_per_lang=100,
        )
        self.assertEqual(rep_bounded.num_languages, 2)
        self.assertEqual(rep_bounded.total_target_sentences, 200)
        self.assertEqual(rep_bounded.total_sentences_processed, 0)
        self.assertEqual(rep_bounded.total_tokens_processed, 0)

        # Unbounded
        rep_unbounded = UDTrainingProgressReporter(
            languages=["English"],
            max_sentences_per_lang=None,
        )
        self.assertIsNone(rep_unbounded.total_target_sentences)

    def test_stage_transitions(self):
        """Tests stage start and completion emissions."""
        out = io.StringIO()
        reporter = UDTrainingProgressReporter(stdout=out)
        reporter.is_tty = False

        reporter.on_stage_start("stage1", "Testing stage start...")
        self.assertIn("[1/2] Testing stage start...", out.getvalue())

        reporter.on_stage_complete("stage1", "Testing stage complete")
        self.assertIn("[1/2] ✓ Testing stage complete", out.getvalue())

    def test_non_tty_heartbeat_emission(self):
        """Tests single-line heartbeat emission in non-interactive / pipe mode."""
        out = io.StringIO()
        reporter = UDTrainingProgressReporter(
            languages=["English"],
            max_sentences_per_lang=100,
            stdout=out,
            heartbeat_interval=0.01,
        )
        reporter.is_tty = False

        reporter.on_language_start("English")
        reporter.last_heartbeat_time = 0
        # Trigger sentence emission
        reporter.on_sentence("English", "en_ewt", 15)

        output = out.getvalue()
        self.assertIn("English/en_ewt", output)
        self.assertIn("Sents", output)
        self.assertIn("Tokens", output)
        self.assertIn("Rate", output)

    def test_tty_in_place_card_rendering(self):
        """Tests ANSI cursor repositioning and line clearing in interactive TTY mode."""
        out = io.StringIO()
        reporter = UDTrainingProgressReporter(
            languages=["English"],
            max_sentences_per_lang=100,
            stdout=out,
            heartbeat_interval=0.01,
        )
        reporter.is_tty = True
        reporter.raw_stdout = out

        reporter.on_language_start("English")
        reporter.last_heartbeat_time = 0
        # First emission creates initial card
        reporter.on_sentence("English", "en_ewt", 20)
        first_output = out.getvalue()
        self.assertIn("Language: English", first_output)
        self.assertIn("Treebank: en_ewt", first_output)

        out.truncate(0)
        out.seek(0)

        # Second emission should prepend cursor up code \033[...A\r
        reporter.last_heartbeat_time = 0
        reporter.on_sentence("English", "en_ewt", 20)
        second_output = out.getvalue()
        self.assertIn("\033[", second_output)
        self.assertIn("A\r", second_output)
        self.assertIn("\033[K\n", second_output)

    def test_language_completion_summary(self):
        """Tests per-language completion output and state reset."""
        out = io.StringIO()
        reporter = UDTrainingProgressReporter(
            languages=["English", "Tamil"],
            max_sentences_per_lang=50,
            stdout=out,
        )
        reporter.is_tty = False

        reporter.on_language_start("English")
        reporter.on_sentence("English", "en_ewt", 100)
        reporter.on_language_end("English")

        output = out.getvalue()
        self.assertIn("Language 'English' complete", output)
        self.assertIn("sentences", output)
        self.assertIn("tokens", output)
        self.assertEqual(reporter._last_card_line_count, 0)

    def test_final_completion_summary(self):
        """Tests end-of-run statistical summary formatting."""
        out = io.StringIO()
        reporter = UDTrainingProgressReporter(
            languages=["English", "Tamil"],
            max_sentences_per_lang=100,
            stdout=out,
        )
        reporter.is_tty = False

        mock_res = {
            "status": "compiled_ud_tensor_bank",
            "saved_to": "dummy_path.pt",
            "sentences_trained": 200,
            "tokens_trained": 4500,
            "total_exemplars": 1500,
            "total_upos_exemplars": 500,
            "total_deprel_exemplars": 600,
            "total_morph_exemplars": 400,
        }

        reporter.print_summary(mock_res)
        summary_text = out.getvalue()
        self.assertIn("UD linguistic training complete", summary_text)
        self.assertIn("Languages       : 2 (English, Tamil)", summary_text)
        self.assertIn("Sentences       : 200", summary_text)
        self.assertIn("Tokens          : 4,500", summary_text)
        self.assertIn("Exemplars       : 1,500 (UPOS: 500, Deprel: 600, Morph: 400)", summary_text)
        self.assertIn("Tensor artifacts:", summary_text)

    def test_ud_supplemental_treebank_metadata(self):
        """Tests that supplemental CoNLL-U streaming attaches treebank metadata."""
        conllu_path = ArisUDIngestionEngine.SUPPLEMENTAL_CONLLU_PATH
        if os.path.exists(conllu_path):
            sentences = list(ArisUDIngestionEngine._stream_conllu_file(conllu_path))
            self.assertGreater(len(sentences), 0)
            for sent in sentences:
                self.assertEqual(sent.metadata.get("treebank"), "ud_supplemental")

    def test_train_from_treebanks_progress_reporter_integration(self):
        """Tests train_from_treebanks with an active progress reporter without corrupting production model."""
        mock_reporter = MagicMock()
        tmp_model_path = os.path.join(tempfile.gettempdir(), "test_ud_temp_tensor_bank.pt")

        try:
            # Run minimal training with reporter to isolated temporary file
            res = ArisUniversalLinguisticEncoder.train_from_treebanks(
                languages=["English"],
                max_sentences_per_lang=2,
                max_exemplars_per_tag=2,
                save_path=tmp_model_path,
                progress_reporter=mock_reporter,
            )

            # Verify lifecycle hooks were called
            mock_reporter.on_stage_start.assert_called()
            mock_reporter.on_language_start.assert_called_with("English")
            mock_reporter.on_sentence.assert_called()
            mock_reporter.on_language_end.assert_called_with("English")
            mock_reporter.on_stage_complete.assert_called()

            # Verify returned summary dict includes exemplar metrics
            self.assertIn("total_exemplars", res)
            self.assertIn("total_upos_exemplars", res)
            self.assertIn("total_deprel_exemplars", res)
            self.assertIn("total_morph_exemplars", res)
        finally:
            if os.path.exists(tmp_model_path):
                try:
                    os.remove(tmp_model_path)
                except Exception:
                    pass
            # Restore production model bank into memory
            ArisUniversalLinguisticEncoder._ensure_loaded(force_reload=True)


if __name__ == "__main__":
    unittest.main()
