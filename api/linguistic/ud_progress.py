"""
AR.I.S — Universal Dependencies Training Observability & Progress Reporter

Provides real-time diagnostic visibility into the UD v2.18 linguistic streaming
and PyTorch tensor compilation pipeline without altering any training or tensor logic.
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional


class UDTrainingProgressReporter:
    """
    Decoupled progress reporter for Universal Dependencies treebank ingestion
    and metric tensor compilation.
    """

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        max_sentences_per_lang: Optional[int] = None,
        stdout: Optional[Any] = None,
        style: Optional[Any] = None,
        heartbeat_interval: float = 5.0,
    ):
        self.languages = languages or ["English", "Tamil", "French", "Hindi", "German"]
        self.num_languages = len(self.languages)
        self.max_sentences_per_lang = max_sentences_per_lang
        self.stdout = stdout or sys.stdout
        self.style = style
        self.heartbeat_interval = heartbeat_interval

        # Global counters
        self.total_target_sentences = (
            max_sentences_per_lang * self.num_languages
            if max_sentences_per_lang is not None
            else None
        )
        self.total_sentences_processed = 0
        self.total_tokens_processed = 0
        self.current_lang_idx = 0

        # Current language state
        self.curr_lang = ""
        self.curr_treebank = "discovering"
        self.curr_lang_sentences = 0
        self.curr_lang_tokens = 0

        # Timing
        self.start_time = time.time()
        self.lang_start_time = time.time()
        self.last_heartbeat_time = self.start_time

        # Exemplars snapshot
        self.exemplar_counts: Dict[str, int] = {}

        # Terminal capabilities & in-place update tracking
        self.raw_stdout = getattr(self.stdout, "_out", self.stdout)
        self.is_tty = hasattr(self.raw_stdout, "isatty") and self.raw_stdout.isatty()
        if not self.is_tty and hasattr(sys.stdout, "isatty"):
            self.is_tty = sys.stdout.isatty()
        self._last_card_line_count = 0
        self._vt100_initialized = False

    def _ensure_vt100(self) -> None:
        """Enables VT100 Virtual Terminal Processing in Windows consoles."""
        if self._vt100_initialized:
            return
        self._vt100_initialized = True
        if sys.platform == "win32":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
                mode = ctypes.c_ulong()
                if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                    kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
            except Exception:
                pass

    def _safe_str(self, text: str) -> str:
        """Ensures characters are safe for the current console encoding."""
        encoding = getattr(self.stdout, "encoding", None) or "utf-8"
        try:
            text.encode(encoding)
            return text
        except (UnicodeEncodeError, LookupError):
            return (
                text.replace("✓", "[OK]")
                .replace("━", "=")
                .replace("─", "-")
                .replace("→", "->")
            )

    def _write(self, msg: str, style_type: Optional[str] = None) -> None:
        safe_msg = self._safe_str(msg)
        if self.style and style_type and hasattr(self.style, style_type):
            formatter = getattr(self.style, style_type)
            output = formatter(safe_msg)
        else:
            output = safe_msg
        self.stdout.write(output)
        if hasattr(self.stdout, "flush"):
            try:
                self.stdout.flush()
            except Exception:
                pass

    def _writeln(self, msg: str = "", style_type: Optional[str] = None) -> None:
        self._write(f"{msg}\n", style_type=style_type)

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Formats seconds into MM:SS or HH:MM:SS."""
        if seconds < 0:
            return "00:00"
        total_sec = int(round(seconds))
        hours = total_sec // 3600
        minutes = (total_sec % 3600) // 60
        secs = total_sec % 60
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def on_stage_start(self, stage_id: str, description: str) -> None:
        """Emits stage-level transition banner."""
        self._last_card_line_count = 0
        self._writeln(f"\n[1/2] {description}")

    def on_stage_complete(self, stage_id: str, description: str) -> None:
        """Emits stage completion banner."""
        self._last_card_line_count = 0
        self._writeln(f"[1/2] ✓ {description}", style_type="SUCCESS")

    def on_language_start(self, lang: str) -> None:
        """Invoked when starting ingestion for a new language."""
        self._last_card_line_count = 0
        self.curr_lang = lang
        self.curr_treebank = "discovering"
        self.curr_lang_sentences = 0
        self.curr_lang_tokens = 0
        self.lang_start_time = time.time()
        self.current_lang_idx += 1

    def on_language_end(self, lang: str) -> None:
        """Invoked when completing ingestion for a language."""
        now = time.time()
        # Emit final progress block for the language
        if (now - self.last_heartbeat_time) > 1.0 or self.curr_lang_sentences >= (self.max_sentences_per_lang or 0):
            self._emit_progress(now)
            self.last_heartbeat_time = now

        self._last_card_line_count = 0
        lang_elapsed = now - self.lang_start_time
        rate = (
            self.curr_lang_sentences / lang_elapsed
            if lang_elapsed > 0
            else 0.0
        )
        self._writeln(
            f"  * Language '{lang}' complete: {self.curr_lang_sentences:,} sentences | "
            f"{self.curr_lang_tokens:,} tokens | {rate:.1f} sent/s | Duration: {self.format_duration(lang_elapsed)}"
        )

    def on_sentence(
        self,
        lang: str,
        treebank: str,
        tokens_in_sentence: int,
        exemplars_counts: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Invoked for each streamed sentence. Evaluates heartbeat interval
        and emits progress reporting when triggered.
        """
        treebank_changed = (treebank != self.curr_treebank and self.curr_lang_sentences > 0)
        self.curr_lang = lang
        self.curr_treebank = treebank
        self.curr_lang_sentences += 1
        self.curr_lang_tokens += tokens_in_sentence
        self.total_sentences_processed += 1
        self.total_tokens_processed += tokens_in_sentence

        if exemplars_counts is not None:
            self.exemplar_counts = exemplars_counts

        now = time.time()
        time_since_heartbeat = now - self.last_heartbeat_time

        # Trigger conditions:
        # 1. Periodic heartbeat interval exceeded (e.g. 5.0s)
        # 2. Significant sentence milestone on short/initial runs
        # 3. Exact target milestone reached
        # 4. Treebank change with reasonable interval (> 2.0s)
        should_emit = False
        if time_since_heartbeat >= self.heartbeat_interval:
            should_emit = True
        elif self.max_sentences_per_lang and self.curr_lang_sentences == self.max_sentences_per_lang:
            should_emit = True
        elif self.curr_lang_sentences in (25, 50, 100, 500, 1000) and time_since_heartbeat >= 2.0:
            should_emit = True
        elif treebank_changed and time_since_heartbeat >= 2.0:
            should_emit = True

        if should_emit:
            self._emit_progress(now)
            self.last_heartbeat_time = now

    def _emit_progress(self, now: float) -> None:
        """Renders either bounded progress block (in-place if TTY) or heartbeat."""
        overall_elapsed = max(1e-4, now - self.start_time)
        lang_elapsed = max(1e-4, now - self.lang_start_time)

        overall_rate = self.total_sentences_processed / overall_elapsed
        lang_rate = self.curr_lang_sentences / lang_elapsed

        if self.max_sentences_per_lang is not None:
            # Bounded progress report
            lang_target = self.max_sentences_per_lang
            lang_pct = min(100.0, (self.curr_lang_sentences / max(1, lang_target)) * 100.0)
            lang_rem_sents = max(0, lang_target - self.curr_lang_sentences)
            lang_eta = lang_rem_sents / max(1e-4, lang_rate) if lang_rate > 0 else 0.0

            overall_target = self.total_target_sentences or (lang_target * self.num_languages)
            overall_pct = min(100.0, (self.total_sentences_processed / max(1, overall_target)) * 100.0)
            overall_rem_sents = max(0, overall_target - self.total_sentences_processed)
            overall_eta = overall_rem_sents / max(1e-4, overall_rate) if overall_rate > 0 else 0.0

            card = [
                f"Language: {self.curr_lang}",
                f"Treebank: {self.curr_treebank}",
                "",
                f"Sentences : {self.curr_lang_sentences:,} / {lang_target:,}",
                f"Tokens    : {self.curr_lang_tokens:,}",
                f"Progress  : {lang_pct:.1f}%",
                f"Rate      : {lang_rate:.1f} sent/s",
                f"Elapsed   : {self.format_duration(lang_elapsed)}",
                f"ETA       : {self.format_duration(lang_eta)}",
                "",
                "Overall",
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"Languages : {self.current_lang_idx} / {self.num_languages}",
                f"Sentences : {self.total_sentences_processed:,} / {overall_target:,}",
                f"Progress  : {overall_pct:.1f}%",
                f"Rate      : {overall_rate:.1f} sent/s",
                f"Elapsed   : {self.format_duration(overall_elapsed)}",
                f"ETA       : {self.format_duration(overall_eta)}",
            ]

            if self.is_tty:
                self._ensure_vt100()
                if self._last_card_line_count > 0:
                    # Move cursor up to overwrite previous card in-place (WITHOUT trailing newline from OutputWrapper)
                    self.raw_stdout.write(f"\033[{self._last_card_line_count}A\r")
                for line in card:
                    safe_line = self._safe_str(line)
                    self.raw_stdout.write(f"{safe_line}\033[K\n")
                self._last_card_line_count = len(card)
                if hasattr(self.raw_stdout, "flush"):
                    self.raw_stdout.flush()
            else:
                # Non-interactive / redirected pipe: concise single-line to avoid log flooding
                time_str = time.strftime("%H:%M:%S", time.localtime(now))
                heartbeat = (
                    f"[{time_str}] {self.curr_lang}/{self.curr_treebank}: "
                    f"Sents {self.curr_lang_sentences:,}/{lang_target:,} ({lang_pct:.1f}%) | "
                    f"Tokens {self.curr_lang_tokens:,} | Rate {lang_rate:.1f} sent/s | "
                    f"Elapsed {self.format_duration(lang_elapsed)} | ETA {self.format_duration(lang_eta)}"
                )
                self._writeln(heartbeat)
        else:
            # Unbounded periodic heartbeat
            time_str = time.strftime("%H:%M:%S", time.localtime(now))
            heartbeat = (
                f"[{time_str}] Processing {self.curr_lang}/{self.curr_treebank}: "
                f"{self.curr_lang_sentences:,} sentences | "
                f"{self.curr_lang_tokens:,} tokens | {lang_rate:.1f} sent/s | "
                f"Overall: {self.total_sentences_processed:,} sents ({self.format_duration(overall_elapsed)})"
            )
            if self.is_tty:
                self._ensure_vt100()
                if self._last_card_line_count > 0:
                    self.raw_stdout.write(f"\033[{self._last_card_line_count}A\r")
                safe_line = self._safe_str(heartbeat)
                self.raw_stdout.write(f"{safe_line}\033[K\n")
                self._last_card_line_count = 1
                if hasattr(self.raw_stdout, "flush"):
                    self.raw_stdout.flush()
            else:
                self._writeln(heartbeat)

    def print_summary(self, res: Dict[str, Any]) -> None:
        """Renders comprehensive completion summary."""
        total_duration = time.time() - self.start_time
        total_sents = res.get("sentences_trained", self.total_sentences_processed)
        total_toks = res.get("tokens_trained", self.total_tokens_processed)
        avg_rate = total_sents / max(1e-4, total_duration)

        saved_path = res.get("saved_to", "")
        file_size_str = ""
        if saved_path and os.path.exists(saved_path):
            size_mb = os.path.getsize(saved_path) / (1024.0 * 1024.0)
            file_size_str = f" ({size_mb:.1f} MB)"

        # Exemplars calculation
        total_exemplars = res.get("total_exemplars")
        upos_ex = res.get("total_upos_exemplars")
        deprel_ex = res.get("total_deprel_exemplars")
        morph_ex = res.get("total_morph_exemplars")

        if total_exemplars is not None and upos_ex is not None:
            exemplars_str = f"{total_exemplars:,} (UPOS: {upos_ex:,}, Deprel: {deprel_ex:,}, Morph: {morph_ex:,})"
        else:
            upos_count = len(res.get("upos_tags", []))
            deprel_count = len(res.get("deprel_relations", []))
            morph_count = res.get("morph_features", 0)
            exemplars_str = f"Prototypes: {upos_count} UPOS tags, {deprel_count} Deprels, {morph_count} Morph features"

        summary_lines = [
            "",
            "✓ UD linguistic training complete",
            "",
            f"Languages       : {len(self.languages)} ({', '.join(self.languages)})",
            f"Sentences       : {total_sents:,}",
            f"Tokens          : {total_toks:,}",
            f"Exemplars       : {exemplars_str}",
            f"Duration        : {self.format_duration(total_duration)}",
            f"Average rate    : {avg_rate:.1f} sent/s",
            "",
            "Tensor artifacts:",
            f"  {saved_path}{file_size_str}",
            "",
        ]
        self._writeln("\n".join(summary_lines), style_type="SUCCESS")
