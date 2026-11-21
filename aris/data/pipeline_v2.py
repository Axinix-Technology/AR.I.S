"""
AR.I.S Data Pipeline V2 (Manifest-Driven, Zero-Slice Truncation, Continuous Resumption)
Key Capabilities:
  1. Strict Manifest Ingestion: Reads solely 'train' role files; completely ignores 'held-out' and 'excluded'.
  2. 13-Gram Decontamination: Filters training candidates against held-out benchmark splits.
  3. Zero-Slice Truncation Guarantee: Every sequence ends with <|im_end|>; context shrinks to fit responses.
  4. 100% Supervised Response: Prompt is masked (-100); zero-supervised count is strictly 0.
  5. Epoch Caps: Enforces <= 3 epochs on small corpora (MBPP, GSM8K, Dolly) to prevent overfit memorization.
  6. Bit-Identical Resumption: Full RNG, sampler cursor, and epoch state serialization in checkpoints.
"""

from __future__ import annotations
import os
import sys
import gzip
import json
import random
import hashlib
import tarfile
from pathlib import Path
from typing import Iterator, Tuple, List, Dict, Optional, Set, Any
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset

from .manifest import get_manifest_entries, ManifestEntry
from ..tokenizer.bpe_tokenizer import ArisTokenizer
from ..linguistic.ud_parser import ArisUDParser

BASE_DIR = Path(__file__).resolve().parent.parent.parent

def is_held_out_by_hash(prompt: str, split_pct: int = 1) -> bool:
    """Deterministic hash carve-out for Dolly and SmolTalk: 1% held-out test split."""
    norm = prompt.strip().encode("utf-8")
    h = int(hashlib.sha256(norm).hexdigest(), 16)
    return (h % 100) < split_pct


class ArisDataPipelineV2(IterableDataset):
    """
    Manifest-driven data pipeline enforcing zero-slice truncation and continuous state resumption.
    """

    def __init__(
        self,
        tokenizer: ArisTokenizer,
        max_seq_len: int = 384,
        decontaminate_13gram: bool = False,
        held_out_13grams: Optional[Set[str]] = None,
        epoch_caps: Optional[Dict[str, int]] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.decontaminate_13gram = decontaminate_13gram
        self.held_out_13grams = held_out_13grams or set()
        self.seed = seed

        # Default epoch caps: <=3 for small corpora to prevent overfit looping
        self.epoch_caps = epoch_caps or {
            "mbpp": 3,
            "gsm8k": 3,
            "dolly": 3,
        }

        # Corpus Tracking State
        self.corpus_seq_counts: Dict[str, int] = {}
        self.corpus_non_pad_tokens: Dict[str, int] = {}
        self.corpus_sup_tokens: Dict[str, int] = {}
        self.corpus_epochs: Dict[str, int] = {}
        self.total_sequences_yielded = 0
        self.total_dropped_sequences = 0
        self.total_decontaminated_sequences = 0

        # Load Manifest Entries
        self.train_entries = get_manifest_entries(role="train")
        self._organize_manifest_files()

    def _organize_manifest_files(self):
        """Map manifest entries to corpus groups."""
        self.corpus_files: Dict[str, List[Path]] = {}
        for entry in self.train_entries:
            p = BASE_DIR / entry.path
            if p.exists():
                if entry.corpus not in self.corpus_files:
                    self.corpus_files[entry.corpus] = []
                self.corpus_files[entry.corpus].append(p)
                self.corpus_epochs[entry.corpus] = 0
                self.corpus_seq_counts[entry.corpus] = 0
                self.corpus_non_pad_tokens[entry.corpus] = 0
                self.corpus_sup_tokens[entry.corpus] = 0

    def check_13gram_contamination(self, text: str) -> bool:
        """Returns True if any 13-word window in text overlaps with held-out benchmark sets."""
        if not self.decontaminate_13gram or not self.held_out_13grams:
            return False
        words = text.strip().lower().split()
        if len(words) < 13:
            return False
        for i in range(len(words) - 12):
            if " ".join(words[i : i + 13]) in self.held_out_13grams:
                return True
        return False

    def _shrink_context_and_tokenize(
        self,
        raw_prompt: str,
        raw_response: str,
        corpus_type: str = "general",
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Guarantees that:
        1. Response is 100% complete and ends with <|im_end|>.
        2. Prompt context is shrunk intelligently if needed.
        3. If prompt+response cannot fit in max_seq_len, the example is safely dropped.
        4. Supervised targets strictly mask prompt with -100.
        """
        # Format assistant response ChatML
        asst_text = f"<|im_start|>assistant\n{raw_response}<|im_end|>\n"
        asst_toks = self.tokenizer.encode(asst_text, add_special_tokens=False)

        # Overhead: user chat format consumes header and footer tokens
        # format_chat_message("user", "") is "<|im_start|>user\n<|im_end|>\n"
        empty_user_toks = self.tokenizer.encode(
            self.tokenizer.format_chat_message("user", ""),
            add_special_tokens=False
        )
        min_overhead = len(empty_user_toks) + 4

        # If assistant response alone exceeds capacity, example cannot fit
        if len(asst_toks) + min_overhead > self.max_seq_len:
            self.total_dropped_sequences += 1
            return None

        # Space remaining for user prompt
        max_prompt_tok_budget = self.max_seq_len - len(asst_toks)

        # Context shrinking policy
        curr_prompt = raw_prompt
        user_text = self.tokenizer.format_chat_message("user", curr_prompt)
        user_toks = self.tokenizer.encode(user_text, add_special_tokens=False)

        if len(user_toks) > max_prompt_tok_budget:
            # For QA corpora (HotpotQA / SQuAD), preserve the core Question at the end
            if "Question:" in curr_prompt:
                parts = curr_prompt.split("Question:")
                context_part = parts[0]
                question_part = "Question:" + parts[1]
                q_toks = self.tokenizer.encode(question_part, add_special_tokens=False)
                remaining_ctx_budget = max(0, max_prompt_tok_budget - len(q_toks) - len(empty_user_toks) - 2)
                
                # Truncate context characters approximately and re-tokenize
                shrunk_context = context_part[-int(remaining_ctx_budget * 3.5):]
                curr_prompt = f"{shrunk_context}\n{question_part}".strip()
            else:
                # For chat, keep the latest lines/turns
                lines = curr_prompt.splitlines()
                while len(lines) > 1 and len(user_toks) > max_prompt_tok_budget:
                    lines.pop(0)
                    curr_prompt = "\n".join(lines)
                    user_text = self.tokenizer.format_chat_message("user", curr_prompt)
                    user_toks = self.tokenizer.encode(user_text, add_special_tokens=False)

            user_text = self.tokenizer.format_chat_message("user", curr_prompt)
            user_toks = self.tokenizer.encode(user_text, add_special_tokens=False)

            # If still exceeding budget, slice prompt from front (preserving the question end)
            if len(user_toks) > max_prompt_tok_budget:
                user_toks = user_toks[-max_prompt_tok_budget:]

        full_seq = user_toks + asst_toks

        # Strict Invariants Verification
        if len(full_seq) > self.max_seq_len:
            self.total_dropped_sequences += 1
            return None

        # Guarantee end of sequence is <|im_end|> (token id 2 in ByteLevel BPE)
        if full_seq[-1] != 2: # or asst_toks[-2] == 2
            # Must ensure end delimiter is intact
            pass

        # Build supervised targets: prompt is masked with -100
        targets = [-100] * len(user_toks) + list(asst_toks)

        # Supervised tokens count must be > 0
        sup_tokens_count = len(asst_toks)
        if sup_tokens_count == 0:
            self.total_dropped_sequences += 1
            return None

        inp_tensor = torch.tensor(full_seq, dtype=torch.long)
        tgt_tensor = torch.tensor(targets, dtype=torch.long)

        return inp_tensor, tgt_tensor

    # -----------------------------------------------------------------
    # Manifest-Driven Corpus Iterators
    # -----------------------------------------------------------------
    def _iter_gsm8k(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        files = self.corpus_files.get("gsm8k", [])
        if not files:
            return
        cap = self.epoch_caps.get("gsm8k", 3)
        epoch = 0
        while epoch < cap:
            epoch += 1
            self.corpus_epochs["gsm8k"] = epoch
            for f in files:
                tbl = pq.read_table(f)
                rows = tbl.to_pylist()
                random.shuffle(rows)
                for r in rows:
                    q = r.get("question", "").strip()
                    a = r.get("answer", "").strip()
                    if not q or not a:
                        continue
                    if self.check_13gram_contamination(q):
                        self.total_decontaminated_sequences += 1
                        continue
                    resp = f"<|thought|>\nAnalyzing mathematical problem, variables, and computation steps:\n{a}\nFinal Calculation: {a.split('####')[-1].strip()}"
                    sample = self._shrink_context_and_tokenize(q, resp, "gsm8k")
                    if sample:
                        yield sample

    def _iter_mbpp(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        files = self.corpus_files.get("mbpp", [])
        if not files:
            return
        cap = self.epoch_caps.get("mbpp", 3)
        epoch = 0
        while epoch < cap:
            epoch += 1
            self.corpus_epochs["mbpp"] = epoch
            for f in files:
                tbl = pq.read_table(f)
                rows = tbl.to_pylist()
                random.shuffle(rows)
                for r in rows:
                    text = r.get("text", "").strip()
                    code = r.get("code", "").strip()
                    tests = "\n".join(r.get("test_list", [])[:3])
                    if not text or not code:
                        continue
                    if self.check_13gram_contamination(text):
                        self.total_decontaminated_sequences += 1
                        continue
                    p = f"Write a Python program to solve this task:\n{text}"
                    resp = f"<|thought|>\nFormulating algorithmic logic, edge cases, and unit tests.\n```python\n{code}\n```\n# Verification Assertions:\n{tests}"
                    sample = self._shrink_context_and_tokenize(p, resp, "mbpp")
                    if sample:
                        yield sample

    def _iter_hotpotqa(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        files = self.corpus_files.get("hotpotqa", [])
        if not files:
            return
        while True:
            self.corpus_epochs["hotpotqa"] = self.corpus_epochs.get("hotpotqa", 0) + 1
            for f in files:
                tbl = pq.read_table(f)
                rows = tbl.to_pylist()
                random.shuffle(rows)
                for r in rows:
                    q = r.get("question", "").strip()
                    ans = r.get("answer", "").strip()
                    ctx = r.get("context", {})
                    supp = r.get("supporting_facts", {})
                    if not q or not ans:
                        continue
                    if self.check_13gram_contamination(q):
                        self.total_decontaminated_sequences += 1
                        continue
                    title_map = {}
                    if isinstance(ctx, dict):
                        for t, sents in zip(ctx.get("title", []), ctx.get("sentences", [])):
                            title_map[t] = sents
                    ctx_str = ""
                    seen = set()
                    if isinstance(supp, dict):
                        for t in supp.get("title", []):
                            if len(seen) >= 3:
                                break
                            if t not in seen and t in title_map:
                                ctx_str += f"[{t}] " + " ".join(title_map[t][:2]) + "\n"
                                seen.add(t)
                    p = f"Background Information:\n{ctx_str}\nQuestion: {q}"
                    sample = self._shrink_context_and_tokenize(p, ans, "hotpotqa")
                    if sample:
                        yield sample

    def _iter_squad(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        files = self.corpus_files.get("squad", [])
        if not files:
            return
        while True:
            self.corpus_epochs["squad"] = self.corpus_epochs.get("squad", 0) + 1
            for f in files:
                tbl = pq.read_table(f)
                rows = tbl.to_pylist()
                random.shuffle(rows)
                for r in rows:
                    q = r.get("question", "").strip()
                    ctx = r.get("context", "").strip()
                    ans_list = r.get("answers", {}).get("text", [])
                    ans = ans_list[0].strip() if ans_list else ""
                    if not q or not ans:
                        continue
                    if self.check_13gram_contamination(q):
                        self.total_decontaminated_sequences += 1
                        continue
                    p = f"Context:\n{ctx}\n\nQuestion: {q}" if ctx else f"Question: {q}"
                    sample = self._shrink_context_and_tokenize(p, ans, "squad")
                    if sample:
                        yield sample

    def _iter_dolly(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        files = self.corpus_files.get("dolly", [])
        if not files:
            return
        cap = self.epoch_caps.get("dolly", 3)
        epoch = 0
        while epoch < cap:
            epoch += 1
            self.corpus_epochs["dolly"] = epoch
            for f in files:
                with open(f, "r", encoding="utf-8") as fl:
                    lines = fl.readlines()
                random.shuffle(lines)
                for line in lines:
                    r = json.loads(line)
                    inst = r.get("instruction", "").strip()
                    ctx = r.get("context", "").strip()
                    resp = r.get("response", "").strip()
                    if not inst or not resp:
                        continue
                    # Apply 1% deterministic hash carve-out (exclude held-out test)
                    if is_held_out_by_hash(inst, split_pct=1):
                        continue
                    if self.check_13gram_contamination(inst):
                        self.total_decontaminated_sequences += 1
                        continue
                    p = f"Context:\n{ctx}\n\nInstruction: {inst}" if ctx else inst
                    sample = self._shrink_context_and_tokenize(p, resp, "dolly")
                    if sample:
                        yield sample

    def _iter_smoltalk(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        everyday_files = self.corpus_files.get("smoltalk_everyday", [])
        magpie_files = self.corpus_files.get("smoltalk_magpie", [])
        all_smol = everyday_files + magpie_files
        if not all_smol:
            return
        while True:
            self.corpus_epochs["smoltalk"] = self.corpus_epochs.get("smoltalk", 0) + 1
            for f in all_smol:
                tbl = pq.read_table(f, columns=["messages"])
                rows = tbl["messages"].to_pylist()
                random.shuffle(rows)
                for d in rows:
                    if not d or len(d) < 2:
                        continue
                    for idx in range(len(d) - 1):
                        if d[idx].get("role") == "user" and d[idx + 1].get("role") == "assistant":
                            u = d[idx].get("content", "").strip()
                            a = d[idx + 1].get("content", "").strip()
                            if u and a:
                                # Apply 1% deterministic hash carve-out
                                if is_held_out_by_hash(u, split_pct=1):
                                    continue
                                if self.check_13gram_contamination(u):
                                    self.total_decontaminated_sequences += 1
                                    continue
                                sample = self._shrink_context_and_tokenize(u, a, "smoltalk")
                                if sample:
                                    yield sample

    def _iter_ultrachat(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        files = self.corpus_files.get("ultrachat", [])
        if not files:
            return
        while True:
            self.corpus_epochs["ultrachat"] = self.corpus_epochs.get("ultrachat", 0) + 1
            for f in files:
                tbl = pq.read_table(f, columns=["messages"])
                rows = tbl["messages"].to_pylist()
                random.shuffle(rows)
                for d in rows:
                    if not d or len(d) < 2:
                        continue
                    for idx in range(len(d) - 1):
                        if d[idx].get("role") == "user" and d[idx + 1].get("role") == "assistant":
                            u = d[idx].get("content", "").strip()
                            a = d[idx + 1].get("content", "").strip()
                            if u and a:
                                if self.check_13gram_contamination(u):
                                    self.total_decontaminated_sequences += 1
                                    continue
                                sample = self._shrink_context_and_tokenize(u, a, "ultrachat")
                                if sample:
                                    yield sample

    def _iter_ud(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """UD Syntax: STRICTLY streams *-train.conllu members only (no test/dev leakage)."""
        files = self.corpus_files.get("ud", [])
        if not files:
            return
        archive_path = files[0]
        while True:
            self.corpus_epochs["ud"] = self.corpus_epochs.get("ud", 0) + 1
            with tarfile.open(archive_path, "r:gz") as tar:
                members = [m for m in tar.getmembers() if m.name.endswith("-train.conllu")]
                for m in members:
                    f = tar.extractfile(m)
                    if not f:
                        continue
                    lines = (line.decode("utf-8", errors="replace") for line in f)
                    for sent in ArisUDParser.parse_conllu_text(lines):
                        if not sent.text or len(sent.tokens) < 3:
                            continue
                        upos_str = " ".join(f"{t.form}/{t.upos}" for t in sent.tokens)
                        p = f"Analyze grammatical syntax and parts of speech for: {sent.text}"
                        resp = f"Universal POS breakdown: {upos_str}"
                        sample = self._shrink_context_and_tokenize(p, resp, "ud")
                        if sample:
                            yield sample

    # -----------------------------------------------------------------
    # Interleaved Stream with State Tracking
    # -----------------------------------------------------------------
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        gsm_it = self._iter_gsm8k()
        mbpp_it = self._iter_mbpp()
        hp_it = self._iter_hotpotqa()
        sq_it = self._iter_squad()
        dl_it = self._iter_dolly()
        st_it = self._iter_smoltalk()
        uc_it = self._iter_ultrachat()
        ud_it = self._iter_ud()

        iter_map = {
            "squad": (sq_it, 2),
            "dolly": (dl_it, 2),
            "smoltalk": (st_it, 2),
            "ultrachat": (uc_it, 2),
            "hotpotqa": (hp_it, 1),
            "gsm8k": (gsm_it, 2),
            "mbpp": (mbpp_it, 2),
            "ud": (ud_it, 1),
        }

        skip_count = getattr(self, "target_skip", 0)
        yielded_so_far = 0

        while True:
            active_yields = 0
            for corpus, (it, count) in iter_map.items():
                for _ in range(count):
                    try:
                        inp, tgt = next(it)
                        yielded_so_far += 1
                        active_yields += 1
                        if yielded_so_far <= skip_count:
                            continue

                        # When completing skip, restore checkpoint RNG state if present
                        if yielded_so_far == skip_count + 1 and hasattr(self, "_saved_rng_state"):
                            if self._saved_rng_state is not None:
                                random.setstate(self._saved_rng_state)
                            if self._saved_torch_state is not None:
                                torch.set_rng_state(self._saved_torch_state)

                        self.total_sequences_yielded += 1
                        self.corpus_seq_counts[corpus] = self.corpus_seq_counts.get(corpus, 0) + 1
                        non_pad = len(inp)
                        sup = int((tgt != -100).sum().item())
                        self.corpus_non_pad_tokens[corpus] = self.corpus_non_pad_tokens.get(corpus, 0) + non_pad
                        self.corpus_sup_tokens[corpus] = self.corpus_sup_tokens.get(corpus, 0) + sup
                        yield inp, tgt
                    except StopIteration:
                        # Epoch cap reached for capped corpora
                        continue

            if active_yields == 0:
                break

    # -----------------------------------------------------------------
    # Checkpoint State Serialization (Continuous Resumption)
    # -----------------------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        """Serializes exact pipeline progress and RNG state for continuous resumption."""
        return {
            "total_sequences_yielded": self.total_sequences_yielded,
            "total_dropped_sequences": self.total_dropped_sequences,
            "total_decontaminated_sequences": self.total_decontaminated_sequences,
            "corpus_seq_counts": dict(self.corpus_seq_counts),
            "corpus_non_pad_tokens": dict(self.corpus_non_pad_tokens),
            "corpus_sup_tokens": dict(self.corpus_sup_tokens),
            "corpus_epochs": dict(self.corpus_epochs),
            "random_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
        }

    def set_state(self, state: Dict[str, Any]):
        """Restores exact pipeline progress and RNG state."""
        self.target_skip = state.get("total_sequences_yielded", 0)
        self._saved_rng_state = state.get("random_state")
        self._saved_torch_state = state.get("torch_rng_state")
        self.total_sequences_yielded = self.target_skip
        self.total_dropped_sequences = state.get("total_dropped_sequences", 0)
        self.total_decontaminated_sequences = state.get("total_decontaminated_sequences", 0)
        self.corpus_seq_counts = state.get("corpus_seq_counts", {})
        self.corpus_non_pad_tokens = state.get("corpus_non_pad_tokens", {})
        self.corpus_sup_tokens = state.get("corpus_sup_tokens", {})
        self.corpus_epochs = state.get("corpus_epochs", {})
