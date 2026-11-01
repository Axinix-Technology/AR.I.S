"""
AR.I.S — Enterprise Multi-Corpus Cognitive DataLoader (2026 Production Grade)
Combines 4 Foundation Pillars:
  1. UltraChat 200k (Conversational Dialogue & Natural Interaction)
  2. Universal Dependencies v2.18 (Grammatical Syntax & Universal Treebanks)
  3. QA-SRL 2020 (Semantic Causal Reasoning: Who, What, Where, When, Why, How)
  4. HotpotQA (Multi-Hop Complex Deductions across Multiple Sentences)
"""

from __future__ import annotations
import gzip
import json
from pathlib import Path
from typing import Iterator, Tuple, List, Optional, Dict, Any
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, DataLoader

from ..tokenizer.bpe_tokenizer import ArisTokenizer
from ..linguistic.ud_parser import ArisUDParser


class EnterpriseArisDataset(IterableDataset):
    """
    Multi-Corpus Cognitive Dataset streaming 4 foundational corpora on-the-fly.
    """

    def __init__(
        self,
        ultrachat_dir: Path,
        ud_archive_path: Optional[Path],
        qasrl_dir: Optional[Path],
        hotpotqa_dir: Optional[Path],
        tokenizer: ArisTokenizer,
        max_seq_len: int = 384,
        split: str = "train",
        focus: str = "all",
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.ultrachat_dir = Path(ultrachat_dir)
        self.ud_archive_path = Path(ud_archive_path) if ud_archive_path else None
        self.qasrl_dir = Path(qasrl_dir) if qasrl_dir else None
        self.hotpotqa_dir = Path(hotpotqa_dir) if hotpotqa_dir else None
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split = split
        self.focus = focus.lower()
        self.rank = rank
        self.world_size = max(1, world_size)

        # UltraChat Parquets
        self.uc_files = sorted(list(self.ultrachat_dir.glob("*.parquet")))

        # HotpotQA Parquets
        self.hp_files = sorted(list(self.hotpotqa_dir.glob("**/*.parquet"))) if self.hotpotqa_dir else []

        # QA-SRL files
        self.qasrl_files = sorted(list(self.qasrl_dir.glob("**/*.jsonl.gz"))) if self.qasrl_dir else []

    def _format_and_tokenize(self, prompt: str, response: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Formats into ChatML prompt/response and produces masked target tensor."""
        user_chat = self.tokenizer.format_chat_message("user", prompt)
        asst_chat = f"<|im_start|>assistant\n{response}<|im_end|>\n"

        user_toks = self.tokenizer.encode(user_chat, add_special_tokens=False)
        asst_toks = self.tokenizer.encode(asst_chat, add_special_tokens=False)

        full_seq = user_toks + asst_toks
        if len(full_seq) > self.max_seq_len:
            full_seq = full_seq[: self.max_seq_len]

        targets = [-100] * min(len(user_toks), len(full_seq))
        remaining = len(full_seq) - len(targets)
        if remaining > 0:
            targets += full_seq[len(targets) : len(targets) + remaining]

        return torch.tensor(full_seq, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

    def _iter_ultrachat(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """1. UltraChat: Conversational dialogue turns."""
        for p_file in self.uc_files:
            table = pq.read_table(p_file, columns=["messages"])
            for dialogue in table["messages"].to_pylist():
                if not dialogue or len(dialogue) < 2:
                    continue
                for idx in range(len(dialogue) - 1):
                    turn_user = dialogue[idx]
                    turn_asst = dialogue[idx + 1]
                    if turn_user.get("role") == "user" and turn_asst.get("role") == "assistant":
                        u_text = turn_user.get("content", "").strip()
                        a_text = turn_asst.get("content", "").strip()
                        if u_text and a_text:
                            yield self._format_and_tokenize(u_text, a_text)

    def _iter_hotpotqa(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """2. HotpotQA: Multi-hop reasoning across multiple sentences with supporting evidence."""
        for p_file in self.hp_files:
            table = pq.read_table(p_file)
            rows = table.to_pylist()
            for r in rows:
                question = r.get("question", "").strip()
                answer = r.get("answer", "").strip()
                context = r.get("context", {})
                supp = r.get("supporting_facts", {})

                if not question or not answer or not context:
                    continue

                ctx_titles = context.get("title", [])
                ctx_sentences = context.get("sentences", [])
                title_map = {t: s for t, s in zip(ctx_titles, ctx_sentences)}

                supp_titles = supp.get("title", [])
                supp_ids = supp.get("sent_id", [])

                # Extract explicit supporting sentences
                evidence_list = []
                for t, sid in zip(supp_titles, supp_ids):
                    if t in title_map and sid < len(title_map[t]):
                        s_text = title_map[t][sid].strip()
                        if s_text:
                            evidence_list.append(f"[{t}] {s_text}")

                # Build compact context with supporting titles prioritizing multi-sentence context
                context_str = ""
                seen_titles = set()
                # Include supporting titles first
                for t in supp_titles:
                    if t in title_map and t not in seen_titles:
                        context_str += f"[{t}] " + " ".join(title_map[t][:3]) + "\n"
                        seen_titles.add(t)
                # Include distractor titles up to 3 total
                for t in ctx_titles:
                    if len(seen_titles) >= 3:
                        break
                    if t not in seen_titles and t in title_map:
                        context_str += f"[{t}] " + " ".join(title_map[t][:2]) + "\n"
                        seen_titles.add(t)

                prompt = f"Background Information:\n{context_str}\nQuestion: {question}"

                if evidence_list:
                    evidence_str = "\n".join(f"- {e}" for e in evidence_list[:3])
                    response = (
                        f"<|thought|>\n"
                        f"Step 1: Connecting multi-sentence supporting facts:\n{evidence_str}\n"
                        f"Step 2: Deducing target conclusion from evidence.\n"
                        f"Answer: {answer}"
                    )
                else:
                    response = f"<|thought|>\nSynthesizing information across background context.\nAnswer: {answer}"

                yield self._format_and_tokenize(prompt, response)

    def _iter_qasrl(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """3. QA-SRL: Word-level semantic causal reasoning (Who, What, Why, Where, How)."""
        for f_path in self.qasrl_files:
            try:
                with gzip.open(f_path, "rt", encoding="utf-8") as f:
                    for line in f:
                        d = json.loads(line)
                        tokens = d.get("sentenceTokens", [])
                        sentence = " ".join(tokens)
                        verb_entries = d.get("verbEntries", {})

                        for v_idx_str, v_data in verb_entries.items():
                            v_idx = int(v_idx_str)
                            verb_word = tokens[v_idx]
                            for q_id, q_info in v_data.get("questionLabels", {}).items():
                                q_str = q_info.get("questionString", "")
                                answers = []
                                for j in q_info.get("answerJudgments", []):
                                    if j.get("isValid"):
                                        for sp in j.get("spans", []):
                                            answers.append(" ".join(tokens[sp[0]:sp[1]]))

                                if q_str and answers:
                                    prompt = f"Sentence: {sentence}\nTarget Action: '{verb_word}'\nQuestion: {q_str}"
                                    response = f"<|thought|>\nAnalyzing semantic argument for action '{verb_word}':\nResolved Argument: {answers[0]}"
                                    yield self._format_and_tokenize(prompt, response)
            except Exception:
                continue

    def _iter_ud(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """4. Universal Dependencies: Universal grammar and dependency syntax trees."""
        if not self.ud_archive_path or not self.ud_archive_path.exists():
            return

        for lang in ["english", "tamil", "hindi", "french", "german"]:
            try:
                for sent in ArisUDParser.stream_from_archive(self.ud_archive_path, language=lang):
                    if not sent.text or len(sent.tokens) < 3:
                        continue
                    upos_str = " ".join(f"{t.form}/{t.upos}" for t in sent.tokens)
                    prompt = f"Analyze grammatical syntax and parts of speech for: {sent.text}"
                    response = f"Universal POS breakdown: {upos_str}"
                    yield self._format_and_tokenize(prompt, response)
            except Exception:
                continue

    def _base_stream(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        uc_iter = self._iter_ultrachat()
        hp_iter = self._iter_hotpotqa()
        qs_iter = self._iter_qasrl()
        ud_iter = self._iter_ud()

        if self.focus == "language":
            # Language Focus: UltraChat (3) + Universal Dependencies (1)
            while True:
                for _ in range(3):
                    try:
                        yield next(uc_iter)
                    except StopIteration:
                        uc_iter = self._iter_ultrachat()
                        yield next(uc_iter)
                try:
                    yield next(ud_iter)
                except StopIteration:
                    ud_iter = self._iter_ud()
                    try:
                        yield next(ud_iter)
                    except StopIteration:
                        pass

        elif self.focus == "reasoning":
            # Reasoning Focus: HotpotQA (2) + QA-SRL (2)
            while True:
                for _ in range(2):
                    try:
                        yield next(hp_iter)
                    except StopIteration:
                        hp_iter = self._iter_hotpotqa()
                        try:
                            yield next(hp_iter)
                        except StopIteration:
                            pass
                for _ in range(2):
                    try:
                        yield next(qs_iter)
                    except StopIteration:
                        qs_iter = self._iter_qasrl()
                        try:
                            yield next(qs_iter)
                        except StopIteration:
                            pass
        else:
            # Default "all": 3 UltraChat : 2 HotpotQA : 2 QA-SRL : 1 Universal Dependencies
            while True:
                for _ in range(3):
                    try:
                        yield next(uc_iter)
                    except StopIteration:
                        uc_iter = self._iter_ultrachat()
                        yield next(uc_iter)

                for _ in range(2):
                    try:
                        yield next(hp_iter)
                    except StopIteration:
                        hp_iter = self._iter_hotpotqa()
                        try:
                            yield next(hp_iter)
                        except StopIteration:
                            pass

                for _ in range(2):
                    try:
                        yield next(qs_iter)
                    except StopIteration:
                        qs_iter = self._iter_qasrl()
                        try:
                            yield next(qs_iter)
                        except StopIteration:
                            pass

                try:
                    yield next(ud_iter)
                except StopIteration:
                    ud_iter = self._iter_ud()
                    try:
                        yield next(ud_iter)
                    except StopIteration:
                        pass

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Streams dataset according to cognitive focus and distributes samples across ranks.
        """
        sample_counter = 0
        for sample in self._base_stream():
            if sample_counter % self.world_size == self.rank:
                yield sample
            sample_counter += 1


def collate_chat_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    pad_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs, targets = zip(*batch)
    max_len = max(len(x) for x in inputs)

    padded_inputs = []
    padded_targets = []

    for inp, tgt in zip(inputs, targets):
        pad_len = max_len - len(inp)
        if pad_len > 0:
            p_inp = torch.cat([inp, torch.full((pad_len,), pad_id, dtype=torch.long)])
            p_tgt = torch.cat([tgt, torch.full((pad_len,), -100, dtype=torch.long)])
        else:
            p_inp = inp
            p_tgt = tgt
        padded_inputs.append(p_inp)
        padded_targets.append(p_tgt)

    return torch.stack(padded_inputs), torch.stack(padded_targets)


class EnterpriseDataLoader:
    @classmethod
    def get_loader(
        cls,
        ultrachat_dir: Path,
        ud_archive_path: Optional[Path],
        qasrl_dir: Optional[Path],
        hotpotqa_dir: Optional[Path],
        tokenizer: ArisTokenizer,
        batch_size: int = 8,
        max_seq_len: int = 384,
        split: str = "train",
        focus: str = "all",
        rank: int = 0,
        world_size: int = 1,
    ) -> DataLoader:
        dataset = EnterpriseArisDataset(
            ultrachat_dir=ultrachat_dir,
            ud_archive_path=ud_archive_path,
            qasrl_dir=qasrl_dir,
            hotpotqa_dir=hotpotqa_dir,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            split=split,
            focus=focus,
            rank=rank,
            world_size=world_size,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=lambda b: collate_chat_batch(b, pad_id=tokenizer.pad_id),
        )
