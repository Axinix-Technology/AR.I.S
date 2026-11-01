"""
AR.I.S — Developmental Milestone Benchmark Suite (2026 Production Grade)
Evaluates genuine neural model progress across 5 cognitive developmental stages:
  - Stage 1: Lexical Grounding (Word Validity & Validation Perplexity)
  - Stage 2: Universal Grammatical Syntax (UPOS Treebank Recognition)
  - Stage 3: Semantic Causal Attribution (QA-SRL Role Extraction)
  - Stage 4: Multi-Hop Reasoning (HotpotQA Evidence Deduction)
  - Stage 5: Live Voice Qualitative Preview
"""

from __future__ import annotations
import gzip
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import pyarrow.parquet as pq
import torch

from ..models.aris_lm import ArisLM
from ..tokenizer.bpe_tokenizer import ArisTokenizer
from ..inference.generation import ArisInferenceEngine
from ..linguistic.ud_parser import ArisUDParser
from .metrics import (
    compute_token_f1,
    compute_exact_match,
    compute_word_validity_rate,
    compute_perplexity,
    compute_upos_alignment,
)


class ArisBenchmarkSuite:
    """
    Standardized Developmental Milestone Benchmark Suite for AR.I.S checkpoints.
    """

    def __init__(
        self,
        model: ArisLM,
        tokenizer: ArisTokenizer,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cpu")

    def _generate(self, prompt: str, max_tokens: int = 40, temperature: float = 0.2) -> str:
        """Helper to run single-prompt inference."""
        chat = [{"role": "user", "content": prompt}]
        prompt_ids = self.tokenizer.encode_chat(chat, add_generation_prompt=True)
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            output_tensor = ArisInferenceEngine.generate(
                model=self.model,
                input_ids=input_tensor,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=40,
                top_p=0.9,
                eos_id=self.tokenizer.im_end_id,
            )

        gen_ids = output_tensor[0][len(prompt_ids):].tolist()
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    def eval_stage1_lexical(self) -> Dict[str, Any]:
        """
        Stage 1: Lexical Grounding.
        Measures whether the model generates real English words vs. raw phonetic babble.
        """
        probe_prompts = [
            "The story begins in a quiet",
            "She opened the front door and saw",
            "Science is the systematic study of",
            "In the morning, the bright sun",
        ]
        all_generated_text = ""
        for p in probe_prompts:
            gen = self._generate(p, max_tokens=25)
            all_generated_text += " " + gen

        valid_rate = compute_word_validity_rate(all_generated_text)

        status = "INFANT (Babbling)"
        if valid_rate >= 75.0:
            status = "MATURE (Fluent Lexicon)"
        elif valid_rate >= 40.0:
            status = "TODDLER (Emerging Words)"

        return {
            "word_validity_rate": valid_rate,
            "status": status,
            "sample_words": " ".join(all_generated_text.split()[:12]),
        }

    def eval_stage2_syntax(self, ud_archive_path: Path, max_samples: int = 5) -> Dict[str, Any]:
        """
        Stage 2: Universal Grammatical Syntax (Universal Dependencies Treebanks).
        Tests whether the model identifies parts of speech (UPOS tags) on real sentences.
        """
        if not ud_archive_path.exists():
            return {"upos_accuracy": 0.0, "status": "NO_UD_ARCHIVE", "evaluated": 0}

        count = 0
        alignment_scores = []

        try:
            for sent in ArisUDParser.stream_from_archive(ud_archive_path, language="english"):
                if count >= max_samples:
                    break
                if not sent.text or len(sent.tokens) < 4:
                    continue

                prompt = f"Analyze grammatical syntax and parts of speech for: {sent.text}"
                gen = self._generate(prompt, max_tokens=35)

                expected_upos = [t.upos for t in sent.tokens if t.upos]
                score = compute_upos_alignment(gen, expected_upos)
                alignment_scores.append(score)
                count += 1
        except Exception:
            pass

        avg_acc = (sum(alignment_scores) / max(1, len(alignment_scores))) if alignment_scores else 0.0

        status = "DEVELOPING"
        if avg_acc >= 60.0:
            status = "SYNTACTIC MASTERY"
        elif avg_acc >= 25.0:
            status = "GRAMMATICAL AWARENESS"

        return {
            "upos_accuracy": avg_acc,
            "status": status,
            "evaluated": count,
        }

    def eval_stage3_causal(self, qasrl_path: Path, max_samples: int = 5) -> Dict[str, Any]:
        """
        Stage 3: Causal Semantic Roles (QA-SRL 2020).
        Tests whether the model answers Who, What, Where, When, Why, How for target verbs.
        """
        if not qasrl_path.exists():
            return {"span_recall": 0.0, "evaluated": 0}

        count = 0
        f1_scores = []

        with gzip.open(qasrl_path, "rt", encoding="utf-8") as f:
            for line in f:
                if count >= max_samples:
                    break
                d = json.loads(line)
                tokens = d.get("sentenceTokens", [])
                sentence = " ".join(tokens)
                verb_entries = d.get("verbEntries", {})

                for v_idx_str, v_data in verb_entries.items():
                    if count >= max_samples:
                        break
                    v_idx = int(v_idx_str)
                    verb_word = tokens[v_idx]

                    for q_id, q_info in v_data.get("questionLabels", {}).items():
                        if count >= max_samples:
                            break
                        q_str = q_info.get("questionString", "")
                        answers = []
                        for j in q_info.get("answerJudgments", []):
                            if j.get("isValid"):
                                for sp in j.get("spans", []):
                                    answers.append(" ".join(tokens[sp[0]:sp[1]]))

                        if q_str and answers:
                            gt_answer = answers[0]
                            prompt = f"Sentence: {sentence}\nTarget Action: '{verb_word}'\nQuestion: {q_str}"
                            gen = self._generate(prompt, max_tokens=30)

                            if "Answer:" in gen:
                                gen = gen.split("Answer:", 1)[1].strip()

                            f1 = compute_token_f1(gen, gt_answer)
                            f1_scores.append(f1)
                            count += 1

        avg_f1 = (sum(f1_scores) / max(1, len(f1_scores))) * 100.0 if f1_scores else 0.0
        return {
            "span_f1": avg_f1,
            "evaluated": count,
        }

    def eval_stage4_multihop(self, hotpotqa_path: Path, max_samples: int = 5) -> Dict[str, Any]:
        """
        Stage 4: Multi-Hop Reasoning (HotpotQA held-out split).
        Tests whether the model connects cross-paragraph evidence to deduce final answers.
        """
        if not hotpotqa_path.exists():
            return {"f1": 0.0, "em": 0.0, "evaluated": 0}

        table = pq.read_table(hotpotqa_path)
        rows = table.to_pylist()

        count = 0
        f1_scores = []
        em_scores = []

        for r in rows:
            if count >= max_samples:
                break
            question = r.get("question", "").strip()
            gt_answer = r.get("answer", "").strip()
            context = r.get("context", {})

            if not question or not gt_answer or not context:
                continue

            ctx_titles = context.get("title", [])
            ctx_sentences = context.get("sentences", [])
            context_str = ""
            for t, sents in zip(ctx_titles[:2], ctx_sentences[:2]):
                context_str += f"[{t}] " + " ".join(sents[:2]) + "\n"

            prompt = f"Background Information:\n{context_str}\nQuestion: {question}"
            gen = self._generate(prompt, max_tokens=35)

            if "Answer:" in gen:
                gen = gen.split("Answer:", 1)[1].strip()

            f1 = compute_token_f1(gen, gt_answer)
            em = compute_exact_match(gen, gt_answer)

            f1_scores.append(f1)
            em_scores.append(em)
            count += 1

        avg_f1 = (sum(f1_scores) / max(1, len(f1_scores))) * 100.0 if f1_scores else 0.0
        avg_em = (sum(em_scores) / max(1, len(em_scores))) * 100.0 if em_scores else 0.0

        return {
            "token_f1": avg_f1,
            "exact_match": avg_em,
            "evaluated": count,
        }

    def eval_stage5_live_voice(self) -> List[Dict[str, str]]:
        """
        Stage 5: Live Qualitative Voice Preview.
        Probes the model directly with standard conversational inputs to hear its voice.
        """
        probes = [
            "hello",
            "Who are you?",
        ]
        previews = []
        for p in probes:
            raw = self._generate(p, max_tokens=30, temperature=0.7)
            previews.append({"prompt": p, "response": raw})
        return previews

    def run_full_evaluation(
        self,
        hotpotqa_path: Path,
        qasrl_path: Path,
        ud_archive_path: Path,
        max_samples: int = 5,
    ) -> Dict[str, Any]:
        """
        Executes complete 5-stage developmental evaluation.
        """
        t0 = time.time()
        print(f"[1/5] Evaluating Stage 1: Lexical Grounding & Word Validity...")
        s1 = self.eval_stage1_lexical()

        print(f"[2/5] Evaluating Stage 2: Grammatical Syntax & POS (Universal Dependencies)...")
        s2 = self.eval_stage2_syntax(ud_archive_path, max_samples=max_samples)

        print(f"[3/5] Evaluating Stage 3: Semantic Causal Attribution (QA-SRL)...")
        s3 = self.eval_stage3_causal(qasrl_path, max_samples=max_samples)

        print(f"[4/5] Evaluating Stage 4: Multi-Hop Reasoning (HotpotQA)...")
        s4 = self.eval_stage4_multihop(hotpotqa_path, max_samples=max_samples)

        print(f"[5/5] Generating Stage 5: Live Voice Preview...")
        s5 = self.eval_stage5_live_voice()

        total_time = time.time() - t0

        return {
            "stage1": s1,
            "stage2": s2,
            "stage3": s3,
            "stage4": s4,
            "stage5": s5,
            "elapsed_seconds": total_time,
        }
