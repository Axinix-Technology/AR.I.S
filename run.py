"""
AR.I.S — Command Line Interface & Cognitive Runtime (2026 Modern Architecture)
Features:
  - BPE Tokenizer training with ChatML support and exact byte-level roundtripping
  - ArisLM training (RoPE + GQA + SwiGLU + RMSNorm + AdamW + Cosine Warmup)
  - Interactive chat with vectorized KV Cache inference
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path
import torch
import pyarrow.parquet as pq

from aris.config import DATA_DIR, MODELS_DIR, TOKENIZER_PATH, LM_CHECKPOINT_PATH, DEVICE
from aris.configs.model_config import ArisConfig
from aris.tokenizer.bpe_tokenizer import ArisTokenizer
from aris.models.aris_lm import ArisLM
from aris.training.trainer import ArisTrainer
from aris.inference.generation import ArisInferenceEngine
from aris.memory import ArisGraphStore, HybridGraphRetriever
from aris.core import AutonomousDeliberationEngine
from aris.evaluation import ArisBenchmarkSuite
from aris.training.merging import ModelMerger


def handle_train_tokenizer(args):
    """
    Trains Byte-Level BPE Tokenizer on text extracted from UltraChat Parquet shards.
    """
    parquet_dir = DATA_DIR / "ultrachat_200k" / "data"
    print(f"\nScanning Parquet shards in: {parquet_dir}...")
    shards = list(parquet_dir.glob("*.parquet"))
    if not shards:
        print(f"Error: No Parquet shards found in {parquet_dir}")
        return

    def text_iterator():
        count = 0
        max_lines = args.max_lines
        for s in shards:
            table = pq.read_table(s, columns=["messages"])
            for dialogue in table["messages"].to_pylist():
                for turn in dialogue:
                    content = turn.get("content", "").strip()
                    if content:
                        yield content
                        count += 1
                        if count >= max_lines:
                            return

    print(f"Training BPE Tokenizer (Vocab Size: {args.vocab_size}, Max Lines: {args.max_lines})...")
    tokenizer = ArisTokenizer()
    tokenizer.train_from_iterator(text_iterator(), vocab_size=args.vocab_size)

    save_path = Path(args.output) if args.output else TOKENIZER_PATH
    tokenizer.save(save_path)
    print(f"Successfully trained and saved BPE Tokenizer to: {save_path}")
    print(f"Actual Vocabulary Size: {tokenizer.vocab_size}")


def handle_train_lm(args):
    """
    Executes neural training of modern ArisLM over UltraChat.
    """
    tok_path = Path(args.tokenizer) if args.tokenizer else TOKENIZER_PATH
    if not tok_path.exists():
        print(f"Error: Tokenizer not found at {tok_path}. Run 'python run.py train-tokenizer' first.")
        return

    print(f"Loading Tokenizer from: {tok_path}")
    tokenizer = ArisTokenizer.load(tok_path)

    parquet_dir = DATA_DIR / "ultrachat_200k" / "data"
    if not parquet_dir.exists():
        print(f"Error: Dataset directory not found: {parquet_dir}")
        return

    config = ArisConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.max_seq_len,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_query_heads=args.num_query_heads,
        num_kv_heads=args.num_kv_heads,
    )

    print(f"Initializing Modern 2026 ArisLM (RoPE + GQA + SwiGLU + RMSNorm)...")
    model = ArisLM(config)

    trainer = ArisTrainer(
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        checkpoint_dir=MODELS_DIR,
    )

    ud_path = Path(args.ud_archive) if args.ud_archive else None
    if ud_path and not ud_path.exists():
        print(f"Warning: UD Archive not found at {ud_path}, proceeding without UD.")
        ud_path = None
    elif ud_path:
        print(f"Loaded Universal Dependencies Archive: {ud_path}")

    qasrl_dir = Path(args.qasrl_dir) if args.qasrl_dir else None
    if qasrl_dir and not qasrl_dir.exists():
        print(f"Warning: QA-SRL directory not found at {qasrl_dir}, proceeding without QA-SRL.")
        qasrl_dir = None
    elif qasrl_dir:
        print(f"Loaded QA-SRL Directory: {qasrl_dir}")

    hotpotqa_dir = Path(args.hotpotqa_dir) if args.hotpotqa_dir else None
    if hotpotqa_dir and not hotpotqa_dir.exists():
        print(f"Warning: HotpotQA directory not found at {hotpotqa_dir}, proceeding without HotpotQA.")
        hotpotqa_dir = None
    elif hotpotqa_dir:
        print(f"Loaded HotpotQA Directory: {hotpotqa_dir}")

    trainer.train(
        parquet_dir=parquet_dir,
        ud_archive_path=ud_path,
        qasrl_dir=qasrl_dir,
        hotpotqa_dir=hotpotqa_dir,
        additional_steps=args.steps,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        max_threads=args.max_threads,
        lr=args.lr,
        max_seq_len=args.max_seq_len,
        log_every=args.log_every,
        save_every=args.save_every,
        checkpoint_name=args.checkpoint_name,
        resume=True,
        focus=args.focus,
        rank=args.rank,
        world_size=args.world_size,
    )


def handle_chat(args):
    """
    Runs autonomous interactive cognitive dialogue directly through ArisLM with KV Caching.
    """
    tok_path = Path(args.tokenizer) if args.tokenizer else TOKENIZER_PATH
    ckpt_path = Path(args.checkpoint) if args.checkpoint else LM_CHECKPOINT_PATH

    if not tok_path.exists():
        print(f"Error: Tokenizer not found at {tok_path}. Please train tokenizer first.")
        return

    if not ckpt_path.exists():
        print(f"Error: Model checkpoint not found at {ckpt_path}. Please run training first.")
        return

    print(f"\n=======================================================")
    print(f"  AR.I.S Autonomous Cognitive Language Agent")
    print(f"  Architecture: RoPE · GQA · SwiGLU · RMSNorm · KV-Cache")
    print(f"=======================================================")
    print(f"Device:     {DEVICE}")
    print(f"Model:      {ckpt_path.name}")
    print(f"Tokenizer:  {tok_path.name}")
    print(f"Type 'exit' or 'quit' to end session.\n")

    tokenizer = ArisTokenizer.load(tok_path)
    model, _ = ArisTrainer.load_model(ckpt_path, device=DEVICE)

    # Initialize Hybrid Neuro-Symbolic Graph RAG & Deliberation Engine
    graph_store = ArisGraphStore()
    retriever = HybridGraphRetriever(graph_store, tokenizer=tokenizer, model=model, device=DEVICE)
    deliberator = AutonomousDeliberationEngine(model=model, tokenizer=tokenizer, retriever=retriever, device=DEVICE)

    history = [
        {"role": "system", "content": "You are AR.I.S, an autonomous, intelligent cognitive language agent."}
    ]

    while True:
        try:
            user_input = input("aris> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Session ended.")
                break

            # Execute autonomous deliberation cycle
            result = deliberator.deliberate(
                user_input=user_input,
                history=history,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )

            if result["thought"]:
                print(f"\n[Thought]: {result['thought']}")
            print(f"\n{result['response']}\n")

            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": result["response"]})

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


def handle_eval(args):
    """
    Runs standardized cognitive evaluation benchmark over model checkpoint.
    """
    tok_path = Path(args.tokenizer) if args.tokenizer else TOKENIZER_PATH
    ckpt_path = Path(args.checkpoint) if args.checkpoint else LM_CHECKPOINT_PATH

    if not tok_path.exists():
        print(f"Error: Tokenizer not found at {tok_path}.")
        return

    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}.")
        return

    hotpot_path = Path(args.hotpotqa_file)
    qasrl_path = Path(args.qasrl_file)
    ud_path = Path(args.ud_archive)

    print(f"\n=======================================================")
    print(f"  AR.I.S Developmental Milestone Evaluation Suite")
    print(f"=======================================================")
    print(f"Device:           {DEVICE}")
    print(f"Model Checkpoint: {ckpt_path.name}")
    print(f"Tokenizer:        {tok_path.name}")
    print(f"Sample Count:     {args.samples} per benchmark")

    tokenizer = ArisTokenizer.load(tok_path)
    model, step = ArisTrainer.load_model(ckpt_path, device=DEVICE)

    suite = ArisBenchmarkSuite(model=model, tokenizer=tokenizer, device=DEVICE)
    report = suite.run_full_evaluation(
        hotpotqa_path=hotpot_path,
        qasrl_path=qasrl_path,
        ud_archive_path=ud_path,
        max_samples=args.samples,
    )

    s1 = report["stage1"]
    s2 = report["stage2"]
    s3 = report["stage3"]
    s4 = report["stage4"]
    s5 = report["stage5"]

    print(f"\n" + "=" * 65)
    print(f"  AR.I.S DEVELOPMENTAL MILESTONE SCORECARD (Step {step})")
    print(f"=" * 65)
    print(f" [STAGE 1] Lexical Grounding & Vocabulary Formation:")
    print(f"    - Real Word Validity:   {s1.get('word_validity_rate', 0.0):.2f}%")
    print(f"    - Status:               {s1.get('status', 'UNKNOWN')}")
    print(f"    - Generated Lexicon:    '{s1.get('sample_words', '')}'")
    print(f" -------------------------------------------------------------")
    print(f" [STAGE 2] Universal Grammatical Syntax (Treebanks):")
    print(f"    - UPOS Alignment Rate:  {s2.get('upos_accuracy', 0.0):.2f}%")
    print(f"    - Status:               {s2.get('status', 'UNKNOWN')}")
    print(f"    - Samples Tested:       {s2.get('evaluated', 0)}")
    print(f" -------------------------------------------------------------")
    print(f" [STAGE 3] Semantic Causal Attribution (QA-SRL Who/What/Why):")
    print(f"    - Causal Span F1:       {s3.get('span_f1', 0.0):.2f}%")
    print(f"    - Samples Tested:       {s3.get('evaluated', 0)}")
    print(f" -------------------------------------------------------------")
    print(f" [STAGE 4] Multi-Hop Reasoning (HotpotQA):")
    print(f"    - Deduction Token F1:   {s4.get('token_f1', 0.0):.2f}%")
    print(f"    - Exact Match (EM):     {s4.get('exact_match', 0.0):.2f}%")
    print(f"    - Samples Tested:       {s4.get('evaluated', 0)}")
    print(f" -------------------------------------------------------------")
    print(f" [STAGE 5] Live Qualitative Voice Preview:")
    for probe in s5:
        print(f"    User:   \"{probe['prompt']}\"")
        resp_clean = probe['response'].replace('\n', ' ')[:80]
        print(f"    AR.I.S: \"{resp_clean}\"")
    print(f"=" * 65)
    print(f" Milestone evaluation completed in {report['elapsed_seconds']:.2f}s.\n")


def handle_merge(args):
    """
    Merges independently trained checkpoints from multiple team members' PCs.
    """
    path_a = Path(args.model_a)
    path_b = Path(args.model_b)
    output_path = Path(args.output) if args.output else LM_CHECKPOINT_PATH

    if not path_a.exists():
        print(f"Error: Model A not found at {path_a}")
        return
    if not path_b.exists():
        print(f"Error: Model B not found at {path_b}")
        return

    print(f"\n=======================================================")
    print(f"  AR.I.S Collaborative Checkpoint Merging Engine")
    print(f"=======================================================")
    print(f"Model A (PC 1):   {path_a}")
    print(f"Model B (PC 2):   {path_b}")
    print(f"Output Target:    {output_path}")
    print(f"Merge Strategy:   {args.method.upper()}")

    ModelMerger.merge_and_save(
        model_paths=[path_a, path_b],
        output_path=output_path,
        method=args.method,
        alpha=args.alpha,
    )

    # Optional Memory Graph Merging
    if args.memory_b:
        mem_b_path = Path(args.memory_b)
        if mem_b_path.exists():
            store = ArisGraphStore()
            store.merge_from(mem_b_path)


def main():
    parser = argparse.ArgumentParser(description="AR.I.S — Artificial Intelligent System CLI (2026)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train Tokenizer
    p_tok = subparsers.add_parser("train-tokenizer", help="Train BPE Tokenizer on UltraChat")
    p_tok.add_argument("--vocab-size", type=int, default=8192)
    p_tok.add_argument("--max-lines", type=int, default=200000)
    p_tok.add_argument("--output", type=str, default=None)

    # Train Language Model
    p_lm = subparsers.add_parser("train-lm", help="Train ArisLM modern transformer on UltraChat")
    p_lm.add_argument("--tokenizer", type=str, default=None)
    p_lm.add_argument("--steps", type=int, default=1000)
    p_lm.add_argument("--batch-size", type=int, default=4)
    p_lm.add_argument("--accum-steps", type=int, default=4, help="Gradient accumulation steps for cool-training")
    p_lm.add_argument("--max-threads", type=int, default=4, help="Limit CPU threads to prevent overheating")
    p_lm.add_argument("--lr", type=float, default=3e-4)
    p_lm.add_argument("--max-seq-len", type=int, default=384)
    p_lm.add_argument("--embed-dim", type=int, default=384)
    p_lm.add_argument("--num-layers", type=int, default=6)
    p_lm.add_argument("--num-query-heads", type=int, default=6)
    p_lm.add_argument("--num-kv-heads", type=int, default=2)
    p_lm.add_argument("--log-every", type=int, default=20)
    p_lm.add_argument("--save-every", type=int, default=200)
    p_lm.add_argument("--checkpoint-name", type=str, default="aris_lm_checkpoint.pt")
    p_lm.add_argument("--ud-archive", type=str, default="ud-treebanks-v2.18.tgz")
    p_lm.add_argument("--qasrl-dir", type=str, default="data/qasrl_gs/qasrl-gs")
    p_lm.add_argument("--hotpotqa-dir", type=str, default="data/hotpotqa/distractor")
    p_lm.add_argument("--focus", type=str, default="all", choices=["all", "language", "reasoning"], help="Cognitive domain focus: 'all', 'language' (UltraChat+UD), or 'reasoning' (HotpotQA+QA-SRL)")
    p_lm.add_argument("--rank", type=int, default=0, help="Worker rank for multi-PC sharding (e.g. 0 for PC1, 1 for PC2)")
    p_lm.add_argument("--world-size", type=int, default=1, help="Total number of distributed training PCs")

    # Chat
    p_chat = subparsers.add_parser("chat", help="Start interactive dialogue session with ArisLM")
    p_chat.add_argument("--tokenizer", type=str, default=None)
    p_chat.add_argument("--checkpoint", type=str, default=None)
    p_chat.add_argument("--temperature", type=float, default=0.7)
    p_chat.add_argument("--top-k", type=int, default=40)
    p_chat.add_argument("--top-p", type=float, default=0.9)
    p_chat.add_argument("--max-tokens", type=int, default=64)

    # Evaluation Benchmark
    p_eval = subparsers.add_parser("eval", help="Run standardized cognitive evaluation benchmark")
    p_eval.add_argument("--tokenizer", type=str, default=None)
    p_eval.add_argument("--checkpoint", type=str, default=None)
    p_eval.add_argument("--samples", type=int, default=30)
    p_eval.add_argument("--hotpotqa-file", type=str, default="data/hotpotqa/distractor/train-00001-of-00002.parquet")
    p_eval.add_argument("--qasrl-file", type=str, default="data/qasrl_gs/qasrl-gs/dev.jsonl.gz")
    p_eval.add_argument("--ud-archive", type=str, default="ud-treebanks-v2.18.tgz")

    # Merge Models
    p_merge = subparsers.add_parser("merge-models", help="Merge independently trained checkpoints from multiple team members' PCs")
    p_merge.add_argument("--model-a", type=str, required=True, help="Path to Checkpoint A (e.g. from your PC)")
    p_merge.add_argument("--model-b", type=str, required=True, help="Path to Checkpoint B (e.g. from teammate's PC)")
    p_merge.add_argument("--output", type=str, default="models/aris_lm_checkpoint.pt", help="Path to write the merged checkpoint")
    p_merge.add_argument("--method", type=str, default="average", choices=["average", "slerp"], help="Merging strategy: 'average' (Model Soup) or 'slerp'")
    p_merge.add_argument("--alpha", type=float, default=0.5, help="Interpolation weight (0.5 = equal 50/50 mix)")
    p_merge.add_argument("--memory-b", type=str, default=None, help="Optional path to teammate's knowledge_graph.sqlite3 to merge memory")

    args = parser.parse_args()

    if args.command == "train-tokenizer":
        handle_train_tokenizer(args)
    elif args.command == "train-lm":
        handle_train_lm(args)
    elif args.command == "chat":
        handle_chat(args)
    elif args.command == "eval":
        handle_eval(args)
    elif args.command == "merge-models":
        handle_merge(args)


if __name__ == "__main__":
    main()