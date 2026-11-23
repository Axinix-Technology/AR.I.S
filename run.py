"""
AR.I.S — Command Line Interface & Cognitive Runtime (2026 Modern Architecture)
Features:
  - BPE Tokenizer V2 management with ChatML support and exact byte-level roundtripping
  - Unified ArisLM training (RoPE + GQA + SwiGLU + RMSNorm + AMP bfloat16 + Cosine Warmup)
  - Interactive chat with vectorized KV Cache inference and repetition penalty
  - Clean held-out benchmark evaluation with strict Decimal parser and sandboxed runner
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path
import torch
import pyarrow.parquet as pq

from aris.config import BASE_DIR, DATA_DIR, MODELS_DIR, TOKENIZER_PATH, LM_CHECKPOINT_PATH, DEVICE
from aris.configs.model_config import ArisConfig
from aris.tokenizer.bpe_tokenizer import ArisTokenizer
from aris.models.aris_lm import ArisLM
from aris.training.trainer import ArisTrainer
from aris.inference.generation import ArisInferenceEngine
from aris.memory import ArisGraphStore, HybridGraphRetriever
from aris.core import AutonomousDeliberationEngine
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
    Executes neural training of modern ArisLM using the proven manifest-driven pipeline.
    """
    backup_dir = None
    if getattr(args, "backup_dir", None):
        backup_dir = Path(args.backup_dir)
    elif Path("/content/drive/MyDrive").exists():
        backup_dir = Path("/content/drive/MyDrive/ARIS_backup")

    w_steps = args.warmup_steps
    if getattr(args, "pilot", False) and args.warmup_steps == 500:
        w_steps = 100

    ArisTrainer.train_proven(
        rung=args.rung,
        pilot=args.pilot,
        budget_mult=args.budget_mult,
        total_steps=args.steps,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=w_steps,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        save_every=args.save_every,
        device_name=args.device,
        checkpoint_name=args.checkpoint_name,
        resume_from=args.resume_from,
        resume=args.resume,
        backup_dir=backup_dir,
    )


def handle_chat(args):
    """
    Runs autonomous interactive cognitive dialogue directly through ArisLM with KV Caching and Repetition Penalty.
    """
    tok_path = Path(args.tokenizer) if args.tokenizer else TOKENIZER_PATH
    ckpt_path = Path(args.checkpoint) if args.checkpoint else LM_CHECKPOINT_PATH

    if not tok_path.exists():
        print(f"Error: Tokenizer not found at {tok_path}.")
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
    print(f"Rep Penalty:{args.repetition_penalty}")
    print(f"Type 'exit' or 'quit' to end session.\n")

    tokenizer = ArisTokenizer.load(tok_path)
    model, step = ArisTrainer.load_model(ckpt_path, device=DEVICE)

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
                repetition_penalty=args.repetition_penalty,
            )

            if result.get("thought"):
                print(f"\n[Thought]: {result['thought']}")
            print(f"\n{result['response']}\n")

            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": result["response"]})

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


def handle_eval(args):
    """
    Runs standardized clean held-out evaluation benchmark suite.
    """
    from aris.evaluation.eval_harness_v2 import evaluate_checkpoints

    tok_path = Path(args.tokenizer) if args.tokenizer else TOKENIZER_PATH
    ckpt_path = Path(args.checkpoint) if args.checkpoint else LM_CHECKPOINT_PATH

    evaluate_checkpoints(
        device_name=args.device,
        custom_ckpt=ckpt_path if ckpt_path.exists() else None,
        custom_tok=tok_path if tok_path.exists() else None,
        max_samples=args.samples,
        output_path=Path(args.output) if args.output else None,
    )


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
    print(f"Model A:       {path_a}")
    print(f"Model B:       {path_b}")
    print(f"Output Target: {output_path}")
    print(f"Strategy:      {args.method.upper()}")

    ModelMerger.merge_and_save(
        model_paths=[path_a, path_b],
        output_path=output_path,
        method=args.method,
        alpha=args.alpha,
    )

    if getattr(args, "memory_b", None):
        mem_b_path = Path(args.memory_b)
        if mem_b_path.exists():
            store = ArisGraphStore()
            store.merge_from(mem_b_path)


def main():
    parser = argparse.ArgumentParser(description="AR.I.S — Artificial Intelligent System CLI (2026 Unified)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train Tokenizer
    p_tok = subparsers.add_parser("train-tokenizer", help="Train BPE Tokenizer on UltraChat")
    p_tok.add_argument("--vocab-size", type=int, default=8192)
    p_tok.add_argument("--max-lines", type=int, default=200000)
    p_tok.add_argument("--output", type=str, default=None)

    # Train Language Model (Proven Manifest-Driven Engine)
    p_lm = subparsers.add_parser("train-lm", help="Train ArisLM with proven manifest-driven pipeline and AMP bfloat16")
    p_lm.add_argument("--rung", type=str, default="small", choices=["small", "medium", "gen3", "pico"], help="Model rung ('small', 'medium'/'gen3', 'pico')")
    p_lm.add_argument("--budget-mult", type=float, default=10.0, help="Chinchilla parameter multiple budget (e.g. 10.0 for 10x, 20.0 for 20x)")
    p_lm.add_argument("--pilot", action="store_true", help="Run 1.0x parameter budget pilot run")
    p_lm.add_argument("--steps", type=int, default=None, help="Explicit total steps override")
    p_lm.add_argument("--batch-size", type=int, default=16, help="Batch size per forward micro-step")
    p_lm.add_argument("--accum-steps", type=int, default=2, help="Gradient accumulation steps")
    p_lm.add_argument("--lr", type=float, default=6e-4, help="Peak learning rate")
    p_lm.add_argument("--min-lr", type=float, default=6e-5, help="Minimum learning rate")
    p_lm.add_argument("--warmup-steps", type=int, default=500, help="Warmup steps for cosine schedule")
    p_lm.add_argument("--grad-clip", type=float, default=1.0, help="Max gradient norm")
    p_lm.add_argument("--save-every", type=int, default=500, help="Save checkpoint frequency in steps")
    p_lm.add_argument("--log-every", type=int, default=50, help="Log metrics frequency in steps")
    p_lm.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device")
    p_lm.add_argument("--checkpoint-name", type=str, default=None, help="Custom checkpoint filename to save")
    p_lm.add_argument("--resume-from", type=str, default=None, help="Explicit checkpoint path to resume from")
    p_lm.add_argument("--no-resume", dest="resume", action="store_false", help="Do not resume from existing checkpoint")
    p_lm.add_argument("--backup-dir", type=str, default=None, help="Directory to mirror checkpoints upon every save (Google Drive)")

    # Chat
    p_chat = subparsers.add_parser("chat", help="Start interactive dialogue session with ArisLM")
    p_chat.add_argument("--tokenizer", type=str, default=None)
    p_chat.add_argument("--checkpoint", type=str, default=None)
    p_chat.add_argument("--temperature", type=float, default=0.7)
    p_chat.add_argument("--top-k", type=int, default=40)
    p_chat.add_argument("--top-p", type=float, default=0.9)
    p_chat.add_argument("--repetition-penalty", type=float, default=1.15, help="Multiplicative penalty for repeating previously generated tokens")
    p_chat.add_argument("--max-tokens", type=int, default=128)

    # Evaluation Benchmark
    p_eval = subparsers.add_parser("eval", help="Run clean held-out cognitive evaluation benchmark")
    p_eval.add_argument("--tokenizer", type=str, default=None)
    p_eval.add_argument("--checkpoint", type=str, default=None)
    p_eval.add_argument("--samples", type=int, default=100, help="Number of samples to evaluate per benchmark stage")
    p_eval.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device")
    p_eval.add_argument("--output", type=str, default=None, help="Path to write JSON results")

    # Merge Models
    p_merge = subparsers.add_parser("merge-models", help="Merge independently trained checkpoints")
    p_merge.add_argument("--model-a", type=str, required=True, help="Path to Checkpoint A")
    p_merge.add_argument("--model-b", type=str, required=True, help="Path to Checkpoint B")
    p_merge.add_argument("--output", type=str, default=None, help="Path to write merged checkpoint")
    p_merge.add_argument("--method", type=str, default="average", choices=["average", "slerp"], help="Merging strategy")
    p_merge.add_argument("--alpha", type=float, default=0.5, help="Interpolation weight")
    p_merge.add_argument("--memory-b", type=str, default=None, help="Optional path to teammate's knowledge graph")

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
