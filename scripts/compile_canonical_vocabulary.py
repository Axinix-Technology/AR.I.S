"""
AR.I.S — Canonical Vocabulary Compilation Script
Extracts discrete vocabulary tokens directly from training corpora and persists them
into models/aris_phase2_vocabulary.json. Zero hardcoded lists.
"""

import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.linguistic.canonical_tokenizer import Vocabulary, ArisCanonicalTokenizer
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "models", "aris_phase2_vocabulary.json")

def compile_vocabulary():
    vocab = Vocabulary()
    raw_tokenizer = ArisCanonicalTokenizer(vocabulary=vocab)

    # 1. declarative_cognitive_training.json
    decl_path = os.path.join(DATA_DIR, "declarative_cognitive_training.json")
    if os.path.exists(decl_path):
        with open(decl_path, "r", encoding="utf-8") as f:
            d = json.load(f)
            for item in d.get("canonical_declarative_corpus", []):
                for tok in raw_tokenizer.TOKEN_REGEX.findall(item.get("raw_text", "")):
                    vocab.add_token(tok)
                for syn in item.get("syntactic_variations", []):
                    for tok in raw_tokenizer.TOKEN_REGEX.findall(syn):
                        vocab.add_token(tok)

    # 2. cognitive_pragmatic_training.json
    prag_path = os.path.join(DATA_DIR, "cognitive_pragmatic_training.json")
    if os.path.exists(prag_path):
        with open(prag_path, "r", encoding="utf-8") as f:
            d = json.load(f)
            for group in ("speech_acts", "deictic_roles", "interrogative_roles", "predicate_roles"):
                for phrase_list in d.get(group, {}).values():
                    for phrase in phrase_list:
                        for tok in raw_tokenizer.TOKEN_REGEX.findall(phrase):
                            vocab.add_token(tok)

    # 3. aris_training_facts.json
    facts_path = os.path.join(DATA_DIR, "aris_training_facts.json")
    if os.path.exists(facts_path):
        with open(facts_path, "r", encoding="utf-8") as f:
            facts = json.load(f)
            for fact in facts:
                stmt = fact.get("statement", "")
                for tok in raw_tokenizer.TOKEN_REGEX.findall(stmt):
                    vocab.add_token(tok)

    # 4. aris_relation_abstraction_benchmark.json
    bench_path = os.path.join(DATA_DIR, "aris_relation_abstraction_benchmark.json")
    if os.path.exists(bench_path):
        with open(bench_path, "r", encoding="utf-8") as f:
            bench = json.load(f)
            for item in bench:
                q = item.get("question", "")
                for tok in raw_tokenizer.TOKEN_REGEX.findall(q):
                    vocab.add_token(tok)

    vocab.save(OUTPUT_FILE)
    print(f"Successfully compiled vocabulary from training corpora: {len(vocab)} tokens -> {OUTPUT_FILE}")

if __name__ == "__main__":
    compile_vocabulary()
