"""
AR.I.S Manifest Hashes & Strict Numeric Parser with Unit Tests
"""

import sys
import re
import hashlib
import multiprocessing
from pathlib import Path
from typing import Optional, Any

DATA_DIR = Path("data")

# =====================================================================
# 3. MANIFEST HASH GENERATION (EMPIRICAL SHA256)
# =====================================================================
def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def generate_all_manifest_hashes():
    print("="*75)
    print("3. EMPIRICAL SHA256 HASHES FOR ALL DATA FILES")
    print("="*75)
    files = [
        "data/gsm8k/train.parquet",
        "data/mbpp/prompt.parquet",
        "data/mbpp/train.parquet",
        "data/mbpp/test.parquet",
        "data/qasrl_gs/qasrl-gs/dev.jsonl.gz",
        "data/qasrl_gs/qasrl-gs/test.jsonl.gz",
        "data/hotpotqa/distractor/train-00000-of-00002.parquet",
        "data/hotpotqa/distractor/train-00001-of-00002.parquet",
        "data/squad_v2/train.parquet",
        "data/dolly/databricks-dolly-15k.jsonl",
        "data/smoltalk/everyday_conversations.parquet",
        "data/smoltalk/smol_magpie_ultra.parquet",
        "data/custom_chat.json",
        "data/vision/detail_23k.json",
        "ud-treebanks-v2.18.tgz",
        "models/aris_bpe_tokenizer.json",
        "aris_lm_checkpoint.pt",
        "models/aris_lm_checkpoint.pt",
    ]
    manifest = {}
    for f_str in files:
        p = Path(f_str)
        if p.exists():
            h = compute_sha256(p)
            manifest[f_str] = h
            print(f"{f_str:55} : {h}")
        else:
            print(f"{f_str:55} : NOT FOUND")
    return manifest


# =====================================================================
# 4. STRICT NUMERIC PARSER (LAST MATCH + NORMALIZATION) & UNIT TESTS
# =====================================================================
def parse_final_calculation(text: str) -> Optional[str]:
    """
    Extracts and normalizes the LAST 'Final Calculation:' match in text.
    Handles:
    - Multiple scratchpad occurrences (takes the LAST one)
    - Comma removal ('1,000' -> '1000')
    - Integer float normalization ('42.0' -> '42')
    - Trailing periods/commas ('42.' -> '42')
    - Negative numbers ('-15.5')
    """
    if not text:
        return None
        
    # Find all occurrences of Final Calculation: <value>
    # Pattern looks for "Final Calculation:" followed by numeric token
    pattern = r"Final Calculation:\s*([+-]?(?:[\d,]+(?:\.\d+)?|\.\d+))"
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    if not matches:
        return None
        
    last_match = matches[-1]
    raw_val = last_match.group(1).strip()
    
    # Strip trailing punctuation
    raw_val = raw_val.rstrip(".,;")
    # Remove thousand separators
    clean_val = raw_val.replace(",", "")
    
    try:
        f_val = float(clean_val)
        # Normalize 42.0 -> '42', 3.14 -> '3.14'
        if f_val.is_integer():
            return str(int(f_val))
        return str(f_val)
    except ValueError:
        return clean_val


def run_parser_unit_tests():
    print("\n" + "="*75)
    print("4. PARSER UNIT TESTS WITH EDGE CASES")
    print("="*75)
    test_cases = [
        # (input_text, expected_output, description)
        ("Final Calculation: 42", "42", "Simple integer"),
        ("Final Calculation: 42.0", "42", "Float integer normalization (42.0 -> 42)"),
        ("Final Calculation: 42.", "42", "Trailing period (42. -> 42)"),
        ("Final Calculation: 1,234,567", "1234567", "Thousands commas (1,234,567 -> 1234567)"),
        ("Final Calculation: -15.5", "-15.5", "Negative decimal"),
        ("Final Calculation: 0.75", "0.75", "Decimal less than 1"),
        (
            "<|thought|>\nFirst try: Final Calculation: 10.\nWait, that's wrong.\nFinal Calculation: 25<|im_end|>",
            "25",
            "Multiple calculations in scratchpad (must take LAST match: 25)"
        ),
        (
            "Final Calculation: 1,000.00\nDone.",
            "1000",
            "Comma and decimal integer (1,000.00 -> 1000)"
        ),
        ("There is no calculation here.", None, "Missing calculation returns None"),
        ("Final Calculation:   99   ", "99", "Whitespace around number"),
    ]
    
    all_passed = True
    for text, expected, desc in test_cases:
        res = parse_final_calculation(text)
        passed = (res == expected)
        if not passed:
            all_passed = False
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {desc:55} -> Expected: {repr(expected):8} | Got: {repr(res)}")
        
    print(f"\nAll Parser Unit Tests Passed: {all_passed}")


# =====================================================================
# 4B. SANDBOXED MBPP EXECUTION RUNNER WITH TIMEOUT
# =====================================================================
def _worker_exec(code: str, test_assertion: str, queue: multiprocessing.Queue):
    """Executes code and assertions in an isolated worker process."""
    exec_globals = {}
    try:
        exec(code, exec_globals)
        exec(test_assertion, exec_globals)
        queue.put({"success": True, "error": None})
    except Exception as e:
        queue.put({"success": False, "error": type(e).__name__ + ": " + str(e)})

def run_sandboxed_python(code: str, test_assertion: str, timeout_seconds: float = 3.0) -> bool:
    """Executes Python code with strict timeout and isolated process namespace."""
    queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_worker_exec, args=(code, test_assertion, queue))
    proc.start()
    proc.join(timeout=timeout_seconds)
    
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return False # Timed out / infinite loop
        
    if not queue.empty():
        res = queue.get()
        return res["success"]
    return False

def test_mbpp_sandbox():
    print("\n" + "="*75)
    print("4B. SANDBOXED MBPP EXECUTION TEST")
    print("="*75)
    # Valid code test
    valid_code = "def is_even(n):\n    return n % 2 == 0"
    valid_test = "assert is_even(4) == True\nassert is_even(5) == False"
    res1 = run_sandboxed_python(valid_code, valid_test, timeout_seconds=2.0)
    print(f"Valid Code Execution: {res1} (Expected: True)")
    
    # Failing assertion test
    failing_test = "assert is_even(5) == True"
    res2 = run_sandboxed_python(valid_code, failing_test, timeout_seconds=2.0)
    print(f"Failing Assertion Execution: {res2} (Expected: False)")
    
    # Infinite loop timeout test
    loop_code = "def is_even(n):\n    while True:\n        pass"
    res3 = run_sandboxed_python(loop_code, valid_test, timeout_seconds=1.0)
    print(f"Infinite Loop Timeout Protection: {res3} (Expected: False)")

if __name__ == "__main__":
    generate_all_manifest_hashes()
    run_parser_unit_tests()
    test_mbpp_sandbox()
