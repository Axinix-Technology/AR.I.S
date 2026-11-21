"""
Unit Tests for AR.I.S Clean Eval Harness V2
Tests:
  1. Strict Decimal Parser edge cases (>2^53 integers, exponential, currency/percent stripping, comma/empty, negative zero)
  2. SQuAD Exact Match & Token F1 metrics
  3. 4-Gram repetition loop detector
  4. Bootstrap 95% Confidence Interval calculation
  5. Sandboxed Python execution and security interceptors
"""

import math
from decimal import Decimal
import pytest
from aris.evaluation.eval_harness_v2 import (
    parse_final_calculation_decimal,
    compute_f1_and_em,
    check_4gram_repetition,
    compute_bootstrap_ci,
    run_sandboxed_candidate,
)


def test_decimal_parser_edge_cases():
    # 1. Standard GSM8K tag
    assert parse_final_calculation_decimal("The answer is #### 42") == "42"
    assert parse_final_calculation_decimal("Final Calculation: 1,234.56") == "1234.56"

    # 2. Currency and percent stripping
    assert parse_final_calculation_decimal("#### $42") == "42"
    assert parse_final_calculation_decimal("#### 42%") == "42"
    assert parse_final_calculation_decimal("Final Calculation: €99.50") == "99.5"

    # 3. > 2^53 integers (float precision limit is 9007199254740992)
    large_int = "9007199254740993123456789"
    assert parse_final_calculation_decimal(f"#### {large_int}") == large_int

    # 4. Exponential notation
    assert parse_final_calculation_decimal("#### 1e3") == "1000"

    # 5. RHS equation extraction
    assert parse_final_calculation_decimal("Final Calculation: 5 + 7 = 12") == "12"

    # 6. Negative zero
    assert parse_final_calculation_decimal("#### -0") == "0"
    assert parse_final_calculation_decimal("#### 0.00") == "0"

    # 7. Invalid comma/empty returns None
    assert parse_final_calculation_decimal("#### ,") is None
    assert parse_final_calculation_decimal("#### ") is None
    assert parse_final_calculation_decimal("") is None
    assert parse_final_calculation_decimal("No answer here") is None


def test_stanford_f1_and_em():
    em, f1 = compute_f1_and_em("The quick brown fox", "the quick brown fox")
    assert em == 1.0
    assert f1 == 1.0

    em, f1 = compute_f1_and_em("A brown fox jumps", "the brown fox")
    assert em == 0.0
    # Overlap is {"brown", "fox"} (2 tokens). pred has 3 ("brown", "fox", "jumps"), gt has 2 ("brown", "fox")
    # precision = 2/3, recall = 2/2 = 1.0 -> f1 = 2 * (2/3 * 1) / (5/3) = 4/5 = 0.8
    assert abs(f1 - 0.8) < 1e-4

    em, f1 = compute_f1_and_em("completely different", "totally unrelated")
    assert em == 0.0
    assert f1 == 0.0


def test_4gram_repetition_detector():
    repeated_text = "the agent is working the agent is working the agent is working"
    assert check_4gram_repetition(repeated_text) is True

    unique_text = "the quick brown fox jumps over the lazy dog in the park"
    assert check_4gram_repetition(unique_text) is False

    short_text = "too short text"
    assert check_4gram_repetition(short_text) is False


def test_bootstrap_ci():
    # Constant values
    scores = [1.0, 1.0, 1.0, 1.0, 1.0]
    mean, low, high = compute_bootstrap_ci(scores, n_boot=100)
    assert mean == 1.0
    assert low == 1.0
    assert high == 1.0

    # Varied values
    scores = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    mean, low, high = compute_bootstrap_ci(scores, n_boot=200, seed=42)
    assert mean == 0.5
    assert low < mean
    assert high > mean


def test_sandboxed_candidate_execution():
    # Passing code
    code = "def add(a, b):\n    return a + b\n"
    asserts = "assert add(2, 3) == 5\n"
    passed, msg = run_sandboxed_candidate(code, test_assertions=asserts, timeout_sec=2.0)
    assert passed is True

    # Failing code (AssertionError)
    failing_asserts = "assert add(2, 3) == 6\n"
    passed, msg = run_sandboxed_candidate(code, test_assertions=failing_asserts, timeout_sec=2.0)
    assert passed is False
    assert "AssertionError" in msg

    # Security check: write access blocked
    exploit_code = "with open('malicious.txt', 'w') as f:\n    f.write('bad')\n"
    passed, msg = run_sandboxed_candidate(exploit_code, timeout_sec=2.0)
    assert passed is False
    assert "PermissionError" in msg or "Defense Guard" in msg
