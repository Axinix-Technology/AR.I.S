# pyright: ignore
"""
AR.I.S Whole-System Capability Audit — Test Instrument

Purpose:
  Pure observation instrument for the AR.I.S Black-Box Whole-System Capability Audit.
  Executes ONLY the real public AR.I.S CLI entry point via subprocess.
  Captures and preserves raw stdout, stderr, execution timing, and metadata.

Strict Integrity Rules:
  - ZERO cognitive or internal AR.I.S imports.
  - ZERO orchestration of cognitive components.
  - ZERO answer injection or knowledge injection.
  - ZERO automated PASS/FAIL decision logic.
  - ZERO automated test looping.
  - ZERO modification or transformation of AR.I.S output.
"""

import sys
import os
import subprocess
import time
import argparse
from typing import Dict, Any, List

# Workspace Root
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Python Interpreter (Virtual Environment)
VENV_PYTHON = os.path.join(WORKSPACE_ROOT, "venv", "Scripts", "python.exe")
if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = sys.executable

# Registry of All Audit Scenarios from Specification
# Each entry specifies the exact public command(s), input text, and expected behavior.
# No internal components are defined here; only public CLI invocations.
AUDIT_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "A1": {
        "group": "Group A: Basic Language Understanding",
        "description": "Simple fact storage and representation",
        "input": "Ravi works at Axinix.",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Ravi works at Axinix." --verbose'
        ],
        "expected": "AR.I.S represents/stores the fact through its normal architecture. Fact stored with tokens, candidates, and relational bindings.",
    },
    "A2": {
        "group": "Group A: Basic Language Understanding",
        "description": "Another relation storage and representation",
        "input": "Anita manages the engineering team.",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Anita manages the engineering team." --verbose'
        ],
        "expected": "AR.I.S handles a different relation through normal architecture.",
    },
    "A3": {
        "group": "Group A: Basic Language Understanding",
        "description": "Entity distinction across two facts",
        "input": "Context: Ravi works at Axinix. Axinix is located in Coimbatore.\nQuestion: Where is Axinix located?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Ravi works at Axinix."',
            f'"{VENV_PYTHON}" manage.py brain store "Axinix is located in Coimbatore."',
            f'"{VENV_PYTHON}" manage.py brain reason "Where is Axinix located?" --verbose'
        ],
        "expected": "Coimbatore",
    },
    "B1": {
        "group": "Group B: Question -> Relevant Relation -> Answer",
        "description": "Subject variable query",
        "input": "Context: Ravi helped Anita.\nQuestion: Who helped Anita?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Ravi helped Anita."',
            f'"{VENV_PYTHON}" manage.py brain reason "Who helped Anita?" --verbose'
        ],
        "expected": "Ravi",
    },
    "B2": {
        "group": "Group B: Question -> Relevant Relation -> Answer",
        "description": "Object variable query (opposite end of same relationship)",
        "input": "Context: Ravi helped Anita.\nQuestion: Who did Ravi help?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Ravi helped Anita."',
            f'"{VENV_PYTHON}" manage.py brain reason "Who did Ravi help?" --verbose'
        ],
        "expected": "Anita",
    },
    "B3": {
        "group": "Group B: Question -> Relevant Relation -> Answer",
        "description": "Locational relation query",
        "input": "Context: Ravi works at Axinix.\nQuestion: Where does Ravi work?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Ravi works at Axinix."',
            f'"{VENV_PYTHON}" manage.py brain reason "Where does Ravi work?" --verbose'
        ],
        "expected": "Axinix",
    },
    "B4": {
        "group": "Group B: Question -> Relevant Relation -> Answer",
        "description": "Subject relation query for work relation",
        "input": "Context: Ravi works at Axinix.\nQuestion: Who works at Axinix?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Ravi works at Axinix."',
            f'"{VENV_PYTHON}" manage.py brain reason "Who works at Axinix?" --verbose'
        ],
        "expected": "Ravi",
    },
    "C1": {
        "group": "Group C: Multiple Identical Occurrences",
        "description": "First action with entity 'apple'",
        "input": "Context: Ravi bought an apple and ate the apple.\nQuestion: What did Ravi buy?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Ravi bought an apple and ate the apple."',
            f'"{VENV_PYTHON}" manage.py brain reason "What did Ravi buy?" --verbose'
        ],
        "expected": "apple (with evidence identifying the buy relation rather than ate)",
    },
    "C2": {
        "group": "Group C: Multiple Identical Occurrences",
        "description": "Second action with identical entity 'apple'",
        "input": "Context: Ravi bought an apple and ate the apple.\nQuestion: What did Ravi eat?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Ravi bought an apple and ate the apple."',
            f'"{VENV_PYTHON}" manage.py brain reason "What did Ravi eat?" --verbose'
        ],
        "expected": "apple (with evidence identifying the eat relation rather than buy)",
    },
    "C3": {
        "group": "Group C: Multiple Identical Occurrences",
        "description": "Two identical entity occurrences in distinct syntactic roles",
        "input": "Context: Ravi told Anita that Anita should call Arun.\nQuestion: Who should call Arun?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Ravi told Anita that Anita should call Arun."',
            f'"{VENV_PYTHON}" manage.py brain reason "Who should call Arun?" --verbose'
        ],
        "expected": "Anita (resolving the embedded clause subject occurrence)",
    },
    "D1": {
        "group": "Group D: Subject / Object",
        "description": "Query for subject",
        "input": "Context: Ravi helped Anita.\nQuestion: Who helped Anita?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Ravi helped Anita."',
            f'"{VENV_PYTHON}" manage.py brain reason "Who helped Anita?" --verbose'
        ],
        "expected": "Ravi",
    },
    "D2": {
        "group": "Group D: Subject / Object",
        "description": "Query for object",
        "input": "Context: Ravi helped Anita.\nQuestion: Who did Ravi help?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Ravi helped Anita."',
            f'"{VENV_PYTHON}" manage.py brain reason "Who did Ravi help?" --verbose'
        ],
        "expected": "Anita",
    },
    "D3": {
        "group": "Group D: Subject / Object",
        "description": "Reversed sentence subject query",
        "input": "Context: Anita helped Ravi.\nQuestion: Who helped Ravi?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Anita helped Ravi."',
            f'"{VENV_PYTHON}" manage.py brain reason "Who helped Ravi?" --verbose'
        ],
        "expected": "Anita",
    },
    "D4": {
        "group": "Group D: Subject / Object",
        "description": "Reversed sentence object query",
        "input": "Context: Anita helped Ravi.\nQuestion: Who did Anita help?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Anita helped Ravi."',
            f'"{VENV_PYTHON}" manage.py brain reason "Who did Anita help?" --verbose'
        ],
        "expected": "Ravi",
    },
    "E1": {
        "group": "Group E: Active / Passive",
        "description": "Active voice question on active context",
        "input": "Context: Ravi helped Anita.\nQuestion: Who helped Anita?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Ravi helped Anita."',
            f'"{VENV_PYTHON}" manage.py brain reason "Who helped Anita?" --verbose'
        ],
        "expected": "Ravi",
    },
    "E2": {
        "group": "Group E: Active / Passive",
        "description": "Active voice question on passive context",
        "input": "Context: Anita was helped by Ravi.\nQuestion: Who helped Anita?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Anita was helped by Ravi."',
            f'"{VENV_PYTHON}" manage.py brain reason "Who helped Anita?" --verbose'
        ],
        "expected": "Ravi",
    },
    "E3": {
        "group": "Group E: Active / Passive",
        "description": "Recipient query on passive context",
        "input": "Context: Anita was helped by Ravi.\nQuestion: Who received help from Ravi?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Anita was helped by Ravi."',
            f'"{VENV_PYTHON}" manage.py brain reason "Who received help from Ravi?" --verbose'
        ],
        "expected": "Anita",
    },
    "F1": {
        "group": "Group F: Same Entity, Different Question",
        "description": "Entity definition query",
        "input": "Context: Google was founded in 1998 by Larry Page and Sergey Brin.\nQuestion: What is Google?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Google was founded in 1998 by Larry Page and Sergey Brin."',
            f'"{VENV_PYTHON}" manage.py brain reason "What is Google?" --verbose'
        ],
        "expected": "Definition / company or UNSUPPORTED from context if not stated",
    },
    "F2": {
        "group": "Group F: Same Entity, Different Question",
        "description": "Temporal founding query",
        "input": "Context: Google was founded in 1998 by Larry Page and Sergey Brin.\nQuestion: When was Google founded?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Google was founded in 1998 by Larry Page and Sergey Brin."',
            f'"{VENV_PYTHON}" manage.py brain reason "When was Google founded?" --verbose'
        ],
        "expected": "1998",
    },
    "F3": {
        "group": "Group F: Same Entity, Different Question",
        "description": "Founder agent query",
        "input": "Context: Google was founded in 1998 by Larry Page and Sergey Brin.\nQuestion: Who founded Google?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Google was founded in 1998 by Larry Page and Sergey Brin."',
            f'"{VENV_PYTHON}" manage.py brain reason "Who founded Google?" --verbose'
        ],
        "expected": "Larry Page and Sergey Brin",
    },
    "F4": {
        "group": "Group F: Same Entity, Different Question",
        "description": "Missing location query on entity",
        "input": "Context: Google was founded in 1998 by Larry Page and Sergey Brin.\nQuestion: Where was Google founded?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Google was founded in 1998 by Larry Page and Sergey Brin."',
            f'"{VENV_PYTHON}" manage.py brain reason "Where was Google founded?" --verbose'
        ],
        "expected": "NOT_DERIVABLE / UNSUPPORTED (context does not contain location)",
    },
    "G1": {
        "group": "Group G: Multi-Hop Reasoning",
        "description": "3-hop relational reasoning chain",
        "input": "Context:\nRavi works at Axinix.\nAxinix is located in Coimbatore.\nCoimbatore is in Tamil Nadu.\nQuestion: Which state is the company Ravi works at located in?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Ravi works at Axinix."',
            f'"{VENV_PYTHON}" manage.py brain store "Axinix is located in Coimbatore."',
            f'"{VENV_PYTHON}" manage.py brain store "Coimbatore is in Tamil Nadu."',
            f'"{VENV_PYTHON}" manage.py brain reason "Which state is the company Ravi works at located in?" --verbose'
        ],
        "expected": "Tamil Nadu (via chain: Ravi -> Axinix -> Coimbatore -> Tamil Nadu)",
    },
    "H1": {
        "group": "Group H: Paraphrase",
        "description": "Paraphrase variant 1",
        "input": "Context: Ravi works at Axinix.\nQuestion: Where does Ravi work?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Ravi works at Axinix."',
            f'"{VENV_PYTHON}" manage.py brain reason "Where does Ravi work?" --verbose'
        ],
        "expected": "Axinix",
    },
    "H2": {
        "group": "Group H: Paraphrase",
        "description": "Paraphrase variant 2",
        "input": "Context: Ravi works at Axinix.\nQuestion: Which company does Ravi work for?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Ravi works at Axinix."',
            f'"{VENV_PYTHON}" manage.py brain reason "Which company does Ravi work for?" --verbose'
        ],
        "expected": "Axinix",
    },
    "H3": {
        "group": "Group H: Paraphrase",
        "description": "Paraphrase variant 3",
        "input": "Context: Ravi works at Axinix.\nQuestion: What company employs Ravi?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Ravi works at Axinix."',
            f'"{VENV_PYTHON}" manage.py brain reason "What company employs Ravi?" --verbose'
        ],
        "expected": "Axinix",
    },
    "H4": {
        "group": "Group H: Paraphrase",
        "description": "Paraphrase variant 4",
        "input": "Context: Ravi works at Axinix.\nQuestion: Ravi is employed by which company?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Ravi works at Axinix."',
            f'"{VENV_PYTHON}" manage.py brain reason "Ravi is employed by which company?" --verbose'
        ],
        "expected": "Axinix",
    },
    "I1": {
        "group": "Group I: Unseen Entities",
        "description": "Unseen entity pair 1",
        "input": "Context: Kiran helped Meena.\nQuestion: Who did Kiran help?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Kiran helped Meena."',
            f'"{VENV_PYTHON}" manage.py brain reason "Who did Kiran help?" --verbose'
        ],
        "expected": "Meena",
    },
    "I2": {
        "group": "Group I: Unseen Entities",
        "description": "Unseen entity pair 2",
        "input": "Context: Devika helped Arjun.\nQuestion: Who did Devika help?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Devika helped Arjun."',
            f'"{VENV_PYTHON}" manage.py brain reason "Who did Devika help?" --verbose'
        ],
        "expected": "Arjun",
    },
    "J1": {
        "group": "Group J: Multi-Relation Context",
        "description": "Query relation 1 in multi-relation context",
        "input": "Context:\nRavi works at Axinix.\nAxinix is located in Coimbatore.\nRavi manages the engineering team.\nAnita works with Ravi.\nQuestion: Where does Ravi work?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Ravi works at Axinix."',
            f'"{VENV_PYTHON}" manage.py brain store "Axinix is located in Coimbatore."',
            f'"{VENV_PYTHON}" manage.py brain store "Ravi manages the engineering team."',
            f'"{VENV_PYTHON}" manage.py brain store "Anita works with Ravi."',
            f'"{VENV_PYTHON}" manage.py brain reason "Where does Ravi work?" --verbose'
        ],
        "expected": "Axinix",
    },
    "J2": {
        "group": "Group J: Multi-Relation Context",
        "description": "Query relation 2 in multi-relation context",
        "input": "Question: Where is Axinix located?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain reason "Where is Axinix located?" --verbose'
        ],
        "expected": "Coimbatore",
    },
    "J3": {
        "group": "Group J: Multi-Relation Context",
        "description": "Query relation 3 in multi-relation context",
        "input": "Question: Who manages the engineering team?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain reason "Who manages the engineering team?" --verbose'
        ],
        "expected": "Ravi",
    },
    "J4": {
        "group": "Group J: Multi-Relation Context",
        "description": "Query relation 4 in multi-relation context",
        "input": "Question: Who works with Ravi?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain reason "Who works with Ravi?" --verbose'
        ],
        "expected": "Anita",
    },
    "K1": {
        "group": "Group K: Ambiguity",
        "description": "Ambiguous pronoun reference",
        "input": "Context: Ravi told Arun that he should leave.\nQuestion: Who should leave?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Ravi told Arun that he should leave."',
            f'"{VENV_PYTHON}" manage.py brain reason "Who should leave?" --verbose'
        ],
        "expected": "AMBIGUOUS (should not arbitrarily pick Ravi or Arun without evidence)",
    },
    "L1": {
        "group": "Group L: Missing Information",
        "description": "Information missing from supplied context",
        "input": "Context: Ravi works at Axinix.\nQuestion: What is the population of Coimbatore?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Ravi works at Axinix."',
            f'"{VENV_PYTHON}" manage.py brain reason "What is the population of Coimbatore?" --verbose'
        ],
        "expected": "NOT_DERIVABLE FROM PROVIDED CONTEXT / UNSUPPORTED / ACQUIRABLE",
    },
    "M1": {
        "group": "Group M: Conflicting Information",
        "description": "Two conflicting relations for same entity",
        "input": "Context:\nRavi works at Axinix.\nRavi works at NovaTech.\nQuestion: Where does Ravi work?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Ravi works at Axinix."',
            f'"{VENV_PYTHON}" manage.py brain store "Ravi works at NovaTech."',
            f'"{VENV_PYTHON}" manage.py brain reason "Where does Ravi work?" --verbose'
        ],
        "expected": "CONFLICT / MULTIPLE CANDIDATES / AMBIGUOUS",
    },
    "N1": {
        "group": "Group N: Generalization",
        "description": "Novel relationship phrasing and entities 1",
        "input": "Context: Devika is employed by Orion Systems.\nQuestion: Which organization employs Devika?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Devika is employed by Orion Systems."',
            f'"{VENV_PYTHON}" manage.py brain reason "Which organization employs Devika?" --verbose'
        ],
        "expected": "Orion Systems",
    },
    "N2": {
        "group": "Group N: Generalization",
        "description": "Novel relationship phrasing and entities 2",
        "input": "Context: Karthik is employed by BluePeak.\nQuestion: Where does Karthik work?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Karthik is employed by BluePeak."',
            f'"{VENV_PYTHON}" manage.py brain reason "Where does Karthik work?" --verbose'
        ],
        "expected": "BluePeak",
    },
    "O1": {
        "group": "Group O: External Knowledge Acquisition",
        "description": "Real question requiring external knowledge",
        "input": "What is Google?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain reason "What is Google?" --verbose'
        ],
        "expected": "Epistemic state evaluated; external acquisition selected autonomously if needed.",
    },
    "P1": {
        "group": "Group P: Procedural / Action Capability",
        "description": "File creation instruction through public interface",
        "input": 'Create a text file named aris_test.txt containing the text "ARIS test".',
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain reason "Create a text file named aris_test.txt containing the text \\"ARIS test\\"." --verbose'
        ],
        "expected": "Capability dispatch or NOT DEMONSTRATED through CLI interface.",
    },
    "Q1": {
        "group": "Group Q: Memory",
        "description": "Explicit memory teaching and retrieval",
        "input": "Teach: Arun's favorite development language is Python.\nAsk: What is Arun's favorite development language?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Arun\'s favorite development language is Python."',
            f'"{VENV_PYTHON}" manage.py brain reason "What is Arun\'s favorite development language?" --verbose'
        ],
        "expected": "Python",
    },
    "Q2": {
        "group": "Group Q: Memory",
        "description": "Paraphrase retrieval from memory",
        "input": "Question: Which programming language does Arun prefer?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain reason "Which programming language does Arun prefer?" --verbose'
        ],
        "expected": "Python",
    },
    "R1": {
        "group": "Group R: Learning / Reuse",
        "description": "Demonstration learning and reuse on structurally similar scenario",
        "input": "Teach: Arun is brother of Bala. -> Who is brother of Bala? -> Arun\nTest Unseen: Who is brother of Qorin?",
        "commands": [
            f'"{VENV_PYTHON}" manage.py brain store "Kiran is brother of Qorin."',
            f'"{VENV_PYTHON}" manage.py brain learn --fact "Arun is brother of Bala." --question "Who is brother of Bala?" --answer "Arun" --verbose',
            f'"{VENV_PYTHON}" manage.py brain reason "Who is brother of Qorin?" --verbose'
        ],
        "expected": "Kiran (reusing the learned reverse traversal template)",
    },
}


def execute_single_command(cmd: str) -> Dict[str, Any]:
    """
    Executes a single command line via subprocess.
    Captures raw stdout and stderr without modifying, parsing, or filtering.
    """
    start_time = time.time()
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=WORKSPACE_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
        duration = time.time() - start_time
        return {
            "cmd": cmd,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
            "duration_sec": round(duration, 3),
        }
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        return {
            "cmd": cmd,
            "stdout": "",
            "stderr": "Command timed out after 120 seconds.",
            "returncode": -1,
            "duration_sec": round(duration, 3),
        }
    except Exception as e:
        duration = time.time() - start_time
        return {
            "cmd": cmd,
            "stdout": "",
            "stderr": f"Execution error: {str(e)}",
            "returncode": -1,
            "duration_sec": round(duration, 3),
        }


def format_test_card(
    test_id: str,
    scenario: Dict[str, Any],
    exec_results: List[Dict[str, Any]],
) -> str:
    """
    Formats the raw execution results into the required Section 5 format:
      TEST ID:
      INPUT:
      AR.I.S COMMAND:
      RAW AR.I.S OUTPUT:
      EXPECTED BEHAVIOR:
      OBSERVED BEHAVIOR:
      RESULT:
    """
    lines = []
    lines.append("=" * 78)
    lines.append(f"TEST ID: {test_id}")
    lines.append(f"GROUP: {scenario.get('group', 'N/A')}")
    lines.append(f"INPUT:\n{scenario.get('input', '')}")
    lines.append("-" * 78)

    all_commands = [res["cmd"] for res in exec_results]
    lines.append(f"AR.I.S COMMAND(S):\n" + "\n".join(f"  $ {c}" for c in all_commands))
    lines.append("-" * 78)

    raw_outputs = []
    total_time = 0.0
    for idx, res in enumerate(exec_results, 1):
        total_time += res.get("duration_sec", 0.0)
        cmd_out = res.get("stdout", "").strip()
        cmd_err = res.get("stderr", "").strip()
        block = f"[Command {idx}] (Duration: {res.get('duration_sec')}s | Exit code: {res.get('returncode')}):\n"
        if cmd_out:
            block += f"{cmd_out}\n"
        if cmd_err:
            block += f"[STDERR]:\n{cmd_err}\n"
        raw_outputs.append(block)

    lines.append("RAW AR.I.S OUTPUT:")
    lines.append("\n".join(raw_outputs).strip())
    lines.append("-" * 78)
    lines.append(f"TOTAL EXECUTION TIME: {total_time:.3f}s")
    lines.append(f"EXPECTED BEHAVIOR:\n{scenario.get('expected', '')}")
    lines.append("-" * 78)
    lines.append("OBSERVED BEHAVIOR:")
    lines.append("  [To be entered by human operator based on raw output above]")
    lines.append("RESULT:")
    lines.append("  [To be assigned by human operator: PASS | PARTIAL | FAIL | UNSUPPORTED | AMBIGUOUS | NOT_DERIVABLE | CONFLICT]")
    lines.append("=" * 78)
    return "\n".join(lines)


def list_scenarios():
    """Prints all scenario IDs, groups, and inputs."""
    print("\n" + "=" * 78)
    print("  AR.I.S WHOLE-SYSTEM CAPABILITY AUDIT — SCENARIO CATALOG")
    print("=" * 78)
    for test_id, data in AUDIT_SCENARIOS.items():
        print(f"[{test_id:<3}] {data['group']}")
        print(f"      Input   : {data['input'].replace(chr(10), ' | ')}")
        print(f"      Expected: {data['expected']}")
        print()
    print("=" * 78)
    print("Usage:")
    print("  python scripts/audit_capability_instrument.py --test <TEST_ID>")
    print("  python scripts/audit_capability_instrument.py --run \"<COMMAND>\"")
    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="AR.I.S Capability Audit Test Instrument (Observer & Capture Only)"
    )
    parser.add_argument(
        "--test",
        "-t",
        type=str,
        default=None,
        help="Execute a specific test scenario by ID (e.g. A1, B1, G1). Executes commands, captures raw output, and displays the Section 5 audit card.",
    )
    parser.add_argument(
        "--run",
        "-r",
        type=str,
        default=None,
        help="Execute an arbitrary single AR.I.S public command and capture raw output in audit format.",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all defined capability audit scenarios and their expected behaviors.",
    )

    args = parser.parse_args()

    if args.list:
        list_scenarios()
        return

    if args.run:
        cmd = args.run
        print(f"\n[AUDIT INSTRUMENT] Executing single command: {cmd}")
        res = execute_single_command(cmd)
        print("\n" + "=" * 78)
        print(f"AR.I.S COMMAND: {cmd}")
        print(f"DURATION: {res['duration_sec']}s (Exit code: {res['returncode']})")
        print("-" * 78)
        print("RAW AR.I.S OUTPUT:")
        print(res["stdout"])
        if res["stderr"]:
            print(f"[STDERR]:\n{res['stderr']}")
        print("=" * 78 + "\n")
        return

    if args.test:
        test_id = args.test.upper()
        if test_id not in AUDIT_SCENARIOS:
            print(f"Error: Unknown test ID '{test_id}'. Available: {list(AUDIT_SCENARIOS.keys())}")
            sys.exit(1)

        scenario = AUDIT_SCENARIOS[test_id]
        print(f"\n[AUDIT INSTRUMENT] Executing single test scenario: {test_id}")
        exec_results = []
        for cmd in scenario["commands"]:
            print(f"  -> Running: {cmd}")
            r = execute_single_command(cmd)
            exec_results.append(r)

        card = format_test_card(test_id, scenario, exec_results)
        print("\n" + card + "\n")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
