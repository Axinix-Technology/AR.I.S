# pyright: ignore
"""
AR.I.S — Brain Terminal CLI Command

Strict Architecture Rule:
CLI -> ArisBrain -> MongoDB
The CLI interacts solely with ArisBrain and contains zero database logic.
Provides clean terminal display focused purely on factual knowledge,
and includes the three-tier cognitive inspection microscope:
  - Tier 1: Atomic Tokens & Positional Coordinates
  - Tier 2: Concept & Entity Candidates (compounds, multi-token names)
  - Tier 3: Relational Bindings (WHO -- RELATION --> WHOM)
"""

from django.core.management.base import BaseCommand
from api.cognitive.brain import ArisBrain


class Command(BaseCommand):
    help = "AR.I.S Brain Terminal Interface: Store, retrieve, count, and inspect facts."

    def add_arguments(self, parser):
        subparsers = parser.add_subparsers(dest="action", help="Action to perform")

        # Subcommand: store
        store_parser = subparsers.add_parser("store", help="Store a new fact into AR.I.S Brain")
        store_parser.add_argument("statement", type=str, help="The fact statement to store")
        store_parser.add_argument("--topic", type=str, default="general", help="Topic/category")
        store_parser.add_argument("--tags", type=str, default="", help="Comma-separated tags")
        store_parser.add_argument("--confidence", type=float, default=1.0, help="Confidence rating (0.0 - 1.0)")
        store_parser.add_argument("--verbose", "-v", action="store_true", help="Display full metadata")

        # Subcommand: retrieve
        retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve facts from AR.I.S Brain")
        retrieve_parser.add_argument("--query", "-q", type=str, default=None, help="Search query string")
        retrieve_parser.add_argument("--topic", "-t", type=str, default=None, help="Filter by topic")
        retrieve_parser.add_argument("--tag", type=str, default=None, help="Filter by tag")
        retrieve_parser.add_argument("--limit", "-l", type=int, default=10, help="Maximum number of facts to return")
        retrieve_parser.add_argument("--verbose", "-v", action="store_true", help="Display full metadata")

        # Subcommand: inspect (Cognitive Debugging Microscope)
        inspect_parser = subparsers.add_parser("inspect", help="Inspect token and candidate representation of a fact")
        inspect_parser.add_argument("fact_id", type=str, help="ID of the fact to inspect")

        # Subcommand: count
        count_parser = subparsers.add_parser("count", help="Count stored facts in AR.I.S Brain")
        count_parser.add_argument("--topic", type=str, default=None, help="Filter count by topic")

        # Subcommand: query (Tier 4 Relational Retrieval)
        query_parser = subparsers.add_parser("query", help="Execute 1-hop relational retrieval query over stored bindings")
        query_parser.add_argument("query_text", nargs="?", type=str, default=None, help="Natural query string (e.g. 'Who leads Axinix...?')")
        query_parser.add_argument("--subject", "-s", type=str, default=None, help="Exact subject anchor")
        query_parser.add_argument("--relation", "-r", type=str, default=None, help="Exact relation span")
        query_parser.add_argument("--object", "-o", dest="obj", type=str, default=None, help="Exact object anchor")
        query_parser.add_argument("--direction", "-d", type=str, default=None, help="Direction (forward, reverse, link)")
        query_parser.add_argument("--limit", "-l", type=int, default=10, help="Maximum number of answers to return")
        query_parser.add_argument("--verbose", "-v", action="store_true", help="Display full structural binding details")

        # Subcommand: traverse (Tier 4-C Multi-Hop Relational Traversal)
        traverse_parser = subparsers.add_parser("traverse", help="Execute explicit multi-hop relational traversal chain")
        traverse_parser.add_argument("start_entity", type=str, help="Starting entity candidate (e.g. 'Axinix Technologies')")
        traverse_parser.add_argument("--hops", "-H", type=str, required=True, help="Comma-separated hops in format 'relation:direction', e.g. 'leads:reverse,is brother of:forward'")
        traverse_parser.add_argument("--limit", "-l", type=int, default=10, help="Maximum number of answers to return")
        traverse_parser.add_argument("--verbose", "-v", action="store_true", help="Display full traversal steps")

        # Subcommand: learn (Tier 5 Operation Learning)
        learn_parser = subparsers.add_parser("learn", help="Teach AR.I.S an abstract relational operation from a demonstration")
        learn_parser.add_argument("--fact", "-f", type=str, required=True, help="Demonstration fact (e.g. 'Arun is brother of Bala.')")
        learn_parser.add_argument("--question", "-q", type=str, required=True, help="Question pattern (e.g. 'Who is brother of Bala?')")
        learn_parser.add_argument("--answer", "-a", type=str, required=True, help="Target answer (e.g. 'Arun')")
        learn_parser.add_argument("--verbose", "-v", action="store_true", help="Display operation abstraction microscope")

        # Subcommand: reason (Tier 5 Autonomous Reasoning)
        reason_parser = subparsers.add_parser("reason", help="Reason an answer to an unseen question using learned operations")
        reason_parser.add_argument("question_text", type=str, help="Question text (e.g. 'Who is brother of Qorin?' or 'Who daxels Veltra?')")
        reason_parser.add_argument("--limit", "-l", type=int, default=10, help="Maximum number of answers to return")
        reason_parser.add_argument("--verbose", "-v", action="store_true", help="Display reasoning microscope")
        reason_parser.add_argument("--no-advisor", dest="use_advisor", action="store_false", default=True, help="Disable sensory structural advisor (ablation testing)")

        # Subcommand: feedback (Pillar 4 Human Feedback Loop)
        feedback_parser = subparsers.add_parser("feedback", help="Provide human feedback (confirm or deny) on a relational hypothesis")
        feedback_parser.add_argument("--subject", "-s", type=str, required=True, help="Subject entity")
        feedback_parser.add_argument("--relation", "-r", type=str, required=True, help="Relation span")
        feedback_parser.add_argument("--object", "-o", dest="obj", type=str, required=True, help="Object entity")
        feedback_parser.add_argument("--deny", action="store_true", help="Explicitly deny/refute hypothesis (confidence=0.0)")
        feedback_parser.add_argument("--confirm", action="store_true", help="Explicitly confirm hypothesis (confidence=1.0)")
        feedback_parser.add_argument("--correction", "-c", type=str, default=None, help="Corrected relation to store and learn reciprocal rule")

        # Subcommand: hypothesize (Pillar 4 GNN Positional Path Reasoning)
        hyp_parser = subparsers.add_parser("hypothesize", help="Evaluate multi-hop graph connectivity and GNN positional path hypothesis")
        hyp_parser.add_argument("subject", type=str, help="Subject entity")
        hyp_parser.add_argument("target_entity", type=str, help="Target entity to find connection to")
        hyp_parser.add_argument("--relation", "-r", type=str, default=None, help="Optional relation to verify")
        hyp_parser.add_argument("--verbose", "-v", action="store_true", help="Display full path steps and provenance")

        # Subcommand: train (Benchmark Knowledge & Supervised QA Training)
        train_parser = subparsers.add_parser("train", help="Train AR.I.S on benchmark facts and supervised QA datasets")
        train_parser.add_argument("--facts", "-f", type=str, default="data/training Dataset.json", help="Path to knowledge facts JSON dataset")
        train_parser.add_argument("--qa", "-q", type=str, default="data/training question dataset.json", help="Path to supervised question dataset")
        train_parser.add_argument("--eval", "-e", action="store_true", help="Run evaluation on QA questions after training")
        train_parser.add_argument("--verbose", "-v", action="store_true", help="Display full training details")

        # Subcommand: evaluate_protocol (Semantic Understanding Protocol)
        eval_proto_parser = subparsers.add_parser("evaluate_protocol", help="Run Semantic Understanding Training & Evaluation Protocol")
        eval_proto_parser.add_argument("--facts", "-f", type=str, default="data/aris_training_facts.json", help="Path to training facts JSON")
        eval_proto_parser.add_argument("--questions", "-q", type=str, default="data/aris_question_training.json", help="Path to question dataset JSON")
        eval_proto_parser.add_argument("--output", "-o", type=str, default="data/aris_evaluation_results.json", help="Path to save evaluation results JSON")
        eval_proto_parser.add_argument("--verbose", "-v", action="store_true", help="Display full per-question diagnostics")

        # Subcommand: evaluate_query_binding (Question -> Semantic Query Binding)
        eqb_parser = subparsers.add_parser("evaluate_query_binding", help="Evaluate Question -> Semantic Query Binding benchmark")
        eqb_parser.add_argument("--dataset", "-d", type=str, default="data/aris_query_binding_dataset.json", help="Path to query binding dataset JSON")
        eqb_parser.add_argument("--facts", "-f", type=str, default="data/aris_training_facts.json", help="Path to training facts JSON")
        eqb_parser.add_argument("--output", "-o", type=str, default="data/aris_query_binding_eval.json", help="Path to save evaluation results JSON")

        # Subcommand: evaluate_relation_abstraction (Semantic Relation Abstraction)
        era_parser = subparsers.add_parser("evaluate_relation_abstraction", help="Evaluate Semantic Relation Abstraction benchmark")
        era_parser.add_argument("--dataset", "-d", type=str, default="data/aris_relation_abstraction_benchmark.json", help="Path to benchmark JSON")
        era_parser.add_argument("--facts", "-f", type=str, default="data/aris_training_facts.json", help="Path to facts JSON")
        era_parser.add_argument("--output", "-o", type=str, default="data/aris_relation_abstraction_eval.json", help="Path to save evaluation JSON")

        # Subcommand: benchmark (14-Tier Generalization Curriculum)
        bench_parser = subparsers.add_parser("benchmark", help="Run 14-Level Generalization Curriculum Benchmark")
        bench_parser.add_argument("--verbose", "-v", action="store_true", help="Display full per-level diagnostics")

        # Subcommand: train-ud (Universal Dependencies Structural Training)
        train_ud_parser = subparsers.add_parser("train-ud", aliases=["train_ud"], help="Train AR.I.S on Universal Dependencies v2.18 multilingual treebanks")
        train_ud_parser.add_argument("--languages", "-l", nargs="+", default=["English", "Tamil", "French", "Hindi", "German"], help="Languages to train on")
        train_ud_parser.add_argument("--max-sentences", "-m", type=int, default=None, help="Maximum sentences per language (default: all)")
        train_ud_parser.add_argument("--max-exemplars", type=int, default=None, help="Maximum prototype exemplars per UPOS tag (default: all)")
        train_ud_parser.add_argument("--max-deprel", type=int, default=None, help="Maximum prototype exemplars per dependency relation (default: all)")
        train_ud_parser.add_argument("--max-morph", type=int, default=None, help="Maximum prototype exemplars per morphological feature (default: all)")
        train_ud_parser.add_argument("--tgz", type=str, default=None, help="Custom path to ud-treebanks-v2.18.tgz")
        train_ud_parser.add_argument("--verbose", "-v", action="store_true", help="Display full training details")

        # Subcommand: interactive
        subparsers.add_parser("interactive", help="Start interactive Brain terminal session")

    def handle(self, *args, **options):
        action = options.get("action")

        if not action:
            self.stdout.write(self.style.WARNING("No action specified. Use --help to view available commands."))
            return

        if action in ("train-ud", "train_ud"):
            self._handle_train_ud(options)
            return

        if action == "store":
            self._handle_store(options)
        elif action == "retrieve":
            self._handle_retrieve(options)
        elif action == "inspect":
            self._handle_inspect(options)
        elif action == "count":
            self._handle_count(options)
        elif action == "query":
            self._handle_query(options)
        elif action == "traverse":
            self._handle_traverse(options)
        elif action == "learn":
            self._handle_learn(options)
        elif action == "reason":
            self._handle_reason(options)
        elif action == "feedback":
            self._handle_feedback(options)
        elif action == "hypothesize":
            self._handle_hypothesize(options)
        elif action == "train":
            self._handle_train(options)
        elif action == "evaluate_protocol":
            self._handle_evaluate_protocol(options)
        elif action == "evaluate_query_binding":
            self._handle_evaluate_query_binding(options)
        elif action == "evaluate_relation_abstraction":
            self._handle_evaluate_relation_abstraction(options)
        elif action == "benchmark":
            self._handle_benchmark(options)
        elif action == "interactive":
            self._handle_interactive()
        else:
            self.stdout.write(self.style.ERROR(f"Unknown action: {action}"))

    def _handle_benchmark(self, options):
        from api.cognitive.training_lab import ArisTrainingLab
        self.stdout.write("Running AR.I.S 14-Level Curriculum Generalization Benchmark...")
        runner = ArisTrainingLab()
        results = runner.run_benchmark()
        s = results["summary"]
        self.stdout.write(self.style.SUCCESS("\n========================================================"))
        self.stdout.write(self.style.SUCCESS("  AR.I.S GENERALIZATION BENCHMARK REPORT"))
        self.stdout.write(self.style.SUCCESS("========================================================"))
        self.stdout.write(f"  Overall Accuracy:      {s['overall_accuracy']}% ({s['total_passed']}/{s['total_tests']} tests passed)")
        self.stdout.write(f"  Seen Knowledge Recall: {s['seen_accuracy']}% ({s['seen_passed']}/{s['seen_total']} in-distribution)")
        self.stdout.write(f"  Zero-Shot Transfer:    {s['unseen_accuracy']}% ({s['unseen_passed']}/{s['unseen_total']} out-of-distribution)")
        self.stdout.write(f"  Duration:              {s['total_duration_sec']}s")
        self.stdout.write("--------------------------------------------------------")
        for lvl in results["levels"]:
            status_tag = "[PASS]" if lvl["accuracy"] == 100 else (f"[{lvl['accuracy']}%]" if lvl["accuracy"] > 0 else "[FAIL]")
            self.stdout.write(f"  Level {lvl['level']:2d}: {lvl['name']:<40} {status_tag}")
            if options.get("verbose"):
                for t in lvl["details"]:
                    icon = "✓" if t["passed"] else "✗"
                    self.stdout.write(f"    {icon} \"{t['query']}\" -> {t['answer']} ({t['latency_ms']}ms)")
        self.stdout.write(self.style.SUCCESS("========================================================\n"))

    def _handle_store(self, options):
        statement = options["statement"]
        topic = options["topic"]
        tags = options["tags"]
        confidence = options["confidence"]
        verbose = options.get("verbose", False)

        try:
            fact = ArisBrain.store_fact(
                statement=statement,
                topic=topic,
                tags=tags,
                source="terminal_cli",
                confidence=confidence,
            )
            self.stdout.write(self.style.SUCCESS("\n[AR.I.S Brain] Fact stored successfully!"))
            self.stdout.write(f"  {fact['statement']}")
            if verbose:
                self.stdout.write(
                    f"  [ID: {fact['id']} | Topic: {fact['topic']} | Tokens: {fact.get('token_count', 0)} "
                    f"| Candidates: {fact.get('candidate_count', 0)} | Relations: {fact.get('relation_count', 0)} "
                    f"| Tags: {', '.join(fact['tags']) if fact['tags'] else 'None'}]"
                )
            self.stdout.write("")
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"[Error storing fact]: {str(e)}"))

    def _handle_retrieve(self, options):
        query = options["query"]
        topic = options["topic"]
        tag = options["tag"]
        limit = options["limit"]
        verbose = options.get("verbose", False)

        facts = ArisBrain.retrieve_facts(query=query, topic=topic, tag=tag, limit=limit)

        if not facts:
            self.stdout.write(self.style.WARNING("\n[AR.I.S Brain] No matching facts found.\n"))
            return

        self.stdout.write(self.style.SUCCESS(f"\n[AR.I.S Brain] Retrieved {len(facts)} fact(s):"))
        self.stdout.write("-" * 60)
        for i, f in enumerate(facts, 1):
            self.stdout.write(f"{i}. {f['statement']}")
            if verbose:
                self.stdout.write(
                    f"   [ID: {f['id']} | Topic: {f['topic']} | Tokens: {f.get('token_count', 0)} "
                    f"| Candidates: {f.get('candidate_count', 0)} | Relations: {f.get('relation_count', 0)} "
                    f"| Date: {f['created_at']}]"
                )
        self.stdout.write("-" * 60 + "\n")

    def _handle_inspect(self, options):
        fact_id = options["fact_id"]
        fact = ArisBrain.inspect_fact(fact_id)

        if not fact:
            self.stdout.write(self.style.ERROR(f"\n[AR.I.S Brain] Fact with ID '{fact_id}' not found.\n"))
            return

        statement = fact.get("statement", "")
        tokens = fact.get("tokens", [])
        token_count = fact.get("token_count", len(tokens))
        candidates = fact.get("candidates", [])
        candidate_count = fact.get("candidate_count", len(candidates))

        # -------------------------------------------------------------
        # Header & Statement
        # -------------------------------------------------------------
        self.stdout.write(self.style.SUCCESS("\n" + "=" * 65))
        self.stdout.write(self.style.SUCCESS("  AR.I.S COGNITIVE INSPECTION MICROSCOPE"))
        self.stdout.write(self.style.SUCCESS("=" * 65))
        self.stdout.write(f"\nStatement:\n{statement}\n")

        # -------------------------------------------------------------
        # Tier 1: Token Inspector
        # -------------------------------------------------------------
        self.stdout.write(self.style.WARNING("TIER 1: ATOMIC TOKENS & POSITIONAL COORDINATES"))
        self.stdout.write("-" * 65)
        self.stdout.write(f"{'Idx':<5}{'Token':<16}{'Type':<8}{'Sentence':<10}{'Char Span':<12}{'Rel.Pos':<7}")
        self.stdout.write(f"{'---':<5}{'-------------':<16}{'------':<8}{'--------':<10}{'----------':<12}{'-------':<7}")

        for t in tokens:
            seq_idx = t.get("sequence_index", 0)
            token_str = t.get("token", "")
            t_type = t.get("type", "word")
            pos = t.get("position", {})
            sent_idx = pos.get("sentence_index", 0)
            span_str = f"{pos.get('char_start', 0)}:{pos.get('char_end', 0)}"
            rel_pos = pos.get("relative_pos", 0.0)
            self.stdout.write(f"{seq_idx:<5}{token_str:<16}{t_type:<8}{sent_idx:<10}{span_str:<12}{rel_pos:<.2f}")

        self.stdout.write(f"Token count: {token_count}\n")

        # -------------------------------------------------------------
        # Tier 2: Concept & Entity Candidate Inspector
        # -------------------------------------------------------------
        self.stdout.write(self.style.WARNING("TIER 2: CONCEPT & ENTITY CANDIDATES (COMPOUNDS & PROPER SPANS)"))
        self.stdout.write("-" * 65)
        self.stdout.write(f"{'Idx':<5}{'Concept / Candidate':<25}{'Type':<20}{'Tokens':<10}{'Char Span':<10}")
        self.stdout.write(f"{'---':<5}{'-----------------------':<25}{'------------------':<20}{'--------':<10}{'---------':<10}")

        if not candidates:
            self.stdout.write("  (No compound concepts or entity candidates detected)")
        else:
            for c in candidates:
                c_idx = c.get("index", 0)
                text_str = c.get("text", "")
                c_type = c.get("candidate_type", "candidate")
                tok_indices = c.get("token_indices", [])
                if len(tok_indices) > 1 and tok_indices[-1] == tok_indices[0] + len(tok_indices) - 1:
                    tok_str = f"[{tok_indices[0]}..{tok_indices[-1]}]"
                elif len(tok_indices) == 1:
                    tok_str = f"[{tok_indices[0]}]"
                else:
                    tok_str = str(tok_indices)
                span_str = f"{c.get('char_start', 0)}:{c.get('char_end', 0)}"
                self.stdout.write(f"{c_idx:<5}{text_str:<25}{c_type:<20}{tok_str:<10}{span_str:<10}")

        self.stdout.write(f"\nCandidate count: {candidate_count}")

        # -------------------------------------------------------------
        # Tier 3: Relational Binding Inspector
        # -------------------------------------------------------------
        relations = fact.get("relations", [])
        relation_count = fact.get("relation_count", len(relations))

        self.stdout.write("")
        self.stdout.write(self.style.WARNING("TIER 3: RELATIONAL BINDINGS (WHO -- RELATION --> WHOM)"))
        self.stdout.write("-" * 65)
        self.stdout.write(f"{'Idx':<4}{'Subject':<22}{'Relation Span':<25}{'Object':<18}")
        self.stdout.write(f"{'---':<4}{'-'*20:<22}{'-'*23:<25}{'-'*16:<18}")

        if not relations:
            self.stdout.write("  (No relational bindings detected — need ≥2 entity candidates in a sentence)")
        else:
            for r in relations:
                subj = r["subject"]["text"]
                rel_span = r["relation"]["text"]
                obj = r["object"]["text"]
                idx = r["index"]
                self.stdout.write(f"{idx:<4}{subj:<22}{rel_span:<25}{obj:<18}")

        self.stdout.write(f"\nRelation count: {relation_count}")
        self.stdout.write("=" * 65 + "\n")

    def _handle_count(self, options):
        topic = options["topic"]
        try:
            total = ArisBrain.count_facts(topic=topic)
            scope = f" in topic '{topic}'" if topic else ""
            self.stdout.write(self.style.SUCCESS(f"\n[AR.I.S Brain] Total facts{scope}: {total}\n"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"[Error counting facts]: {str(e)}"))

    def _handle_query(self, options):
        query_text = options.get("query_text")
        subject = options.get("subject")
        relation = options.get("relation")
        obj = options.get("obj")
        direction = options.get("direction")
        limit = options.get("limit", 10)
        verbose = options.get("verbose", False)

        try:
            result = ArisBrain.query_relations(
                query_text=query_text,
                subject=subject,
                relation=relation,
                obj=obj,
                direction=direction,
                limit=limit,
            )

            q_info = result["query"]
            subj_disp = q_info["subject"] or "?"
            rel_disp = q_info["relation"] or "?"
            obj_disp = q_info["object"] or "?"

            self.stdout.write("\n=================================================================")
            self.stdout.write("  AR.I.S 1-HOP RELATIONAL RETRIEVAL (TIER 4)")
            self.stdout.write("=================================================================")
            self.stdout.write(f"Query Pattern:  {subj_disp} --[{rel_disp}]--> {obj_disp}")
            self.stdout.write(f"Direction:      {q_info['direction'].upper()}  |  Target Variable: {q_info['target_variable'].upper()}")
            self.stdout.write(f"Matches Found:  {result['matches_count']}")
            self.stdout.write("-----------------------------------------------------------------")

            if result["matches_count"] == 0:
                self.stdout.write(self.style.WARNING("  No matching relational bindings found in memory."))
                self.stdout.write("=================================================================\n")
                return

            for idx, item in enumerate(result["answers"], start=1):
                ans = item["answer"]
                ans_str = "True (Confirmed)" if ans is True else str(ans)
                self.stdout.write(self.style.SUCCESS(f"{idx}. ANSWER: {ans_str}"))
                self.stdout.write(f"   Source Fact: \"{item['statement']}\"")
                self.stdout.write(f"   Fact ID:     {item['fact_id']}  |  Confidence: {item['confidence']:.2f}")

                if verbose and "positional_alignment" in item:
                    pa = item["positional_alignment"]
                    self.stdout.write("\n   === POSITIONAL RELATION MICROSCOPE ===")
                    self.stdout.write("   Role      Query Position          Memory Position         Span Verification")
                    self.stdout.write("   --------  ----------------------  ----------------------  -----------------")

                    stmt = item["statement"]
                    for role in ["subject", "relation", "object"]:
                        slot = pa[role]
                        q_toks = str(slot.get("query_tokens", []))
                        q_chars = f"{slot['query_chars'][0]}:{slot['query_chars'][1]}" if slot.get("query_chars") else "N/A"
                        q_disp = f"Tok {q_toks} ({q_chars})" if slot.get("query_tokens") else "N/A"

                        m_toks = str(slot.get("memory_tokens", []))
                        m_c0, m_c1 = slot["memory_chars"]
                        m_disp = f"Tok {m_toks} ({m_c0}:{m_c1})"

                        mem_slice = stmt[m_c0:m_c1]
                        is_exact = mem_slice == slot["memory_text"]
                        status_str = f"'{mem_slice}' [{'OK' if is_exact else 'FAIL'}]"

                        self.stdout.write(f"   {role.capitalize():<8}  {q_disp:<22}  {m_disp:<22}  {status_str}")

                    self.stdout.write("   ======================================\n")
                elif verbose and "matched_binding" in item:
                    mb = item["matched_binding"]
                    self.stdout.write(
                        f"   Binding:     [{mb['subject']['text']}] --[{mb['relation']['text']}]--> [{mb['object']['text']}]"
                    )
                self.stdout.write("")

            self.stdout.write("=================================================================\n")
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"[Error querying relations]: {str(e)}"))

    def _handle_traverse(self, options):
        start_entity = options["start_entity"]
        hops_str = options["hops"]
        limit = options.get("limit", 10)
        verbose = options.get("verbose", False)

        hops = []
        for raw_hop in hops_str.split(","):
            raw_hop = raw_hop.strip()
            if not raw_hop:
                continue
            parts = raw_hop.split(":")
            rel = parts[0].strip()
            direction = parts[1].strip() if len(parts) > 1 else "forward"
            hops.append({"relation": rel, "direction": direction})

        try:
            result = ArisBrain.traverse_multi_hop(start_entity=start_entity, hops=hops, limit=limit)

            self.stdout.write("\n=================================================================")
            self.stdout.write("  AR.I.S MULTI-HOP RELATIONAL TRAVERSAL (TIER 4-C)")
            self.stdout.write("=================================================================")
            self.stdout.write(f"Start Entity:   {start_entity}")
            self.stdout.write(f"Hops Defined:   {len(hops)}")
            for idx, h in enumerate(hops, 1):
                self.stdout.write(f"  Hop {idx}:       [{h['relation']}] ({h['direction'].upper()})")
            self.stdout.write(f"Matches Found:  {result['matches_count']}")
            self.stdout.write("-----------------------------------------------------------------")

            if result["matches_count"] == 0:
                self.stdout.write(self.style.WARNING("  Traversal halted: No valid path found through stored bindings."))
                self.stdout.write("=================================================================\n")
                return

            for idx, item in enumerate(result["answers"], start=1):
                self.stdout.write(self.style.SUCCESS(f"{idx}. TARGET ANSWER: {item['answer']}"))
                self.stdout.write(f"   Traversal Path: {item['path_summary']}")

                if verbose:
                    self.stdout.write("   Step Details:")
                    for step in item["path"]:
                        self.stdout.write(
                            f"     Hop {step['hop']}: {step['from_entity']} --[{step['relation']} ({step['direction']})]--> {step['to_entity']} "
                            f"(Source: \"{step['statement']}\")"
                        )
                self.stdout.write("")

            self.stdout.write("=================================================================\n")
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"[Error executing multi-hop traversal]: {str(e)}"))

    def _handle_learn(self, options):
        fact = options["fact"]
        question = options["question"]
        answer = options["answer"]
        verbose = options.get("verbose", False)

        try:
            res = ArisBrain.learn_demonstration(fact=fact, question=question, answer=answer)
            tpl = res["template"]
            align = res["discovered_alignment"]

            self.stdout.write("\n=================================================================")
            self.stdout.write("  AR.I.S COGNITIVE OPERATION LEARNER (TIER 5)")
            self.stdout.write("=================================================================")
            self.stdout.write(f"Teaching Signal (Fact):      \"{fact}\"")
            self.stdout.write(f"Teaching Signal (Question):  \"{question}\"")
            self.stdout.write(f"Teaching Signal (Answer):    \"{answer}\"")
            self.stdout.write("-----------------------------------------------------------------")
            self.stdout.write(self.style.SUCCESS(f"Learned Operation: {tpl['name']} (ID: {tpl['template_id']})"))
            self.stdout.write(f"Operation Type:    {tpl['operation_type'].upper()}")
            self.stdout.write(f"Variable Role:     {tpl['source_role'].upper()} (in Question)")
            self.stdout.write(f"Target Role:       {tpl['target_role'].upper()} (in Fact)")
            self.stdout.write(f"Constraint Anchor: {tpl['anchor_role'].upper()}")

            if verbose:
                self.stdout.write("\n  === OPERATION ABSTRACTION MICROSCOPE ===")
                self.stdout.write("  [De-lexicalized Schema]:")
                self.stdout.write(f"    Fact Binding:     [S_FACT] --[R]--> [O_FACT]")
                self.stdout.write(f"    Target Answer:    Found at slot [{align['answer_slot_in_fact'].upper()}]")
                self.stdout.write(f"    Question Anchor:  Constrained at slot [{align['anchor_slot_in_question'].upper()}]")
                self.stdout.write(f"    Learned Rule:     Query(? in {align['variable_slot_in_question']}) ==> Extract({align['answer_slot_in_fact']}) via {align['operation']}")
                self.stdout.write("  ========================================\n")

            self.stdout.write("=================================================================\n")
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"[Error in cognitive learning]: {str(e)}"))

    def _handle_reason(self, options):
        question_text = options["question_text"]
        limit = options.get("limit", 10)
        verbose = options.get("verbose", False)
        use_advisor = options.get("use_advisor", True)

        try:
            res = ArisBrain.reason(question_text=question_text, limit=limit, use_advisor=use_advisor)

            self.stdout.write("\n=================================================================")
            self.stdout.write("  AR.I.S AUTONOMOUS REASONING (TIER 5)")
            self.stdout.write("=================================================================")
            self.stdout.write(f"Question:       \"{question_text}\"")
            self.stdout.write(f"Matches Found:  {res['matches_count']}")
            self.stdout.write("-----------------------------------------------------------------")

            if res["matches_count"] == 0:
                self.stdout.write(self.style.WARNING("  No matching relational bindings could be reasoned from memory."))
                if verbose and "reasoning_trace" in res:
                    note = res["reasoning_trace"].get("note")
                    status = res["reasoning_trace"].get("status")
                    if note:
                        self.stdout.write(self.style.WARNING(f"  [Reason]: {note} (Status: {status})"))
                self.stdout.write("=================================================================\n")
                return

            tpl = res.get("applied_template")
            if tpl:
                self.stdout.write(f"Applied Template: {tpl['name']} ({tpl['operation_type']})")

            for idx, item in enumerate(res["answers"], start=1):
                ans = item["answer"]
                self.stdout.write(self.style.SUCCESS(f"{idx}. AUTONOMOUS ANSWER: {ans}"))
                if "path_summary" in item:
                    self.stdout.write(f"   Traversal Path: {item['path_summary']}")
                    if verbose and item.get("path"):
                        self.stdout.write("   Step Details:")
                        for step in item["path"]:
                            s_hop = step["hop"]
                            s_from = step["from_entity"]
                            s_rel = step["relation"]
                            s_dir = step["direction"]
                            s_to = step["to_entity"]
                            s_stmt = step.get("statement", "")
                            self.stdout.write(f"     Hop {s_hop}: {s_from} --[{s_rel} ({s_dir})]--> {s_to} (Source: \"{s_stmt}\")")
                else:
                    self.stdout.write(f"   Source:      \"{item.get('statement', '')}\"")
                self.stdout.write(f"   Confidence:  {item.get('confidence', 1.0):.2f}")

                if verbose and "reasoning_trace" in res:
                    trace = res["reasoning_trace"]
                    sensory = trace.get("sensory_observation", {})
                    decision = trace.get("cognitive_decision", {})
                    epistemic = trace.get("epistemic_cognition", {})

                    if epistemic and epistemic.get("state"):
                        # "goal_detail" holds the structured dict {target_type, anchor_entity, ...}.
                        # "goal" is a plain string (goal.target_variable) — never call .get() on it.
                        goal_info = epistemic.get("goal_detail", {})
                        self.stdout.write(
                            f"   [Epistemic State]: State='{epistemic.get('state')}', "
                            f"Deficit='{epistemic.get('deficit')}', "
                            f"Target='{goal_info.get('target_type')}', "
                            f"Anchor='{goal_info.get('anchor_entity')}'"
                        )

                    if sensory.get("type") in {"parameterized_expression", "teacher_guided_capability", "parameterized_execution"}:
                        bound_p = decision.get("bound_parameters") or sensory.get("parameters") or {}
                        expr = sensory.get("expression") or bound_p.get("expression") or bound_p.get("query")
                        if expr:
                            self.stdout.write(
                                f"   [Parameter Frame]: Parameter='{expr}' "
                                f"(Source: {sensory.get('source')})"
                            )
                        else:
                            self.stdout.write(
                                f"   [Parameter Frame]: {bound_p} "
                                f"(Source: {sensory.get('source')})"
                            )
                        self.stdout.write(
                            f"   [Capability Dispatch]: Capability='{decision.get('capability')}', "
                            f"Bound Parameters={bound_p}, "
                            f"Result='{decision.get('result')}'"
                        )
                        ingested_count = decision.get("ingested_facts_count", 0)
                        if ingested_count > 0:
                            toks = decision.get("total_tokens", 0)
                            rels = decision.get("total_relations", 0)
                            self.stdout.write(
                                self.style.SUCCESS(
                                    f"   [Cognitive Memory Ingestion]: {ingested_count} facts stored into brain "
                                    f"({toks} tokens, {rels} relational bindings)"
                                )
                            )

                    else:
                        if sensory:
                            self.stdout.write(
                                f"   [Sensory Observation]: Variable Slot='{sensory.get('variable_slot')}', "
                                f"Anchor Slot='{sensory.get('anchor_slot')}', Source='{sensory.get('source')}'"
                            )
                        if decision:
                            self.stdout.write(
                                f"   [Cognitive Decision]: Operation='{decision.get('operation')}', "
                                f"Anchor='{decision.get('anchor')}', Target Slot='{decision.get('target_slot')}', "
                                f"Relation='{decision.get('relation')}'"
                            )
                self.stdout.write("")


            self.stdout.write("=================================================================\n")
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"[Error in autonomous reasoning]: {str(e)}"))

    def _handle_feedback(self, options):
        subject = options["subject"]
        relation = options["relation"]
        obj = options["obj"]
        is_deny = options.get("deny", False)
        is_confirm = options.get("confirm", False)
        correction = options.get("correction")

        action = "deny" if is_deny or not is_confirm else "confirm"

        try:
            res = ArisBrain.provide_feedback(
                subject=subject,
                relation=relation,
                obj=obj,
                action=action,
                correction=correction,
            )
            self.stdout.write("\n=================================================================")
            self.stdout.write("  AR.I.S HUMAN FEEDBACK LOOP (PILLAR 4)")
            self.stdout.write("=================================================================")
            self.stdout.write(f"Assertion:   '{subject}' --[{relation}]--> '{obj}'")
            self.stdout.write(f"Feedback:    {action.upper()}")
            self.stdout.write(f"Confidence:  {res.get('confidence', 0.0)}")
            if action == "deny":
                self.stdout.write(self.style.WARNING("Status:      DENIED (Inhibited from future suggestions)"))
            else:
                self.stdout.write(self.style.SUCCESS("Status:      CONFIRMED (Knowledge reinforced)"))

            if correction:
                self.stdout.write(self.style.SUCCESS(f"Correction:  '{subject}' {correction} '{obj}'"))
                self.stdout.write(self.style.SUCCESS(f"Rule Learned: Reciprocal ({relation} <-> {correction})"))

            self.stdout.write("=================================================================\n")
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"[Error in human feedback]: {str(e)}"))

    def _handle_hypothesize(self, options):
        subject = options["subject"]
        target_entity = options["target_entity"]
        relation = options.get("relation")
        verbose = options.get("verbose", False)

        try:
            res = ArisBrain.hypothesize(
                subject=subject,
                target_entity=target_entity,
                relation=relation,
            )
            self.stdout.write("\n=================================================================")
            self.stdout.write("  AR.I.S GNN POSITIONAL PATH REASONING (PILLAR 4)")
            self.stdout.write("=================================================================")
            self.stdout.write(f"Subject:       '{subject}'")
            self.stdout.write(f"Target Entity: '{target_entity}'")
            self.stdout.write(f"Status:        {res.get('status', '').upper()}")
            self.stdout.write(f"Confidence:    {res.get('confidence', 0.0)}")
            self.stdout.write(f"Statement:     {res.get('statement')}")

            if res.get("path_summary"):
                self.stdout.write(f"Path Summary:  {res['path_summary']}")

            if verbose and res.get("provenance"):
                self.stdout.write("Provenance Statements:")
                for stmt in res["provenance"]:
                    self.stdout.write(f"  * \"{stmt}\"")

            self.stdout.write("=================================================================\n")
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"[Error in hypothesis evaluation]: {str(e)}"))

    def _handle_train(self, options):
        facts_path = options.get("facts", "data/training Dataset.json")
        qa_path = options.get("qa", "data/training question dataset.json")
        run_eval = options.get("eval", False)
        verbose = options.get("verbose", False)

        self.stdout.write("\n=================================================================")
        self.stdout.write("  AR.I.S BENCHMARK & SUPERVISED QA TRAINING ENGINE")
        self.stdout.write("=================================================================")

        # --- Phase 1: Train Facts Knowledge Graph ---
        self.stdout.write(f"\n[Phase 1] Ingesting Knowledge Facts: '{facts_path}'...")
        try:
            res_facts = ArisBrain.train_dataset(facts_path)
            self.stdout.write(self.style.SUCCESS(f"  Status: SUCCESS"))
            self.stdout.write(f"  Dataset:              {res_facts['dataset_name']}")
            self.stdout.write(f"  Total Triples:        {res_facts['total_facts']}")
            self.stdout.write(f"  New Facts Stored:     {res_facts['new_stored']}")
            self.stdout.write(f"  Already Stored:       {res_facts['already_stored']}")
            self.stdout.write(f"  Atomic Tokens:        {res_facts['total_tokens']}")
            self.stdout.write(f"  Candidate Spans:      {res_facts['total_candidates']}")
            self.stdout.write(f"  Relational Bindings:  {res_facts['total_relations']}")

            # Run recommended benchmark tests
            rec_tests = res_facts.get("recommended_tests", [])
            if rec_tests:
                self.stdout.write("\n  --- Benchmark Verification Tests ---")
                pass_count = 0
                for t in rec_tests:
                    q = t["query"]
                    exp = t["expected"]
                    hyp = ArisBrain.hypothesize(subject=q[0], target_entity=q[1])
                    status = "CONNECTED" if hyp.get("status") in ("plausible_hypothesis", "verified_symmetric") else "UNSUPPORTED"
                    passed = (status == exp)
                    if passed:
                        pass_count += 1
                    symbol = "[PASS]" if passed else "[FAIL]"
                    style_fn = self.style.SUCCESS if passed else self.style.ERROR
                    hops_info = f"({hyp.get('hop_count')} hops)" if hyp.get("hop_count") else ""
                    self.stdout.write(style_fn(f"  {symbol} {q[0]} -> {q[1]}: {status} {hops_info}"))
                    if verbose and hyp.get("path_summary"):
                        self.stdout.write(f"        Path: {hyp['path_summary']}")
                self.stdout.write(f"  Verification Result:  {pass_count}/{len(rec_tests)} tests passed")

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"  [Error training facts]: {str(e)}"))

        # --- Phase 2: Train Supervised QA & Reciprocal Supervision ---
        if qa_path:
            self.stdout.write(f"\n[Phase 2] Training Supervised QA & Reciprocal Rules: '{qa_path}'...")
            try:
                res_qa = ArisBrain.train_qa_dataset(qa_path)
                self.stdout.write(self.style.SUCCESS(f"  Status: SUCCESS"))
                self.stdout.write(f"  Dataset:                  {res_qa['dataset_name']}")
                self.stdout.write(f"  Total QA Examples:        {res_qa['total_examples']}")
                self.stdout.write(f"  Reciprocal Rules Learned: {res_qa['reciprocal_rules_learned']}")
                self.stdout.write(f"  Direct Operation Tpls:    {res_qa['direct_templates_learned']}")
                self.stdout.write(f"  Multi-Hop Traversal Tpls: {res_qa['multihop_templates_learned']}")

                if run_eval:
                    self.stdout.write("\n  --- Evaluating Supervised QA Benchmark ---")
                    eval_res = ArisBrain.evaluate_qa_dataset(qa_path)
                    self.stdout.write(f"  Total Evaluated: {eval_res['total_evaluated']}")
                    self.stdout.write(f"  Passed:          {eval_res['passed']}")
                    self.stdout.write(f"  Failed:          {eval_res['failed']}")
                    self.stdout.write(self.style.SUCCESS(f"  Accuracy:        {eval_res['accuracy'] * 100:.2f}%"))
                    for k, v in eval_res.get("by_type", {}).items():
                        self.stdout.write(f"    * {k}: {v['passed']}/{v['total']} passed")

            except Exception as e:
                self.stdout.write(self.style.ERROR(f"  [Error training QA]: {str(e)}"))

        self.stdout.write("\n=================================================================\n")

    def _handle_train_ud(self, options):
        languages = options.get("languages") or ["English", "Tamil", "French", "Hindi", "German"]
        max_sentences = options.get("max_sentences")
        max_exemplars = options.get("max_exemplars")
        max_deprel = options.get("max_deprel")
        max_morph = options.get("max_morph")
        tgz_path = options.get("tgz")
        verbose = options.get("verbose", False)

        self.stdout.write("\n=================================================================")
        self.stdout.write("  AR.I.S UNIVERSAL DEPENDENCIES (UD v2.18) LINGUISTIC TRAINING")
        self.stdout.write("=================================================================")
        self.stdout.write(f"Target Languages:       {', '.join(languages)}")
        self.stdout.write(f"Max Sentences / Lang:   {max_sentences if max_sentences is not None else 'Unlimited'}")
        self.stdout.write(f"Max Exemplars / Tag:    {max_exemplars if max_exemplars is not None else 'Unlimited'}")
        self.stdout.write(f"Max Exemplars / Deprel: {max_deprel if max_deprel is not None else 'Unlimited'}")
        self.stdout.write(f"Max Exemplars / Morph:  {max_morph if max_morph is not None else 'Unlimited'}")
        self.stdout.write(f"Archive:                {tgz_path or 'Default Universal Dependencies 2.18'}")

        try:
            from api.linguistic.ud_progress import UDTrainingProgressReporter
            from api.linguistic.ud_tensor_engine import ArisUniversalLinguisticEncoder

            reporter = UDTrainingProgressReporter(
                languages=languages,
                max_sentences_per_lang=max_sentences,
                stdout=self.stdout,
                style=self.style,
                heartbeat_interval=5.0,
            )

            res = ArisUniversalLinguisticEncoder.train_from_treebanks(
                languages=languages,
                max_sentences_per_lang=max_sentences,
                max_exemplars_per_tag=max_exemplars,
                max_exemplars_per_deprel=max_deprel,
                max_exemplars_per_morph=max_morph,
                progress_reporter=reporter,
            )

            self.stdout.write("\n[2/2] In-Memory Brain Substrate Verification:")
            # Test English contextual prediction
            tag_en, sim_en = ArisUniversalLinguisticEncoder.predict_upos_contextual("founded", "Arun", "Axinix")
            rel_en, sim_rel = ArisUniversalLinguisticEncoder.predict_dependency_relation("founded", "Axinix")
            self.stdout.write(f"  * Sample Context: 'Arun [founded] Axinix' -> Predicted UPOS: {tag_en} (confidence: {sim_en})")
            self.stdout.write(f"  * Sample Relation: 'founded -> Axinix' -> Predicted Deprel: {rel_en} (confidence: {sim_rel})")

            # Completion summary
            reporter.print_summary(res)
            self.stdout.write(self.style.SUCCESS("[Universal Dependencies Training Completed Successfully]\n"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"\n[Error in Universal Dependencies Training]: {str(e)}\n"))

    def _handle_evaluate_protocol(self, options):
        facts_file = options.get("facts", "data/aris_training_facts.json")
        questions_file = options.get("questions", "data/aris_question_training.json")
        output_file = options.get("output", "data/aris_evaluation_results.json")
        verbose = options.get("verbose", False)

        from api.semantic.semantic_evaluator import ArisSemanticEvaluator

        self.stdout.write("\n=================================================================")
        self.stdout.write("  AR.I.S SEMANTIC UNDERSTANDING TRAINING & EVALUATION PROTOCOL")
        self.stdout.write("=================================================================")
        self.stdout.write(f"Knowledge Facts File:    {facts_file}")
        self.stdout.write(f"Questions Dataset File:  {questions_file}")
        self.stdout.write(f"Results Output File:     {output_file}")
        self.stdout.write("-----------------------------------------------------------------")

        summary = ArisSemanticEvaluator.evaluate_protocol(
            facts_file=facts_file,
            questions_file=questions_file,
            output_file=output_file,
        )

        metrics = summary["metrics"]
        self.stdout.write("\n--- EVALUATION METRICS ---")
        for k, v in metrics.items():
            if isinstance(v, float):
                disp = f"{v * 100:.1f}%"
                self.stdout.write(f"  {k:<32}: {disp:>7}")
            else:
                self.stdout.write(f"  {k:<32}: {v:>7}")

        self.stdout.write("\n--- FAILURE DISTRIBUTION ---")
        fail_dist = summary.get("failure_distribution", {})
        if not fail_dist:
            self.stdout.write(self.style.SUCCESS("  No failures recorded!"))
        else:
            for fc, cnt in sorted(fail_dist.items(), key=lambda x: x[1], reverse=True):
                self.stdout.write(f"  {fc:<32}: {cnt} questions")

        self.stdout.write("\n-----------------------------------------------------------------")
        self.stdout.write(self.style.WARNING(f"PRIMARY TRAINING TARGET:\n  {summary['primary_training_target']}"))
        self.stdout.write("-----------------------------------------------------------------")
        self.stdout.write(self.style.SUCCESS(f"\nFull diagnostic records saved to: {output_file}"))
        self.stdout.write("=================================================================\n")

    def _handle_evaluate_query_binding(self, options):
        from api.semantic.query_evaluator import ArisQueryEvaluator

        dataset_path = options.get("dataset", "data/aris_query_binding_dataset.json")
        facts_path = options.get("facts", "data/aris_training_facts.json")
        output_path = options.get("output", "data/aris_query_binding_eval.json")

        self.stdout.write("\n=================================================================")
        self.stdout.write("  AR.I.S QUESTION -> SEMANTIC QUERY BINDING EVALUATION")
        self.stdout.write("=================================================================")
        self.stdout.write(f"Dataset File:    {dataset_path}")
        self.stdout.write(f"Facts File:      {facts_path}")
        self.stdout.write(f"Output File:     {output_path}")
        self.stdout.write("-----------------------------------------------------------------\n")

        results = ArisQueryEvaluator.evaluate_stages(
            dataset_path=dataset_path,
            facts_path=facts_path,
            output_path=output_path,
        )

        metrics = results["metrics"]
        failures = results["failure_distribution"]

        self.stdout.write("AR.I.S QUERY BINDING REPORT\n")
        self.stdout.write(f"Entity extraction:       {metrics['Entity extraction']:>5.1f}%")
        self.stdout.write(f"Question type:           {metrics['Question type']:>5.1f}%")
        self.stdout.write(f"Relation extraction:     {metrics['Relation extraction']:>5.1f}%")
        self.stdout.write(f"Canonical relation:      {metrics['Canonical relation']:>5.1f}%")
        self.stdout.write(f"Subject binding:         {metrics['Subject binding']:>5.1f}%")
        self.stdout.write(f"Object binding:          {metrics['Object binding']:>5.1f}%")
        self.stdout.write(f"Answer slot:             {metrics['Answer slot']:>5.1f}%\n")
        self.stdout.write(f"Semantic Query Accuracy: {metrics['Semantic Query Accuracy']:>5.1f}%\n")
        self.stdout.write(f"Graph Path Accuracy:     {metrics['Graph Path Accuracy']:>5.1f}%")
        self.stdout.write(f"Final Answer Accuracy:   {metrics['Final Answer Accuracy']:>5.1f}%\n")

        self.stdout.write("Failure distribution:")
        for cat, count in failures.items():
            self.stdout.write(f"  {cat:<26}: {count:>3} questions")

        self.stdout.write("\n=================================================================")
        self.stdout.write(f"Full diagnostic records saved to: {output_path}")
        self.stdout.write("=================================================================\n")

    def _handle_evaluate_relation_abstraction(self, options):
        from api.relational.relation_evaluator import ArisRelationEvaluator

        dataset_path = options.get("dataset", "data/aris_relation_abstraction_benchmark.json")
        facts_path = options.get("facts", "data/aris_training_facts.json")
        output_path = options.get("output", "data/aris_relation_abstraction_eval.json")

        self.stdout.write("\n=================================================================")
        self.stdout.write("  AR.I.S SEMANTIC RELATION ABSTRACTION EVALUATION")
        self.stdout.write("=================================================================")
        self.stdout.write(f"Dataset File:    {dataset_path}")
        self.stdout.write(f"Facts File:      {facts_path}")
        self.stdout.write(f"Output File:     {output_path}")
        self.stdout.write("-----------------------------------------------------------------\n")

        results = ArisRelationEvaluator.evaluate_benchmark(
            dataset_path=dataset_path,
            facts_path=facts_path,
            output_path=output_path,
        )

        levels = results.get("levels", {})
        metrics = results.get("metrics", {})
        failures = results.get("failure_distribution", {})
        status = results.get("experiment_status", "UNKNOWN")
        bottleneck = results.get("primary_bottleneck", "NONE")
        next_cap = results.get("next_capability", "NONE")

        self.stdout.write("## LEVEL 1–10 EVALUATION RESULTS\n")
        for lvl_name, lvl_data in levels.items():
            status_str = "[PASS]" if lvl_data.get("passed") else "[FAIL]"
            self.stdout.write(f"{status_str} {lvl_name}")
            for m_k, m_v in lvl_data.get("metrics", {}).items():
                self.stdout.write(f"       - {m_k}: {m_v}")
            self.stdout.write("")

        self.stdout.write("## Failure distribution")
        for cat, count in failures.items():
            pct = (count / max(metrics.get('total', 100), 1)) * 100.0
            self.stdout.write(f"  {cat:<30}: {count:>3} ({pct:>5.1f}%)")

        self.stdout.write("\n=================================================================")
        self.stdout.write(f"Full diagnostic records saved to: {output_path}")
        self.stdout.write("=================================================================\n")

        self.stdout.write(f"EXPERIMENT STATUS: {status}")
        self.stdout.write(f"PRIMARY BOTTLENECK: {bottleneck}")
        self.stdout.write(f"NEXT CAPABILITY: {next_cap}\n")

    def _handle_interactive(self):
        self.stdout.write(self.style.SUCCESS("\n=== AR.I.S Brain Interactive Terminal ==="))
        self.stdout.write("Commands:")
        self.stdout.write("  store <statement>                  Store a new fact")
        self.stdout.write("  retrieve [query]                   Search stored facts")
        self.stdout.write("  reason <query>                     Reason on any question or entity pair")
        self.stdout.write("  deny <subject> | <relation> | <object>  Inhibit a false hypothesis (confidence=0.0)")
        self.stdout.write("  confirm <subj> | <rel> | <obj>     Confirm hypothesis (confidence=1.0)")
        self.stdout.write("  train [facts.json] [qa.json]       Train AR.I.S on benchmark datasets")
        self.stdout.write("  evaluate [facts] [questions]       Run Semantic Protocol evaluation")
        self.stdout.write("  inspect <id>                       Inspect token/candidate breakdown")
        self.stdout.write("  count                              Show total fact count")
        self.stdout.write("  exit                               Quit interactive session\n")

        while True:
            try:
                user_input = input("aris-brain> ").strip()
                if not user_input:
                    continue

                if user_input.lower() in ("exit", "quit", "q"):
                    self.stdout.write(self.style.SUCCESS("Exiting AR.I.S Brain session."))
                    break

                if user_input.startswith("store "):
                    stmt = user_input[6:].strip()
                    fact = ArisBrain.store_fact(statement=stmt, source="interactive_terminal")
                    self.stdout.write(
                        self.style.SUCCESS(f"-> Fact saved: {fact['statement']} (Tokens: {fact['token_count']}, Candidates: {fact.get('candidate_count', 0)})")
                    )
                elif user_input.startswith("retrieve"):
                    q = user_input[8:].strip() or None
                    facts = ArisBrain.retrieve_facts(query=q, limit=10)
                    if not facts:
                        self.stdout.write(self.style.WARNING("-> No facts found."))
                    else:
                        for i, f in enumerate(facts, 1):
                            self.stdout.write(f"  {i}. {f['statement']}")
                elif user_input.startswith("reason "):
                    q_text = user_input[7:].strip()
                    res = ArisBrain.reason(question_text=q_text)
                    if res.get("status") == "denied":
                        self.stdout.write(self.style.WARNING(f"-> Status: DENIED / INHIBITED by human feedback (Confidence: 0.0)"))
                    elif res["matches_count"] == 0:
                        self.stdout.write(self.style.WARNING(f"-> No matches found. (Status: {res.get('status', 'unsupported')})"))
                    else:
                        for ans in res["answers"]:
                            if res.get("status") == "plausible_hypothesis":
                                self.stdout.write(self.style.SUCCESS(f"-> Plausible Hypothesis: {ans['statement']} (Confidence: {ans.get('confidence')})"))
                                if res.get("provenance"):
                                    self.stdout.write(f"   Provenance: {res['provenance']}")
                                elif ans.get("provenance"):
                                    self.stdout.write(f"   Provenance: {ans['provenance']}")
                            else:
                                self.stdout.write(self.style.SUCCESS(f"-> {ans['statement']} (Confidence: {ans.get('confidence', 1.0)})"))
                elif user_input.startswith("deny ") or user_input.startswith("confirm "):
                    parts = user_input.split(" ", 1)
                    cmd = parts[0]
                    slots = [s.strip() for s in parts[1].split("|")]
                    if len(slots) >= 3:
                        s, r, o = slots[0], slots[1], slots[2]
                        corr = slots[3] if len(slots) > 3 else None
                        fb_res = ArisBrain.provide_feedback(subject=s, relation=r, obj=o, action=cmd, correction=corr)
                        self.stdout.write(self.style.SUCCESS(f"-> Feedback recorded: {cmd.upper()} for '{s}' {r} '{o}' (Confidence: {fb_res.get('confidence')})"))
                    else:
                        self.stdout.write(self.style.WARNING("Format: deny <subject> | <relation> | <object> [| <correction>]"))
                elif user_input.startswith("inspect "):
                    fid = user_input[8:].strip()
                    self._handle_inspect({"fact_id": fid})
                elif user_input == "count":
                    count = ArisBrain.count_facts()
                    self.stdout.write(f"-> Total facts: {count}")
                elif user_input.startswith("train"):
                    parts = user_input.split()
                    f_path = parts[1] if len(parts) > 1 else "data/training Dataset.json"
                    q_path = parts[2] if len(parts) > 2 else "data/training question dataset.json"
                    self._handle_train({"facts": f_path, "qa": q_path, "eval": False, "verbose": False})
                elif user_input.startswith("evaluate"):
                    parts = user_input.split()
                    f_path = parts[1] if len(parts) > 1 else "data/aris_training_facts.json"
                    q_path = parts[2] if len(parts) > 2 else "data/aris_question_training.json"
                    self._handle_evaluate_protocol({"facts": f_path, "questions": q_path, "output": "data/aris_evaluation_results.json", "verbose": False})
                else:
                    self.stdout.write(self.style.WARNING("Unknown command. Type 'reason <text>', 'store <text>', 'retrieve [query]', or 'exit'."))
            except (KeyboardInterrupt, EOFError):
                self.stdout.write("\nSession ended.")
                break
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error: {e}"))
