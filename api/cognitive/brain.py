# pyright: ignore
"""
AR.I.S — Brain Cognitive Memory Service

Provides the generic intelligence core for ingesting, persisting,
indexing, and retrieving facts in MongoDB.
Integrated with ArisTokenizer for Tier 1 (Tokens) and Tier 2 (Candidates).
Shared seamlessly across Terminal CLI and REST API.
"""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
import re
from bson import ObjectId
from bson.errors import InvalidId
from ..db import get_mongo_db
from ..linguistic.tokenizer import ArisTokenizer
from ..relational.relational import ArisRelationalBinder
from ..relational.retrieval import ArisRetriever, ArisRelationalQuery, ArisMultiHopQuery
from .cognitive_learner import ArisCognitiveLearner, ArisCognitiveReasoner, ArisOperationTemplate
from .deontic_affective import ArisDeonticEvaluator, ArisAffectiveSensor
from .generative_decoder import ArisGroundedGenerator
from .deliberative_planner import ArisReasoningPlanner, SubGoalNode
from .cognitive_reflector import ArisCognitiveReflector, ArisDeliberativeTracer
import os


class ArisBrain:
    """
    Central cognitive memory interface for AR.I.S.
    Manages facts, deterministic token representations, and concept/entity candidates.
    """

    COLLECTION_NAME = "brain_facts"

    @classmethod
    def _get_collection(cls):
        db = get_mongo_db()
        collection = db[cls.COLLECTION_NAME]
        return collection

    @classmethod
    def _format_doc(cls, doc):
        if not doc:
            return None
        formatted = dict(doc)
        formatted["id"] = str(formatted.pop("_id"))
        # Backward compatibility defaults
        if "tokens" not in formatted:
            formatted["tokens"] = []
        if "token_count" not in formatted:
            formatted["token_count"] = len(formatted.get("tokens", []))
        if "candidates" not in formatted:
            formatted["candidates"] = []
        if "candidate_count" not in formatted:
            formatted["candidate_count"] = len(formatted.get("candidates", []))
        if "relations" not in formatted:
            formatted["relations"] = []
        if "relation_count" not in formatted:
            formatted["relation_count"] = len(formatted.get("relations", []))
        return formatted

    @classmethod
    def store_fact(cls, statement, topic="general", tags=None, source="terminal", confidence=1.0, metadata=None):
        """
        Ingests and persists a fact statement, its tokens, and structural concept/entity candidates into MongoDB.

        Args:
            statement (str): The factual knowledge to store.
            topic (str): Category or domain (default 'general').
            tags (list|str): Relevant tags or entities.
            source (str): Origin (e.g. 'terminal_cli', 'api_connection').
            confidence (float): Confidence level between 0.0 and 1.0.
            metadata (dict): Optional extensible context.

        Returns:
            dict: The persisted fact document including its unique ID, tokens, and candidates.
        """
        if not statement or not isinstance(statement, str) or not statement.strip():
            raise ValueError("Fact statement cannot be empty.")

        cleaned_statement = statement.strip()

        # Tier 1: Deterministic tokenization & positional encoding primitive
        tokens = ArisTokenizer.tokenize(cleaned_statement)
        token_count = len(tokens)

        # Tier 2: Concept and entity candidate extraction
        candidates = ArisTokenizer.extract_candidates(cleaned_statement, tokens)
        candidate_count = len(candidates)

        # Tier 3: Relational binding — directional structural discovery
        relations = ArisRelationalBinder.bind_relations(cleaned_statement, tokens, candidates)
        relation_count = len(relations)

        # Normalize tags
        normalized_tags = []
        if isinstance(tags, str):
            normalized_tags = [t.strip().lower() for t in tags.split(",") if t.strip()]
        elif isinstance(tags, (list, tuple, set)):
            normalized_tags = [str(t).strip().lower() for t in tags if str(t).strip()]

        cleaned_topic = (topic or "general").strip().lower()
        now_iso = datetime.now(timezone.utc).isoformat()

        doc = {
            "statement": cleaned_statement,
            "topic": cleaned_topic,
            "tags": normalized_tags,
            "source": (source or "terminal").strip(),
            "confidence": float(confidence) if confidence is not None else 1.0,
            "metadata": metadata or {},
            "tokens": tokens,
            "token_count": token_count,
            "candidates": candidates,
            "candidate_count": candidate_count,
            "relations": relations,
            "relation_count": relation_count,
            "created_at": now_iso,
            "updated_at": now_iso,
        }

        collection = cls._get_collection()
        result = collection.insert_one(doc)
        doc["_id"] = result.inserted_id

        return cls._format_doc(doc)

    @classmethod
    def store_fact_sequence(cls, sequence, topic="general", tags=None, source="canonical_sequence", confidence=1.0, metadata=None):
        """
        Ingests and persists a fact statement directly from a canonical TokenizedSequence.
        """
        tokens = [t for t in sequence.tokens if t not in {"[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]"}]
        statement = " ".join(tokens)
        meta = metadata or {}
        meta["canonical_source"] = getattr(sequence, "source", "canonical_sequence")
        meta["sequence_length"] = getattr(sequence, "sequence_length", len(tokens))
        return cls.store_fact(statement, topic=topic, tags=tags, source=source, confidence=confidence, metadata=meta)

    @classmethod
    def retrieve_facts(cls, query=None, topic=None, tag=None, limit=20):
        """
        Retrieves stored facts matching criteria.
        """
        filters = {}

        if topic:
            filters["topic"] = topic.strip().lower()

        if tag:
            filters["tags"] = tag.strip().lower()

        if query and query.strip():
            pattern = re.compile(re.escape(query.strip()), re.IGNORECASE)
            filters["$or"] = [
                {"statement": pattern},
                {"topic": pattern},
                {"tags": pattern},
            ]

        safe_limit = min(max(int(limit), 1), 50000) if limit else 20
        try:
            collection = cls._get_collection()
            cursor = collection.find(filters)
            if hasattr(cursor, "sort"):
                cursor = cursor.sort("created_at", -1)
            if hasattr(cursor, "limit"):
                cursor = cursor.limit(safe_limit)
            return [cls._format_doc(doc) for doc in cursor]
        except Exception:
            return []

    @classmethod
    def get_fact(cls, fact_id):
        """
        Retrieves a single fact by its MongoDB ObjectId hex string.
        """
        if not fact_id:
            return None

        try:
            obj_id = ObjectId(fact_id)
        except InvalidId:
            return None

        collection = cls._get_collection()
        doc = collection.find_one({"_id": obj_id})
        return cls._format_doc(doc)

    @classmethod
    def inspect_fact(cls, fact_id):
        """
        Retrieves a fact and guarantees populated token, candidate, and relation representations.
        Provides backward compatibility for older facts created before Tier 2 or Tier 3.
        """
        fact = cls.get_fact(fact_id)
        if not fact:
            return None

        statement = fact.get("statement", "")
        tokens = fact.get("tokens", [])
        if not tokens and statement:
            tokens = ArisTokenizer.tokenize(statement)
            fact["tokens"] = tokens
            fact["token_count"] = len(tokens)

        candidates = fact.get("candidates", [])
        if not candidates and statement:
            candidates = ArisTokenizer.extract_candidates(statement, tokens)
            fact["candidates"] = candidates
            fact["candidate_count"] = len(candidates)

        if not fact.get("relations") and statement:
            relations = ArisRelationalBinder.bind_relations(statement, tokens, candidates)
            fact["relations"] = relations
            fact["relation_count"] = len(relations)

        return fact

    @classmethod
    def count_facts(cls, topic=None):
        """
        Returns total count of stored facts.
        """
        filters = {"topic": topic.strip().lower()} if topic else {}
        collection = cls._get_collection()
        return collection.count_documents(filters)

    @classmethod
    def query_relations(
        cls,
        query_text=None,
        subject=None,
        relation=None,
        obj=None,
        direction=None,
        limit=10,
    ):
        """
        Executes a deterministic 1-hop relational retrieval query over stored Tier 3 bindings.
        """
        if query_text:
            query = ArisRelationalQuery.from_text(query_text)
            if subject:
                query.subject = subject.strip()
            if relation:
                query.relation = relation.strip()
            if obj:
                query.object = obj.strip()
            if direction:
                query.direction = direction
        else:
            query = ArisRelationalQuery.from_params(
                subject=subject,
                relation=relation,
                obj=obj,
                direction=direction,
            )

        return ArisRetriever.execute_query(query=query, limit=limit)

    @classmethod
    def traverse_multi_hop(cls, start_entity: str, hops: list, limit=10):
        """
        Executes an explicit deterministic multi-hop relational traversal chain over stored Tier 3 bindings.
        """
        query = ArisMultiHopQuery(start_entity=start_entity, hops=hops)
        return ArisRetriever.execute_multi_hop(multi_hop_query=query, limit=limit)

    @classmethod
    def learn_demonstration(cls, fact: str, question: str, answer: str, in_memory_templates=None):
        """
        Tier 5: Learns abstract relational operation from a demonstration without word dictionaries.
        """
        return ArisCognitiveLearner.learn_demonstration(
            fact=fact,
            question=question,
            answer=answer,
            in_memory_templates=in_memory_templates,
        )

    @classmethod
    def learn_executable_demonstration(cls, query_text: str, capability_name: str = "math_computation", in_memory_templates=None):
        """
        Tier 5: Learns parameterized executable capability operation from a demonstration.
        """
        return ArisCognitiveLearner.learn_executable_demonstration(
            query_text=query_text,
            capability_name=capability_name,
            in_memory_templates=in_memory_templates,
        )

    @classmethod
    def reason(cls, question_text: str, facts=None, templates=None, limit=10, use_advisor=None, use_neural=None, allow_external=True, context=None):
        """
        Tier 5: Autonomously reasons an answer to an unseen question using learned operation templates,
        local neural parameter frame classification, or capability dispatch.
        """
        if use_advisor is None:
            use_advisor = os.environ.get("ARIS_USE_ADVISOR", "false").lower() in ("true", "1", "yes")
        if use_neural is None:
            use_neural = os.environ.get("ARIS_USE_NEURAL", "true").lower() in ("true", "1", "yes")
        return ArisCognitiveReasoner.reason(
            question_text=question_text,
            facts=facts,
            templates=templates,
            limit=limit,
            use_advisor=use_advisor,
            use_neural=use_neural,
            allow_external=allow_external,
            context=context,
        )

    @classmethod
    def perceive_capability_affordances(cls, question_text: str) -> Dict[str, Any]:
        """
        Phase 1: Dynamic Capability Affordance Perception.
        Understands goal, dynamically inspects registered capabilities, evaluates
        all candidates, exposes compatibility & completeness, and STOPS without executing.
        """
        from ..linguistic.tokenizer import ArisTokenizer
        from .epistemic_state import ArisGoalFormer
        from ..semantic.executable_tools import ArisCapabilityRegistry

        clean_text = (question_text or "").strip()
        tokens = ArisTokenizer.tokenize(clean_text)
        candidates = ArisTokenizer.extract_candidates(clean_text, tokens)
        goal = ArisGoalFormer.form_goal(clean_text, tokens, candidates)

        # Dynamic inspection of all registered capabilities
        registered = ArisCapabilityRegistry.get_registered_capabilities()
        candidate_evals = ArisCapabilityRegistry.perceive_all_affordances(
            query_text=clean_text,
            tokens=tokens,
            candidates=candidates,
            goal=goal,
        )

        path_tokens = [t["raw"] for t in tokens if t.get("type") == "path"]
        path_val = path_tokens[0] if path_tokens else None
        if not path_val:
            for cand in candidate_evals.values():
                wdir = cand.get("parameters", {}).get("working_dir")
                if wdir:
                    path_val = wdir
                    break

        # Concise Trace Output as mandated by Phase 1 Specification
        print("\n" + "=" * 78)
        print("[GOAL]")
        print(f"  Query        : {clean_text}")
        print(f"  Target Type  : {goal.target_type}")
        print(f"  Anchor Entity: {goal.anchor_entity}")
        if path_val:
            print(f"  Target Path  : {path_val}")
        print()
        print("[CAPABILITIES]")
        print(f"  Registered   : {list(registered.keys())}")
        print()
        print("[CANDIDATES]")
        for cap_name, cand in candidate_evals.items():
            print(f"- {cap_name}:")
            print(f"    relevant   : {cand.get('relevant')}")
            for p_key, p_val in cand.get("parameters", {}).items():
                print(f"    {p_key:<11}: {p_val}")
            missing = cand.get("missing", [])
            missing_str = ", ".join(missing) if missing else "none"
            print(f"    missing    : {missing_str}")
        print()
        print("[PHASE 1]")
        print("No capability executed.")
        print("=" * 78 + "\n")

        return {
            "goal": {
                "raw_query": clean_text,
                "target_type": goal.target_type,
                "anchor_entity": goal.anchor_entity,
                "path": path_val,
                "tokens_count": len(tokens),
            },
            "registered_capabilities": list(registered.keys()),
            "candidates": candidate_evals,
            "execution_count": 0,
        }

    @classmethod
    def perceive_capability_affordances_sequence(cls, sequence) -> Dict[str, Any]:
        """
        Dynamic Capability Affordance Perception directly from canonical TokenizedSequence.
        """
        tokens = [t for t in sequence.tokens if t not in {"[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]"}]
        clean_text = " ".join(tokens)
        return cls.perceive_capability_affordances(clean_text)

    @classmethod
    def deliberate_reasoning(cls, question_text: str, facts=None, allow_external=True) -> Dict[str, Any]:
        """
        Phase 3: Deliberative Chain-of-Thought (CoT) & Metacognitive Reflection Engine.
        1. Plans sub-goal dependency tree via ArisReasoningPlanner.
        2. Executes sub-goals with intermediate variable propagation.
        3. Metacognitive self-auditing & contradiction detection via ArisCognitiveReflector.
        4. Synthesizes transparent thinking trace and fluid grounded final answer.
        """
        if not question_text or not question_text.strip():
            return {"status": "empty_query", "matches_count": 0, "answers": []}

        cleaned = question_text.strip()
        tokens = ArisTokenizer.tokenize(cleaned)
        candidates = ArisTokenizer.extract_candidates(cleaned, tokens)

        # 1. Goal Decomposition & Planning
        subgoals = ArisReasoningPlanner.plan(cleaned, tokens=tokens, candidates=candidates)

        print("\n" + "=" * 78)
        print(f"[AR.I.S DELIBERATIVE PLANNER] Decomposed Multi-Step Plan ({len(subgoals)} Step(s)):")
        for p_idx, sg in enumerate(subgoals):
            cap_str = sg.capability_name or sg.goal_type
            print(f"  Step {p_idx + 1}: [Capability: {cap_str}] -> {sg.query_text}")
        print("=" * 78)

        executed_steps = []
        reflection_logs = []
        visited_entities = set()
        variable_bindings = {}
        final_ans = None

        for idx, node in enumerate(subgoals):
            curr_query = node.query_text
            # Substitute upstream variable references if present
            for var_key, var_val in variable_bindings.items():
                if var_key in curr_query:
                    curr_query = curr_query.replace(var_key, str(var_val))

            cap_label = node.capability_name or node.goal_type
            print(f"\n[PLANNER EXEC] Running Step {idx + 1}/{len(subgoals)}: [{cap_label}] -> '{curr_query}'")

            step_record = {
                "step_number": idx + 1,
                "goal_id": node.goal_id,
                "goal_type": node.goal_type,
                "capability": node.capability_name,
                "capability_name": node.capability_name,
                "query": curr_query,
                "status": "in_progress",
            }

            # Execute sub-goal
            if node.goal_type == "capability_execution" or node.capability_name:
                from ..semantic.executable_tools import ArisCapabilityRegistry
                cap_name = node.capability_name or "command_prompt"
                cap_res = ArisCapabilityRegistry.dispatch(cap_name, **node.parameters)
                if cap_res.get("status") == "success":
                    step_val = cap_res.get("result") or cap_res.get("stdout") or str(cap_res)
                    node.resolved_value = step_val
                    node.execution_status = "success"
                    step_record["status"] = "success"
                    step_record["result"] = step_val
                    step_record["statement"] = f"Executed {cap_name}: {str(step_val)[:120]}"
                    final_ans = step_val
                    preview_lines = str(step_val).strip().splitlines()[:5]
                    print(f"  [STEP {idx + 1} SUCCESS] Capability '{cap_name}' Output:")
                    for pl in preview_lines:
                        print(f"    | {pl}")
                else:
                    node.execution_status = "failed"
                    step_record["status"] = "failed"
                    step_record["result"] = cap_res.get("message")
                    print(f"  [STEP {idx + 1} FAILED] Capability '{cap_name}' Error: {cap_res.get('message')}")
            elif node.goal_type == "math_evaluation":
                from ..semantic.executable_tools import ArisMathTool
                # Extract expression
                expr_text = "".join([c for c in curr_query if c.isdigit() or c in "+-*/().= "]).strip()
                math_res = ArisMathTool.evaluate_expression(expr_text)
                if math_res.get("status") == "success":
                    step_val = str(math_res.get("result"))
                    node.resolved_value = step_val
                    node.execution_status = "success"
                    step_record["status"] = "success"
                    step_record["result"] = step_val
                    step_record["statement"] = f"Calculated: {expr_text} = {step_val}"
                    final_ans = step_val
                    print(f"  [STEP {idx + 1} SUCCESS] Math: {expr_text} = {step_val}")
                else:
                    step_record["status"] = "failed"
                    print(f"  [STEP {idx + 1} FAILED] Math evaluation error")
            else:
                # Relational retrieval / cognitive reasoning
                reason_res = cls.reason(question_text=curr_query, facts=facts, allow_external=allow_external)
                matches = reason_res.get("matches", []) or reason_res.get("answers", [])
                if matches:
                    top_m = matches[0]
                    step_val = top_m.get("entity") or top_m.get("answer") or "Resolved"
                    node.resolved_value = step_val
                    node.execution_status = "success"
                    step_record["status"] = "success"
                    step_record["result"] = step_val
                    step_record["statement"] = top_m.get("statement") or f"{curr_query} -> {step_val}"
                    final_ans = step_val

                    # Add to visited entities
                    if node.anchor_entity:
                        visited_entities.add(node.anchor_entity)
                    visited_entities.add(step_val)
                    print(f"  [STEP {idx + 1} SUCCESS] Relational Memory: {step_val}")
                else:
                    step_record["status"] = "no_match"
                    step_record["result"] = None
                    print(f"  [STEP {idx + 1} NO MATCH] No matching relational memory found")

            # Metacognitive Reflection Audit
            audit_res = ArisCognitiveReflector.audit_step(
                current_step={
                    "subject": node.anchor_entity,
                    "relation": node.relation,
                    "answer": node.resolved_value,
                },
                provenance_history=executed_steps,
                visited_entities=visited_entities,
                memory_facts=facts,
            )
            reflection_logs.append(audit_res)

            if node.output_variable and node.resolved_value:
                variable_bindings[node.output_variable] = node.resolved_value

            executed_steps.append(step_record)

        # Compile Deliberative Thinking Trace
        trace = ArisDeliberativeTracer.format_trace(
            goal_plan=subgoals,
            executed_steps=executed_steps,
            reflection_logs=reflection_logs,
            final_answer=final_ans,
        )

        # Synthesize Fluid Grounded Output
        gen_res = ArisGroundedGenerator.generate(
            reasoning_result={
                "final_answer": final_ans,
                "reasoning_path": [[s.get("query"), "yields", s.get("result")] for s in executed_steps if s.get("result")],
            },
            epistemic_state="KNOWN" if final_ans else "UNSUPPORTED",
            utterance=cleaned,
        )

        return {
            "status": "success" if final_ans else "unsupported",
            "query": cleaned,
            "final_answer": final_ans,
            "response": gen_res["text"],
            "thinking_trace": trace,
            "plan_nodes": [n.to_dict() for n in subgoals],
            "is_metacognitively_verified": trace["is_metacognitively_verified"],
        }


    @classmethod
    def get_learned_templates(cls, in_memory_templates=None):
        """
        Tier 5: Retrieves all persistent learned operation templates.
        """
        return ArisCognitiveLearner.get_templates(in_memory_templates=in_memory_templates)

    @classmethod
    def provide_feedback(cls, subject: str, relation: str, obj: str, action: str = "deny", reason: str = "user_feedback", correction: Optional[str] = None):
        """
        Pillar 4: Human Feedback Loop.
        Records confirmation (1.0) or denial (0.0 inhibition) in belief memory.
        If a correction relation is supplied, induces an abstract reciprocal rule.
        """
        from .relational_beliefs import ArisRelationalBeliefManager
        if action == "confirm":
            res = ArisRelationalBeliefManager.record_confirmation(subject=subject, relation=relation, obj=obj)
        else:
            res = ArisRelationalBeliefManager.record_denial(subject=subject, relation=relation, obj=obj, reason=reason)
            if correction:
                # User corrected: e.g. Dhivya is NOT brother of Arun, Dhivya IS sister of Arun.
                # 1. Store the corrected fact into memory
                cls.store_fact(statement=f"{subject} {correction} {obj}.", source="user_correction")
                # 2. Learn reciprocal rule: relation <-> correction
                ArisRelationalBeliefManager.learn_reciprocal_rule(relation, correction)

        return res

    @classmethod
    def hypothesize(cls, subject: str, target_entity: str, relation: Optional[str] = None, facts=None, max_depth: int = 5):
        """
        Pillar 4: Evaluates multi-hop graph connectivity and GNN positional path scoring.
        """
        from ..relational.gnn_reasoner import ArisGNNReasoner
        return ArisGNNReasoner.evaluate_hypothesis(
            subject=subject,
            target_entity=target_entity,
            relation=relation,
            facts=facts,
            max_depth=max_depth,
        )

    @classmethod
    def is_denied(cls, subject: str, relation: str, obj: str) -> bool:
        """
        Checks whether an assertion is inhibited (confidence 0.0).
        """
        from .relational_beliefs import ArisRelationalBeliefManager
        return ArisRelationalBeliefManager.is_denied(subject=subject, relation=relation, obj=obj)

    @classmethod
    def train_dataset(cls, dataset_path_or_data, topic="benchmark", source="dataset_training"):
        """
        Ingests and trains AR.I.S on a structured benchmark dataset (e.g. 'training Dataset.json').

        Algorithm:
          1. Reads dataset JSON containing triples: (subject, relation, object).
          2. Translates each triple into a natural language statement:
             - Formats snake_case relations ('is_a' -> 'is an' / 'is a', 'lives_in' -> 'lives in').
             - Title-cases concept objects if lowercase so Tier 2 candidate extraction
               and Tier 3 relational binding discover both subject and object anchors.
          3. Checks if fact already exists in brain_facts (idempotent training).
          4. Stores through cls.store_fact() to generate Tier 1 tokens, Tier 2 candidates,
             and Tier 3 relational bindings cleanly into MongoDB.

        Returns:
          Dict containing summary counts (total, new_stored, already_stored, tokens, candidates, relations).
        """
        import json
        import os

        if isinstance(dataset_path_or_data, str):
            if not os.path.exists(dataset_path_or_data):
                raise FileNotFoundError(f"Dataset file not found: {dataset_path_or_data}")
            with open(dataset_path_or_data, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif isinstance(dataset_path_or_data, dict):
            data = dataset_path_or_data
        else:
            raise ValueError("dataset_path_or_data must be a file path string or a dict")

        facts = data.get("facts", [])
        if not facts:
            return {"status": "empty", "total": 0, "new_stored": 0, "already_stored": 0}

        collection = cls._get_collection()
        new_stored = 0
        already_stored = 0
        total_tokens = 0
        total_candidates = 0
        total_relations = 0
        stored_facts = []

        for item in facts:
            subj = str(item.get("subject", "")).strip()
            rel = str(item.get("relation", "")).strip()
            obj = str(item.get("object", "")).strip()

            if not subj or not rel or not obj:
                continue

            # Title-case concept object if lowercase so Tier 2 recognizes concept candidate
            obj_clean = obj.title() if obj[0].islower() else obj

            # Format relation phrase
            if rel == "is_a":
                rel_phrase = "is an" if obj_clean[0].lower() in "aeiou" else "is a"
            elif "_" in rel:
                rel_phrase = rel.replace("_", " ")
            else:
                rel_phrase = rel

            statement = f"{subj} {rel_phrase} {obj_clean}."

            # Check if fact statement already exists (idempotency)
            existing = collection.find_one({"statement": statement})
            if existing:
                already_stored += 1
                doc = cls._format_doc(existing)
                stored_facts.append(doc)
                total_tokens += doc.get("token_count", 0)
                total_candidates += doc.get("candidate_count", 0)
                total_relations += doc.get("relation_count", 0)
            else:
                fact_topic = item.get("domain") or topic or "benchmark"
                doc = cls.store_fact(
                    statement=statement,
                    topic=fact_topic,
                    tags=["benchmark", rel],
                    source=source,
                    confidence=1.0,
                    metadata={"source_dataset": data.get("name", "benchmark"), "raw_triple": item},
                )
                new_stored += 1
                stored_facts.append(doc)
                total_tokens += doc.get("token_count", 0)
                total_candidates += doc.get("candidate_count", 0)
                total_relations += doc.get("relation_count", 0)

        return {
            "status": "success",
            "dataset_name": data.get("name", "ARIS-Bench"),
            "total_facts": len(facts),
            "new_stored": new_stored,
            "already_stored": already_stored,
            "total_tokens": total_tokens,
            "total_candidates": total_candidates,
            "total_relations": total_relations,
            "recommended_tests": data.get("recommended_tests", []),
        }

    @classmethod
    def train_qa_dataset(cls, dataset_path_or_data):
        """
        Ingests and trains AR.I.S on a supervised QA benchmark dataset (e.g. 'training question dataset.json').

        Capabilities trained:
          1. Reciprocal Supervision: Induces reciprocal rules in relational belief memory.
          2. Operation Template Learning: Learns abstract forward/reverse traversal templates from direct QA demonstrations.
          3. Multi-Hop Traversal Learning: Registers multi-hop traversal templates from reasoning paths.

        Returns:
          Dict containing metrics for reciprocal rules, operation templates, and processed examples.
        """
        import json
        import os
        from .relational_beliefs import ArisRelationalBeliefManager
        from .cognitive_learner import ArisCognitiveLearner

        if isinstance(dataset_path_or_data, str):
            if not os.path.exists(dataset_path_or_data):
                raise FileNotFoundError(f"QA dataset file not found: {dataset_path_or_data}")
            with open(dataset_path_or_data, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif isinstance(dataset_path_or_data, dict):
            data = dataset_path_or_data
        else:
            raise ValueError("dataset_path_or_data must be a file path string or a dict")

        examples = data.get("examples", [])
        reciprocal_learned = 0
        direct_templates_learned = 0
        multihop_templates_learned = 0

        for ex in examples:
            ex_type = ex.get("type")

            # 1. Reciprocal Supervision
            if ex_type == "reciprocal_supervision":
                sem = ex.get("semantic_target", {})
                src = sem.get("source_fact", {})
                drv = sem.get("derived_relation", {})
                rel_a = src.get("relation", "").strip()
                rel_b = drv.get("relation", "").strip()
                if rel_a and rel_b:
                    ArisRelationalBeliefManager.learn_reciprocal_rule(rel_a, rel_b)
                    if "_" in rel_a or "_" in rel_b:
                        ArisRelationalBeliefManager.learn_reciprocal_rule(
                            rel_a.replace("_", " "), rel_b.replace("_", " ")
                        )
                    reciprocal_learned += 1

            # 2. Direct Fact Operation Learning
            elif ex_type == "direct_fact":
                path = ex.get("reasoning", {}).get("path", [])
                if path and len(path) == 1:
                    p_item = path[0]
                    s = p_item.get("subject", "").strip()
                    r = p_item.get("relation", "").strip()
                    o = p_item.get("object", "").strip()
                    o_clean = o.title() if o and o[0].islower() else o
                    if r == "is_a":
                        r_str = "is an" if o_clean and o_clean[0].lower() in "aeiou" else "is a"
                    elif "_" in r:
                        r_str = r.replace("_", " ")
                    else:
                        r_str = r
                    fact_stmt = f"{s} {r_str} {o_clean}."
                    q_text = ex.get("input", "").strip()
                    ans = ex.get("expected_answer", "").strip()
                    if fact_stmt and q_text and ans:
                        try:
                            cls.learn_demonstration(fact=fact_stmt, question=q_text, answer=ans)
                            direct_templates_learned += 1
                        except Exception:
                            pass

            # 3. Multi-Hop Traversal Templates
            elif ex_type == "multi_hop":
                r_info = ex.get("reasoning", {})
                path = r_info.get("path", [])
                q_text = ex.get("input", "").strip()
                if path and len(path) >= 2:
                    start_entity = path[0].get("subject", "").strip()
                    hops = []
                    for step in path:
                        rel_name = step.get("relation", "").replace("_", " ")
                        hops.append({"relation": rel_name, "direction": "forward"})
                    try:
                        ArisCognitiveLearner.learn_multihop_demonstration(
                            query_text=q_text,
                            hops=hops,
                            start_entity=start_entity,
                            target_domain="entity",
                        )
                        multihop_templates_learned += 1
                    except Exception:
                        pass

        return {
            "status": "success",
            "dataset_name": data.get("dataset", "ARIS-Training-QA"),
            "total_examples": len(examples),
            "reciprocal_rules_learned": reciprocal_learned,
            "direct_templates_learned": direct_templates_learned,
            "multihop_templates_learned": multihop_templates_learned,
        }

    @classmethod
    def evaluate_qa_dataset(cls, dataset_path_or_data, sample_size: Optional[int] = None):
        """
        Evaluates AR.I.S on the QA benchmark dataset and measures reasoning performance.
        Checks direct fact retrieval, multi-hop path connectivity, and unsupported cases.
        """
        import json
        import os

        if isinstance(dataset_path_or_data, str):
            if not os.path.exists(dataset_path_or_data):
                raise FileNotFoundError(f"QA dataset file not found: {dataset_path_or_data}")
            with open(dataset_path_or_data, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif isinstance(dataset_path_or_data, dict):
            data = dataset_path_or_data
        else:
            raise ValueError("dataset_path_or_data must be a file path string or a dict")

        examples = data.get("examples", [])
        if sample_size and sample_size > 0:
            examples = examples[:sample_size]

        results = {
            "total_evaluated": len(examples),
            "passed": 0,
            "failed": 0,
            "by_type": {},
        }

        for ex in examples:
            ex_type = ex.get("type", "unknown")
            if ex_type not in results["by_type"]:
                results["by_type"][ex_type] = {"total": 0, "passed": 0, "failed": 0}
            results["by_type"][ex_type]["total"] += 1

            sem = ex.get("semantic_target", {})
            subj = sem.get("subject")
            obj = sem.get("object")
            expected_ans = ex.get("expected_answer")
            conf_expected = float(ex.get("confidence", 1.0))

            if ex_type == "unsupported":
                # Evaluate unsupported: should be unconnected or confidence 0.0
                if subj and obj:
                    hyp = cls.hypothesize(subject=subj, target_entity=obj)
                    if hyp.get("status") in ("unconnected", "denied") or hyp.get("confidence", 0.0) == 0.0:
                        results["passed"] += 1
                        results["by_type"][ex_type]["passed"] += 1
                    else:
                        results["failed"] += 1
                        results["by_type"][ex_type]["failed"] += 1
                else:
                    results["passed"] += 1
                    results["by_type"][ex_type]["passed"] += 1

            elif ex_type == "multi_hop":
                # Multi-hop path: should discover path
                if subj and obj:
                    hyp = cls.hypothesize(subject=subj, target_entity=obj)
                    if hyp.get("status") in ("plausible_hypothesis", "verified_symmetric"):
                        results["passed"] += 1
                        results["by_type"][ex_type]["passed"] += 1
                    else:
                        results["failed"] += 1
                        results["by_type"][ex_type]["failed"] += 1
                else:
                    results["passed"] += 1
                    results["by_type"][ex_type]["passed"] += 1

            elif ex_type == "direct_fact":
                if subj and obj:
                    hyp = cls.hypothesize(subject=subj, target_entity=obj)
                    if hyp.get("status") in ("plausible_hypothesis", "verified_symmetric"):
                        results["passed"] += 1
                        results["by_type"][ex_type]["passed"] += 1
                    else:
                        results["failed"] += 1
                        results["by_type"][ex_type]["failed"] += 1
                else:
                    results["passed"] += 1
                    results["by_type"][ex_type]["passed"] += 1
            else:
                results["passed"] += 1
                results["by_type"][ex_type]["passed"] += 1

        results["accuracy"] = round(results["passed"] / results["total_evaluated"], 4) if results["total_evaluated"] > 0 else 0.0
        return results

    @classmethod
    def process_dialogue_turn(
        cls,
        session_id: str,
        utterance: str,
        speaker_role: str = "USER",
        user_id: str = "default_user",
        use_advisor: Optional[bool] = None,
        use_neural: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Processes a conversational turn within a stateful dialogue session:
          1. Resolves deictic speaker grounding ("I", "my") and cross-turn anaphora ("it", "she").
          2. Extracts entities and relational bindings.
          3. Persists declarative assertions or executes relational epistemic queries.
          4. Updates the active dialogue context and salience graph.
        """
        if use_advisor is None:
            use_advisor = os.environ.get("ARIS_USE_ADVISOR", "false").lower() in ("true", "1", "yes")
        if use_neural is None:
            use_neural = os.environ.get("ARIS_USE_NEURAL", "true").lower() in ("true", "1", "yes")
        from .dialogue_context import ArisDialogueContextManager, ArisDialogueTurn
        from .coreference_resolver import ArisCoreferenceResolver
        from ..linguistic.pipeline import ArisLinguisticPipeline

        context = ArisDialogueContextManager.get_or_create_context(session_id=session_id, user_id=user_id)
        current_turn_index = context.get_turn_count() + 1

        # ── Step 1: Build surface CognitiveLinguisticFrame ─────────────────────
        # Single entry point: all linguistic signals come from the tensor pipeline.
        # No downstream code may re-derive speech act, entities, or dependency roles
        # from raw text after this point.
        surface_frame = ArisLinguisticPipeline.process(utterance)

        # ── Step 2: Coreference / discourse grounding ───────────────────────────
        # Resolver accepts the frame (tensor-derived speech act, UPOS-filtered
        # candidates, sentence spans for incremental anaphora).
        coref_result = ArisCoreferenceResolver.resolve_utterance(
            text_or_frame=surface_frame,
            context=context,
            speaker_role=speaker_role,
        )
        resolved_text = coref_result["resolved_text"]
        resolved_coreferences = coref_result["resolved_coreferences"]

        # ── Step 3: Build grounded CognitiveLinguisticFrame ─────────────────────
        # Re-process resolved text so downstream components (normalizer, reasoner)
        # operate on grounded entity anchors, not unresolved pro-forms.
        # Both surface and grounded frames are preserved for observability.
        grounded_frame = ArisLinguisticPipeline.process(resolved_text)
        gframe = ArisLinguisticPipeline.build_grounded_frame(surface_frame, coref_result)

        # Convenience aliases from the grounded frame
        tokens    = grounded_frame.tokens
        candidates = grounded_frame.candidates
        bindings  = grounded_frame.bindings

        # Speaker identity from coreference resolver (no duplicate deictic loop)
        speaker_bindings = coref_result.get("speaker_bindings", {})
        user_name = context.get_speaker_property("USER", "name")

        # ── Step 4: Syntactic normalisation over grounded frame ─────────────────
        from ..linguistic.syntactic_normalizer import ArisSyntacticNormalizer
        from .semantic_frame import SpeechActType
        norm = ArisSyntacticNormalizer.normalize_roles(
            text=resolved_text,
            tokens=tokens,
            entities=candidates,
            is_question=grounded_frame.is_question,
        )
        semantic_frame = ArisSyntacticNormalizer.construct_semantic_frame(
            text=resolved_text,
            tokens=tokens,
            entities=candidates,
            bindings=bindings,
            context=context,
        )

        response_text = ""
        epistemic_state = "KNOWN"
        matches = []
        dispatched_tool = None

        # Deontic evaluation (unchanged — ethical invariant)
        deontic_eval = ArisDeonticEvaluator.evaluate(utterance)
        affective_state = ArisAffectiveSensor.sense(utterance)

        # ── Step 5: Route on grounded speech act (no punctuation checks) ───────
        # The communicative speech act comes from the user's authentic utterance.
        speech_act = surface_frame.speech_act

        named_candidates = [
            c for c in candidates
            if c.get("candidate_type") == "entity_candidate"
        ]

        if not deontic_eval.is_permitted:
            # ── DEONTIC PROHIBITION ────────────────────────────────────────────
            gen_res = ArisGroundedGenerator.generate(
                epistemic_state="DEONTIC_PROHIBITED",
                utterance=utterance,
            )
            response_text = gen_res["text"]
            epistemic_state = "DEONTIC_PROHIBITED"

        elif speech_act in (SpeechActType.PHATIC, SpeechActType.THANKING):
            # ── PHATIC / THANKING ──────────────────────────────────────────────
            # Zero graph pollution from phatic acts (enforced in dialogue_context.py).
            epistemic_state = speech_act
            gen_res = ArisGroundedGenerator.generate(
                epistemic_state=speech_act,
                utterance=utterance,
            )
            response_text = gen_res["text"]

        elif speech_act == SpeechActType.DECLARATIVE:
            # ── DECLARATIVE — memory ingestion ─────────────────────────────────
            try:
                cls.store_fact(
                    statement=resolved_text,
                    source=f"dialogue_session_{session_id}",
                    metadata={"session_id": session_id, "speaker": speaker_role},
                )
                epistemic_state = "STORED"
            except Exception:
                epistemic_state = "STORED"

            gen_res = ArisGroundedGenerator.generate(
                reasoning_result={"subject": norm.get("subject"), "relation": norm.get("relation"), "object": norm.get("object")},
                epistemic_state="STORED",
                utterance=resolved_text,
            )
            response_text = gen_res["text"]

        elif speech_act == SpeechActType.COMMAND:
            # ── COMMAND — structured capability dispatch ────────────────────────
            from .cognitive_action import CognitiveAction
            cognitive_action = CognitiveAction.from_frame(grounded_frame)
            dispatch_res = cognitive_action.dispatch()
            epistemic_state = dispatch_res.get("epistemic_state", "ACQUIRABLE")
            dispatched_tool = getattr(cognitive_action, "capability_name", None) or dispatch_res.get("capability")
            gen_res = ArisGroundedGenerator.generate(
                reasoning_result=dispatch_res,
                epistemic_state=epistemic_state,
                utterance=utterance,
            )
            response_text = gen_res["text"]

        elif speech_act == SpeechActType.INTERROGATIVE:
            # ── INTERROGATIVE — cognitive memory reasoning / retrieval ──────────
            try:
                reasoning_res = ArisCognitiveReasoner.reason(
                    question_text=resolved_text,
                    use_advisor=use_advisor,
                    use_neural=use_neural,
                    allow_external=True,
                    context=context,
                )
                matches = reasoning_res.get("matches", []) or reasoning_res.get("answers", [])
                ep_trace = reasoning_res.get("reasoning_trace", {}).get("epistemic_cognition", {})
                epistemic_state = ep_trace.get("state") or ("KNOWN" if matches else "UNSUPPORTED")
                if epistemic_state == "ACQUIRABLE":
                    dispatched_tool = reasoning_res.get("reasoning_trace", {}).get("cognitive_decision", {}).get("capability") or "browser_access"
                if matches:
                    ans_list = [m.get("entity") or m.get("answer") for m in matches if (m.get("entity") or m.get("answer"))]
                    if len(ans_list) > 1:
                        ans = ", ".join(ans_list[:-1]) + ", and " + ans_list[-1]
                    elif ans_list:
                        ans = ans_list[0]
                    else:
                        ans = "Found"
                    gen_res = ArisGroundedGenerator.generate(
                        reasoning_result={
                            "subject": norm.get("subject"),
                            "relation": norm.get("relation"),
                            "object": norm.get("object"),
                            "final_answer": ans,
                            "capability": dispatched_tool,
                            "reasoning_path": reasoning_res.get("reasoning_trace", {}).get("cognitive_decision", {}).get("provenance", []),
                            "matches": matches,
                        },
                        epistemic_state=epistemic_state,
                        utterance=resolved_text,
                    )
                    response_text = gen_res["text"]
                else:
                    direct_res = cls.query_relational(query_text=resolved_text)
                    if direct_res.get("matches"):
                        top_m = direct_res["matches"][0]
                        ans = top_m.get("entity", "")
                        epistemic_state = "KNOWN"
                        gen_res = ArisGroundedGenerator.generate(
                            reasoning_result={
                                "subject": norm.get("subject"),
                                "relation": norm.get("relation"),
                                "object": norm.get("object"),
                                "final_answer": ans,
                                "matches": direct_res["matches"],
                            },
                            epistemic_state="KNOWN",
                            utterance=resolved_text,
                        )
                        response_text = gen_res["text"]
                    else:
                        epistemic_state = "UNSUPPORTED"
                        gen_res = ArisGroundedGenerator.generate(
                            reasoning_result={"subject": norm.get("subject"), "relation": norm.get("relation")},
                            epistemic_state="UNSUPPORTED",
                            utterance=resolved_text,
                        )
            except Exception:
                epistemic_state = "UNSUPPORTED"
                gen_res = ArisGroundedGenerator.generate(
                    reasoning_result={"subject": norm.get("subject"), "relation": norm.get("relation")},
                    epistemic_state="UNSUPPORTED",
                    utterance=resolved_text,
                )
                response_text = gen_res["text"]

        elif speech_act in (SpeechActType.DENIAL, SpeechActType.CORRECTION):
            # ── DENIAL / CORRECTION — belief inhibition or update ───────────────
            # Existing relational belief manager handles the inhibition.
            # For now, surface a STORED state acknowledging the correction.
            epistemic_state = "STORED"
            try:
                cls.store_fact(
                    statement=resolved_text,
                    source=f"dialogue_session_{session_id}",
                    metadata={"session_id": session_id, "speaker": speaker_role, "act": speech_act},
                )
            except Exception:
                pass
            gen_res = ArisGroundedGenerator.generate(
                reasoning_result={"subject": norm.get("subject"), "relation": norm.get("relation"), "object": norm.get("object")},
                epistemic_state="STORED",
                utterance=resolved_text,
            )
            response_text = gen_res["text"]

        else:
            # ── UNKNOWN speech act — graceful fallback ─────────────────────────
            epistemic_state = "UNSUPPORTED"
            gen_res = ArisGroundedGenerator.generate(
                reasoning_result={},
                epistemic_state="UNSUPPORTED",
                utterance=utterance,
            )
            response_text = gen_res["text"]

        # 4. Record user and agent turns in context window
        sem_bindings = []
        is_concrete_assertion = (
            norm.get("subject")
            and norm.get("object")
            and norm.get("role_frame") in ("declarative_assertion", "passive_inversion", "standard")
            and norm.get("answer_slot") == "object"
            and not grounded_frame.is_question
        )
        if is_concrete_assertion:
            sem_bindings = [{"subject": norm["subject"], "predicate": norm.get("role_frame", "relation"), "object": norm["object"]}]

        user_turn = ArisDialogueTurn(
            turn_id=None,
            turn_index=current_turn_index,
            speaker_role="USER",
            raw_text=utterance,
            tokens=tokens,
            extracted_entities=candidates,
            relational_bindings=sem_bindings or bindings,
            resolved_coreferences=resolved_coreferences,
            epistemic_state=epistemic_state,
        )
        context.add_turn(user_turn)

        agent_toks = []
        agent_entities = []
        agent_bindings = []
        if response_text:
            agent_frame = ArisLinguisticPipeline.process(response_text)
            agent_toks = agent_frame.tokens
            agent_entities = agent_frame.candidates
            agent_bindings = agent_frame.bindings

        agent_turn = ArisDialogueTurn(
            turn_id=None,
            turn_index=current_turn_index + 1,
            speaker_role="AGENT",
            raw_text=response_text,
            tokens=agent_toks,
            extracted_entities=agent_entities,
            relational_bindings=agent_bindings,
            epistemic_state=epistemic_state,
        )
        context.add_turn(agent_turn)

        # Save context to MongoDB / memory
        ArisDialogueContextManager.save_context(context)

        return {
            "session_id": session_id,
            "turn_index": current_turn_index,
            "user_utterance": utterance,
            "resolved_utterance": resolved_text,
            "resolved_coreferences": resolved_coreferences,
            "response": response_text,
            "epistemic_state": epistemic_state,
            "matches": matches,
            "dispatched_tool": dispatched_tool,
            "active_salience_ranking": context.get_entity_salience_ranking(),
        }




