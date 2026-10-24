"""
AR.I.S — 1-Hop Relational Retrieval Engine (Tier 4)

Provides deterministic, bidirectional graph traversal over stored Tier 3 relational bindings.

Key Architectural Invariants:
  - Zero Spoon-Feeding: No English grammar dictionaries, no hardcoded interrogative lists (no 'who', 'what', 'does').
  - Pure Structural Alignment: Compares query structural bindings against stored memory bindings.
  - Complete Structural Constraints: Matches exact normalized relation text and exact entity names.
  - Bidirectional Traversal:
      1. Forward:        (Subject, Relation, ?)      -> Object
      2. Reverse:        (?, Relation, Object)      -> Subject
      3. Link Discovery: (Subject, ?, Object)      -> Relation
      4. Verification:   (Subject, Relation, Object) -> True / False (Match / No Match)
"""

from typing import Optional, List, Dict, Any
from ..linguistic.tokenizer import ArisTokenizer
from .relational import ArisRelationalBinder


class ArisRelationalQuery:
    """
    Encapsulates a 1-hop relational query pattern.
    """

    def __init__(
        self,
        subject: Optional[str] = None,
        relation: Optional[str] = None,
        obj: Optional[str] = None,
        direction: Optional[str] = None,
        target_variable: Optional[str] = None,
        raw_query: Optional[str] = None,
        query_binding: Optional[Dict[str, Any]] = None,
    ):
        self.subject = subject.strip() if subject else None
        self.relation = relation.strip() if relation else None
        self.object = obj.strip() if obj else None
        self.raw_query = raw_query
        self.query_binding = query_binding

        # Deduce target_variable and direction if not explicitly provided
        if target_variable:
            self.target_variable = target_variable
        elif self.subject is None and self.relation and self.object:
            self.target_variable = "subject"
        elif self.object is None and self.subject and self.relation:
            self.target_variable = "object"
        elif self.relation is None and self.subject and self.object:
            self.target_variable = "relation"
        elif self.subject and self.relation and self.object:
            self.target_variable = "verification"
        else:
            self.target_variable = "unknown"

        if direction:
            self.direction = direction
        else:
            if self.target_variable == "subject":
                self.direction = "reverse"
            elif self.target_variable == "object":
                self.direction = "forward"
            elif self.target_variable == "relation":
                self.direction = "link"
            elif self.target_variable == "verification":
                self.direction = "verification"
            else:
                self.direction = "unknown"

    @classmethod
    def from_params(
        cls,
        subject: Optional[str] = None,
        relation: Optional[str] = None,
        obj: Optional[str] = None,
        direction: Optional[str] = None,
    ) -> "ArisRelationalQuery":
        """
        Constructs a query directly from explicit slot parameters.
        """
        return cls(
            subject=subject,
            relation=relation,
            obj=obj,
            direction=direction,
        )

    @classmethod
    def from_text(cls, query_text: str) -> "ArisRelationalQuery":
        """
        Derives structural query slots strictly using AR.I.S's own native
        Tier 1 tokens, Tier 2 candidates, and Tier 3 bindings.

        Zero spoon-feeding: No hardcoded interrogative words, no English grammar heuristics.
        Whatever AR.I.S's native representation produces is what gets queried.
        """
        if not query_text or not query_text.strip():
            return cls(raw_query=query_text)

        cleaned = query_text.strip()
        tokens = ArisTokenizer.tokenize(cleaned)
        candidates = ArisTokenizer.extract_candidates(cleaned, tokens)
        bindings = ArisRelationalBinder.bind_relations(cleaned, tokens, candidates)

        # If AR.I.S Tier 3 natively produced a structural binding:
        # e.g. "Who leads Axinix Technologies Development Team?"
        # -> subject="Who", relation="leads", object="Axinix Technologies Development Team"
        if bindings:
            primary_binding = bindings[0]
            subj_text = primary_binding["subject"]["text"]
            rel_text = primary_binding["relation"]["normalized"]
            obj_text = primary_binding["object"]["text"]

            return cls(
                subject=subj_text,
                relation=rel_text,
                obj=obj_text,
                direction="structural_intersection",
                target_variable="structural_intersection",
                raw_query=cleaned,
                query_binding=primary_binding,
            )

        # If the query does not form a native Tier 3 binding, return without forced slots
        return cls(raw_query=cleaned)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "subject": self.subject or "?",
            "relation": self.relation or "?",
            "object": self.object or "?",
            "direction": self.direction,
            "target_variable": self.target_variable,
            "raw_query": self.raw_query,
        }
        if self.query_binding:
            data["query_binding"] = self.query_binding
        return data

    def __repr__(self) -> str:
        s = self.subject or "?"
        r = self.relation or "?"
        o = self.object or "?"
        return f"<ArisRelationalQuery: {s} --[{r}]--> {o} (dir={self.direction}, target={self.target_variable})>"


class ArisRetriever:
    """
    Tier 4 Deterministic 1-Hop Relational Retrieval Engine.
    Executes exact structural graph traversal over stored Tier 3 bindings.
    """

    @classmethod
    def execute_query(
        cls,
        query: ArisRelationalQuery,
        facts: Optional[List[Dict[str, Any]]] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Executes a 1-hop relational query over facts containing Tier 3 bindings.

        Args:
            query: The ArisRelationalQuery specifying constraints.
            facts: Optional list of fact documents to search directly.
                   If None, queries MongoDB via ArisBrain.
            limit: Maximum number of answers to return.

        Returns:
            Dict containing:
              - query: Normalized query pattern metadata
              - matches_count: Total answers found
              - answers: List of matched answers with source context
        """
        if facts is None:
            from ..cognitive.brain import ArisBrain
            collection = ArisBrain._get_collection()
            cursor = collection.find({"relation_count": {"$gt": 0}}).sort("created_at", -1)
            facts = [ArisBrain._format_doc(doc) for doc in cursor]

        matches: List[Dict[str, Any]] = []
        seen_answers = set()

        q_subj = query.subject.strip().lower() if query.subject else None
        q_rel = query.relation.strip().lower() if query.relation else None
        q_obj = query.object.strip().lower() if query.object else None
        target = query.target_variable

        # Build corpus of structurally credible relation spans from loaded facts.
        # Filter: span must be at least 2 characters and contain at least one
        # alphabetic character — this removes noise like 'of', '/', '-', ','.
        # No specific words or relations are hardcoded.
        known_spans: set = {
            s for s in (
                binding["relation"]["normalized"].strip().lower()
                for fact in facts
                for binding in fact.get("relations", [])
                if binding.get("relation", {}).get("normalized")
            )
            if " " in s and any(c.isalpha() for c in s)
        }
        if q_rel:
            q_rel = ArisRelationalBinder.canonicalize_to_known(q_rel, known_spans)

        for fact in facts:
            relations = fact.get("relations", [])
            for binding in relations:
                b_subj = binding["subject"]["text"].strip().lower()
                b_rel = binding["relation"]["normalized"].strip().lower()
                b_obj = binding["object"]["text"].strip().lower()

                is_match = False
                answer_value = None
                matched_role = target

                # Mode 1: Structural intersection from a natural query binding
                # When both a query binding and stored binding are compared:
                # The query binding has (Q_subj, Q_rel, Q_obj).
                # Memory has (M_subj, M_rel, M_obj).
                if target == "structural_intersection":
                    # Case A: Relation and Object match -> Subject in memory is the answer
                    if q_rel and q_obj and b_rel == q_rel and b_obj == q_obj:
                        if b_subj != q_subj:  # Differing slot is the answer
                            is_match = True
                            answer_value = binding["subject"]["text"]
                            matched_role = "subject"
                    # Case B: Subject and Relation match -> Object in memory is the answer
                    elif q_subj and q_rel and b_subj == q_subj and b_rel == q_rel:
                        if b_obj != q_obj:  # Differing slot is the answer
                            is_match = True
                            answer_value = binding["object"]["text"]
                            matched_role = "object"
                    # Case C: Subject and Object match -> Relation in memory is the answer
                    elif q_subj and q_obj and b_subj == q_subj and b_obj == q_obj:
                        if b_rel != q_rel:
                            is_match = True
                            answer_value = binding["relation"]["text"]
                            matched_role = "relation"

                # Mode 2: Explicit slot queries
                elif target == "object":
                    # Forward Query: Subject + Relation known, find Object
                    if q_subj and q_rel and b_subj == q_subj and b_rel == q_rel:
                        is_match = True
                        answer_value = binding["object"]["text"]
                        matched_role = "object"

                elif target == "subject":
                    # Reverse Query: Relation + Object known, find Subject
                    if q_rel and q_obj and b_rel == q_rel and b_obj == q_obj:
                        is_match = True
                        answer_value = binding["subject"]["text"]
                        matched_role = "subject"

                elif target == "relation":
                    # Link Discovery: Subject + Object known, find Relation
                    if q_subj and q_obj and b_subj == q_subj and b_obj == q_obj:
                        is_match = True
                        answer_value = binding["relation"]["text"]
                        matched_role = "relation"

                elif target == "verification":
                    # Full Constraint Check: Subject + Relation + Object all specified
                    if q_subj and q_rel and q_obj:
                        if b_subj == q_subj and b_rel == q_rel and b_obj == q_obj:
                            is_match = True
                            answer_value = True
                            matched_role = "verification"

                if is_match and answer_value is not None:
                    dedup_key = (str(answer_value), fact.get("id"))
                    if dedup_key not in seen_answers:
                        seen_answers.add(dedup_key)

                        q_bind = query.query_binding or {}
                        alignment = {
                            "subject": {
                                "query_text": q_bind.get("subject", {}).get("text"),
                                "query_tokens": q_bind.get("subject", {}).get("token_indices", []),
                                "query_chars": [q_bind.get("subject", {}).get("char_start"), q_bind.get("subject", {}).get("char_end")] if "subject" in q_bind else None,
                                "memory_text": binding.get("subject", {}).get("text"),
                                "memory_tokens": binding.get("subject", {}).get("token_indices", []),
                                "memory_chars": [binding.get("subject", {}).get("char_start", 0), binding.get("subject", {}).get("char_end", 0)],
                            },
                            "relation": {
                                "query_text": q_bind.get("relation", {}).get("text") or query.relation,
                                "query_tokens": q_bind.get("relation", {}).get("token_indices", []),
                                "query_chars": [q_bind.get("relation", {}).get("char_start"), q_bind.get("relation", {}).get("char_end")] if "relation" in q_bind else None,
                                "memory_text": binding.get("relation", {}).get("text"),
                                "memory_tokens": binding.get("relation", {}).get("token_indices", []),
                                "memory_chars": [binding.get("relation", {}).get("char_start", 0), binding.get("relation", {}).get("char_end", 0)],
                            },
                            "object": {
                                "query_text": q_bind.get("object", {}).get("text"),
                                "query_tokens": q_bind.get("object", {}).get("token_indices", []),
                                "query_chars": [q_bind.get("object", {}).get("char_start"), q_bind.get("object", {}).get("char_end")] if "object" in q_bind else None,
                                "memory_text": binding.get("object", {}).get("text"),
                                "memory_tokens": binding.get("object", {}).get("token_indices", []),
                                "memory_chars": [binding.get("object", {}).get("char_start", 0), binding.get("object", {}).get("char_end", 0)],
                            },
                        }

                        matches.append({
                            "answer": answer_value,
                            "target_role": matched_role,
                            "fact_id": fact.get("id"),
                            "statement": fact.get("statement"),
                            "confidence": fact.get("confidence", 1.0),
                            "matched_binding": binding,
                            "positional_alignment": alignment,
                        })

                        if len(matches) >= limit:
                            break

            if len(matches) >= limit:
                break

        return {
            "query": query.to_dict(),
            "matches_count": len(matches),
            "answers": matches,
        }

    @classmethod
    def execute_multi_hop(
        cls,
        multi_hop_query: "ArisMultiHopQuery",
        facts: Optional[List[Dict[str, Any]]] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Executes an explicit multi-hop relational traversal chain over stored Tier 3 bindings.
        At each hop, feeds the output entity / entities into the next hop.
        Tracks full traversal paths and provenance.
        """
        if facts is None:
            from ..cognitive.brain import ArisBrain
            collection = ArisBrain._get_collection()
            cursor = collection.find({"relation_count": {"$gt": 0}}).sort("created_at", -1)
            facts = [ArisBrain._format_doc(doc) for doc in cursor]

        current_paths = [
            {
                "current_entity": multi_hop_query.start_entity,
                "steps": [],
            }
        ]

        for hop_idx, hop_spec in enumerate(multi_hop_query.hops, start=1):
            next_paths = []
            rel_name = hop_spec.get("relation", "").strip()
            direction = hop_spec.get("direction", "forward").strip().lower()

            for path_state in current_paths:
                curr_entity = path_state["current_entity"]

                if direction == "reverse":
                    q = ArisRelationalQuery(
                        subject=None,
                        relation=rel_name,
                        obj=curr_entity,
                        direction="reverse",
                        target_variable="subject",
                    )
                else:
                    q = ArisRelationalQuery(
                        subject=curr_entity,
                        relation=rel_name,
                        obj=None,
                        direction="forward",
                        target_variable="object",
                    )

                step_result = cls.execute_query(query=q, facts=facts, limit=limit)

                for item in step_result["answers"]:
                    next_entity = item["answer"]
                    step_record = {
                        "hop": hop_idx,
                        "from_entity": curr_entity,
                        "relation": rel_name,
                        "direction": direction,
                        "to_entity": next_entity,
                        "fact_id": item["fact_id"],
                        "statement": item["statement"],
                        "binding": item["matched_binding"],
                    }
                    next_paths.append({
                        "current_entity": next_entity,
                        "steps": path_state["steps"] + [step_record],
                    })

            current_paths = next_paths
            if not current_paths:
                break

        final_answers = []
        seen = set()
        for p in current_paths:
            ans = p["current_entity"]
            if ans not in seen:
                seen.add(ans)
                final_answers.append({
                    "answer": ans,
                    "path": p["steps"],
                    "path_summary": " -> ".join(
                        [multi_hop_query.start_entity]
                        + [f"[{s['relation']} ({s['direction']})] -> {s['to_entity']}" for s in p["steps"]]
                    ),
                })
                if len(final_answers) >= limit:
                    break

        return {
            "query": multi_hop_query.to_dict(),
            "matches_count": len(final_answers),
            "answers": final_answers,
        }


class ArisMultiHopQuery:
    """
    Encapsulates an explicitly specified multi-hop relational traversal chain.
    Zero English parsing: Pure structural traversal definition.

    Example:
      start_entity = "Axinix Technologies"
      hops = [
          {"relation": "leads", "direction": "reverse"},         # ? --leads--> Axinix Technologies => Arunbharathi
          {"relation": "is brother of", "direction": "forward"}, # Arunbharathi --is brother of--> ? => Bharathikannan
      ]
    """

    def __init__(self, start_entity: str, hops: List[Dict[str, str]]):
        self.start_entity = start_entity.strip()
        self.hops = hops

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_entity": self.start_entity,
            "hops": self.hops,
            "hop_count": len(self.hops),
        }
