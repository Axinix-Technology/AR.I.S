# pyright: ignore
"""
AR.I.S — API Views

Strict Architecture Flow:
REST API Views -> Serializers (Validation) -> ArisBrain -> MongoDB

Authentication & Permissions sit exclusively at the API boundary.
Views delegate all cognitive storage and retrieval logic directly to ArisBrain.
"""

from datetime import datetime, timezone
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny

from .db import check_mongo_connection
from .cognitive.brain import ArisBrain
from .serializers import FactInputSerializer, FactOutputSerializer


class ConnectionStatusView(APIView):
    """
    Foundational gateway endpoint for users to establish and verify API connection.
    Reports API readiness and MongoDB backend connectivity.
    """
    permission_classes = [AllowAny]

    def get(self, request, format=None):
        mongo_status = check_mongo_connection()

        data = {
            "system": "AR.I.S (Artificial Intelligent System)",
            "status": "operational",
            "version": "1.0.0",
            "message": "API gateway connection established successfully.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mongodb": mongo_status,
            "client_ip": request.META.get('REMOTE_ADDR', 'unknown'),
        }

        return Response(data, status=status.HTTP_200_OK)


class BrainFactListCreateView(APIView):
    """
    Token-Ready Brain Endpoint:
    - GET:  Retrieve stored facts from ArisBrain with optional query filters.
    - POST: Store a new fact into ArisBrain.
    """
    permission_classes = [AllowAny]

    def get(self, request, format=None):
        query = request.query_params.get('q') or request.query_params.get('query')
        topic = request.query_params.get('topic')
        tag = request.query_params.get('tag')
        limit = request.query_params.get('limit', 20)

        try:
            limit = int(limit)
        except (ValueError, TypeError):
            limit = 20

        facts = ArisBrain.retrieve_facts(query=query, topic=topic, tag=tag, limit=limit)
        total_count = ArisBrain.count_facts(topic=topic)

        # Serialize results for presentation
        serialized_facts = FactOutputSerializer(facts, many=True).data

        return Response({
            "count": len(serialized_facts),
            "total": total_count,
            "results": serialized_facts,
        }, status=status.HTTP_200_OK)

    def post(self, request, format=None):
        serializer = FactInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {"errors": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST
            )

        validated = serializer.validated_data

        try:
            fact = ArisBrain.store_fact(
                statement=validated["statement"],
                topic=validated.get("topic", "general"),
                tags=validated.get("tags", []),
                source="api_connection",
                confidence=validated.get("confidence", 1.0),
                metadata=validated.get("metadata", {}),
            )
            output = FactOutputSerializer(fact).data
            return Response({
                "message": "Fact stored successfully in AR.I.S Brain.",
                "fact": output
            }, status=status.HTTP_201_CREATED)
        except Exception as e:
            return Response(
                {"error": f"Failed to store fact in AR.I.S Brain: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class BrainFactDetailView(APIView):
    """
    Token-Ready Brain Detail Endpoint:
    - GET: Retrieve a single fact by its ID from ArisBrain.
    """
    permission_classes = [AllowAny]

    def get(self, request, fact_id, format=None):
        fact = ArisBrain.get_fact(fact_id)
        if not fact:
            return Response(
                {"error": f"Fact with id '{fact_id}' not found in AR.I.S Brain."},
                status=status.HTTP_404_NOT_FOUND
            )
        output = FactOutputSerializer(fact).data
        return Response({"fact": output}, status=status.HTTP_200_OK)


class BrainRelationalQueryView(APIView):
    """
    Tier 4 Relational Retrieval Endpoint:
    - GET: Execute a deterministic 1-hop relational query over stored Tier 3 bindings.
      Query parameters:
        - q / query: Natural query text (e.g. 'Who leads Axinix Technologies Development Team?')
        - subject: Exact subject anchor text
        - relation: Exact relation span text
        - object: Exact object anchor text
        - direction: Direction ('forward', 'reverse', 'link', 'verification')
        - limit: Maximum results to return (default 10)
    """
    permission_classes = [AllowAny]

    def get(self, request, format=None):
        query_text = request.query_params.get('q') or request.query_params.get('query')
        subject = request.query_params.get('subject')
        relation = request.query_params.get('relation')
        obj = request.query_params.get('object') or request.query_params.get('obj')
        direction = request.query_params.get('direction')
        limit = request.query_params.get('limit', 10)

        try:
            limit = int(limit)
        except (ValueError, TypeError):
            limit = 10

        try:
            result = ArisBrain.query_relations(
                query_text=query_text,
                subject=subject,
                relation=relation,
                obj=obj,
                direction=direction,
                limit=limit,
            )
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"error": f"Failed to execute relational query: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class BrainTraceView(APIView):
    """
    Tier 5 Cognitive Pipeline Trace Endpoint:
    - POST: Executes reasoning over query and extracts detailed stage-by-stage snapshots
      for real-time pipeline visualization.
    """
    permission_classes = [AllowAny]

    def post(self, request, format=None):
        query_text = (request.data.get("query") or request.data.get("q") or "").strip()
        if not query_text:
            return Response(
                {"error": "A 'query' string parameter is required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        from .linguistic.tokenizer import ArisTokenizer
        from .relational.relational import ArisRelationalBinder
        from .cognitive.epistemic_state import ArisEpistemicEvaluator

        # Stage 1: Raw Input
        stage_input = {
            "stage_id": "input",
            "title": "Raw Query Ingestion",
            "description": "Natural language query entered into the cognitive system",
            "data": {
                "query": query_text,
                "char_length": len(query_text),
                "word_count": len(query_text.split()),
            }
        }

        # Stage 2: Tokenization (Tier 1)
        tokens = ArisTokenizer.tokenize(query_text)
        token_data = [
            {
                "index": t.get("sequence_index", idx),
                "raw": t.get("raw", ""),
                "normalized": t.get("token", ""),
                "type": t.get("type", "word"),
                "char_start": t.get("position", {}).get("char_start", 0),
                "char_end": t.get("position", {}).get("char_end", 0),
            }
            for idx, t in enumerate(tokens)
        ]
        stage_tokens = {
            "stage_id": "tokens",
            "title": "Lexical Tokenization (Tier 1)",
            "description": "Deterministic segmenter tokenizing raw character stream with structural boundaries",
            "data": {
                "count": len(token_data),
                "tokens": token_data,
            }
        }

        # Stage 3: Candidate Spans (Tier 2)
        candidates = ArisTokenizer.extract_candidates(query_text, tokens=tokens)
        cand_data = [
            {
                "text": c.get("text", ""),
                "normalized": c.get("normalized", ""),
                "type": c.get("candidate_type", "entity_candidate"),
                "token_indices": c.get("token_indices", []),
                "char_start": c.get("char_start", 0),
                "char_end": c.get("char_end", 0),
            }
            for c in candidates
        ]
        stage_candidates = {
            "stage_id": "candidates",
            "title": "Candidate Entity Spans (Tier 2)",
            "description": "Dynamic multi-token candidate extraction without fixed dictionary lookups",
            "data": {
                "count": len(cand_data),
                "candidates": cand_data,
            }
        }


        # Stage 4: Relational Binding (Tier 3)
        bindings = ArisRelationalBinder.bind_relations(text=query_text, tokens=tokens, candidates=candidates)
        binding_data = []
        for b in bindings:
            binding_data.append({
                "subject": b["subject"]["text"] if b.get("subject") else None,
                "relation": b["relation"]["normalized"] if b.get("relation") else None,
                "object": b["object"]["text"] if b.get("object") else None,
                "confidence": b.get("confidence", 1.0),
                "pattern": b.get("pattern_type", "sro_binding"),
            })

        stage_bindings = {
            "stage_id": "bindings",
            "title": "Relational Binding (Tier 3)",
            "description": "Subject-Relation-Object structural binding with zero hardcoded phrase dictionaries",
            "data": {
                "count": len(binding_data),
                "bindings": binding_data,
            }
        }

        from .cognitive.epistemic_state import ArisGoalFormer, ArisEpistemicEvaluator

        # Epistemic pre-assessment
        goal = ArisGoalFormer.form_goal(
            query_text=query_text,
            tokens=tokens,
            candidates=candidates,
            bindings=bindings,
        )
        epistemic_eval = ArisEpistemicEvaluator.evaluate(goal=goal)
        epistemic_dict = {
            "state": epistemic_eval.get("state").value if hasattr(epistemic_eval.get("state"), "value") else str(epistemic_eval.get("state")),
            "deficit": epistemic_eval.get("deficit"),
            "goal": goal.target_variable,
            "target_type": goal.target_type,
            "anchor_entity": goal.anchor_entity,
            "relational_constraint": goal.relational_constraint,
            "matched_capability": epistemic_eval.get("required_capability"),
        }

        # Tier 5 Full Cognitive Reasoning Execution
        reason_result = ArisBrain.reason(question_text=query_text)


        trace = reason_result.get("reasoning_trace", {})
        epistemic_cognition = trace.get("epistemic_cognition", epistemic_dict)

        # Stage 5: Epistemic State
        stage_epistemic = {
            "stage_id": "epistemic",
            "title": "Epistemic State Assessment",
            "description": "Self-reflective cognitive appraisal (KNOWN / DERIVABLE / ACQUIRABLE / UNSUPPORTED)",
            "data": epistemic_cognition,
        }

        # Stage 6: Capability Selection / Cognitive Decision
        decision = trace.get("cognitive_decision", {})
        sensory = trace.get("sensory_observation", {})
        capability_name = trace.get("matched_capability") or epistemic_cognition.get("matched_capability") or "cognitive_reasoning"
        stage_decision = {
            "stage_id": "decision",
            "title": "Capability & Operation Selection",
            "description": "Autonomous dispatch to neural classifier, relational traversal, or capability executor",
            "data": {
                "matched_capability": capability_name,
                "cognitive_decision": decision,
                "sensory_observation": sensory,
            }
        }

        # Stage 7: Graph / Retrieval
        matches_count = reason_result.get("matches_count", 0)
        matched_facts = reason_result.get("matched_facts", [])
        traversal = trace.get("traversal", {})
        stage_retrieval = {
            "stage_id": "retrieval",
            "title": "Graph Traversal & Fact Retrieval",
            "description": "Multi-hop symbolic graph traversal and belief memory verification",
            "data": {
                "matches_count": matches_count,
                "matched_facts": matched_facts,
                "traversal": traversal,
                "applied_template": reason_result.get("applied_template"),
            }
        }

        from .cognitive.response_learner import ArisResponseLearner

        # Stage 8: Answer Synthesis with Natural Language Realization
        answers = reason_result.get("answers", [])
        clean_answers = []
        for ans in answers:
            if isinstance(ans, dict):
                clean_answers.append(ans.get("answer") or ans.get("statement") or str(ans))
            else:
                clean_answers.append(str(ans))

        final_ans_str = clean_answers[0] if clean_answers else (str(reason_result.get("result", "")) if "result" in reason_result else "")

        # Extract structural components for natural sentence generator
        subj_val = None
        rel_val = None
        obj_val = None
        slot_val = None

        if bindings:
            subj_val = bindings[0].get("subject")
            rel_val = bindings[0].get("relation")
            obj_val = bindings[0].get("object")

        if decision:
            if decision.get("target_slot") == "object":
                slot_val = "object"
                subj_val = decision.get("anchor") or subj_val
                rel_val = decision.get("relation") or rel_val
            elif decision.get("target_slot") == "subject":
                slot_val = "subject"
                obj_val = decision.get("anchor") or obj_val
                rel_val = decision.get("relation") or rel_val

        resp_learner_input = {
            "subject": subj_val,
            "relation": rel_val,
            "object": obj_val,
            "answer_slot": slot_val,
            "final_answer": final_ans_str,
            "reasoning_path": trace.get("traversal", {}).get("path") or (answers[0].get("path") if answers and isinstance(answers[0], dict) else []),
        }

        # Realize natural language sentence
        sentence = ""
        if answers and isinstance(answers[0], dict) and answers[0].get("statement"):
            sentence = answers[0]["statement"]
        elif "result" in reason_result and reason_result.get("result"):
            sentence = f"The result is {reason_result['result']}."
        elif final_ans_str:
            sentence = ArisResponseLearner.generate(resp_learner_input)
        else:
            sentence = ArisResponseLearner.generate(resp_learner_input)

        provenance_list = []
        if matched_facts:
            provenance_list = [f.get("statement", str(f)) for f in matched_facts]
        elif answers and isinstance(answers[0], dict) and answers[0].get("provenance"):
            provenance_list = answers[0].get("provenance")
        elif answers and isinstance(answers[0], dict) and answers[0].get("path_summary"):
            provenance_list = [f"Path: {answers[0]['path_summary']}"]

        stage_answer = {
            "stage_id": "answer",
            "title": "Cognitive Answer Synthesis",
            "description": "Final deduced output with epistemic confidence and provenance",
            "data": {
                "answers": clean_answers,
                "sentence": sentence,
                "status": "success" if (clean_answers or sentence) else ("unsupported" if epistemic_cognition.get("state") == "UNSUPPORTED" else "no_match"),
                "provenance": provenance_list,
            }
        }

        response_payload = {
            "query": query_text,
            "answers": clean_answers,
            "sentence": sentence,
            "stages": [
                stage_input,
                stage_tokens,
                stage_candidates,
                stage_bindings,
                stage_epistemic,
                stage_decision,
                stage_retrieval,
                stage_answer,
            ],
            "full_result": reason_result,
        }


        return Response(response_payload, status=status.HTTP_200_OK)


from django.shortcuts import render
from django.views import View
from .cognitive.training_lab import ArisTrainingLab


class VisualizerPageView(View):
    """
    Renders the AR.I.S Real-Time Cognitive Pipeline Visualizer.
    """
    def get(self, request):
        return render(request, "aris_visualizer.html")


class TrainingLabPageView(View):
    """
    Renders the AR.I.S 14-Level Generalization Training Lab dashboard.
    """
    def get(self, request):
        return render(request, "aris_training_lab.html")


class TrainingLabRunView(APIView):
    """
    Executes the 14-Level Curriculum Generalization Benchmark.
    """
    permission_classes = [AllowAny]

    def get(self, request):
        runner = ArisTrainingLab()
        results = runner.run_benchmark()
        return Response(results, status=status.HTTP_200_OK)

    def post(self, request):
        runner = ArisTrainingLab()
        results = runner.run_benchmark()
        return Response(results, status=status.HTTP_200_OK)

