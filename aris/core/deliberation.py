"""
AR.I.S — Autonomous Cognitive Deliberation Engine (2026 Production Grade)
Orchestrates internal multi-hop reasoning, self-directed Graph RAG memory retrieval,
vectorized KV-Cache inference, and continuous memory consolidation.
"""

from __future__ import annotations
import re
from typing import List, Dict, Any, Tuple, Optional
import torch

from ..models.aris_lm import ArisLM
from ..tokenizer.bpe_tokenizer import ArisTokenizer
from ..inference.generation import ArisInferenceEngine
from ..memory.retriever import HybridGraphRetriever
from ..memory.schema import EntityNode, RelationEdge
from .governance import EpistemicGovernor


class AutonomousDeliberationEngine:
    """
    Autonomous Cognitive Deliberation Engine:
    Decodes internally before responding, consults its own Graph RAG memory,
    and consolidates learned knowledge continuously under Epistemic Governance.
    """

    def __init__(
        self,
        model: ArisLM,
        tokenizer: ArisTokenizer,
        retriever: HybridGraphRetriever,
        governor: Optional[EpistemicGovernor] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.governor = governor or EpistemicGovernor(graph_store=retriever.graph_store)
        self.device = device or torch.device("cpu")

    def deliberate(
        self,
        user_input: str,
        history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """
        Executes a 5-step cognitive deliberation cycle:
          1. Self-directed memory search across the Neuro-Symbolic Graph.
          2. Multi-hop context compilation.
          3. Neural prompt formatting with ChatML.
          4. Vectorized KV-Cache auto-regressive generation.
          5. Memory consolidation of new conversational entities into the Graph.
        """
        history = list(history) if history else []

        # 1 & 2. Self-directed retrieval from internal Graph RAG
        memory_context = self.retriever.build_context_narrative(user_input, top_k=3)

        augmented_user_input = user_input
        if memory_context:
            augmented_user_input = (
                f"Background Information:\n{memory_context}\n"
                f"Question: {user_input}"
            )

        # 3. Construct ChatML prompt
        active_turns = history + [{"role": "user", "content": augmented_user_input}]
        prompt_ids = self.tokenizer.encode_chat(active_turns, add_generation_prompt=True)
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        # 4. Neural generation with KV caching
        output_tensor = ArisInferenceEngine.generate(
            model=self.model,
            input_ids=input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_id=self.tokenizer.im_end_id,
        )

        gen_ids = output_tensor[0][len(prompt_ids):].tolist()
        raw_response = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Parse <|thought|> chain and answer if present
        thought = ""
        response_text = raw_response

        if "<|thought|>" in raw_response:
            parts = raw_response.split("<|thought|>")
            if len(parts) > 1:
                thought_body = parts[1]
                if "Answer:" in thought_body:
                    t_part, a_part = thought_body.split("Answer:", 1)
                    thought = t_part.strip()
                    response_text = a_part.strip()
                else:
                    thought = thought_body.strip()
                    response_text = thought_body.strip()

        # 5. Continuous Memory Consolidation
        self.consolidate_knowledge(user_input, response_text)

        return {
            "response": response_text,
            "thought": thought,
            "raw_output": raw_response,
            "memory_context": memory_context,
        }

    def consolidate_knowledge(self, user_text: str, assistant_text: str):
        """
        Stores salient entities into the knowledge graph under Epistemic Authority gating.
        """
        words = user_text.split()
        for w in words:
            clean_w = w.strip(".,!?:;\"'()[]{}")
            if clean_w and clean_w[0].isupper() and len(clean_w) > 2 and clean_w.lower() not in ("what", "where", "when", "which", "how", "why", "who", "the", "this", "that"):
                ent_id = clean_w.lower()
                
                # Check Epistemic Governor authority
                gov_eval = self.governor.evaluate_candidate_memory(clean_w, description=user_text)
                perm = self.governor.check_action_permission("memory_write", {"confidence": gov_eval["confidence"]})
                
                if not perm.get("allowed", False):
                    continue

                existing = self.retriever.graph_store.get_entity(ent_id)
                if not existing:
                    emb = self.retriever.encode_text(clean_w)
                    node = EntityNode(
                        id=ent_id,
                        name=clean_w,
                        entity_type=gov_eval["status"],
                        description=f"[{gov_eval['status'].upper()}] Observed in dialogue: {user_text[:80]}",
                        embedding=emb,
                        access_count=1,
                    )
                    self.retriever.graph_store.add_entity(node)
                else:
                    existing.access_count += 1
                    # Promote status if corroborated
                    if gov_eval["status"] == "verified":
                        existing.entity_type = "verified"
                    self.retriever.graph_store.add_entity(existing)

