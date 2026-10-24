# pyright: ignore
"""
AR.I.S — Sensory Structural Advisor Primitive (Layer 0)

Core Architectural Invariant:
  "The advisor observes. AR.I.S learns, identifies patterns, reasons,
   allocates work, and answers."

The Structural Advisor acts strictly as a sensory receptor (eye/ear).
It observes the syntactic/structural roles of slots in a query sentence
(whether a slot represents an open variable '?' or a bound reference anchor).

Strict Constraints:
  - It NEVER receives AR.I.S memory facts.
  - It NEVER answers the question.
  - It NEVER decides the retrieval/traversal operation (forward vs reverse).
  - It NEVER re-interprets relational spans (Tier 3 owns S-R-O).
  - Clean Failure Policy: If unavailable or disabled, returns status 'unavailable'.
    AR.I.S produces 0 matches (no hardcoded word list fallbacks).
"""

import os
import json
import re
import httpx
from typing import Dict, Any, Optional


class ArisStructuralAdvisor:
    """
    Sensory receptor that observes structural query slots without reasoning or answering.
    """

    SYSTEM_PROMPT = (
        "You are a structural syntactic sensor.\n"
        "You are given two syntactic slots from a sentence: 'subject' and 'object'.\n"
        "One slot contains a known entity anchor (e.g. 'A', 'B', 'Axinix').\n"
        "The other slot contains an open question variable / unknown placeholder (e.g. 'Who', 'who', 'what', '?').\n"
        "Identify which slot is the 'variable_slot' and which is the 'anchor_slot'.\n"
        "Output ONLY raw JSON:\n"
        '{"variable_slot": "subject" | "object", "anchor_slot": "subject" | "object"}'
    )

    @classmethod
    def _get_api_config(cls) -> Optional[Dict[str, str]]:
        """
        Resolves active API endpoint and credentials from environment.
        Prefers OPENROUTER_API_KEY if available; falls back to OPENAI_API_KEY.
        """
        from dotenv import load_dotenv
        load_dotenv()

        openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()

        # Prioritize OpenRouter if OpenAI is missing or known to be credit-limited
        if openrouter_key and openrouter_key.startswith("sk-or-"):
            m = os.getenv("OPENROUTER_MODEL", "")
            chosen_model = m if (m and m != "openrouter/auto") else "openai/gpt-4o-mini"
            return {
                "url": "https://openrouter.ai/api/v1/chat/completions",
                "key": openrouter_key,
                "model": chosen_model,
            }
        elif openai_key:
            return {
                "url": "https://api.openai.com/v1/chat/completions",
                "key": openai_key,
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            }
        return None

    @classmethod
    def get_system_prompt(cls) -> str:
        """
        Builds dynamic system prompt containing the knowledge of registered capabilities.
        The advisor acts as a general structural, syntactic, and capability perception sensor.
        Contains zero problem-specific constraints and zero narrow hardcodes.
        """
        from ..semantic.executable_tools import ArisCapabilityRegistry
        cap_knowledge = ArisCapabilityRegistry.get_capabilities_knowledge()
        return (
            "You are the structural and cognitive perception sensor for AR.I.S (Artificial Intelligent System).\n"
            "You observe the syntactic, relational, and computational structure of any input query.\n"
            "You NEVER answer questions, NEVER access memory, and NEVER invent facts.\n\n"
            "Analyze the sentence and extract all available structural data in JSON format:\n"
            "- 'target_domain': The semantic category/type of the expected target (e.g. 'entity', 'location', 'temporal', 'quantity', 'property', 'boolean', 'definition').\n"
            "- 'target_variable': The open question variable or placeholder (e.g. 'who', 'where', 'when', 'what', 'how much', '?').\n"
            "- 'anchors': List of concrete named entities, concepts, or values mentioned in the query that anchor it.\n"
            "- 'type': The primary structural intent ('executable_capability' | 'relational_traversal' | 'factual_lookup').\n"
            "- 'capability': Name of registered capability if matching an executable tool.\n"
            "- 'parameters': Dictionary of extracted capability arguments conforming to its schema.\n"
            "- 'start_entity': The concrete starting entity for relational navigation.\n"
            "- 'hops': The ordered sequence of relational steps required to traverse from start_entity to the unknown target. Each hop has:\n"
            "    * 'relation': Canonical predicate or relationship.\n"
            "    * 'direction': 'forward' (entity -> target) or 'reverse' (target -> entity).\n"
            "- 'variable_slot': 'subject' or 'object' if analyzing a direct 2-slot relation.\n"
            "- 'anchor_slot': 'subject' or 'object' corresponding to the known entity anchor.\n\n"
            f"Known Executable Capabilities:\n{cap_knowledge}\n\n"
            "Output ONLY raw JSON containing the identified fields."
        )


    @classmethod
    def observe_query(
        cls,
        query_text: str,
        subject: Optional[str] = None,
        obj: Optional[str] = None,
        enabled: bool = True,
    ) -> Dict[str, Any]:
        """
        Observes all structural, relational, and computational dimensions of a query.
        Returns all extracted fields without limiting the AI.
        """
        if not enabled or not query_text or not query_text.strip():
            return {
                "status": "unavailable",
                "type": None,
                "variable_slot": None,
                "anchor_slot": None,
                "parameters": None,
            }

        config = cls._get_api_config()
        if not config:
            return {
                "status": "unavailable",
                "type": None,
                "variable_slot": None,
                "anchor_slot": None,
                "parameters": None,
            }

        headers = {
            "Authorization": f"Bearer {config['key']}",
            "Content-Type": "application/json",
        }

        user_content = f"Query sentence: '{query_text.strip()}'"
        if subject and obj:
            user_content += f"\nExtracted entity slots -> subject: '{subject.strip()}', object: '{obj.strip()}'"

        payload = {
            "model": config["model"],
            "messages": [
                {"role": "system", "content": cls.get_system_prompt()},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": 300,
            "temperature": 0.0,
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(config["url"], headers=headers, json=payload)
                if response.status_code != 200:
                    return {"status": "unavailable"}

                data = response.json()
                msg = data.get("choices", [{}])[0].get("message", {})
                content = (msg.get("content") or "").strip()
                if not content:
                    return {"status": "unavailable"}

                clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE).strip()
                parsed = json.loads(clean_json)

                obs_type = parsed.get("type")
                if not obs_type:
                    if parsed.get("capability"):
                        obs_type = "executable_capability"
                    elif parsed.get("hops") or parsed.get("start_entity"):
                        obs_type = "relational_traversal"
                    elif "variable_slot" in parsed:
                        obs_type = "relational_traversal"
                    else:
                        obs_type = "factual_lookup"

                start_ent = parsed.get("start_entity")
                anchors = parsed.get("anchors") or []
                if not start_ent and anchors:
                    start_ent = anchors[0]

                var_slot = parsed.get("variable_slot")
                anc_slot = parsed.get("anchor_slot")

                # Derive variable_slot and anchor_slot if model specified start_entity, anchors, or hops
                if not anc_slot and subject and obj:
                    s_low = subject.strip().lower()
                    o_low = obj.strip().lower()
                    if start_ent and start_ent.strip().lower() == o_low:
                        anc_slot = "object"
                        var_slot = "subject"
                    elif start_ent and start_ent.strip().lower() == s_low:
                        anc_slot = "subject"
                        var_slot = "object"
                    elif anchors:
                        if any(a.strip().lower() == o_low for a in anchors):
                            anc_slot = "object"
                            var_slot = "subject"
                        elif any(a.strip().lower() == s_low for a in anchors):
                            anc_slot = "subject"
                            var_slot = "object"
                    elif parsed.get("hops"):
                        first_hop = parsed["hops"][0]
                        if first_hop.get("direction") == "reverse":
                            anc_slot = "object"
                            var_slot = "subject"
                        elif first_hop.get("direction") == "forward":
                            anc_slot = "subject"
                            var_slot = "object"

                return {
                    "status": "ok",
                    "type": obs_type,
                    "target_domain": parsed.get("target_domain"),
                    "target_variable": parsed.get("target_variable"),
                    "anchors": anchors,
                    "start_entity": start_ent,
                    "hops": parsed.get("hops", []),
                    "capability": parsed.get("capability"),
                    "parameters": parsed.get("parameters", {}),
                    "variable_slot": var_slot,
                    "anchor_slot": anc_slot,
                    "raw_observation": parsed,
                }
        except Exception:
            pass

        return {"status": "unavailable"}

    @classmethod
    def observe_slots(
        cls,
        subject: str,
        obj: str,
        query_text: Optional[str] = None,
        enabled: bool = True,
    ) -> Dict[str, Any]:
        """
        Sensory observation for relational slots (backward compatible).
        """
        obs = cls.observe_query(
            query_text=query_text or f"{subject} {obj}",
            subject=subject,
            obj=obj,
            enabled=enabled,
        )
        if obs.get("status") == "ok":
            return {
                "status": "ok",
                "variable_slot": obs.get("variable_slot"),
                "anchor_slot": obs.get("anchor_slot"),
                "raw_observation": obs.get("raw_observation"),
            }
        return {
            "status": "unavailable",
            "variable_slot": None,
            "anchor_slot": None,
            "raw_observation": None,
        }

    @classmethod
    def decompose_multi_hop(
        cls,
        query_text: str,
        enabled: bool = True,
    ) -> Dict[str, Any]:
        """
        Sensory receptor that observes relational inquiries and extracts
        start_entity, target_domain, and ordered relational hops via unified observation.
        """
        obs = cls.observe_query(query_text=query_text, enabled=enabled)
        if obs.get("status") == "ok":
            return {
                "status": "ok",
                "start_entity": obs.get("start_entity"),
                "target_domain": obs.get("target_domain") or "entity",
                "hops": obs.get("hops", []),
                "raw_observation": obs.get("raw_observation"),
            }

        return {"status": "unavailable", "start_entity": None, "hops": [], "target_domain": "unknown"}


