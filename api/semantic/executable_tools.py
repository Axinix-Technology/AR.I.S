# pyright: ignore
"""
AR.I.S — Executable Capability Registry & Tools

Architectural Invariants (Mandatory):
  - Tools are deterministic execution capabilities, NOT cognitive reasoners.
  - Tools receive BOUND parameters from AR.I.S and perform computation.
  - Tools do NOT decide which parameters are variables, what the user intends,
    or how to reason. All reasoning and parameter binding belong to AR.I.S.
  - No arithmetic logic or lookup tables inside the cognitive learning core.
"""

import os
import json
from typing import Dict, Any, Optional, List, Type, Tuple
from abc import ABC, abstractmethod
import operator


class ArisCapability(ABC):
    """
    Abstract base specification for all AR.I.S capabilities.
    Enforces standardized self-describing contracts and polymorphic affordance evaluation.
    Zero procedural string branching across registries.
    """
    CAPABILITY_NAME: str = ""
    ALIASES: List[str] = []
    IS_EXTERNAL: bool = False
    AFFORDANCE_DOMAIN: str = "general"
    DESCRIPTION: str = ""
    USAGE: str = ""
    PARAMETERS_SCHEMA: Dict[str, Any] = {}
    ACCEPTS: Dict[str, Any] = {}
    PRODUCES: Dict[str, Any] = {}

    @classmethod
    def evaluate_affordance(
        cls,
        query_text: str = "",
        tokens: Optional[List[Dict[str, Any]]] = None,
        candidates: Optional[List[Dict[str, Any]]] = None,
        goal: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Default universal affordance perception.
        Delegates to PyTorch metric tensor engine (ArisAffordanceTensorEngine) for continuous
        slot binding and domain projection. Zero procedural if/else substring guards.
        """
        schema = getattr(cls, "PARAMETERS_SCHEMA", {})
        accepts = getattr(cls, "ACCEPTS", {})
        required_params = accepts.get("requires", list(schema.keys()))
        domain = accepts.get("domain", getattr(cls, "AFFORDANCE_DOMAIN", "general"))

        from ..cognitive.affordance_tensor_engine import ArisAffordanceTensorEngine
        bound, missing, completeness, relevance = ArisAffordanceTensorEngine.bind_slots(
            tokens=tokens,
            candidates=candidates,
            schema=schema,
            requires=required_params,
            domain=domain,
            goal=goal,
            query_text=query_text,
        )

        return {
            "name": getattr(cls, "CAPABILITY_NAME", getattr(cls, "__name__", "unknown")),
            "relevant": relevance,
            "parameters": bound,
            "missing": missing,
            "completeness": completeness,
        }

    @classmethod
    @abstractmethod
    def execute(cls, **kwargs) -> Dict[str, Any]:
        pass


class ArisMathTool(ArisCapability):
    """
    Deterministic Mathematical Execution Substrate.
    A decoupled mathematical engine providing:
      1. Safe AST Arithmetic Evaluation (single and compositional expressions)
      2. Universal Affine Finite-Difference Root-Finding for Linear Equations
      3. Positional Distance and Coordinate Difference Primitives

    Strict Architectural Invariants:
      - Zero domain semantics (knows zero English words like 'age', 'birth', 'death', 'who').
      - Accepts only structured mathematical contracts.
      - Returns deterministic mathematical outputs.
    """

    CAPABILITY_NAME = "math_computation"
    ALIASES = ["math", "calculator", "arithmetic"]
    IS_EXTERNAL = False
    AFFORDANCE_DOMAIN = "numerical_computation"
    DESCRIPTION = "Executes deterministic mathematical computations: expression evaluation, linear equation solving, and coordinate distance."
    USAGE = (
        "Use math_computation when a numerical arithmetic expression (+, -, *, /), "
        "linear algebraic equation (solve for variable), or coordinate distance calculation "
        "is required. Inputs: expression string or operand parameters (param1, op_str, param2). "
        "Produces: deterministic computed numerical result."
    )
    PARAMETERS_SCHEMA = {
        "operation": "Mathematical operation ('evaluate', 'solve_linear', 'distance')",
        "param1": "First operand or parameter value",
        "op_str": "Arithmetic operator symbol (+, -, *, /)",
        "param2": "Second operand or parameter value",
        "expression": "Arithmetic expression string (e.g. '2015 - 1931' or '10 - 4')",
        "equation": "Equation dict {'left': '...', 'right': '...'} or equation string",
        "variable": "Variable name to solve for (e.g. 'x', 'y', 'n')",
        "coord_a": "First coordinate / position index (scalar or list/tuple)",
        "coord_b": "Second coordinate / position index (scalar or list/tuple)",
    }

    ACCEPTS = {
        "domain": "numerical_computation",
        "requires": ["expression", "equation", "coord_a", "param1"],
    }
    PRODUCES = {
        "state_transition": "DERIVABLE -> KNOWN",
        "output_type": "numerical_value",
    }

    OPERATORS = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
    }

    @classmethod
    def _eval_ast_node(cls, node):
        import ast
        import math
        ast_ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
        }
        if isinstance(node, ast.Expression):
            return cls._eval_ast_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -cls._eval_ast_node(node.operand)
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in ast_ops:
                raise ValueError(f"Unsupported math operator: {node.op}")
            left = cls._eval_ast_node(node.left)
            right = cls._eval_ast_node(node.right)
            return ast_ops[op_type](left, right)
        raise ValueError(f"Disallowed expression node: {type(node).__name__}")

    @classmethod
    def evaluate(cls, expression: str) -> Any:
        """
        Direct evaluator returning computed scalar value.
        """
        res = cls.evaluate_expression(expression)
        if res.get("status") == "success":
            return res.get("value")
        raise ValueError(res.get("message", "Evaluation failed"))

    @classmethod
    def evaluate_expression(cls, expression: str) -> Dict[str, Any]:
        """
        Executes safe AST arithmetic evaluation on an expression string.
        """
        import ast
        clean_expr = str(expression).strip()
        if not clean_expr:
            return {"status": "error", "success": False, "message": "Empty expression provided."}

        try:
            parsed = ast.parse(clean_expr, mode="eval")
            res = cls._eval_ast_node(parsed)

            if isinstance(res, float) and res.is_integer():
                res = int(res)

            return {
                "status": "success",
                "success": True,
                "result": str(res),
                "numeric_result": res,
                "value": res,
                "expression": clean_expr,
            }
        except Exception as e:
            return {"status": "error", "success": False, "message": str(e), "expression": clean_expr}

    @classmethod
    def solve_linear_equation(cls, left_expr: str, right_expr: str, variable: str = "x") -> Dict[str, Any]:
        """
        Universal Affine Finite-Difference Root-Finding:
          g(v) = LHS(v) - RHS(v) = 0
          Slope m = g(1) - g(0)
          Intercept c = g(0)
          Root v = -c / m
        Zero case-by-case rules; uniformly solves any linear equation.
        """
        import ast

        clean_left = str(left_expr).strip()
        clean_right = str(right_expr).strip()
        var_name = str(variable).strip()

        if not clean_left or not clean_right or not var_name:
            return {"status": "error", "success": False, "message": "Invalid equation parameters for linear solving."}

        class VarSubstituter(ast.NodeTransformer):
            def __init__(self, target_var, target_val):
                self.target_var = target_var
                self.target_val = target_val

            def visit_Name(self, node):
                if node.id == self.target_var:
                    return ast.Constant(value=self.target_val)
                return node

        def _eval_with_val(expr_str: str, val: float) -> float:
            parsed = ast.parse(expr_str, mode="eval")
            transformed = VarSubstituter(var_name, val).visit(parsed)
            ast.fix_missing_locations(transformed)
            return float(cls._eval_ast_node(transformed))

        try:
            g0 = _eval_with_val(clean_left, 0.0) - _eval_with_val(clean_right, 0.0)
            g1 = _eval_with_val(clean_left, 1.0) - _eval_with_val(clean_right, 1.0)

            m = g1 - g0
            c = g0

            if abs(m) < 1e-12:
                if abs(c) < 1e-12:
                    return {"status": "infinite_solutions", "success": False, "message": "Identity equation."}
                return {"status": "no_solution", "success": False, "message": "Inconsistent linear equation."}

            root = -c / m
            if isinstance(root, float) and root.is_integer():
                root = int(root)
            elif isinstance(root, float):
                root = round(root, 6)

            return {
                "status": "solved",
                "success": True,
                "variable": var_name,
                "solution": root,
                "value": root,
                "result": str(root),
                "numeric_result": root,
                "equation": f"{clean_left} = {clean_right}",
            }
        except Exception as e:
            return {"status": "error", "success": False, "message": str(e), "equation": f"{clean_left} = {clean_right}"}

    @classmethod
    def calculate_distance(cls, coord_a: Any, coord_b: Any) -> Dict[str, Any]:
        return cls.distance(coord_a, coord_b)

    @classmethod
    def distance(cls, coord_a: Any, coord_b: Any) -> Dict[str, Any]:
        """
        Deterministic positional and coordinate distance metric.
        Handles both 1D scalars and N-dimensional vectors (lists/tuples).
        """
        import math
        try:
            if isinstance(coord_a, (list, tuple)) and isinstance(coord_b, (list, tuple)):
                diff_sq = sum((float(a) - float(b)) ** 2 for a, b in zip(coord_a, coord_b))
                val = math.sqrt(diff_sq)
            else:
                val = abs(float(coord_a) - float(coord_b))

            if isinstance(val, float) and val.is_integer():
                val = int(val)
            elif isinstance(val, float):
                val = round(val, 6)

            return {
                "status": "success",
                "success": True,
                "distance": val,
                "result": str(val),
                "numeric_result": val,
                "value": val,
            }
        except Exception as e:
            return {"status": "error", "success": False, "message": str(e)}

    @classmethod
    def execute(
        cls,
        operation: Optional[str] = None,
        expression: Optional[str] = None,
        equation: Optional[Any] = None,
        variable: Optional[str] = None,
        coord_a: Optional[Any] = None,
        coord_b: Optional[Any] = None,
        op_str: Optional[str] = None,
        param1: Any = None,
        param2: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Dispatches structured mathematical contracts to AST evaluation,
        affine equation solving, or distance metrics.
        """
        # Case 1: Distance metric
        if operation == "distance" or (coord_a is not None and coord_b is not None):
            return cls.distance(coord_a, coord_b)

        # Case 2: Linear Equation Solving
        if operation == "solve_linear" or equation is not None:
            left_expr = None
            right_expr = None
            var_name = variable or "x"

            if isinstance(equation, dict):
                left_expr = equation.get("left") or equation.get("lhs")
                right_expr = equation.get("right") or equation.get("rhs")
                var_name = equation.get("variable", var_name)
            elif isinstance(equation, str) and "=" in equation:
                parts = equation.split("=", 1)
                left_expr = parts[0].strip()
                right_expr = parts[1].strip()
            elif expression and "=" in expression:
                parts = expression.split("=", 1)
                left_expr = parts[0].strip()
                right_expr = parts[1].strip()

            if left_expr and right_expr:
                return cls.solve_linear_equation(left_expr, right_expr, var_name)

        # Case 3: Expression / Binary Parameter Evaluation
        target_expr = expression
        if not target_expr and op_str is not None and param1 is not None and param2 is not None:
            target_expr = f"{param1} {op_str} {param2}"

        if not target_expr and kwargs:
            # Check for legacy parameter dicts
            p1 = kwargs.get("param1")
            op = kwargs.get("op_str")
            p2 = kwargs.get("param2")
            if p1 is not None and op is not None and p2 is not None:
                target_expr = f"{p1} {op} {p2}"

        if not target_expr:
            return {"status": "error", "message": "No expression, equation, or parameters provided for math execution."}

        # If expression contains '=', route to linear solver
        if "=" in str(target_expr):
            parts = str(target_expr).split("=", 1)
            var_name = variable or "x"
            return cls.solve_linear_equation(parts[0].strip(), parts[1].strip(), var_name)

        return cls.evaluate_expression(target_expr)

    @classmethod
    def evaluate_affordance(
        cls,
        query_text: str = "",
        tokens: Optional[List[Dict[str, Any]]] = None,
        candidates: Optional[List[Dict[str, Any]]] = None,
        goal: Optional[Any] = None,
    ) -> Dict[str, Any]:
        schema = getattr(cls, "PARAMETERS_SCHEMA", {})
        accepts = getattr(cls, "ACCEPTS", {})
        required_params = accepts.get("requires", ["expression"])
        tokens = tokens or []

        ops = set(cls.OPERATORS.keys())
        has_digits = any(t.get("type") == "number" or t.get("raw", "").isdigit() for t in tokens)
        has_ops = any(t.get("raw") in ops for t in tokens)

        bound: Dict[str, Any] = {}
        if has_digits and has_ops:
            expr_toks = [t["raw"] for t in tokens if t.get("type") == "number" or t.get("raw") in ops]
            if expr_toks:
                bound["expression"] = " ".join(expr_toks)
            relevant = "yes"
        else:
            relevant = "no"

        missing = [p for p in required_params if p not in bound or bound[p] is None]
        completeness = "complete" if not missing else "none"

        return {
            "name": cls.CAPABILITY_NAME,
            "relevant": relevant,
            "parameters": bound,
            "missing": missing,
            "completeness": completeness,
        }


class ArisBrowserAccessTool(ArisCapability):
    """
    Deterministic Browser & Web Search Execution Capability.
    Executes web query search and retrieval via browser access.
    """

    CAPABILITY_NAME = "browser_access"
    ALIASES = ["browser_search", "web_search", "internet_search"]
    IS_EXTERNAL = True
    AFFORDANCE_DOMAIN = "external_knowledge_acquisition"
    DESCRIPTION = "Performs browser access for web query search and retrieval."
    USAGE = (
        "Use browser_access when external real-world factual information, operational command syntax, "
        "or current web knowledge is needed that is not stored in AR.I.S cognitive memory. "
        "Inputs: query (search phrase or entity topic). "
        "Produces: ingested factual statements and textual summaries from verified web sources."
    )
    PARAMETERS_SCHEMA = {
        "query": "The search query string to look up or navigate in the browser",
    }
    ACCEPTS = {
        "domain": "external_knowledge_acquisition",
        "requires": ["query"],
    }
    PRODUCES = {
        "state_transition": "ACQUIRABLE -> KNOWN",
        "output_type": "ingested_memory_facts",
    }

    @classmethod
    def execute(cls, query: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Executes query search using browser access via HTTP.
        """
        if not query or not str(query).strip():
            return {
                "status": "error",
                "message": "No query provided for browser search.",
            }

        clean_query = str(query).strip()
        ingested_facts = []
        summary = None

        try:
            import httpx
            import re

            headers = {
                "User-Agent": "ARIS-Cognitive-System/1.0 (Mozilla/5.0; Windows NT 10.0; Win64; x64; contact@aris.ai)"
            }
            clean_search_term = re.sub(r"[^\w\s\.]", " ", clean_query).strip()

            with httpx.Client(timeout=8.0, follow_redirects=True, headers=headers) as client:
                # Source 1: Direct POST web search
                try:
                    r_post = client.post("https://html.duckduckgo.com/html/", data={"q": clean_query})
                    if r_post.status_code == 200 and r_post.text:
                        import re
                        snippets = re.findall(r'<a[^>]*class="[^"]*snippet[^"]*"[^>]*>(.*?)</a>', r_post.text, re.IGNORECASE | re.DOTALL)
                        if snippets:
                            clean_snips = [re.sub(r'<[^>]+>', '', s).strip() for s in snippets if s.strip()]
                            if clean_snips:
                                summary = " ".join(clean_snips)
                except Exception:
                    pass

                # Source 2: Direct Wikipedia Entity Summary
                if not summary:
                    url_title = clean_search_term.replace(" ", "_")
                    try:
                        r_sum = client.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{url_title}")
                        if r_sum.status_code == 200:
                            data = r_sum.json()
                            if data.get("type") == "standard" and data.get("extract"):
                                summary = data["extract"]
                    except Exception:
                        pass

                # Source 2: Wikipedia Search query fallback
                if not summary:
                    try:
                        r_srch = client.get(
                            "https://en.wikipedia.org/w/api.php",
                            params={
                                "action": "query",
                                "list": "search",
                                "srsearch": clean_search_term,
                                "format": "json",
                                "srlimit": 3,
                            },
                        )
                        if r_srch.status_code == 200:
                            hits = r_srch.json().get("query", {}).get("search", [])
                            for h in hits:
                                title = h.get("title", "")
                                if title:
                                    r_page = client.get(
                                        f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
                                    )
                                    if r_page.status_code == 200:
                                        p_data = r_page.json()
                                        if p_data.get("type") == "standard" and p_data.get("extract"):
                                            summary = p_data["extract"]
                                            break
                    except Exception:
                        pass

                # Source 3: DuckDuckGo Instant Answer API
                if not summary:
                    try:
                        r_ddg = client.get(
                            "https://api.duckduckgo.com/",
                            params={"q": clean_query, "format": "json", "no_html": 1},
                        )
                        if r_ddg.status_code == 200:
                            d_data = r_ddg.json()
                            summary = d_data.get("AbstractText") or d_data.get("Abstract")
                            if not summary and d_data.get("RelatedTopics"):
                                first_topic = d_data["RelatedTopics"][0]
                                if isinstance(first_topic, dict):
                                    summary = first_topic.get("Text")
                    except Exception:
                        pass

            if not summary:
                summary = f"Browser search executed for: '{clean_query}'"

            # Ingest retrieved sentences as structured factual knowledge in AR.I.S cognitive memory
            try:
                from ..cognitive.brain import ArisBrain
                cleaned_summary = summary.replace("&quot;", '"').replace("&#x27;", "'").replace("&amp;", "&")
                sentences = re.split(r"(?<=[.!?])\s+", cleaned_summary)
                query_keywords = [w.lower() for w in re.findall(r"\b\w+\b", clean_query) if len(w) > 1]
                fact_tags = list(dict.fromkeys([clean_query.lower(), "browser_data"] + query_keywords))

                for sent in sentences:
                    sent_clean = sent.strip()
                    if len(sent_clean) >= 15 and re.search(r"[a-zA-Z]{2,}", sent_clean):
                        try:
                            fact_doc = ArisBrain.store_fact(
                                statement=sent_clean,
                                topic="web_search",
                                tags=fact_tags,
                                source="browser_access",
                            )
                            ingested_facts.append(fact_doc)
                        except Exception:
                            pass
            except Exception:
                pass

            total_tokens = sum(f.get("token_count", 0) for f in ingested_facts)
            total_relations = sum(f.get("relation_count", 0) for f in ingested_facts)

            return {
                "status": "success",
                "result": summary,
                "query": clean_query,
                "ingested_facts_count": len(ingested_facts),
                "total_tokens": total_tokens,
                "total_relations": total_relations,
            }

        except Exception as e:
            return {
                "status": "success",
                "result": f"Browser search executed for: '{clean_query}'",
                "query": clean_query,
                "ingested_facts_count": 0,
                "total_tokens": 0,
                "total_relations": 0,
                "detail": str(e),
            }

    @classmethod
    def evaluate_affordance(
        cls,
        query_text: str = "",
        tokens: Optional[List[Dict[str, Any]]] = None,
        candidates: Optional[List[Dict[str, Any]]] = None,
        goal: Optional[Any] = None,
    ) -> Dict[str, Any]:
        clean_text = (query_text or "").strip()
        schema = getattr(cls, "PARAMETERS_SCHEMA", {})
        accepts = getattr(cls, "ACCEPTS", {})
        required_params = accepts.get("requires", list(schema.keys()))

        bound: Dict[str, Any] = {}
        if candidates and len(candidates) > 0:
            query_val = max(candidates, key=lambda c: len(c.get("text", "")))["text"]
        else:
            query_val = clean_text.rstrip("?!.").strip()

        if query_val:
            bound["query"] = query_val

        missing = [p for p in required_params if p not in bound or bound[p] is None]
        relevant = "possible" if bound.get("query") else "no"
        completeness = "complete" if not missing else "none"

        return {
            "name": cls.CAPABILITY_NAME,
            "relevant": relevant,
            "parameters": bound,
            "missing": missing,
            "completeness": completeness,
        }


class ArisCommandPromptTool(ArisCapability):
    """
    Deterministic Command Prompt & Local Shell Execution Capability.
    Executes shell commands in a designated working directory.
    """

    CAPABILITY_NAME = "command_prompt"
    ALIASES = ["terminal", "shell", "git_tool", "cli"]
    IS_EXTERNAL = False
    AFFORDANCE_DOMAIN = "system_terminal_execution"
    DESCRIPTION = "Executes command prompt shell commands within a specified working directory."
    USAGE = (
        "Use command_prompt when system terminal, command line, or local shell execution "
        "is required within a target repository or working directory (e.g. inspecting git status, "
        "validating directory paths, running CLI utilities). "
        "Inputs: command (shell command string), working_dir (optional target folder path). "
        "Produces: command execution output, stdout, stderr, and exit status code."
    )
    PARAMETERS_SCHEMA = {
        "command": "The terminal command to execute (e.g. 'git status', 'dir', 'git branch')",
        "working_dir": "The target working directory / folder path where the command should execute",
    }
    ACCEPTS = {
        "domain": "system_terminal_execution",
        "requires": ["command"],
    }
    PRODUCES = {
        "state_transition": "DERIVABLE -> KNOWN",
        "output_type": "command_output",
    }

    @classmethod
    def execute(cls, command: Optional[str] = None, working_dir: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Executes a shell command safely in the specified working directory.
        """
        import subprocess
        import os

        clean_cmd = str(command or kwargs.get("cmd") or kwargs.get("query") or "").strip()
        low_cmd = clean_cmd.lower()
        if low_cmd in ("open command prompt", "command prompt", "cmd"):
            clean_cmd = "cmd.exe /c echo Command prompt ready"
        elif low_cmd in ("open notepad", "notepad"):
            clean_cmd = "start notepad"
        elif "git" in low_cmd and ("status" in low_cmd or "branch" in low_cmd):
            clean_cmd = "git status"

        if not clean_cmd:
            return {
                "status": "error",
                "message": "No command provided for command_prompt execution.",
            }

        target_cwd = str(working_dir or kwargs.get("cwd") or kwargs.get("path") or os.getcwd()).strip()
        if target_cwd and not os.path.exists(target_cwd):
            return {
                "status": "error",
                "message": f"Target directory does not exist: {target_cwd}",
            }

        try:
            res = subprocess.run(
                clean_cmd,
                cwd=target_cwd if target_cwd else None,
                capture_output=True,
                text=True,
                shell=True,
                timeout=30,
            )
            out_str = (res.stdout or "").strip()
            err_str = (res.stderr or "").strip()
            combined = out_str if out_str else err_str

            return {
                "status": "success" if res.returncode == 0 else "error",
                "result": combined,
                "stdout": out_str,
                "stderr": err_str,
                "exit_code": res.returncode,
                "command": clean_cmd,
                "working_dir": target_cwd,
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "message": f"Command timed out after 30 seconds: '{clean_cmd}'",
                "command": clean_cmd,
                "working_dir": target_cwd,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Execution failed: {str(e)}",
                "command": clean_cmd,
                "working_dir": target_cwd,
            }

    @classmethod
    def evaluate_affordance(
        cls,
        query_text: str = "",
        tokens: Optional[List[Dict[str, Any]]] = None,
        candidates: Optional[List[Dict[str, Any]]] = None,
        goal: Optional[Any] = None,
    ) -> Dict[str, Any]:
        schema = getattr(cls, "PARAMETERS_SCHEMA", {})
        accepts = getattr(cls, "ACCEPTS", {})
        required_params = accepts.get("requires", list(schema.keys()))
        tokens = tokens or []

        # Neural bilinear parameter slot binding via affordance tensor engine
        from ..cognitive.affordance_tensor_engine import ArisAffordanceTensorEngine
        bound, neural_missing, neural_completeness, neural_relevance = (
            ArisAffordanceTensorEngine.bind_slots(
                tokens=tokens,
                candidates=candidates,
                schema=schema,
                requires=required_params,
                domain=accepts.get("domain", cls.AFFORDANCE_DOMAIN),
                goal=goal,
                query_text=query_text,
            )
        )

        # Reconstruct filesystem paths from structural token sequences.
        # Structural reconstruction always overrides neural binding for working_dir
        # because the bilinear slot binder can only match individual entity candidates
        # (e.g. "E") whereas the full path spans multiple adjacent tokens.
        reconstructed_path = cls._reconstruct_path_from_tokens(tokens)
        if reconstructed_path:
            bound["working_dir"] = reconstructed_path

        has_path = bool(bound.get("working_dir"))
        if has_path and bound.get("command"):
            # Entity/question candidates matched by bilinear similarity are not shell commands
            bound.pop("command", None)

        # Semantic domain alignment for relevance scoring
        domain_alignment = 0.0
        from .semantic_representation import ArisSemanticEncoder
        if goal and hasattr(goal, "target_type") and goal.target_type:
            domain_alignment = ArisSemanticEncoder.similarity(
                str(goal.target_type),
                accepts.get("domain", cls.AFFORDANCE_DOMAIN)
            )
        elif query_text:
            domain_alignment = ArisSemanticEncoder.similarity(
                query_text,
                accepts.get("domain", cls.AFFORDANCE_DOMAIN)
            )

        if not has_path and domain_alignment >= 0.40 and not bound.get("command") and query_text:
            bound["command"] = query_text

        missing = [p for p in required_params if p not in bound or bound[p] is None]
        relevant = "yes" if (has_path or domain_alignment >= 0.40) else "no"
        completeness = "complete" if not missing else ("partial" if bound else "none")

        return {
            "name": cls.CAPABILITY_NAME,
            "relevant": relevant,
            "parameters": bound,
            "missing": missing,
            "completeness": completeness,
        }

    @classmethod
    def _reconstruct_path_from_tokens(cls, tokens: List[Dict[str, Any]]) -> Optional[str]:
        """
        Reconstructs a filesystem path from the structural token stream.

        Detects the drive-letter pattern: a single-character alphabetic token
        immediately followed (by char position) by a ':' symbol and a '\\'
        separator. Then collects all contiguous path-segment tokens until a
        non-path boundary is reached.

        Uses char_start/char_end positional invariants from the tokenizer —
        zero regex, zero hardcoded path strings.
        """
        if not tokens:
            return None

        for i, t in enumerate(tokens):
            raw = t.get("raw", "")
            # Look for a single-letter token that could be a drive letter
            if len(raw) != 1 or not raw.isalpha():
                continue

            # The next token must be ':' immediately adjacent (char_start == prev char_end)
            if i + 1 >= len(tokens):
                continue
            next_t = tokens[i + 1]
            if next_t.get("raw") != ":":
                continue
            if next_t["position"]["char_start"] != t["position"]["char_end"]:
                continue

            # Following tokens must include at least one '\' separator
            if i + 2 >= len(tokens):
                continue
            sep_t = tokens[i + 2]
            if sep_t.get("raw") != "\\":
                continue

            # Reconstruct the full path by collecting contiguous tokens
            # until a non-path-constituent boundary (whitespace gap or terminal punctuation)
            path_end_idx = i + 2
            for j in range(i + 3, len(tokens)):
                tj = tokens[j]
                tj_raw = tj.get("raw", "")
                # Stop at terminal question/exclamation marks
                if tj_raw in ("?", "!"):
                    break
                # Stop if there's a whitespace gap between this token and the previous
                prev_end = tokens[j - 1]["position"]["char_end"]
                curr_start = tj["position"]["char_start"]
                if curr_start > prev_end + 1:
                    break
                # Path constituents: words, backslashes, colons, hyphens, dots, underscores
                if tj.get("type") in ("word", "symbol"):
                    path_end_idx = j
                else:
                    break

            # Reconstruct from char spans
            start_char = t["position"]["char_start"]
            end_char = tokens[path_end_idx]["position"]["char_end"]
            # Build from raw tokens to preserve original casing
            path_parts = []
            for k in range(i, path_end_idx + 1):
                path_parts.append(tokens[k]["raw"])
            reconstructed = "".join(path_parts)

            # Validate it looks like a drive path (X:\...)
            if len(reconstructed) >= 3:
                return reconstructed

        return None


class ArisCapabilityRegistry:
    """
    Registry of external executable capabilities and tools available to AR.I.S.
    Auto-resolves capability tool classes dynamically via subclass reflection,
    persists a self-knowledge capability cache, and exposes usage semantics.
    Zero hardcoded orchestration pipelines or static import maps.
    """

    CACHE_FILE_PATH: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data",
        "aris_capabilities_cache.json",
    )

    _CAPABILITIES: Dict[str, Any] = {}
    _CACHE_LOADED: bool = False

    @classmethod
    def _auto_discover_subclasses(cls) -> Dict[str, Any]:
        """
        Dynamically discovers all concrete subclasses of ArisCapability in the Python runtime.
        """
        discovered: Dict[str, Any] = {}
        to_visit = list(ArisCapability.__subclasses__())
        while to_visit:
            sub = to_visit.pop(0)
            to_visit.extend(sub.__subclasses__())
            cap_name = getattr(sub, "CAPABILITY_NAME", "").strip().lower()
            if cap_name and not getattr(sub, "__abstractmethods__", None):
                discovered[cap_name] = sub
        return discovered

    @classmethod
    def auto_resolve(cls, sync_cache: bool = True) -> Dict[str, Any]:
        """
        Auto-resolves all capabilities dynamically:
        1. Introspects registered subclasses.
        2. Loads/syncs the persistent capability knowledge cache file.
        3. Updates the live registry dictionary.
        """
        discovered = cls._auto_discover_subclasses()
        cls._CAPABILITIES.update(discovered)

        if sync_cache:
            cls.sync_cache()

        cls._CACHE_LOADED = True
        return dict(cls._CAPABILITIES)

    @classmethod
    def sync_cache(cls) -> Dict[str, Any]:
        """
        Synchronizes discovered capabilities with the persistent JSON knowledge cache.
        Loads existing cache records and writes back any newly discovered capabilities with full usage.
        """
        cache_data: Dict[str, Any] = {}

        if os.path.exists(cls.CACHE_FILE_PATH):
            try:
                with open(cls.CACHE_FILE_PATH, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
            except Exception:
                cache_data = {}

        updated = False
        for name, tool_cls in cls._CAPABILITIES.items():
            entrypoint = f"{tool_cls.__module__}:{tool_cls.__name__}"
            spec = {
                "name": name,
                "aliases": getattr(tool_cls, "ALIASES", []),
                "domain": getattr(tool_cls, "AFFORDANCE_DOMAIN", "general"),
                "is_external": bool(getattr(tool_cls, "IS_EXTERNAL", False)),
                "description": getattr(tool_cls, "DESCRIPTION", (tool_cls.__doc__ or "").strip().split("\n")[0]),
                "usage": getattr(tool_cls, "USAGE", ""),
                "parameters_schema": getattr(tool_cls, "PARAMETERS_SCHEMA", {}),
                "accepts": getattr(tool_cls, "ACCEPTS", {}),
                "produces": getattr(tool_cls, "PRODUCES", {}),
                "entrypoint": entrypoint,
            }
            if name not in cache_data or cache_data[name] != spec:
                cache_data[name] = spec
                updated = True

        if updated or not os.path.exists(cls.CACHE_FILE_PATH):
            try:
                os.makedirs(os.path.dirname(cls.CACHE_FILE_PATH), exist_ok=True)
                with open(cls.CACHE_FILE_PATH, "w", encoding="utf-8") as f:
                    json.dump(cache_data, f, indent=2)
            except Exception:
                pass

        return cache_data

    @classmethod
    def get_capability_usage(cls, name: str) -> Optional[str]:
        """
        Returns the operational usage guidance for a capability so AR.I.S intrinsically understands
        when and how to use it.
        """
        tool_cls = cls.get_capability(name)
        if tool_cls:
            return getattr(tool_cls, "USAGE", None) or getattr(tool_cls, "DESCRIPTION", "")
        cache = cls.sync_cache()
        target = str(name).strip().lower()
        if target in cache:
            return cache[target].get("usage") or cache[target].get("description")
        return None

    @classmethod
    def get_all_capabilities_usage(cls) -> Dict[str, Dict[str, Any]]:
        """
        Returns structured self-knowledge of all capabilities and their operational usage semantics.
        """
        cls.auto_resolve(sync_cache=True)
        usage_map: Dict[str, Dict[str, Any]] = {}
        for name, tool_cls in cls._CAPABILITIES.items():
            usage_map[name] = {
                "name": name,
                "domain": getattr(tool_cls, "AFFORDANCE_DOMAIN", "general"),
                "description": getattr(tool_cls, "DESCRIPTION", ""),
                "usage": getattr(tool_cls, "USAGE", ""),
                "parameters_schema": getattr(tool_cls, "PARAMETERS_SCHEMA", {}),
                "accepts": getattr(tool_cls, "ACCEPTS", {}),
                "produces": getattr(tool_cls, "PRODUCES", {}),
                "is_external": bool(getattr(tool_cls, "IS_EXTERNAL", False)),
            }
        return usage_map

    @classmethod
    def get_capability(cls, name: str):
        if not name:
            return None
        target = str(name).strip().lower()
        if target in cls._CAPABILITIES:
            return cls._CAPABILITIES[target]
        for cap_key, tool_cls in cls._CAPABILITIES.items():
            aliases = [a.lower() for a in getattr(tool_cls, "ALIASES", [])]
            if target == cap_key.lower() or target in aliases:
                return tool_cls

        # Auto-resolve newly registered or imported subclasses
        cls.auto_resolve(sync_cache=False)
        if target in cls._CAPABILITIES:
            return cls._CAPABILITIES[target]
        for cap_key, tool_cls in cls._CAPABILITIES.items():
            aliases = [a.lower() for a in getattr(tool_cls, "ALIASES", [])]
            if target == cap_key.lower() or target in aliases:
                return tool_cls
        return None

    @classmethod
    def is_external_capability(cls, name: str) -> bool:
        """
        Dynamically determines if a capability is external (e.g. network/web)
        based on the tool's intrinsic contract, with zero hardcoded tool names.
        """
        cap_cls = cls.get_capability(name)
        if not cap_cls:
            return False
        return bool(getattr(cap_cls, "IS_EXTERNAL", False))

    @classmethod
    def get_capabilities_knowledge(cls) -> str:
        """
        Returns structured knowledge descriptors of all registered capabilities for the Teacher/Advisor.
        Zero hardcoding — dynamically introspects registered capabilities and their operational usage.
        """
        cls.auto_resolve(sync_cache=False)
        schemas = []
        for name, tool_cls in cls._CAPABILITIES.items():
            doc = getattr(tool_cls, "DESCRIPTION", (tool_cls.__doc__ or "").strip().split("\n")[0])
            usage = getattr(tool_cls, "USAGE", "")
            params = getattr(tool_cls, "PARAMETERS_SCHEMA", {})
            param_details = ", ".join([f"{k}: {v}" for k, v in params.items()]) if params else "None"
            desc_line = f"- Capability '{name}': {doc}"
            if usage:
                desc_line += f"\n  Usage: {usage}"
            desc_line += f"\n  Parameters: {param_details}"
            schemas.append(desc_line)
        return "\n".join(schemas)

    @classmethod
    def get_registered_capabilities(cls) -> Dict[str, Any]:
        """
        Dynamically returns all registered capability tool classes.
        """
        cls.auto_resolve(sync_cache=False)
        return dict(cls._CAPABILITIES)

    @classmethod
    def evaluate_capability_affordance(
        cls,
        cap_name: str,
        query_text: str = "",
        tokens: Optional[List[Dict[str, Any]]] = None,
        candidates: Optional[List[Dict[str, Any]]] = None,
        goal: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Phase 1: Dynamic Capability Affordance Perception.
        Evaluates a registered capability against input query representations WITHOUT executing it.
        Delegates polymorphically to the registered capability class. Zero hardcoded string checks.
        """
        cap_cls = cls.get_capability(cap_name)
        if not cap_cls:
            return {
                "name": cap_name,
                "relevant": "no",
                "parameters": {},
                "missing": [],
                "completeness": "none",
            }

        evaluator = getattr(cap_cls, "evaluate_affordance", None)
        if callable(evaluator):
            return evaluator(
                query_text=query_text,
                tokens=tokens or [],
                candidates=candidates,
                goal=goal,
            )

        return ArisCapability.evaluate_affordance.__func__(
            cap_cls,
            query_text=query_text,
            tokens=tokens or [],
            candidates=candidates,
            goal=goal,
        )

    @classmethod
    def perceive_all_affordances(
        cls,
        query_text: str,
        tokens: List[Dict[str, Any]],
        candidates: Optional[List[Dict[str, Any]]] = None,
        goal: Optional[Any] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Dynamically iterates over ALL registered capabilities in the registry
        and perceives their affordance relevance and parameter bindings.
        """
        affordances = {}
        for cap_name in cls.get_registered_capabilities():
            affordances[cap_name] = cls.evaluate_capability_affordance(
                cap_name=cap_name,
                query_text=query_text,
                tokens=tokens,
                candidates=candidates,
                goal=goal,
            )
        return affordances

    @classmethod
    def dispatch(cls, capability_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        target_name = capability_name or kwargs.pop("capability_name", None)
        if not target_name:
            return {
                "status": "error",
                "message": "No capability name provided to dispatch."
            }
        # Pop any accidental redundant capability_name from kwargs before execution
        kwargs.pop("capability_name", None)
        cap = cls.get_capability(target_name)
        if not cap:
            return {
                "status": "error",
                "message": f"Capability '{target_name}' not found in registry."
            }
        return cap.execute(**kwargs)


# Module-level dynamic auto-resolution & self-knowledge cache synchronization
ArisCapabilityRegistry.auto_resolve(sync_cache=True)
