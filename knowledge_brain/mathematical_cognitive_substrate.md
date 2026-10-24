# Mathematical Cognitive Substrate & Algebraic Root-Finding

## 1. Overview & Core Directives
The Mathematical Cognitive Substrate in AR.I.S (`ArisMathTool`) is an internal, deterministic computation engine decoupled from domain natural language. It serves as a foundational cognitive primitive for:
1. **Deterministic AST Arithmetic Evaluation**: Safe syntax-tree traversal across addition, subtraction, multiplication, division, and exponentiation.
2. **Universal Affine Finite-Difference Root-Finding**: Exact solving of single-variable linear algebraic equations without case-by-case procedural branching.
3. **Multi-Layer Positional & Geometric Distance**: Scalar and $N$-dimensional Euclidean metric computations between tokens, candidates, relational graph nodes, and embedding tensors.

Strictly complies with:
* **Rule 1 (Zero Hard Patches)**: Contains zero English keyword checks (`"plus"`, `"minus"`, `"age"`, `"birth"`), zero ad-hoc equation branches, and zero regex workarounds.
* **Rule 4 (Generic Cognitive Intelligence)**: Treats mathematical operations as a generic cognitive substrate invoked through structured mathematical contracts.

---

## 2. Universal Affine Finite-Difference Root Theorem
Rather than writing procedural symbol manipulation algorithms or regex string replacements for linear algebra, AR.I.S applies the **Universal Affine Finite-Difference Theorem**:

Given any linear equation $\text{LHS}(v) = \text{RHS}(v)$, define the difference function:
$$g(v) = \text{LHS}(v) - \text{RHS}(v) = 0$$

Because $g(v)$ is linear ($g(v) = m \cdot v + c$), we determine its slope and intercept in two function evaluations:
$$\text{Intercept } c = g(0)$$
$$\text{Slope } m = g(1) - g(0)$$

The exact algebraic root is computed deterministically:
$$v = -\frac{c}{m} = -\frac{g(0)}{g(1) - g(0)}$$

### Universal Invariance Proof
* $1 + x = 3 \implies g(0) = 1 - 3 = -2,\; g(1) = 2 - 3 = -1,\; m = 1,\; x = -(-2)/1 = 2$.
* $100 - n = 37 \implies g(0) = 100 - 37 = 63,\; g(1) = 99 - 37 = 62,\; m = -1,\; n = -(63)/(-1) = 63$.
* $3 \cdot y + 15 = 45 \implies g(0) = -30,\; g(1) = -27,\; m = 3,\; y = -(-30)/3 = 10$.
* $z / 4 - 5 = 5 \implies g(0) = -10,\; g(1) = -9.75,\; m = 0.25,\; z = -(-10)/0.25 = 40$.

Variable name agnostic: works equally for $x$, $y$, $n$, $\alpha$, or any custom identifier without regex tailoring.

---

## 3. Cognitive Pipeline Integration
1. **Sensory Ingestion & Goal Formation** (`ArisGoalFormer`):
   Perceives mathematical structures ($LHS = RHS$ with a variable) and tags `target_type = "algebraic_equation"`.
2. **Epistemic Evaluation** (`ArisEpistemicEvaluator`):
   Identifies self-contained mathematical problems as `DERIVABLE` without requiring external database lookup.
3. **Capability Dispatch** (`ArisCapabilityRegistry`):
   Dispatches structured mathematical contracts (`evaluate`, `solve_linear`, `distance`) to `ArisMathTool`.
4. **Answer Synthesis & Realization** (`ArisResponseLearner`):
   Produces a natural language statement grounded in deterministic mathematical provenance.

---

## 4. Multi-Layer Coordinate Distance Evolution
The distance primitive $D(u, v)$ operates seamlessly across all cognitive layers:
* **Layer 1 & 2 (Lexical & Concept Spans)**: 1D character distance $|c_1 - c_2|$ and token offset $|t_1 - t_2|$.
* **Layer 3 & 4 (Relational Graph Structure)**: 2D/3D topological coordinates and shortest-path graph hops.
* **Layer 5 (Neural Tensor Embeddings)**: $N$-dimensional vector distance $\|u - v\|_2$.
