# Phase 0: Tokenizer-First Cognitive Foundation & Empirical Sequence Learning Substrate

Phase 0 delivers empirical evidence that a PyTorch sequence model can learn the provided token-to-target mappings from discrete token inputs.

## Foundational Principle & Invariant
Every piece of language entering AR.I.S must pass through the canonical tokenizer before touching the neural substrate:

$$\text{Language} \xrightarrow{\text{ArisCanonicalTokenizer.encode}} \text{Token IDs} \in \mathbb{N}^L \xrightarrow{\text{torch.tensor}} \text{Neural Substrate}$$

No cognitive component may process raw language, regex rules, keyword detectors, or character-level heuristics. Token IDs are discrete serialized coordinates in $\{0, \dots, |\mathcal{V}|-1\}$; they do not contain intrinsic meaning.

---

## 1. Current-State Audit Summary

| Component | Status in Legacy AR.I.S | Phase 0 Implementation |
| :--- | :--- | :--- |
| **Tokenizer** | Regex string chunker outputting string dicts (`ArisTokenizer`) | `ArisCanonicalTokenizer` (`api/linguistic/canonical_tokenizer.py`) |
| **Vocabulary** | Did not exist | `Vocabulary` with special tokens (`[PAD]=0`, `[UNK]=1`, `[BOS]=2`, `[EOS]=3`, `[MASK]=4`) |
| **Token IDs** | Did not exist (only sequential offset index) | Discrete integer IDs mapped directly from $\mathcal{V}$ |
| **Embedding Layer** | Did not exist (used MD5 n-gram hash projection in `ArisSemanticEncoder`) | Learnable `nn.Embedding(vocab_size, embed_dim)` |
| **Positional Embedding** | Metadata float `relative_pos` only | Discrete learnable `nn.Embedding(max_seq_len, embed_dim)` |
| **PyTorch Substrate** | Isolated MLPs on hand-crafted scalars or MD5 hashes | TransformerBlock: Multi-head self-attention with GELU feed-forward network, LayerNorm, and 4 attention heads (`ArisPhase0Model`) |
| **Serialization Boundary** | Did not exist (passed raw Python strings) | Exact roundtrip `encode(str) -> list[int]` and `decode(list[int]) -> str` |

---

## 2. Measurable Empirical Evidence (Phase 0 Experiment)

Experiment executed via `scripts/run_phase0_experiment.py` on PyTorch 2.14.0:

### 2.1 Vocabulary & Serialization
- **Vocabulary Size $|\mathcal{V}|$**: 25 symbols (5 special tokens, 16 entities $e_0 \dots e_{15}$, 4 relations $r_{\text{succ}}, r_{\text{pred}}, r_{\text{swap}}, r_{\text{jump}}$).
- **Sample Serialization**:
  - Raw Input: `"[BOS] e_5 r_jump [MASK] [EOS]"`
  - Token IDs: `[2, 10, 24, 4, 3]`
  - Sequence Length: $L = 5$
  - Decoded Back: `"[BOS] e_5 r_jump [MASK] [EOS]"` (Exact bijective roundtrip)

### 2.2 Tensor Shapes Across the Pipeline
- **Input Token IDs Tensor**: `(102, 5)` (Batch size = 102, Sequence length = 5)
- **Token Embeddings $E_{\text{tok}}$**: `(102, 5, 64)` ($D = 64$)
- **Positional Embeddings $E_{\text{pos}}$**: `(102, 5, 64)`
- **Hidden States**: `(102, 5, 64)` — Contextual token representations produced after self-attention and feed-forward transformation.
- **Output Prediction Logits**: `(102, 5, 25)`
- **Target Tensor**: `(102,)`
- **Total Learnable Parameters**: 69,184

### 2.3 Optimization & Learning Trajectory
- **Theoretical Random Loss**: $-\ln(1/25) = 3.2189$
- **Initial Measured Untrained Loss**: $3.2684$ (Accuracy = $0.0\%$)
- **Final Measured Trained Loss**: $0.0010$
- **Final Trained Accuracy**: $100.0\%$
  - *This accuracy is not sufficient to establish generalization because the reported evaluation must be explicitly confirmed as held-out from training at the structural-combination level.*
- **Loss Reduction**: $\Delta \mathcal{L} = 3.2674$
- **Parameter Update Proof (Frobenius Norm)**:
  - Token Embedding Delta: $\|\Delta E_{\text{tok}}\|_F = 4.2206 > 0$
  - Self-Attention Weights Delta: $\|\Delta W_{\text{attn}}\|_F = 4.8123 > 0$
  - Confirms non-mocked gradient backpropagation reaching the embedding and attention parameters.

### 2.4 Observed Prediction Outputs
| Input Sequence | Target Token | Predicted Token | Model Confidence | Status |
| :--- | :--- | :--- | :--- | :--- |
| `[BOS] e_5 r_jump [MASK] [EOS]` | `e_9` | `e_9` | $99.92\%$ | Observed in training |
| `[BOS] r_jump e_5 [MASK] [EOS]` | `e_9` | `e_9` | $99.92\%$ | Observed in training |
| `[BOS] e_6 r_pred [MASK] [EOS]` | `e_5` | `e_5` | $99.90\%$ | Observed in training |
| `[BOS] r_pred e_6 [MASK] [EOS]` | `e_5` | `e_5` | $99.90\%$ | Observed in training |
| `[BOS] e_1 r_jump [MASK] [EOS]` | `e_5` | `e_5` | $99.91\%$ | Observed in training |
| `[BOS] e_5 r_swap [MASK] [EOS]` | `e_10` | `e_10` | $99.90\%$ | Observed in training |

---

## 3. The Source of "Meaning": Experimental Dissection

| Factor | Nature | Role in Substrate | Empirical Finding in Phase 0 |
| :--- | :--- | :--- | :--- |
| **Token Identity** | Discrete integer ($i \in \mathcal{V}$) | Arbitrary categorical coordinate. | Contains zero semantics, zero ordering, and zero distance metrics. |
| **Token Position** | Coordinate ($p \in \{0, \dots, L-1\}$) | Positional coordinates in sequence. | Provides positional information that allows the model to distinguish locations within the serialized sequence. The experiment does not yet establish that the model has learned semantic argument roles from position alone. |
| **Context** | Sequence Interaction ($\text{Attn}(Q, K, V)$) | Cross-position information flow. | Multi-head self-attention provides a mechanism through which the representation at the prediction position can incorporate information from other sequence positions. |
| **Embedding** | Continuous lookup matrix ($E \in \mathbb{R}^{|\mathcal{V}| \times D}$) | Continuous substrate for gradient updates. | Initial embeddings are random Gaussian vectors ($\sim \mathcal{N}(0, 0.02)$) with zero relational knowledge. |
| **Learned Parameters** | Weights ($W_Q, W_K, W_V, W_O, W_{\text{ffn}}$) | Geometric operators mapping context to target. | Gradient descent physically updated parameters ($\|\Delta W\| > 0$) to align with the demonstrated prediction task. |
| **Training Objective** | Loss function ($\mathcal{L} = -\sum \log P$) | Directional supervisory signal. | Without the objective, parameters remain random noise. |

**Critical Scientific Distinction**:
The experiment establishes learnability of the supplied mapping, not yet the existence of an abstract relation concept independent of the token identities used during training.

**Scientific Conclusion**:
The experiment demonstrates that token IDs themselves contain no learned semantic geometry, while the trainable embedding and transformation parameters can be modified by gradient descent to encode information sufficient for the demonstrated token-to-target prediction task. Whether this learned structure represents reusable relational knowledge requires held-out compositional and permutation-based generalization experiments.

---

## 4. Scientific Boundary & Current Status Matrix

```text
Tokenizer serialization       [x] demonstrated
Token IDs                     [x] demonstrated
Embedding lookup              [x] demonstrated
Gradient reaches embedding    [x] demonstrated
Transformer learning          [x] demonstrated
Prediction                    [x] demonstrated
Training convergence          [x] demonstrated

Generalization                [ ] not yet demonstrated
Compositional reasoning       [ ] not yet demonstrated
Token-ID permutation transfer [ ] not yet demonstrated
Unseen relation composition   [ ] not yet demonstrated
Natural-language meaning      [ ] not yet demonstrated
```

---

## 5. Phase 0 Stop Condition

The Phase 0 tokenization and trainable sequence-learning substrate has been implemented and empirically verified for the demonstrated task. Generalization beyond the observed training mappings remains unverified and is explicitly deferred to the next phase.
