# Model Merging Protocol (Collaborative Distributed Training)

## 1. Overview & Mathematical Rationale
Training deep foundation models collaboratively across distributed teammates typically requires expensive multi-node GPU interconnects (Infiniband / NVLink). AR.I.S uses **Asynchronous Weight Merging (Model Soups & Task Vector Interpolation)**.

When Team Member A trains on PC 1 and Team Member B trains on PC 2 starting from the same base checkpoint $\theta_{\text{base}}$:
- Both trajectories explore the same low-loss optimization basin.
- Parameter averaging cancels out individual stochastic gradient noise and regularizes the network.
- Mathematically:
  $$\theta_{\text{merged}} = \alpha \theta_A + (1 - \alpha) \theta_B$$
  Or via Task Vector arithmetic:
  $$\Delta_A = \theta_A - \theta_{\text{base}}, \quad \Delta_B = \theta_B - \theta_{\text{base}}$$
  $$\theta_{\text{merged}} = \theta_{\text{base}} + \lambda (\Delta_A + \Delta_B)$$

## 2. Merging Methods
1. **Model Soup Linear Averaging (`average`)**:
   - Computes weighted mean across matching tensor parameter keys.
   - Ideal when both members train across balanced general corpora.
2. **Spherical Linear Interpolation (`slerp`)**:
   - Interpolates along the spherical geodesic between high-dimensional weight vectors, preserving parameter vector magnitudes and attention head geometry.
3. **Task-Vector Arithmetic (`task_vector`)**:
   - Isolates the fine-tuned directional shifts from the base checkpoint and combines them linearly.

## 3. Knowledge Graph Memory Merging
In addition to neural weights, teammates can merge their SQLite Knowledge Graphs (`knowledge_graph.sqlite3`):
- Merges entity nodes using `INSERT ... ON CONFLICT(id) DO UPDATE`.
- Preserves unique relational edges across both team members' conversational discoveries.

## 4. Usage Command
```bash
python run.py merge-models --model-a models/aris_pc1.pt --model-b models/aris_pc2.pt --output models/aris_lm_checkpoint.pt
```
Outputs a unified checkpoint combining the cumulative training steps and weights of both team members.

