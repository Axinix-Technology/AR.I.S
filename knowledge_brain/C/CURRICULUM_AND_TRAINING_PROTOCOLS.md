# Curriculum & Training Protocols

## 1. The 5-Stage Developmental Curriculum
AR.I.S develops language and reasoning capabilities progressively, mirroring developmental cognitive progression:

1. **Stage 1: Infancy (Steps 0 – 2,000)**:
   - Word boundaries, Byte-Level BPE tokenization, loss convergence. *(Completed)*
2. **Stage 2: Toddlerhood (Steps 2,000 – 5,000)**:
   - Grammatical syntax, dependency trees, Universal POS tags via Universal Dependencies.
3. **Stage 3: Early Childhood (Steps 5,000 – 15,000)**:
   - Semantic causal attribution: answering *Who, What, Where, When, Why, How* via QA-SRL.
4. **Stage 4: School Age (Steps 15,000 – 30,000)**:
   - Multi-hop reasoning, connecting disparate facts across paragraphs via HotpotQA `<|thought|>` chains.
5. **Stage 5: Maturity (Steps 30,000+)**:
   - Multi-turn conversation flow (UltraChat), lifelong knowledge graph consolidation, and autonomous problem solving.

## 2. Operational Training Protocols

### Protocol 1: Thermal & Gradient Accumulation Protocol (Cool & Unattended Training)
- **Problem**: Large batch sizes cause excessive CPU heat, and Windows default power management puts the PC to sleep after 15–30 minutes of inactivity, pausing training while away.
- **Protocol**:
  - Run small micro-batches (e.g. `batch_size = 2` or `4`).
  - Limit CPU execution threads (`torch.set_num_threads(max_threads)`, default 4).
  - Accumulate gradients across $N$ micro-steps (`accum_steps = 4`) before performing `optimizer.step()`.
  - Effective batch size = `batch_size * accum_steps` with cool, quiet CPU temperatures.
  - **Windows Keep-Awake Kernel Lock**: `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)` automatically prevents Windows from entering sleep mode while training is active, then releases the lock upon completion. Screen can sleep to save power, but CPU continues uninterrupted.

### Protocol 2: Gatekeeper Evaluation Protocol
- **Protocol**: After every milestone training block (e.g., +1,000 steps), automatically trigger `run.py eval --samples 15`.
- Ensure Token F1, Exact Match, and Graph RAG MRR metrics are monotonically advancing before promoting model checkpoints.

### Protocol 3: Cumulative Plasticity Protocol (Zero Knowledge Loss)
- **Protocol**: Model weights, optimizer states, and cumulative step counts must persist across training sessions.
- Resuming training stacks knowledge without resetting step counters.

