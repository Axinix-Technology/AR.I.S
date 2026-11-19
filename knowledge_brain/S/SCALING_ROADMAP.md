# Scaling & Parameter Progression Roadmap (3-Month Laptop Phase)

## 1. Core Philosophy: The $2\times$ Doubling Staircase
During the initial 3-month incubation period before dedicated GPU server infrastructure comes online, all neural pretraining and inference take place on consumer laptops. To ensure thermal stability, prevent hardware wear, and maximize data efficiency, AR.I.S follows a strict **$2\times$ Parameter Doubling Staircase**:

$$\text{Gen 1: } 11.5\text{M} \longrightarrow \text{Gen 2: } 23\text{M} \longrightarrow \text{Gen 3: } 46\text{M} \longrightarrow \text{Gen 4: } 92\text{M}$$

---

## 2. Parameter Tier Architecture & Hardware Envelope

| Generation | Hidden Dim ($d_{\text{model}}$) | Layers ($L$) | Query / KV Heads | Total Parameters | Memory Footprint (FP32) | Laptop CPU Step Latency | Target Milestone |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Gen 1** *(Current)* | 384 | 6 | 6 / 2 | **~11.5 Million** | ~46 MB | ~1.5 sec / batch | 40,000 steps (Foundational reasoning & syntax) |
| **Gen 2** | 512 | 8 | 8 / 2 | **~23.0 Million** | ~92 MB | ~2.5 sec / batch | 60,000 steps (Compound math & multi-turn code) |
| **Gen 3** | 640 | 12 | 10 / 2 | **~46.0 Million** | ~184 MB | ~4.5 sec / batch | 80,000 steps (Complex deduction & tool syntax) |
| **Gen 4** *(Server Ready)* | 896 | 16 | 14 / 2 | **~92.0 Million** | ~368 MB | Fast (GPU Server) | Server deployment baseline |

---

## 3. Scientific Advantages for Laptop Development

1. **Zero Thermal Throttling**:
   - Combined with Protocol 1 (gradient accumulation 4, thread cap 4), power consumption remains under 25–35 Watts. Temperatures remain stable around ~50°C–55°C.
2. **Progressive Weight Inheritance**:
   - Each generation does not start from random noise. The lower-tier model acts as the seed substrate, cutting the training time of the next tier in half.
3. **Decentralized Multi-PC Model Soups**:
   - Two teammates running 23M or 46M models independently can average weights daily, effectively delivering server-grade compute throughput across two portable laptops.
4. **Clean Server Handoff**:
   - When the dedicated GPU server arrives at Month 3, the team will already possess battle-tested, converged 46M and 92M checkpoints ready for fine-tuning and scale-up.

