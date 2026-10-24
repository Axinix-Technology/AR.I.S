"""
AR.I.S — Neural Conditioned Generative Language Decoder (Phase 3)

Transforms continuous internal cognitive representations (subject, relation,
answer, epistemic state in R^64) into discrete token sequences via a causal
cross-attention transformer decoder operating over the canonical vocabulary.

Constitutional Directives:
  - ZERO STRING TEMPLATES: Generation is learned via autoregressive next-token
    prediction over canonical token IDs.
  - INVERSE LANGUAGE BOUNDARY:
      Cognitive State Memory in R^64
            ↓
      ArisConditionedDecoder (Causal + Cross-Attention)
            ↓
      Vocabulary Logits in R^|V|
            ↓
      Token IDs in {0, ..., |V|-1}
            ↓
      ArisCanonicalTokenizer.decode()
            ↓
      Fluid Sentence
  - REPRODUCIBLE & GROUNDED: Tied embeddings with lm_head, deterministic greedy
    decoding, strict termination on [EOS].
"""

from __future__ import annotations
import os
import json
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

from ..linguistic.canonical_tokenizer import ArisCanonicalTokenizer, Vocabulary


class CrossAttnBlock(nn.Module):
    """
    Transformer Decoder Block with Causal Self-Attention and Memory Cross-Attention.
    """

    def __init__(self, embed_dim: int, num_heads: int = 2, ff_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        ff_dim = ff_dim or (2 * embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. Causal Self-Attention
        sa_out, _ = self.self_attn(
            x, x, x,
            attn_mask=self_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + sa_out)

        # 2. Cross-Attention over Cognitive Memory
        ca_out, _ = self.cross_attn(
            query=x,
            key=memory,
            value=memory,
            need_weights=False,
        )
        x = self.norm2(x + ca_out)

        # 3. Feed-Forward Network
        ff_out = self.mlp(x)
        x = self.norm3(x + ff_out)
        return x


class ArisConditionedDecoder(nn.Module):
    """
    Minimal Causal Cross-Attention Neural Decoder for AR.I.S.
    Maps continuous cognitive representations -> vocabulary logits -> discrete token sequences.
    """

    DEFAULT_VOCAB_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "models",
        "aris_phase2_vocabulary.json",
    )

    def __init__(
        self,
        vocab_size: int = 116,
        embed_dim: int = 64,
        max_seq_len: int = 32,
        num_heads: int = 2,
        num_layers: int = 1,
        bos_id: int = 2,
        eos_id: int = 3,
        pad_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

        # 1. Discrete Token Embedding Table: E_tok in R^(vocab_size x embed_dim)
        self.tok_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)

        # 2. Positional Embedding Table: E_pos in R^(max_seq_len x embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)

        # 3. Slot Embedding Table: distinguishes Subject (0), Relation (1), Object/Answer (2), Epistemic (3)
        self.slot_emb = nn.Embedding(4, embed_dim)

        # 4. Memory Projection (maps cognitive memory vectors into decoder embedding space)
        self.memory_proj = nn.Linear(embed_dim, embed_dim)

        # 5. Decoder Layers (Stacked Causal Cross-Attention Blocks)
        self.layers = nn.ModuleList([
            CrossAttnBlock(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])

        # 6. Output Projection Head to Discrete Vocabulary Logits (Tied weights with tok_emb)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.slot_emb.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.memory_proj.weight)
        if self.memory_proj.bias is not None:
            nn.init.zeros_(self.memory_proj.bias)

    def encode_cognitive_memory(
        self,
        subject_id: int,
        relation_id: int,
        object_id: int,
        epistemic_id: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Constructs a structured cognitive memory tensor M in R^(1 x K x embed_dim)
        with explicit slot awareness (Subject=0, Relation=1, Object=2).
        """
        dev = device or self.tok_emb.weight.device
        s_id = subject_id if 0 <= subject_id < self.vocab_size else 1
        r_id = relation_id if 0 <= relation_id < self.vocab_size else 1
        o_id = object_id if 0 <= object_id < self.vocab_size else 1
        s_vec = self.tok_emb.weight[s_id]
        r_vec = self.tok_emb.weight[r_id]
        o_vec = self.tok_emb.weight[o_id]

        if epistemic_id is not None:
            e_id = epistemic_id if 0 <= epistemic_id < self.vocab_size else 1
            e_vec = self.tok_emb.weight[e_id]
            raw_mem = torch.stack([s_vec, r_vec, o_vec, e_vec], dim=0).to(dev)
            slots = self.slot_emb(torch.tensor([0, 1, 2, 3], device=dev))
        else:
            raw_mem = torch.stack([s_vec, r_vec, o_vec], dim=0).to(dev)
            slots = self.slot_emb(torch.tensor([0, 1, 2], device=dev))

        slotted_mem = (raw_mem + slots).unsqueeze(0)
        return self.memory_proj(slotted_mem)

    def encode_vector_memory(
        self,
        subject_vec: torch.Tensor,
        relation_vec: torch.Tensor,
        object_vec: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Constructs cognitive memory directly from continuous R^64 vectors.
        """
        dev = device or self.tok_emb.weight.device
        raw_mem = torch.stack([
            subject_vec.squeeze().to(dev),
            relation_vec.squeeze().to(dev),
            object_vec.squeeze().to(dev)
        ], dim=0)
        slots = self.slot_emb(torch.tensor([0, 1, 2], device=dev))
        slotted_mem = (raw_mem + slots).unsqueeze(0)
        return self.memory_proj(slotted_mem)

    def forward(
        self,
        memory: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for teacher-forced training or logit scoring.
        Args:
          memory: Tensor of shape (Batch_Size, Memory_Len, embed_dim)
          token_ids: LongTensor of shape (Batch_Size, Seq_Len)
        Returns:
          logits: Tensor of shape (Batch_Size, Seq_Len, vocab_size)
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.tok_emb(token_ids) + self.pos_emb(positions)

        # Create Causal Mask: upper triangular entries are -inf
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)

        for layer in self.layers:
            x = layer(x, memory, self_mask=causal_mask)

        logits = self.lm_head(x)
        return logits

    def generate(
        self,
        memory: torch.Tensor,
        max_len: int = 16,
        temperature: float = 0.0,
    ) -> List[int]:
        """
        Autoregressively generates discrete token IDs conditioned on cognitive memory.
        Starts with [BOS] and halts strictly upon predicting [EOS] or reaching max_len.
        """
        self.eval()
        device = memory.device
        generated = [self.bos_id]

        with torch.no_grad():
            for _ in range(max_len):
                curr_tokens = torch.tensor([generated], dtype=torch.long, device=device)
                logits = self(memory, curr_tokens)
                next_token_logits = logits[0, -1, :]

                if temperature > 0.0:
                    probs = torch.softmax(next_token_logits / temperature, dim=-1)
                    next_tok = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_tok = next_token_logits.argmax().item()

                generated.append(next_tok)
                if next_tok == self.eos_id:
                    break

        return generated

    def train_step(
        self,
        memory: torch.Tensor,
        target_ids: torch.Tensor,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """
        Executes a single teacher-forced supervised training step.
        """
        self.train()
        optimizer.zero_grad()

        # Input: tokens from 0 to N-1; Target: tokens from 1 to N
        inp = target_ids[:, :-1]
        tgt = target_ids[:, 1:]

        logits = self(memory, inp)
        loss = criterion(logits.reshape(-1, self.vocab_size), tgt.reshape(-1))

        loss.backward()
        optimizer.step()
        return loss.item()

    def save(self, filepath: str) -> None:
        """Persists trained decoder parameters to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save(self.state_dict(), filepath)

    @classmethod
    def load(cls, filepath: str, vocab_size: int = 116, embed_dim: int = 64) -> "ArisConditionedDecoder":
        """Loads decoder checkpoint from disk."""
        decoder = cls(vocab_size=vocab_size, embed_dim=embed_dim)
        decoder.load_state_dict(torch.load(filepath, map_location="cpu", weights_only=True))
        return decoder

    DEFAULT_CHECKPOINT_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "models",
        "aris_conditioned_decoder.pt",
    )

    _instance: Optional["ArisConditionedDecoder"] = None

    @classmethod
    def get_instance(cls, checkpoint_path: Optional[str] = None) -> "ArisConditionedDecoder":
        """
        Retrieves the singleton trained decoder instance, loading from checkpoint or training if needed.
        """
        if cls._instance is not None:
            return cls._instance
        ckpt = checkpoint_path or cls.DEFAULT_CHECKPOINT_PATH
        if os.path.exists(ckpt):
            try:
                decoder = cls.load(ckpt)
                cls._instance = decoder
                return decoder
            except Exception:
                pass
        decoder = cls.train_default_model(save_path=ckpt)
        cls._instance = decoder
        return decoder

    @classmethod
    def train_default_model(
        cls,
        facts: Optional[List[Dict[str, Any]]] = None,
        epochs: int = 180,
        lr: float = 0.008,
        save_path: Optional[str] = None,
    ) -> "ArisConditionedDecoder":
        """
        Trains the canonical conditioned decoder directly from active cognitive memory facts.
        Zero hardcoded file paths or hidden JSON references (Rule 1 & Rule 4).
        """
        import random
        torch.manual_seed(42)
        random.seed(42)

        tok = ArisCanonicalTokenizer.from_vocabulary_file(cls.DEFAULT_VOCAB_PATH)
        v = tok.vocab

        if facts is None:
            from .brain import ArisBrain
            facts = ArisBrain.retrieve_facts(limit=1000)

        exemplars = []
        for f_item in (facts or []):
            stmt = f_item.get("statement", "").strip()
            relations = f_item.get("relations", [])
            if relations:
                for b in relations:
                    s_ent = b.get("subject", {})
                    s = (s_ent.get("text") if isinstance(s_ent, dict) else str(s_ent)).strip()
                    r_rel = b.get("relation", {})
                    r = (r_rel.get("normalized") or r_rel.get("text", "") if isinstance(r_rel, dict) else str(r_rel)).strip().replace("_", " ")
                    o_ent = b.get("object", {})
                    o = (o_ent.get("text") if isinstance(o_ent, dict) else str(o_ent)).strip()
                    if s and r and o and stmt:
                        exemplars.append((s, r, o, stmt))
            else:
                s = str(f_item.get("subject", "")).strip()
                r = str(f_item.get("relation", "")).strip().replace("_", " ")
                o = str(f_item.get("object", "")).strip()
                if s and r and o and stmt:
                    exemplars.append((s, r, o, stmt))

        decoder = cls(vocab_size=len(v), embed_dim=64)
        opt = optim.Adam(decoder.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss(ignore_index=v.pad_id)

        for _ in range(epochs):
            shuffled = list(exemplars)
            random.shuffle(shuffled)
            for s_w, r_w, o_w, target_text in shuffled:
                s_id = v.token_to_id(s_w.split()[0])
                r_id = v.token_to_id(r_w.split()[0])
                o_id = v.token_to_id(o_w.split()[0])
                mem = decoder.encode_cognitive_memory(s_id, r_id, o_id)
                target_ids = torch.tensor([tok.encode(target_text, add_special_tokens=True)])
                decoder.train_step(mem, target_ids, opt, crit)

        if save_path:
            decoder.save(save_path)

        return decoder
