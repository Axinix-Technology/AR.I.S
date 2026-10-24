"""
Universal Dependencies UPOS and Syntax Inspector.
Tests the live in-memory tensor engine using self-scaling sqrt(N) manifold density.
"""
from api.linguistic.tokenizer import ArisTokenizer

def main():
    query = "What are the other companies Elon Musk leads?"
    print(f"\nQuery: '{query}'\n" + "=" * 55)
    raw_tokens = ArisTokenizer.tokenize(query)
    from api.linguistic.pipeline import ArisLinguisticPipeline
    frame = ArisLinguisticPipeline.process(query)
    print(f"Speech Act: {frame.speech_act} | is_question: {frame.is_question}")
    print(f"Candidates: {[c['text'] for c in frame.candidates]}")
    tokens = frame.tokens
    for i, t in enumerate(tokens):
        raw = t.get("raw", "")
        upos = t.get("upos", "")
        conf = t.get("upos_confidence", 0.0)
        print(f"  {raw:<14} -> {upos:<8} (confidence: {conf})")
        if raw.lower() in ("companies", "owned"):
            # Show all tag scores for this token
            left_ctx = tokens[i - 1]["raw"] if i > 0 else ""
            right_ctx = tokens[i + 1]["raw"] if i + 1 < len(tokens) else ""
            from api.linguistic.ud_tensor_engine import ArisUniversalLinguisticEncoder
            import torch
            norm_vec = ArisUniversalLinguisticEncoder.encode_contextual_token(raw, left_ctx, right_ctx)
            sims = []
            for tag, mat in ArisUniversalLinguisticEncoder._upos_matrices.items():
                all_s = torch.matmul(mat, norm_vec)
                max_1 = float(torch.max(all_s).item())
                k_val = max(1, int(all_s.shape[0] ** 0.5))
                k_mean = float(torch.topk(all_s, k=k_val).values.mean().item())
                sims.append((tag, round(max_1, 4), round(k_mean, 4)))
            sims.sort(key=lambda x: -x[2])
            print(f"       >> BREAKDOWN FOR '{raw}' (sorted by sqrt(N) mean):")
            for t_name, m1, km in sims[:5]:
                print(f"          - {t_name:<6}: sqrt(N) mean = {km} | max (1-NN) = {m1}")
    print("=" * 55 + "\n")

if __name__ == "__main__":
    main()
