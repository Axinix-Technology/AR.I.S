# Vision-Language & Multimodal Architecture

## 1. Overview & Purpose
A 20-year autonomous cognitive system must possess multimodal perception: the ability to understand visual environments, spatial relationships, object attributes, and image descriptions. In AR.I.S, the foundational visual scene and image understanding stream is powered by **LLaVA-Instruct** (`liuhaotian/LLaVA-Instruct-150K`).

---

## 2. Dataset Mechanics & Sourcing

- **Source**: `liuhaotian/LLaVA-Instruct-150K` (`detail_23k.json`)
- **Location on Disk**: `data/vision/detail_23k.json` (19.57 MB, 23,240 detailed visual scene examples)
- **Structure**:
  - `id`: Image identifier from the COCO / vision corpus.
  - `image`: Image filename reference (`<id>.jpg`).
  - `conversations`: Human visual prompt paired with detailed visual attribute descriptions (foreground, background, colors, textures, lighting, object locations).

---

## 3. Cognitive Integration & Neural Tokenization

The data is streamed directly into ChatML format via `_iter_vision()` in `EnterpriseArisDataset`:

```
<|im_start|>user
[Image Context: {image_id}]
{human_question}
<|im_end|>
<|im_start|>assistant
<|thought|>
Analyzing visual features, spatial relationships, and object attributes:
{detailed_scene_analysis}
<|im_end|>
```

### Key Substrate Benefits:
1. **Spatial & Relational Vocabulary**: Grounds words like `above`, `adjacent`, `foreground`, `background`, `center`, `perpendicular`.
2. **Multimodal Alignment Readiness**: Prepares AR.I.S's internal hidden representations to connect with vision encoder embeddings (e.g., SigLIP or CLIP projection layers) in subsequent milestones.
3. **Modular Cognitive Focus**: Activated independently via `--focus vision`.

