import torch

# ── Config ────────────────────────────────────────────────────────────────────
checkpoint_path = "models/aris_ud_tensor_bank.pt"
markdown_path   = "model_structure.md"
# ─────────────────────────────────────────────────────────────────────────────


def write_section(f, data, heading: str, depth: int = 2):
    """Recursively write a dict/list/tensor section into the markdown file."""
    hashes = "#" * depth
    f.write(f"{hashes} {heading}\n\n")

    if isinstance(data, torch.Tensor):
        t = data
        dtype = str(t.dtype).replace("torch.", "")
        if t.numel() == 0:
            f.write(f"> **Tensor** — shape={list(t.shape)}  dtype={dtype}  (empty)\n\n")
        else:
            mn   = t.min().item()
            mx   = t.max().item()
            mean = t.float().mean().item()
            f.write(
                f"> **Tensor** — shape={list(t.shape)}  dtype={dtype}  "
                f"min={mn:.4f}  max={mx:.4f}  mean={mean:.4f}\n\n"
            )
        return

    if isinstance(data, dict):
        tensor_rows = {}
        nested      = {}
        scalar_rows = {}

        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                tensor_rows[k] = v
            elif isinstance(v, dict):
                nested[k] = v
            else:
                scalar_rows[k] = v

        # Scalar / list / string metadata
        if scalar_rows:
            f.write("| Key | Value |\n")
            f.write("| :--- | :--- |\n")
            for k, v in scalar_rows.items():
                val_str = str(v)
                if len(val_str) > 120:
                    val_str = val_str[:120] + " …"
                f.write(f"| `{k}` | {val_str} |\n")
            f.write("\n")

        # Tensors table
        if tensor_rows:
            f.write("| Key / Layer | Shape | dtype | min | max | mean |\n")
            f.write("| :--- | :--- | :--- | ---: | ---: | ---: |\n")
            for k, t in tensor_rows.items():
                dtype = str(t.dtype).replace("torch.", "")
                if t.numel() == 0:
                    f.write(f"| `{k}` | {list(t.shape)} | {dtype} | — | — | — |\n")
                else:
                    mn   = t.min().item()
                    mx   = t.max().item()
                    mean = t.float().mean().item()
                    f.write(
                        f"| `{k}` | {list(t.shape)} | {dtype} "
                        f"| {mn:.4f} | {mx:.4f} | {mean:.4f} |\n"
                    )
            f.write("\n")

        # Nested dicts → recursive sub-sections
        for k, v in nested.items():
            write_section(f, v, heading=f"{heading} › {k}", depth=depth + 1)
        return

    if isinstance(data, list):
        f.write(f"**List** ({len(data)} items)\n\n")
        for i, item in enumerate(data[:30]):
            f.write(f"- `[{i}]` {item}\n")
        if len(data) > 30:
            f.write(f"- … *({len(data) - 30} more items)*\n")
        f.write("\n")
        return

    # Fallback: plain value
    f.write(f"```\n{data}\n```\n\n")


# ── Count helpers ─────────────────────────────────────────────────────────────
total_params  = 0
total_tensors = 0


def count_tensors(obj):
    global total_params, total_tensors
    if isinstance(obj, torch.Tensor):
        total_tensors += 1
        total_params  += obj.numel()
    elif isinstance(obj, dict):
        for v in obj.values():
            count_tensors(v)
    elif isinstance(obj, list):
        for v in obj:
            count_tensors(v)


# ── Main ──────────────────────────────────────────────────────────────────────
print(f"Loading {checkpoint_path} …")
data = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)

count_tensors(data)

with open(markdown_path, "w", encoding="utf-8") as f:
    f.write("# AR.I.S — PyTorch Checkpoint Structure\n\n")
    f.write(f"- **Source:** `{checkpoint_path}`\n")
    f.write(f"- **Top-level keys:** {list(data.keys()) if isinstance(data, dict) else type(data).__name__}\n")
    f.write(f"- **Total tensors:** {total_tensors:,}\n")
    f.write(f"- **Total parameters (elements):** {total_params:,}\n\n")
    f.write("---\n\n")

    if isinstance(data, dict):
        for top_key, top_val in data.items():
            write_section(f, top_val, heading=top_key, depth=2)
            f.write("---\n\n")
    else:
        f.write(f"Loaded object type: `{type(data).__name__}`\n\n")
        f.write(f"```\n{data}\n```\n")

print(f"Done — Markdown written to: {markdown_path}")
print(f"  Tensors  : {total_tensors:,}")
print(f"  Elements : {total_params:,}")