from __future__ import annotations

from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from property_to_selfies.ape_tokenizer import APETokenizer


def load_hf_or_local_tokenizer(
    hf_repo_id: str,
    hf_filename: str,
    local_path: str | Path,
    hf_local_path: str | Path | None = None,
) -> APETokenizer:
    target = Path(local_path)
    if target.exists():
        return APETokenizer.from_pretrained(target)

    if hf_local_path is not None:
        local_hf_file = Path(hf_local_path)
        if local_hf_file.exists():
            tokenizer = APETokenizer.from_pretrained(local_hf_file)
            target.parent.mkdir(parents=True, exist_ok=True)
            tokenizer.save_vocabulary(target)
            return tokenizer

    target.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(repo_id=hf_repo_id, filename=hf_filename)
    tokenizer = APETokenizer.from_pretrained(downloaded)
    tokenizer.save_vocabulary(target)
    return tokenizer


def diagnose_tokenizer(
    tokenizer: APETokenizer,
    selfies: list[str],
    max_length: int | None = None,
) -> dict[str, float]:
    lengths = []
    unk_count = 0
    total_count = 0
    truncated = 0

    for item in selfies:
        ids = tokenizer.encode(item, add_special_tokens=True)
        lengths.append(len(ids))
        unk_count += sum(token_id == tokenizer.unk_token_id for token_id in ids)
        total_count += len(ids)
        if max_length is not None and len(ids) > max_length:
            truncated += 1

    arr = np.asarray(lengths, dtype=np.float64)
    return {
        "num_samples": float(len(selfies)),
        "vocab_size": float(len(tokenizer)),
        "unk_ratio": float(unk_count / max(total_count, 1)),
        "avg_length": float(arr.mean()) if len(arr) else 0.0,
        "p95_length": float(np.percentile(arr, 95)) if len(arr) else 0.0,
        "max_length": float(arr.max()) if len(arr) else 0.0,
        "truncated_ratio": float(truncated / max(len(selfies), 1)),
    }
