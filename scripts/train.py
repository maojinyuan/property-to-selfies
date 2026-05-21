from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from property_to_selfies.config import load_config, save_config
from property_to_selfies.data import (
    ConditionalSELFIESDataset,
    fit_property_scaler,
    load_property_data,
    save_scaler,
    split_indices,
)
from property_to_selfies.model import PropertyConditionedDecoder
from property_to_selfies.tokenizer_utils import load_hf_or_local_tokenizer
from property_to_selfies.ape_tokenizer import APETokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tokenizer(cfg: dict) -> APETokenizer:
    tok_cfg = cfg["tokenizer"]
    if tok_cfg["source"] == "hf":
        return load_hf_or_local_tokenizer(
            tok_cfg["hf_repo_id"],
            tok_cfg["hf_filename"],
            tok_cfg["local_path"],
            tok_cfg.get("hf_local_path"),
        )
    return APETokenizer.from_pretrained(tok_cfg["trained_path"])


def run_epoch(model, loader, optimizer, device, train: bool) -> float:
    model.train(train)
    total_loss = 0.0
    total_items = 0
    iterator = tqdm(loader, leave=False)
    for batch in iterator:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            output = model(**batch)
            loss = output["loss"]
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        batch_size = batch["input_ids"].size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size
        iterator.set_postfix(loss=total_loss / max(total_items, 1))
    return total_loss / max(total_items, 1)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config.json")

    data = load_property_data(
        cfg["csv_path"],
        selfies_column=cfg["selfies_column"],
        property_columns=cfg["property_columns"],
        molwt_column=cfg["molwt_column"],
    )
    scaler = fit_property_scaler(data.properties)
    properties = scaler.transform(data.properties).astype(np.float32)
    save_scaler(scaler, data.property_columns, output_dir / "property_scaler.json")

    tokenizer = load_tokenizer(cfg)
    dataset = ConditionalSELFIESDataset(
        data.selfies,
        properties,
        tokenizer,
        max_length=cfg["max_length"],
    )
    train_idx, val_idx = split_indices(len(dataset), cfg["train_fraction"], cfg["seed"])
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg["model"]
    model = PropertyConditionedDecoder(
        vocab_size=len(tokenizer),
        num_properties=properties.shape[1],
        pad_token_id=tokenizer.pad_token_id,
        max_length=cfg["max_length"],
        **model_cfg,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    best_val = float("inf")
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False)
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(record)
        print(json.dumps(record))

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "vocab_size": len(tokenizer),
                "num_properties": properties.shape[1],
                "pad_token_id": tokenizer.pad_token_id,
                "max_length": cfg["max_length"],
                **model_cfg,
            },
            "tokenizer_path": cfg["tokenizer"]["local_path"]
            if cfg["tokenizer"]["source"] == "hf"
            else cfg["tokenizer"]["trained_path"],
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, output_dir / "best.pt")

    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
