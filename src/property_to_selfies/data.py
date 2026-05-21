from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PropertyData:
    selfies: list[str]
    properties: np.ndarray
    property_columns: list[str]


def load_property_data(
    csv_path: str | Path,
    selfies_column: str = "SELFIES",
    property_columns: list[str] | None = None,
    molwt_column: str = "MolWt",
) -> PropertyData:
    df = pd.read_csv(Path(csv_path).expanduser())
    if selfies_column not in df.columns:
        raise ValueError(f"SELFIES column {selfies_column!r} not found.")

    selected = property_columns or list(df.columns[-15:])
    missing = [col for col in selected if col not in df.columns]
    if missing:
        raise ValueError(f"Property columns not found: {missing}")

    work = df[[selfies_column, *selected]].dropna().copy()
    if molwt_column in selected:
        work[molwt_column] = work[molwt_column].round().astype(int)

    selfies = work[selfies_column].astype(str).tolist()
    properties = work[selected].astype(float).to_numpy(dtype=np.float32)
    return PropertyData(selfies=selfies, properties=properties, property_columns=selected)


def fit_property_scaler(properties: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(properties)
    return scaler


def save_scaler(scaler: StandardScaler, property_columns: list[str], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "property_columns": property_columns,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_scaler(path: str | Path) -> tuple[StandardScaler, list[str]]:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        payload = json.load(f)
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(payload["mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(payload["scale"], dtype=np.float64)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)
    return scaler, list(payload["property_columns"])


def split_indices(n_items: int, train_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n_items)
    train_idx, val_idx = train_test_split(
        indices,
        train_size=train_fraction,
        random_state=seed,
        shuffle=True,
    )
    return train_idx, val_idx


class ConditionalSELFIESDataset(Dataset):
    def __init__(
        self,
        selfies: list[str],
        properties: np.ndarray,
        tokenizer,
        max_length: int,
    ) -> None:
        self.selfies = selfies
        self.properties = properties.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.selfies)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.selfies[idx],
            add_special_tokens=True,
            padding=True,
            max_length=self.max_length,
        )
        input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.bool)
        labels = input_ids.clone()
        labels[~attention_mask] = -100

        return {
            "properties": torch.tensor(self.properties[idx], dtype=torch.float32),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
