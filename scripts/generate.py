from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from property_to_selfies.data import load_scaler
from property_to_selfies.model import PropertyConditionedDecoder, generate
from property_to_selfies.ape_tokenizer import APETokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/runs/baseline/best.pt")
    parser.add_argument("--scaler", default="outputs/runs/baseline/property_scaler.json")
    parser.add_argument("--properties", required=True, help="JSON list with 15 property values.")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    scaler, property_columns = load_scaler(args.scaler)

    values = np.asarray(json.loads(args.properties), dtype=np.float32)
    if values.shape != (len(property_columns),):
        raise ValueError(f"Expected {len(property_columns)} properties: {property_columns}")
    if property_columns[-1] == "MolWt":
        values[-1] = round(float(values[-1]))

    scaled = scaler.transform(values.reshape(1, -1)).astype(np.float32)
    scaled = np.repeat(scaled, args.num_samples, axis=0)
    properties = torch.tensor(scaled, dtype=torch.float32)

    tokenizer = APETokenizer.from_pretrained(checkpoint["tokenizer_path"])
    model = PropertyConditionedDecoder(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])

    generated = generate(
        model,
        properties,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_length=checkpoint["model_config"]["max_length"],
        temperature=args.temperature,
        top_k=args.top_k,
    )
    for row in generated.tolist():
        print(tokenizer.decode(row, skip_special_tokens=True))


if __name__ == "__main__":
    main()
