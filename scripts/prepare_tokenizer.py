from __future__ import annotations

import argparse
import json
from pathlib import Path

from property_to_selfies.config import load_config
from property_to_selfies.data import load_property_data
from property_to_selfies.tokenizer_utils import diagnose_tokenizer, load_hf_or_local_tokenizer
from property_to_selfies.ape_tokenizer import APETokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--source", choices=["hf", "train"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    tok_cfg = cfg["tokenizer"]
    source = args.source or tok_cfg["source"]

    print(f"loading data from {cfg['csv_path']}", flush=True)
    data = load_property_data(
        cfg["csv_path"],
        selfies_column=cfg["selfies_column"],
        property_columns=cfg["property_columns"],
        molwt_column=cfg["molwt_column"],
    )
    print(
        f"loaded {len(data.selfies)} SELFIES; found {len(data.property_columns)} property columns for config validation",
        flush=True,
    )

    if source == "hf":
        print(f"loading Hugging Face tokenizer {tok_cfg['hf_repo_id']}", flush=True)
        tokenizer = load_hf_or_local_tokenizer(
            tok_cfg["hf_repo_id"],
            tok_cfg["hf_filename"],
            tok_cfg["local_path"],
            tok_cfg.get("hf_local_path"),
        )
        output_path = Path(tok_cfg["local_path"])
    else:
        print(
            "training APE tokenizer "
            f"(max_vocab_size={tok_cfg['max_vocab_size']}, "
            f"min_freq_for_merge={tok_cfg['min_freq_for_merge']})",
            flush=True,
        )
        tokenizer = APETokenizer()
        tokenizer.train(
            data.selfies,
            max_vocab_size=tok_cfg["max_vocab_size"],
            min_freq_for_merge=tok_cfg["min_freq_for_merge"],
            show_progress=True,
        )
        output_path = Path(tok_cfg["trained_path"])
        print(f"saving tokenizer to {output_path}", flush=True)
        tokenizer.save_vocabulary(output_path)

    print("running tokenizer diagnostics", flush=True)
    report = diagnose_tokenizer(tokenizer, data.selfies, max_length=cfg["max_length"])
    report_path = output_path.with_name("diagnostics.json")
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"tokenizer: {output_path}")
    print(f"diagnostics: {report_path}")


if __name__ == "__main__":
    main()
