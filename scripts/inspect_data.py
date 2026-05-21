from __future__ import annotations

import argparse

from property_to_selfies.config import load_config
from property_to_selfies.data import load_property_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    return parser.parse_args()


def main() -> None:
    cfg = load_config(parse_args().config)
    data = load_property_data(
        cfg["csv_path"],
        selfies_column=cfg["selfies_column"],
        property_columns=cfg["property_columns"],
        molwt_column=cfg["molwt_column"],
    )
    print(f"samples: {len(data.selfies)}")
    print("property columns:")
    for column in data.property_columns:
        print(f"- {column}")
    print("first SELFIES:")
    print(data.selfies[0])
    print("first properties:")
    print(data.properties[0].tolist())


if __name__ == "__main__":
    main()
