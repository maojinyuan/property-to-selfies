# Property to SELFIES

This project trains a property-conditioned Transformer decoder that generates SELFIES from the last 15 molecular property columns in `PI1M_with_features.csv`.

## Data Contract

Default input file:

```text
/home/adminstrator/Desktop/PI1M_with_features.csv
```

Expected columns:

- `SELFIES` is the target sequence.
- The last 15 columns are used as conditioning properties.
- `MolWt` is rounded to the nearest integer before scaling, as requested.

Default property columns:

```text
NumHAcceptors
NumHDonors
NHOHCount
NOCount
NumAliphaticCarbocycles
NumAliphaticHeterocycles
NumAliphaticRings
NumAromaticCarbocycles
NumAromaticHeterocycles
NumAromaticRings
RingCount
NumRotatableBonds
NumHeteroatoms
HeavyAtomCount
MolWt
```

## Project Layout

```text
configs/                  JSON configs
scripts/                  CLI entry points
src/property_to_selfies/  tokenizer, data, model code
tests/                    lightweight smoke tests
outputs/                  generated locally, ignored by git
```

## Setup

```bash
cd ~/Desktop/property-to-selfies
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 1. Prepare Tokenizer

Use the author's `mikemayuare/SELFYAPE` tokenizer as the baseline:

```bash
python scripts/prepare_tokenizer.py --config configs/default.json --source hf
```

Or train an APE tokenizer from your PI1M SELFIES:

```bash
python scripts/prepare_tokenizer.py --config configs/default.json --source train
```

The script prints diagnostics:

- vocabulary size
- unknown token ratio
- average tokenized length
- 95th percentile length
- truncation ratio at `max_length`

If the Hugging Face tokenizer has a low `<unk>` ratio on your data, keep it as the first baseline.

## 2. Train Conditional Decoder

```bash
python scripts/train.py --config configs/default.json
```

The model conditions on the 15 property values by converting them into learned property tokens and prepending those embeddings before the SELFIES token sequence.

Training artifacts are saved under:

```text
outputs/runs/baseline/
```

Important files:

- `best.pt`
- `last.pt`
- `property_scaler.json`
- `config.json`
- `history.json`

## 3. Generate SELFIES

Pass 15 property values in the same order as `property_scaler.json`. The last value, `MolWt`, is rounded before scaling.

```bash
python scripts/generate.py \
  --checkpoint outputs/runs/baseline/best.pt \
  --scaler outputs/runs/baseline/property_scaler.json \
  --properties '[5,1,1,5,0,0,0,0,0,0,0,18,8,22,355.256]' \
  --num-samples 4
```

## Notes

- This is a conditional generation task: properties are the condition, SELFIES is the generated target.
- The first baseline uses `mikemayuare/SELFYAPE` tokenizer.
- A project-local APE tokenizer is included so the project does not depend on the original local `apetokenizer` checkout.
- SELFIES validity checking is intentionally not included yet because it requires adding the `selfies` package and deciding how strict generation filtering should be.
