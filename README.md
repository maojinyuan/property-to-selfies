# Property to SELFIES

This project trains a property-conditioned Transformer decoder that generates SELFIES from molecular property values in `PI1M_with_features.csv`.

The basic workflow is:

```text
prepare tokenizer -> train conditional generator -> generate SELFIES from properties
```

## Data Contract

The training data is configured by `csv_path` in `configs/default.json`. Set it to the CSV file on your machine before preparing the tokenizer or training.

Example:

```json
"csv_path": "/path/to/PI1M_with_features.csv"
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
external/                 optional local external assets, such as downloaded tokenizer files
```

## Setup

```bash
git clone <your-repo-url>
cd property-to-selfies
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then edit `configs/default.json` and set `csv_path` to your local `PI1M_with_features.csv` path.

## 1. Prepare Tokenizer

Prepare a tokenizer before training or generation. The default config uses the author's `mikemayuare/SELFYAPE` tokenizer as the baseline:

```bash
python scripts/prepare_tokenizer.py --config configs/default.json --source hf
```

With `--source hf`, the script loads the tokenizer in this order:

1. Use `outputs/tokenizers/selfyape/tokenizer.json` if it already exists.
2. Otherwise use `external/SELFYAPE/tokenizer.json` if that local file exists.
3. Otherwise download `tokenizer.json` from Hugging Face repo `mikemayuare/SELFYAPE`.

The prepared tokenizer is saved to:

```text
outputs/tokenizers/selfyape/tokenizer.json
```

The diagnostics report is saved to:

```text
outputs/tokenizers/selfyape/diagnostics.json
```

`external/SELFYAPE/tokenizer.json` is only an optional local source file. `outputs/tokenizers/selfyape/tokenizer.json` is the project-local tokenizer path used by training when `tokenizer.source` is `hf`.

Or train an APE tokenizer from your PI1M SELFIES:

```bash
python scripts/prepare_tokenizer.py --config configs/default.json --source train
```

With `--source train`, the trained tokenizer is saved to:

```text
outputs/tokenizers/pi1m_ape/tokenizer.json
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

`scripts/train.py` trains the actual SELFIES generator. It is not a tokenizer training script.

The model learns this mapping:

```text
15 molecular property values -> SELFIES token sequence
```

The model conditions on the property values by converting them into learned property tokens and prepending those embeddings before the SELFIES token sequence. Training then uses standard next-token prediction loss.

The data source is configured in `configs/default.json`:

```json
"csv_path": "/path/to/PI1M_with_features.csv"
```

By default, `property_columns` is `null`, so the loader uses the last 15 columns in the CSV as property columns. `MolWt` is rounded to the nearest integer before scaling.

Training artifacts are saved under:

```text
outputs/runs/baseline/
```

Important files:

- `best.pt`: checkpoint with the lowest validation loss.
- `last.pt`: checkpoint from the final epoch.
- `property_scaler.json`: means/scales used to normalize property values.
- `config.json`: copy of the config used for the run.
- `history.json`: train and validation loss for each epoch.

Use `best.pt` for generation unless you specifically want the final epoch checkpoint.

## 3. Generate SELFIES

After training, generate SELFIES with `scripts/generate.py`.

Minimal example:

```bash
python scripts/generate.py \
  --checkpoint outputs/runs/baseline/best.pt \
  --scaler outputs/runs/baseline/property_scaler.json \
  --properties '[5,1,1,5,0,0,0,0,0,0,0,18,8,22,355.256]' \
  --num-samples 4
```

The default checkpoint and scaler paths are already:

```text
outputs/runs/baseline/best.pt
outputs/runs/baseline/property_scaler.json
```

So this shorter command is equivalent if you use the default run directory:

```bash
python scripts/generate.py \
  --properties '[5,1,1,5,0,0,0,0,0,0,0,18,8,22,355.256]' \
  --num-samples 4
```

The `--properties` value must be a JSON list with exactly 15 numbers. The order must match `property_scaler.json`.

Default order:

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

The last value, `MolWt`, is rounded before scaling. For example, `355.256` is treated as `355`.

Generation parameters:

- `--num-samples`: how many SELFIES strings to generate for the same property values.
- `--temperature`: sampling randomness. Lower is more conservative; higher is more random.
- `--top-k`: number of highest-probability next tokens allowed at each step.

Recommended starting point:

```bash
python scripts/generate.py \
  --properties '[5,1,1,5,0,0,0,0,0,0,0,18,8,22,355]' \
  --num-samples 20 \
  --temperature 0.8 \
  --top-k 30
```

Useful ranges:

```text
--num-samples 4 to 10       quick test
--num-samples 20 to 100     normal interactive generation
--num-samples 500+          larger candidate search

--temperature 0.7 to 0.9    more conservative
--temperature 1.0           default-style sampling
--temperature > 1.0         more diverse, often noisier

--top-k 1                   greedy decoding, usually identical outputs
--top-k 10 to 50            practical sampling range
--top-k 0                   sample from the full vocabulary
```

If all generated samples are identical, check whether `--top-k` is set to `1`. With `top-k=1`, there is only one candidate token at each step, so generation becomes deterministic for the same property values.

## Notes

- This is a conditional generation task: properties are the condition, SELFIES is the generated target.
- The first baseline uses `mikemayuare/SELFYAPE` tokenizer.
- A project-local APE tokenizer is included so the project does not depend on the original local `apetokenizer` checkout.
- SELFIES validity checking is intentionally not included yet because it requires adding the `selfies` package and deciding how strict generation filtering should be.
