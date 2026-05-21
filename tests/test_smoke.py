import numpy as np
import torch

from property_to_selfies.ape_tokenizer import APETokenizer
from property_to_selfies.data import ConditionalSELFIESDataset
from property_to_selfies.model import PropertyConditionedDecoder


def test_tokenizer_model_smoke():
    tokenizer = APETokenizer()
    tokenizer.train(["[C][C][O]", "[C][C][N]"], max_vocab_size=32, min_freq_for_merge=1)
    dataset = ConditionalSELFIESDataset(
        ["[C][C][O]"],
        np.asarray([[1.0] * 15], dtype=np.float32),
        tokenizer,
        max_length=12,
    )
    batch = dataset[0]
    model = PropertyConditionedDecoder(
        vocab_size=len(tokenizer),
        num_properties=15,
        pad_token_id=tokenizer.pad_token_id,
        d_model=32,
        n_heads=4,
        n_layers=1,
        dim_feedforward=64,
        max_length=12,
        num_property_tokens=2,
    )
    output = model(
        properties=batch["properties"].unsqueeze(0),
        input_ids=batch["input_ids"].unsqueeze(0),
        attention_mask=batch["attention_mask"].unsqueeze(0),
        labels=batch["labels"].unsqueeze(0),
    )
    assert torch.isfinite(output["loss"])
