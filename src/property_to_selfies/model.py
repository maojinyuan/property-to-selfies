from __future__ import annotations

import math

import torch
from torch import nn


class PropertyConditionedDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_properties: int,
        pad_token_id: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_length: int = 160,
        num_property_tokens: int = 4,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_properties = num_properties
        self.pad_token_id = pad_token_id
        self.d_model = d_model
        self.max_length = max_length
        self.num_property_tokens = num_property_tokens

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_length + num_property_tokens, d_model)
        self.property_encoder = nn.Sequential(
            nn.Linear(num_properties, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_property_tokens * d_model),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def _causal_mask(self, total_length: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((total_length, total_length), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        properties: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        prop_emb = self.property_encoder(properties)
        prop_emb = prop_emb.view(batch_size, self.num_property_tokens, self.d_model)

        token_emb = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        hidden = torch.cat([prop_emb, token_emb], dim=1)

        positions = torch.arange(hidden.size(1), device=input_ids.device).unsqueeze(0)
        hidden = hidden + self.position_embedding(positions)

        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id)
        prop_mask = torch.ones(
            batch_size,
            self.num_property_tokens,
            dtype=torch.bool,
            device=input_ids.device,
        )
        key_padding_mask = ~torch.cat([prop_mask, attention_mask.bool()], dim=1)

        hidden = self.transformer(
            hidden,
            mask=self._causal_mask(hidden.size(1), input_ids.device),
            src_key_padding_mask=key_padding_mask,
        )
        token_hidden = self.norm(hidden[:, self.num_property_tokens :, :])
        logits = self.lm_head(token_hidden)

        output = {"logits": logits}
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss
        return output


@torch.no_grad()
def generate(
    model: PropertyConditionedDecoder,
    properties: torch.Tensor,
    bos_token_id: int,
    eos_token_id: int,
    max_length: int,
    temperature: float = 1.0,
    top_k: int = 50,
) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device
    properties = properties.to(device)
    input_ids = torch.full(
        (properties.size(0), 1),
        bos_token_id,
        dtype=torch.long,
        device=device,
    )

    for _ in range(max_length - 1):
        attention_mask = input_ids.ne(model.pad_token_id)
        logits = model(properties, input_ids, attention_mask)["logits"][:, -1, :]
        logits = logits / max(temperature, 1e-6)
        if top_k > 0:
            values, indices = torch.topk(logits, k=min(top_k, logits.size(-1)))
            probs = torch.softmax(values, dim=-1)
            next_ids = indices.gather(-1, torch.multinomial(probs, num_samples=1))
        else:
            probs = torch.softmax(logits, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_ids], dim=1)
        if torch.all(next_ids.squeeze(1).eq(eos_token_id)):
            break

    return input_ids
