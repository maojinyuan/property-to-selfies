from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


class APETokenizer:
    def __init__(
        self,
        pad_token: str = "<pad>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
        mask_token: str = "<mask>",
    ) -> None:
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.mask_token = mask_token
        self.special_tokens = {
            self.bos_token: 0,
            self.pad_token: 1,
            self.eos_token: 2,
            self.unk_token: 3,
            self.mask_token: 4,
        }
        self.vocabulary: dict[str, int] = dict(self.special_tokens)
        self.vocabulary_frequency: dict[str, int] = {}
        self._refresh_cache()

    @property
    def bos_token_id(self) -> int:
        return self.vocabulary[self.bos_token]

    @property
    def eos_token_id(self) -> int:
        return self.vocabulary[self.eos_token]

    @property
    def pad_token_id(self) -> int:
        return self.vocabulary[self.pad_token]

    @property
    def unk_token_id(self) -> int:
        return self.vocabulary[self.unk_token]

    def __len__(self) -> int:
        return len(self.vocabulary)

    def _refresh_cache(self) -> None:
        self.reverse_vocabulary = {idx: token for token, idx in self.vocabulary.items()}
        self.sorted_tokens = sorted(
            (token for token in self.vocabulary if token not in self.special_tokens),
            key=len,
            reverse=True,
        )

    def pre_tokenize(self, molecule: str) -> list[str]:
        tokens = re.findall(r"\[[^\]]+]", molecule)
        if tokens:
            return tokens
        pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        return re.findall(pattern, molecule)

    def train(
        self,
        corpus: list[str],
        max_vocab_size: int = 5000,
        min_freq_for_merge: int = 50,
    ) -> None:
        sequences = [self.pre_tokenize(text) for text in corpus]
        token_counts = Counter(token for seq in sequences for token in seq)

        while len(self.special_tokens) + len(token_counts) < max_vocab_size:
            pair_counts: Counter[tuple[str, str]] = Counter()
            for seq in sequences:
                pair_counts.update(zip(seq, seq[1:]))
            if not pair_counts:
                break

            (left, right), freq = pair_counts.most_common(1)[0]
            if freq < min_freq_for_merge:
                break

            merged = left + right
            new_sequences: list[list[str]] = []
            for seq in sequences:
                merged_seq: list[str] = []
                i = 0
                while i < len(seq):
                    if i + 1 < len(seq) and seq[i] == left and seq[i + 1] == right:
                        merged_seq.append(merged)
                        i += 2
                    else:
                        merged_seq.append(seq[i])
                        i += 1
                new_sequences.append(merged_seq)

            sequences = new_sequences
            token_counts[merged] = freq

        self.vocabulary_frequency = dict(token_counts)
        self.vocabulary = dict(self.special_tokens)
        for token in token_counts:
            if token not in self.vocabulary:
                self.vocabulary[token] = len(self.vocabulary)
        self._refresh_cache()

    def encode(
        self,
        text: str,
        padding: bool | str = False,
        max_length: int | None = None,
        add_special_tokens: bool = False,
    ) -> list[int]:
        token_ids: list[int] = []
        if add_special_tokens:
            token_ids.append(self.bos_token_id)

        i = 0
        while i < len(text):
            match = None
            for token in self.sorted_tokens:
                if text.startswith(token, i):
                    match = token
                    break
            if match is None:
                token_ids.append(self.unk_token_id)
                i += 1
            else:
                token_ids.append(self.vocabulary[match])
                i += len(match)

        if add_special_tokens:
            token_ids.append(self.eos_token_id)

        if max_length is not None:
            token_ids = token_ids[:max_length]

        if padding:
            if max_length is None:
                raise ValueError("max_length is required when padding is enabled.")
            token_ids = token_ids + [self.pad_token_id] * max(0, max_length - len(token_ids))

        return token_ids

    def __call__(
        self,
        text: str,
        padding: bool | str = False,
        max_length: int | None = None,
        add_special_tokens: bool = False,
    ) -> dict[str, list[int]]:
        input_ids = self.encode(
            text,
            padding=padding,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )
        attention_mask = [0 if token_id == self.pad_token_id else 1 for token_id in input_ids]
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        if isinstance(tokens, str):
            return self.vocabulary.get(tokens, self.unk_token_id)
        return [self.vocabulary.get(token, self.unk_token_id) for token in tokens]

    def convert_ids_to_tokens(self, token_ids: int | list[int]) -> str | list[str]:
        if isinstance(token_ids, int):
            return self.reverse_vocabulary.get(token_ids, self.unk_token)
        return [self.reverse_vocabulary.get(token_id, self.unk_token) for token_id in token_ids]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        tokens = self.convert_ids_to_tokens(token_ids)
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.special_tokens]
        return "".join(tokens)

    def save_vocabulary(self, file_path: str | Path) -> None:
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump(self.vocabulary, f, ensure_ascii=False, indent=2)

    def load_vocabulary(self, file_path: str | Path) -> None:
        with Path(file_path).expanduser().open("r", encoding="utf-8") as f:
            self.vocabulary = {str(k): int(v) for k, v in json.load(f).items()}
        self.special_tokens = {
            token: self.vocabulary[token]
            for token in [self.bos_token, self.pad_token, self.eos_token, self.unk_token, self.mask_token]
            if token in self.vocabulary
        }
        self._refresh_cache()

    def save_pretrained(self, save_directory: str | Path) -> None:
        target = Path(save_directory)
        target.mkdir(parents=True, exist_ok=True)
        self.save_vocabulary(target / "tokenizer.json")

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "APETokenizer":
        tokenizer = cls()
        candidate = Path(path).expanduser()
        vocab_path = candidate / "tokenizer.json" if candidate.is_dir() else candidate
        tokenizer.load_vocabulary(vocab_path)
        return tokenizer
