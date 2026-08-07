#!/usr/bin/env python3
"""
Multilingual UD UPOS fine-tuning with locally downloaded CANINE-c.

Experiment design
-----------------
- Automatically discovers every treebank directory beside this script that
  contains dataset_dict.json.
- Mixes all available train splits into one training set.
- Mixes all available validation/dev/eval splits into one validation set.
- Evaluates every available test split separately by treebank.
- Uses the original UD token strings; no comturk/uroman conversion is applied.
- Gives the model no language ID and no treebank ID.
- Fine-tunes the full CANINE-c encoder by default.
- Uses mean pooling over the contextual character representations belonging
  to each gold UD token, followed by dropout and a linear UPOS classifier.
- Selects the best model by validation macro-F1.
- Keeps the best trainable parameters only in CPU memory.
- Writes no checkpoint, prediction, cache, metric, or log file.

Expected directory layout
-------------------------
turkic_ud_2_18/
├── all_canine_c_pos_train.py
├── canine-c/
│   ├── config.json
│   ├── model.safetensors
│   └── ...
├── az_tuecl/
│   ├── dataset_dict.json
│   ├── test/
│   └── ...
└── ...

Notes
-----
CANINE operates on Unicode code points. Each sentence is reconstructed by
joining its gold UD tokens with one ASCII space. The gold token boundaries are
used only to pool character-level representations into one vector per token.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import gc
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    CanineModel,
    CanineTokenizer,
    get_linear_schedule_with_warmup,
)


IGNORE_INDEX = -100
VALIDATION_SPLIT_NAMES = (
    "validation",
    "valid",
    "val",
    "dev",
    "development",
    "eval",
    "evaluation",
)
TRAIN_SPLIT_NAMES = ("train", "training")
TEST_SPLIT_NAMES = ("test", "testing")

# These are used only as a sanity check. The actual label-to-ID mapping is
# constructed from the training data, exactly as in the fastText experiment.
UNIVERSAL_UPOS = {
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
}


@dataclass(slots=True)
class SentenceChunk:
    tokens: list[str]
    labels: list[str]
    treebank: str


class PosDataset(torch.utils.data.Dataset):
    def __init__(self, examples: Sequence[SentenceChunk]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> SentenceChunk:
        return self.examples[index]


class CaninePosTagger(nn.Module):
    """
    CANINE-c encoder + mean character pooling + linear UPOS classifier.

    token_starts and token_ends use half-open indices into the CANINE sequence:
        [start, end)

    They already include the offset introduced by the leading special token.
    """

    def __init__(
        self,
        model_dir: Path,
        num_labels: int,
        classifier_dropout: float,
        gradient_checkpointing: bool,
    ) -> None:
        super().__init__()

        self.encoder = CanineModel.from_pretrained(
            model_dir,
            local_files_only=True,
            use_safetensors=True,
        )

        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Match the initialization convention used by Transformers models.
        initializer_range = float(self.encoder.config.initializer_range)
        nn.init.normal_(
            self.classifier.weight,
            mean=0.0,
            std=initializer_range,
        )
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_starts: torch.Tensor,
        token_ends: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        character_states = outputs.last_hidden_state

        # Prefix sums let us mean-pool every variable-length token span without
        # a Python loop over tokens.
        batch_size, _, hidden_size = character_states.shape
        zero = character_states.new_zeros((batch_size, 1, hidden_size))
        prefix = torch.cat(
            (zero, character_states.cumsum(dim=1)),
            dim=1,
        )

        start_index = token_starts.unsqueeze(-1).expand(
            -1,
            -1,
            hidden_size,
        )
        end_index = token_ends.unsqueeze(-1).expand(
            -1,
            -1,
            hidden_size,
        )

        token_sums = (
            prefix.gather(1, end_index)
            - prefix.gather(1, start_index)
        )
        token_lengths = (
            token_ends - token_starts
        ).clamp_min(1).unsqueeze(-1)

        token_states = token_sums / token_lengths
        token_states = self.dropout(token_states)
        return self.classifier(token_states)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=here,
        help="Directory containing all saved DatasetDict treebanks.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=here / "canine-c",
        help="Local google/canine-c directory.",
    )

    # Full fine-tuning defaults.
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--classifier-dropout", type=float, default=0.10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--min-delta", type=float, default=1e-4)

    # Character sequences are much longer than word/subword sequences.
    # These conservative defaults are intended for Apple Silicon MPS.
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=24)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=2048,
        help=(
            "Maximum CANINE sequence length including the two special tokens. "
            "Long UD sentences are split at gold token boundaries, with no "
            "tokens discarded."
        ),
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Reduce activation memory at the cost of slower training.",
    )

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cuda", "cpu"),
        default="auto",
    )

    # Optional speed-test controls. Zero means no limit.
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="For a speed smoke test only; 0 uses the complete training set.",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=0,
        help="For a speed smoke test only; 0 uses complete validation/test sets.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable.")
        return torch.device("mps")

    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable.")
        return torch.device("cuda")

    if name == "cpu":
        return torch.device("cpu")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def find_split(
    dataset_dict: DatasetDict,
    candidates: Sequence[str],
) -> str | None:
    normalized = {
        str(name).strip().lower(): name
        for name in dataset_dict.keys()
    }

    for candidate in candidates:
        if candidate in normalized:
            return str(normalized[candidate])

    return None


def extract_class_names(feature: object) -> list[str] | None:
    """
    Recursively find ClassLabel.names inside Sequence/List feature wrappers.
    """
    names = getattr(feature, "names", None)
    if names is not None:
        return [str(name) for name in names]

    for attribute in ("feature", "value_type"):
        child = getattr(feature, attribute, None)
        if child is not None and child is not feature:
            result = extract_class_names(child)
            if result is not None:
                return result

    return None


def decode_upos_values(
    raw_values: object,
    feature: object,
    treebank: str,
    split_name: str,
    row_index: int,
) -> list[str]:
    if not isinstance(raw_values, (list, tuple)):
        raise TypeError(
            f"{treebank}/{split_name}, row {row_index}: "
            "upos must be a list or tuple."
        )

    class_names = extract_class_names(feature)
    labels: list[str] = []

    for value in raw_values:
        if value is None:
            labels.append("_")
            continue

        if isinstance(value, (int, np.integer)):
            integer_value = int(value)

            if integer_value < 0:
                labels.append("_")
                continue

            if class_names is None:
                raise RuntimeError(
                    f"{treebank}/{split_name}, row {row_index}: "
                    "upos contains integer IDs, but its Dataset feature "
                    "does not expose ClassLabel names."
                )

            if integer_value >= len(class_names):
                raise ValueError(
                    f"{treebank}/{split_name}, row {row_index}: "
                    f"UPOS ID {integer_value} is outside "
                    f"0..{len(class_names) - 1}."
                )

            labels.append(class_names[integer_value])
            continue

        text = str(value).strip()
        labels.append(text if text else "_")

    return labels


def sentence_character_length(tokens: Sequence[str]) -> int:
    if not tokens:
        return 0

    return sum(len(token) for token in tokens) + len(tokens) - 1


def split_sentence_at_token_boundaries(
    tokens: Sequence[str],
    labels: Sequence[str],
    treebank: str,
    max_text_characters: int,
) -> list[SentenceChunk]:
    """
    Split overlong sentences without dropping any token or label.

    Chunks are joined with one ASCII space at model input time. Therefore, a
    chunk containing n tokens has:
        sum(len(token)) + (n - 1)
    raw characters.
    """
    chunks: list[SentenceChunk] = []
    current_tokens: list[str] = []
    current_labels: list[str] = []
    current_length = 0

    for token, label in zip(tokens, labels):
        if not token:
            raise ValueError(
                f"{treebank}: encountered an empty token, which cannot be "
                "aligned to a character span."
            )

        added_length = len(token)
        if current_tokens:
            added_length += 1

        if current_tokens and current_length + added_length > max_text_characters:
            chunks.append(
                SentenceChunk(
                    tokens=current_tokens,
                    labels=current_labels,
                    treebank=treebank,
                )
            )
            current_tokens = []
            current_labels = []
            current_length = 0
            added_length = len(token)

        if added_length > max_text_characters:
            raise ValueError(
                f"{treebank}: one token has {len(token):,} characters, "
                f"longer than the allowed text length "
                f"{max_text_characters:,}: {token[:100]!r}"
            )

        current_tokens.append(token)
        current_labels.append(label)
        current_length += added_length

    if current_tokens:
        chunks.append(
            SentenceChunk(
                tokens=current_tokens,
                labels=current_labels,
                treebank=treebank,
            )
        )

    return chunks


def read_split(
    dataset: Dataset,
    treebank: str,
    split_name: str,
    max_text_characters: int,
) -> tuple[list[SentenceChunk], int]:
    if "tokens" not in dataset.column_names:
        raise RuntimeError(
            f"{treebank}/{split_name} has no 'tokens' column. "
            f"Columns: {dataset.column_names}"
        )

    if "upos" not in dataset.column_names:
        raise RuntimeError(
            f"{treebank}/{split_name} has no 'upos' column. "
            f"Columns: {dataset.column_names}"
        )

    upos_feature = dataset.features["upos"]
    chunks: list[SentenceChunk] = []

    for row_index, record in enumerate(dataset):
        raw_tokens = record["tokens"]

        if not isinstance(raw_tokens, (list, tuple)):
            raise TypeError(
                f"{treebank}/{split_name}, row {row_index}: "
                "tokens must be a list or tuple."
            )

        tokens = [
            "" if token is None else str(token)
            for token in raw_tokens
        ]
        labels = decode_upos_values(
            record["upos"],
            upos_feature,
            treebank,
            split_name,
            row_index,
        )

        if len(tokens) != len(labels):
            raise ValueError(
                f"{treebank}/{split_name}, row {row_index}: "
                f"tokens={len(tokens)}, upos={len(labels)}"
            )

        if not tokens:
            continue

        if not any(label != "_" for label in labels):
            continue

        chunks.extend(
            split_sentence_at_token_boundaries(
                tokens,
                labels,
                treebank,
                max_text_characters,
            )
        )

    return chunks, len(dataset)


def discover_data(
    data_dir: Path,
    max_sequence_length: int,
) -> tuple[
    list[SentenceChunk],
    list[SentenceChunk],
    dict[str, list[SentenceChunk]],
    dict[str, int],
]:
    max_text_characters = max_sequence_length - 2

    treebank_paths = sorted(
        path
        for path in data_dir.iterdir()
        if path.is_dir()
        and (path / "dataset_dict.json").is_file()
    )

    if not treebank_paths:
        raise FileNotFoundError(
            f"No saved DatasetDict treebanks found in {data_dir}"
        )

    all_train: list[SentenceChunk] = []
    all_validation: list[SentenceChunk] = []
    tests: dict[str, list[SentenceChunk]] = {}
    test_sentence_counts: dict[str, int] = {}

    print(f"数据目录：{data_dir}")
    print(f"树库目录：{len(treebank_paths)}")
    print()
    print(
        f"{'树库':<24}"
        f"{'train':>12}"
        f"{'validation':>16}"
        f"{'test':>12}"
        f"{'chunks':>12}"
    )
    print("-" * 76)

    for treebank_path in treebank_paths:
        treebank = treebank_path.name
        dataset_dict = load_from_disk(
            treebank_path,
            keep_in_memory=False,
        )

        if not isinstance(dataset_dict, DatasetDict):
            raise TypeError(
                f"{treebank_path} did not load as a DatasetDict."
            )

        train_name = find_split(dataset_dict, TRAIN_SPLIT_NAMES)
        validation_name = find_split(
            dataset_dict,
            VALIDATION_SPLIT_NAMES,
        )
        test_name = find_split(dataset_dict, TEST_SPLIT_NAMES)

        train_chunks: list[SentenceChunk] = []
        validation_chunks: list[SentenceChunk] = []
        test_chunks: list[SentenceChunk] = []

        train_sentence_count = 0
        validation_sentence_count = 0
        test_sentence_count = 0

        if train_name is not None:
            train_chunks, train_sentence_count = read_split(
                dataset_dict[train_name],
                treebank,
                train_name,
                max_text_characters,
            )
            all_train.extend(train_chunks)

        if validation_name is not None:
            (
                validation_chunks,
                validation_sentence_count,
            ) = read_split(
                dataset_dict[validation_name],
                treebank,
                validation_name,
                max_text_characters,
            )
            all_validation.extend(validation_chunks)

        if test_name is not None:
            test_chunks, test_sentence_count = read_split(
                dataset_dict[test_name],
                treebank,
                test_name,
                max_text_characters,
            )

            if test_chunks:
                tests[treebank] = test_chunks
                test_sentence_counts[treebank] = test_sentence_count

        total_chunks = (
            len(train_chunks)
            + len(validation_chunks)
            + len(test_chunks)
        )

        print(
            f"{treebank:<24}"
            f"{train_sentence_count:>12,}"
            f"{validation_sentence_count:>16,}"
            f"{test_sentence_count:>12,}"
            f"{total_chunks:>12,}"
        )

        del dataset_dict
        gc.collect()

    print("-" * 76)
    print(
        f"{'混合总计':<24}"
        f"{sum(1 for _ in all_train):>12,}"
        f"{sum(1 for _ in all_validation):>16,}"
        f"{sum(test_sentence_counts.values()):>12,}"
        f"{len(all_train) + len(all_validation) + sum(len(x) for x in tests.values()):>12,}"
    )
    print()

    if not all_train:
        raise RuntimeError("No train split was found.")

    if not all_validation:
        raise RuntimeError(
            "No validation/dev/eval split was found."
        )

    if not tests:
        raise RuntimeError("No test split was found.")

    return (
        all_train,
        all_validation,
        tests,
        test_sentence_counts,
    )


def build_label_vocab(
    train: Sequence[SentenceChunk],
) -> tuple[dict[str, int], list[str], Counter[str]]:
    counts: Counter[str] = Counter()

    for sentence in train:
        counts.update(
            label
            for label in sentence.labels
            if label != "_"
        )

    labels = sorted(counts)

    if not labels:
        raise RuntimeError("No UPOS labels found in training data.")

    unexpected = set(labels) - UNIVERSAL_UPOS
    if unexpected:
        raise RuntimeError(
            f"Unexpected UPOS labels found: {sorted(unexpected)}"
        )

    return (
        {label: index for index, label in enumerate(labels)},
        labels,
        counts,
    )


def check_unseen_labels(
    examples: Iterable[SentenceChunk],
    label_to_id: dict[str, int],
    split_name: str,
) -> None:
    unseen = {
        label
        for sentence in examples
        for label in sentence.labels
        if label != "_" and label not in label_to_id
    }

    if unseen:
        raise RuntimeError(
            f"{split_name} contains labels unseen in train: "
            f"{sorted(unseen)}"
        )


def build_text_and_spans(
    tokens: Sequence[str],
) -> tuple[str, list[int], list[int]]:
    text_parts: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    cursor = 0

    for index, token in enumerate(tokens):
        if index > 0:
            text_parts.append(" ")
            cursor += 1

        starts.append(cursor)
        text_parts.append(token)
        cursor += len(token)
        ends.append(cursor)

    return "".join(text_parts), starts, ends


def make_collate(
    tokenizer: CanineTokenizer,
    label_to_id: dict[str, int],
):
    pad_token_id = int(tokenizer.pad_token_id)

    def collate(batch: Sequence[SentenceChunk]):
        encoded_inputs: list[list[int]] = []
        batch_starts: list[list[int]] = []
        batch_ends: list[list[int]] = []
        batch_labels: list[list[int]] = []

        maximum_sequence_length = 0
        maximum_token_count = 0

        for sentence in batch:
            text, starts, ends = build_text_and_spans(
                sentence.tokens
            )

            character_ids = [ord(character) for character in text]
            input_ids = tokenizer.build_inputs_with_special_tokens(
                character_ids
            )

            if len(input_ids) != len(text) + 2:
                raise RuntimeError(
                    "CANINE special-token construction did not produce "
                    "exactly one leading and one trailing token."
                )

            # The first text character begins after the leading special token.
            starts = [start + 1 for start in starts]
            ends = [end + 1 for end in ends]

            label_ids = [
                (
                    label_to_id[label]
                    if label != "_"
                    else IGNORE_INDEX
                )
                for label in sentence.labels
            ]

            encoded_inputs.append(input_ids)
            batch_starts.append(starts)
            batch_ends.append(ends)
            batch_labels.append(label_ids)

            maximum_sequence_length = max(
                maximum_sequence_length,
                len(input_ids),
            )
            maximum_token_count = max(
                maximum_token_count,
                len(label_ids),
            )

        input_ids_tensor = torch.full(
            (len(batch), maximum_sequence_length),
            pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (len(batch), maximum_sequence_length),
            dtype=torch.long,
        )
        token_starts = torch.zeros(
            (len(batch), maximum_token_count),
            dtype=torch.long,
        )
        token_ends = torch.ones(
            (len(batch), maximum_token_count),
            dtype=torch.long,
        )
        label_ids_tensor = torch.full(
            (len(batch), maximum_token_count),
            IGNORE_INDEX,
            dtype=torch.long,
        )

        character_count = 0

        for row, (
            input_ids,
            starts,
            ends,
            labels,
        ) in enumerate(
            zip(
                encoded_inputs,
                batch_starts,
                batch_ends,
                batch_labels,
            )
        ):
            sequence_length = len(input_ids)
            token_count = len(labels)

            input_ids_tensor[row, :sequence_length] = torch.tensor(
                input_ids,
                dtype=torch.long,
            )
            attention_mask[row, :sequence_length] = 1
            token_starts[row, :token_count] = torch.tensor(
                starts,
                dtype=torch.long,
            )
            token_ends[row, :token_count] = torch.tensor(
                ends,
                dtype=torch.long,
            )
            label_ids_tensor[row, :token_count] = torch.tensor(
                labels,
                dtype=torch.long,
            )

            character_count += sequence_length - 2

        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask,
            "token_starts": token_starts,
            "token_ends": token_ends,
            "labels": label_ids_tensor,
            "characters": character_count,
        }

    return collate


def make_loader(
    examples: Sequence[SentenceChunk],
    tokenizer: CanineTokenizer,
    label_to_id: dict[str, int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = None

    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        PosDataset(examples),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=make_collate(tokenizer, label_to_id),
        generator=generator,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )


def update_confusion(
    confusion: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_labels: int,
) -> None:
    mask = labels != IGNORE_INDEX

    if not mask.any():
        return

    gold = labels[mask].detach().cpu().long()
    predicted = predictions[mask].detach().cpu().long()

    flat = gold * num_labels + predicted
    counts = torch.bincount(
        flat,
        minlength=num_labels * num_labels,
    )
    confusion += counts.reshape(num_labels, num_labels)


def summarize_metrics(
    loss_sum: float,
    token_count: int,
    confusion: torch.Tensor,
    character_count: int,
) -> dict[str, object]:
    matrix = confusion.double()
    true_positive = matrix.diag()
    predicted_count = matrix.sum(0)
    gold_count = matrix.sum(1)

    precision = true_positive / predicted_count.clamp_min(1)
    recall = true_positive / gold_count.clamp_min(1)
    f1 = (
        2 * precision * recall
        / (precision + recall).clamp_min(1e-12)
    )
    present = gold_count > 0

    correct = int(true_positive.sum())

    return {
        "loss": (
            loss_sum / token_count
            if token_count
            else math.nan
        ),
        "accuracy": (
            correct / token_count
            if token_count
            else 0.0
        ),
        "macro_f1": (
            float(f1[present].mean())
            if present.any()
            else 0.0
        ),
        "tokens": token_count,
        "characters": character_count,
        "confusion": confusion,
    }


def limited_batch_count(
    loader: DataLoader,
    maximum_batches: int,
) -> int:
    if maximum_batches <= 0:
        return len(loader)

    return min(len(loader), maximum_batches)


def run_epoch(
    model: CaninePosTagger,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    num_labels: int,
    optimizer: AdamW | None = None,
    scheduler=None,
    gradient_accumulation_steps: int = 1,
    grad_clip: float = 1.0,
    maximum_batches: int = 0,
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)

    loss_sum = 0.0
    token_count = 0
    character_count = 0
    confusion = torch.zeros(
        (num_labels, num_labels),
        dtype=torch.long,
    )

    number_of_batches = limited_batch_count(
        loader,
        maximum_batches,
    )

    if training:
        optimizer.zero_grad(set_to_none=True)

    context = (
        torch.enable_grad()
        if training
        else torch.inference_mode()
    )

    with context:
        for batch_index, batch in enumerate(loader, start=1):
            if batch_index > number_of_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_starts = batch["token_starts"].to(device)
            token_ends = batch["token_ends"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_starts=token_starts,
                token_ends=token_ends,
            )

            loss = criterion(
                logits.reshape(-1, num_labels),
                labels.reshape(-1),
            )

            if training:
                (
                    loss / gradient_accumulation_steps
                ).backward()

                should_step = (
                    batch_index % gradient_accumulation_steps == 0
                    or batch_index == number_of_batches
                )

                if should_step:
                    nn.utils.clip_grad_norm_(
                        model.parameters(),
                        grad_clip,
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    if scheduler is not None:
                        scheduler.step()

            valid_tokens = int(
                (labels != IGNORE_INDEX).sum().item()
            )

            loss_sum += loss.detach().item() * valid_tokens
            token_count += valid_tokens
            character_count += int(batch["characters"])

            update_confusion(
                confusion,
                logits.argmax(-1),
                labels,
                num_labels,
            )

            if training and (
                batch_index % 100 == 0
                or batch_index == number_of_batches
            ):
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"    batch {batch_index:,}/{number_of_batches:,}  "
                    f"loss={loss.detach().item():.4f}  "
                    f"lr={current_lr:.2e}",
                    flush=True,
                )

    return summarize_metrics(
        loss_sum,
        token_count,
        confusion,
        character_count,
    )


def metric_text(metrics: dict[str, object]) -> str:
    return (
        f"loss={metrics['loss']:.4f}  "
        f"accuracy={metrics['accuracy'] * 100:.2f}%  "
        f"macro-F1={metrics['macro_f1'] * 100:.2f}%  "
        f"tokens={metrics['tokens']:,}  "
        f"chars={metrics['characters']:,}"
    )


def copy_trainable_state_to_cpu(
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def restore_trainable_state(
    model: nn.Module,
    copied_state: dict[str, torch.Tensor],
) -> None:
    named_parameters = dict(model.named_parameters())

    with torch.no_grad():
        for name, saved_parameter in copied_state.items():
            target = named_parameters[name]
            target.copy_(
                saved_parameter.to(
                    device=target.device,
                    dtype=target.dtype,
                )
            )


def build_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> AdamW:
    decay_parameters: list[nn.Parameter] = []
    no_decay_parameters: list[nn.Parameter] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        normalized_name = name.lower()
        use_no_decay = (
            normalized_name.endswith("bias")
            or "layernorm.weight" in normalized_name
            or "layer_norm.weight" in normalized_name
        )

        if use_no_decay:
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)

    return AdamW(
        [
            {
                "params": decay_parameters,
                "weight_decay": weight_decay,
            },
            {
                "params": no_decay_parameters,
                "weight_decay": 0.0,
            },
        ],
        lr=learning_rate,
        foreach=False,
    )


def main() -> None:
    args = parse_args()

    if args.max_sequence_length < 4:
        raise ValueError("--max-sequence-length must be at least 4.")

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "--gradient-accumulation-steps must be at least 1."
        )

    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in [0, 1).")

    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    data_dir = args.data_dir.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    device = choose_device(args.device)

    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Local CANINE-c directory not found: {model_dir}"
        )

    print("=" * 78)
    print("CANINE-c multilingual UD UPOS fine-tuning")
    print("=" * 78)
    print(f"设备：{device}")
    print(f"随机种子：{args.seed}")
    print(f"本地模型：{model_dir}")
    print("微调方式：全参数微调")
    print(
        "词级表示：gold UD token 所覆盖字符的最终层表示取均值"
    )
    print("语言/treebank 特征：不提供")
    print("checkpoint/日志/预测文件：不写入")
    print()

    (
        train,
        validation,
        tests,
        test_sentence_counts,
    ) = discover_data(
        data_dir,
        args.max_sequence_length,
    )

    label_to_id, id_to_label, label_counts = build_label_vocab(
        train
    )

    print(f"训练集 UPOS 标签（{len(id_to_label)}）：")
    print(", ".join(id_to_label))
    print("标签频数：")
    print(
        "  "
        + "  ".join(
            f"{label}={label_counts[label]:,}"
            for label in id_to_label
        )
    )
    print()

    check_unseen_labels(
        validation,
        label_to_id,
        "validation",
    )
    check_unseen_labels(
        (
            sentence
            for group in tests.values()
            for sentence in group
        ),
        label_to_id,
        "test",
    )

    tokenizer = CanineTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
    )

    # Verify the exact one-character-to-one-ID behavior relied on by the
    # gold-token span construction.
    alignment_probe = "AӘÜت"
    probe_ids = tokenizer.build_inputs_with_special_tokens(
        [ord(character) for character in alignment_probe]
    )
    if len(probe_ids) != len(alignment_probe) + 2:
        raise RuntimeError(
            "Unexpected CANINE tokenizer behavior: character alignment "
            "cannot be guaranteed."
        )

    train_loader = make_loader(
        train,
        tokenizer,
        label_to_id,
        args.batch_size,
        True,
        args.num_workers,
        args.seed,
    )
    validation_loader = make_loader(
        validation,
        tokenizer,
        label_to_id,
        args.eval_batch_size,
        False,
        args.num_workers,
        args.seed,
    )

    model = CaninePosTagger(
        model_dir=model_dir,
        num_labels=len(id_to_label),
        classifier_dropout=args.classifier_dropout,
        gradient_checkpointing=args.gradient_checkpointing,
    ).to(device)

    optimizer = build_optimizer(
        model,
        args.learning_rate,
        args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(
        ignore_index=IGNORE_INDEX,
    )

    train_batches_per_epoch = limited_batch_count(
        train_loader,
        args.max_train_batches,
    )
    optimizer_steps_per_epoch = math.ceil(
        train_batches_per_epoch
        / args.gradient_accumulation_steps
    )
    total_optimizer_steps = (
        optimizer_steps_per_epoch * args.epochs
    )
    warmup_steps = int(
        total_optimizer_steps * args.warmup_ratio
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("模型：CANINE-c → token character mean → dropout → Linear")
    print(f"总参数：{total_parameters:,}")
    print(f"可训练参数：{trainable_parameters:,}")
    print(f"训练 batch：{len(train_loader):,}")
    print(f"验证 batch：{len(validation_loader):,}")
    print(
        "梯度累积："
        f"{args.gradient_accumulation_steps}；"
        f"有效 batch≈{args.batch_size * args.gradient_accumulation_steps}"
    )
    print(
        f"每轮 optimizer steps：{optimizer_steps_per_epoch:,}"
    )
    print(
        f"总 optimizer steps：{total_optimizer_steps:,}；"
        f"warmup steps：{warmup_steps:,}"
    )

    if args.gradient_checkpointing:
        print("gradient checkpointing：开启")
    else:
        print("gradient checkpointing：关闭")

    if args.max_train_batches or args.max_eval_batches:
        print()
        print(
            "警告：当前启用了 batch 数量限制，只适合测速，"
            "不能作为正式实验结果。"
        )

    print()

    best_f1 = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    no_improvement = 0

    for epoch in range(1, args.epochs + 1):
        started = time.time()

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d}  "
            f"start_lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            num_labels=len(id_to_label),
            optimizer=optimizer,
            scheduler=scheduler,
            gradient_accumulation_steps=(
                args.gradient_accumulation_steps
            ),
            grad_clip=args.grad_clip,
            maximum_batches=args.max_train_batches,
        )

        validation_metrics = run_epoch(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
            num_labels=len(id_to_label),
            maximum_batches=args.max_eval_batches,
        )

        elapsed = time.time() - started

        print(
            f"  time={elapsed / 60:.2f} min  "
            f"end_lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        print(f"  train: {metric_text(train_metrics)}")
        print(f"  valid: {metric_text(validation_metrics)}")

        validation_f1 = float(
            validation_metrics["macro_f1"]
        )

        if validation_f1 > best_f1 + args.min_delta:
            best_f1 = validation_f1
            best_epoch = epoch
            no_improvement = 0

            del best_state
            gc.collect()

            best_state = copy_trainable_state_to_cpu(model)
            print(
                "  验证集提升；完整最佳参数仅保存在 CPU 内存中。"
            )
        else:
            no_improvement += 1
            print(
                f"  未提升：{no_improvement}/{args.patience}"
            )

        print()

        if no_improvement >= args.patience:
            print("触发 early stopping。")
            print()
            break

    if best_state is None:
        raise RuntimeError(
            "No best in-memory model was produced."
        )

    restore_trainable_state(model, best_state)
    del best_state
    gc.collect()

    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    print(
        f"使用第 {best_epoch} 轮的最佳内存参数；"
        f"validation macro-F1={best_f1 * 100:.2f}%"
    )
    print()

    print("各树库 test 结果")
    print("=" * 104)
    print(
        f"{'树库':<24}"
        f"{'句子':>12}"
        f"{'chunks':>12}"
        f"{'token':>12}"
        f"{'loss':>12}"
        f"{'accuracy':>14}"
        f"{'macro-F1':>14}"
    )
    print("-" * 104)

    combined_confusion = torch.zeros(
        (len(id_to_label), len(id_to_label)),
        dtype=torch.long,
    )
    combined_loss_sum = 0.0
    combined_tokens = 0
    combined_characters = 0

    for treebank in sorted(tests):
        loader = make_loader(
            tests[treebank],
            tokenizer,
            label_to_id,
            args.eval_batch_size,
            False,
            args.num_workers,
            args.seed,
        )

        metrics = run_epoch(
            model=model,
            loader=loader,
            criterion=criterion,
            device=device,
            num_labels=len(id_to_label),
            maximum_batches=args.max_eval_batches,
        )

        print(
            f"{treebank:<24}"
            f"{test_sentence_counts[treebank]:>12,}"
            f"{len(tests[treebank]):>12,}"
            f"{metrics['tokens']:>12,}"
            f"{metrics['loss']:>12.4f}"
            f"{metrics['accuracy'] * 100:>13.2f}%"
            f"{metrics['macro_f1'] * 100:>13.2f}%"
        )

        combined_confusion += metrics["confusion"]
        combined_loss_sum += (
            metrics["loss"] * metrics["tokens"]
        )
        combined_tokens += int(metrics["tokens"])
        combined_characters += int(metrics["characters"])

        del loader
        gc.collect()

    combined = summarize_metrics(
        combined_loss_sum,
        combined_tokens,
        combined_confusion,
        combined_characters,
    )

    print("-" * 104)
    print(
        f"{'全部 test 混合':<24}"
        f"{'-':>12}"
        f"{sum(len(x) for x in tests.values()):>12,}"
        f"{combined['tokens']:>12,}"
        f"{combined['loss']:>12.4f}"
        f"{combined['accuracy'] * 100:>13.2f}%"
        f"{combined['macro_f1'] * 100:>13.2f}%"
    )
    print("=" * 104)
    print()
    print(
        "完成：未写入 checkpoint、预测结果、缓存或日志文件。"
    )


if __name__ == "__main__":
    main()


































