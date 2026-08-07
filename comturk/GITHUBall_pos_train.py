#!/usr/bin/env python3
"""
Multilingual UD UPOS training with frozen fastText vectors.

- Automatically scans every *.jsonl beside this script.
- Mixes every train split into one training set.
- Mixes every validation/dev/eval split into one validation set.
- Tests every JSONL file's test split separately.
- Uses no language/treebank feature during training or validation.
- Writes no checkpoint, prediction, or log file.
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import gc
import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import fasttext
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset


IGNORE_INDEX = -100
PAD_TOKEN = "<PAD>"

TRAIN_PREFIXES = ("train", "training", "trn")
TEST_PREFIXES = ("test", "testing", "tst")
VALID_PREFIXES = (
    "val", "valid", "validation",
    "dev", "develop", "development",
    "eva", "eval", "evaluate", "evaluation",
)


@dataclass(slots=True)
class Sentence:
    tokens: list[str]
    labels: list[str]
    source: str


@dataclass(slots=True)
class IndexedSentence:
    token_ids: list[int]
    label_ids: list[int]


class PosDataset(Dataset):
    def __init__(self, examples: Sequence[IndexedSentence]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> IndexedSentence:
        return self.examples[index]


class PosTagger(nn.Module):
    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        num_labels: int,
        hidden_size: int,
        num_layers: int,
        input_dropout: float,
        lstm_dropout: float,
        classifier_dropout: float,
    ) -> None:
        super().__init__()
        embedding_dim = embedding_matrix.shape[1]

        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=True,
            padding_idx=0,
        )
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.input_dropout = nn.Dropout(input_dropout)

        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )

        output_dim = hidden_size * 2
        self.output_norm = nn.LayerNorm(output_dim)
        self.classifier_dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(output_dim, num_labels)

    def forward(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embedding(token_ids)
        x = self.embedding_norm(x)
        x = self.input_dropout(x)

        packed = pack_padded_sequence(
            x,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=True,
        )
        packed_output, _ = self.bilstm(packed)
        x, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=token_ids.shape[1],
        )

        x = self.output_norm(x)
        x = self.classifier_dropout(x)
        return self.classifier(x)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=here)
    parser.add_argument(
        "--fasttext-model",
        type=Path,
        default=here.parent / "turkic_comturk.bin",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--input-dropout", type=float, default=0.20)
    parser.add_argument("--lstm-dropout", type=float, default=0.30)
    parser.add_argument("--classifier-dropout", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default= ) #1，41，42，43，72
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cuda", "cpu"),
        default="auto",
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


def canonical_split(value: object) -> str | None:
    if value is None:
        return None

    compact = "".join(
        character
        for character in str(value).strip().lower()
        if character.isalnum()
    )

    if compact.startswith(TRAIN_PREFIXES):
        return "train"
    if compact.startswith(TEST_PREFIXES):
        return "test"
    if compact.startswith(VALID_PREFIXES):
        return "validation"
    return None


def read_jsonl(path: Path) -> tuple[list[Sentence], list[Sentence], list[Sentence]]:
    train: list[Sentence] = []
    validation: list[Sentence] = []
    test: list[Sentence] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path.name}, line {line_number}: invalid JSON: {error}"
                ) from error

            split = canonical_split(record.get("split"))
            if split is None:
                continue

            raw_tokens = record.get("tokens")
            raw_labels = record.get("upos")
            if not isinstance(raw_tokens, list) or not isinstance(raw_labels, list):
                raise ValueError(
                    f"{path.name}, line {line_number}: tokens/upos must be lists."
                )

            tokens = ["" if item is None else str(item) for item in raw_tokens]
            labels = [
                "_" if item is None or not str(item).strip() else str(item).strip()
                for item in raw_labels
            ]

            if len(tokens) != len(labels):
                raise ValueError(
                    f"{path.name}, line {line_number}: "
                    f"tokens={len(tokens)}, upos={len(labels)}"
                )
            if not tokens or not any(label != "_" for label in labels):
                continue

            example = Sentence(tokens=tokens, labels=labels, source=path.stem)
            if split == "train":
                train.append(example)
            elif split == "validation":
                validation.append(example)
            else:
                test.append(example)

    return train, validation, test


def discover_data(
    data_dir: Path,
) -> tuple[list[Sentence], list[Sentence], dict[str, list[Sentence]]]:
    paths = sorted(data_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No JSONL files found in {data_dir}")

    all_train: list[Sentence] = []
    all_validation: list[Sentence] = []
    tests: dict[str, list[Sentence]] = {}

    print(f"数据目录：{data_dir}")
    print(f"JSONL 文件：{len(paths)}")
    print()
    print(f"{'树库':<24}{'train':>12}{'validation':>16}{'test':>12}")
    print("-" * 64)

    for path in paths:
        train, validation, test = read_jsonl(path)
        all_train.extend(train)
        all_validation.extend(validation)
        if test:
            tests[path.stem] = test

        print(
            f"{path.stem:<24}"
            f"{len(train):>12,}"
            f"{len(validation):>16,}"
            f"{len(test):>12,}"
        )

    print("-" * 64)
    print(
        f"{'混合总计':<24}"
        f"{len(all_train):>12,}"
        f"{len(all_validation):>16,}"
        f"{sum(len(x) for x in tests.values()):>12,}"
    )
    print()

    if not all_train:
        raise RuntimeError("No train split was found.")
    if not all_validation:
        raise RuntimeError("No validation/dev/eval split was found.")
    if not tests:
        raise RuntimeError("No test split was found.")

    return all_train, all_validation, tests


def build_label_vocab(
    train: Sequence[Sentence],
) -> tuple[dict[str, int], list[str], Counter[str]]:
    counts: Counter[str] = Counter()
    for sentence in train:
        counts.update(label for label in sentence.labels if label != "_")

    labels = sorted(counts)
    if not labels:
        raise RuntimeError("No UPOS labels found in train data.")

    return {label: index for index, label in enumerate(labels)}, labels, counts


def check_unseen_labels(
    examples: Iterable[Sentence],
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
            f"{split_name} contains labels unseen in train: {sorted(unseen)}"
        )


def build_token_vocab(examples: Iterable[Sentence]) -> tuple[dict[str, int], list[str]]:
    vocabulary: set[str] = set()
    for sentence in examples:
        vocabulary.update(sentence.tokens)

    id_to_token = [PAD_TOKEN] + sorted(vocabulary)
    token_to_id = {token: index for index, token in enumerate(id_to_token)}
    return token_to_id, id_to_token


def build_embedding_matrix(path: Path, id_to_token: Sequence[str]) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"fastText model not found: {path}")

    print(f"加载 fastText：{path}")
    model = fasttext.load_model(str(path))
    dimension = model.get_dimension()
    matrix = np.zeros((len(id_to_token), dimension), dtype=np.float32)

    print(f"fastText 维度：{dimension}")
    print(f"词表大小：{len(id_to_token):,}")
    print("构建冻结词向量矩阵……")

    report_every = max(10_000, (len(id_to_token) - 1) // 20)
    started = time.time()

    for index in range(1, len(id_to_token)):
        matrix[index] = model.get_word_vector(id_to_token[index])
        if index % report_every == 0 or index == len(id_to_token) - 1:
            print(
                f"  {index:,}/{len(id_to_token) - 1:,} "
                f"({time.time() - started:.1f}s)"
            )

    del model
    gc.collect()
    return torch.from_numpy(matrix)


def index_sentences(
    examples: Sequence[Sentence],
    token_to_id: dict[str, int],
    label_to_id: dict[str, int],
) -> list[IndexedSentence]:
    return [
        IndexedSentence(
            token_ids=[token_to_id[token] for token in sentence.tokens],
            label_ids=[
                label_to_id[label] if label != "_" else IGNORE_INDEX
                for label in sentence.labels
            ],
        )
        for sentence in examples
    ]


def collate(batch: Sequence[IndexedSentence]):
    # Sort by length for efficient packed LSTM execution.
    batch = sorted(batch, key=lambda item: len(item.token_ids), reverse=True)
    lengths = torch.tensor([len(item.token_ids) for item in batch], dtype=torch.long)
    max_length = int(lengths.max())

    token_ids = torch.zeros((len(batch), max_length), dtype=torch.long)
    label_ids = torch.full(
        (len(batch), max_length),
        IGNORE_INDEX,
        dtype=torch.long,
    )

    for row, item in enumerate(batch):
        length = len(item.token_ids)
        token_ids[row, :length] = torch.tensor(item.token_ids, dtype=torch.long)
        label_ids[row, :length] = torch.tensor(item.label_ids, dtype=torch.long)

    return token_ids, label_ids, lengths


def make_loader(
    examples: Sequence[IndexedSentence],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        PosDataset(examples),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=num_workers,
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
    counts = torch.bincount(flat, minlength=num_labels * num_labels)
    confusion += counts.reshape(num_labels, num_labels)


def summarize_metrics(
    loss_sum: float,
    token_count: int,
    confusion: torch.Tensor,
) -> dict[str, object]:
    matrix = confusion.double()
    true_positive = matrix.diag()
    predicted_count = matrix.sum(0)
    gold_count = matrix.sum(1)

    precision = true_positive / predicted_count.clamp_min(1)
    recall = true_positive / gold_count.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    present = gold_count > 0

    correct = int(true_positive.sum())
    return {
        "loss": loss_sum / token_count if token_count else math.nan,
        "accuracy": correct / token_count if token_count else 0.0,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "tokens": token_count,
        "confusion": confusion,
    }


def run_epoch(
    model: PosTagger,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    num_labels: int,
    optimizer: AdamW | None = None,
    grad_clip: float = 1.0,
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)

    loss_sum = 0.0
    token_count = 0
    confusion = torch.zeros((num_labels, num_labels), dtype=torch.long)

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for token_ids, label_ids, lengths in loader:
            token_ids = token_ids.to(device)
            label_ids = label_ids.to(device)
            lengths = lengths.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(token_ids, lengths)
            loss = criterion(
                logits.reshape(-1, num_labels),
                label_ids.reshape(-1),
            )

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            valid_tokens = int((label_ids != IGNORE_INDEX).sum())
            loss_sum += loss.detach().item() * valid_tokens
            token_count += valid_tokens

            update_confusion(
                confusion,
                logits.argmax(-1),
                label_ids,
                num_labels,
            )

    return summarize_metrics(loss_sum, token_count, confusion)


def metric_text(metrics: dict[str, object]) -> str:
    return (
        f"loss={metrics['loss']:.4f}  "
        f"accuracy={metrics['accuracy'] * 100:.2f}%  "
        f"macro-F1={metrics['macro_f1'] * 100:.2f}%  "
        f"tokens={metrics['tokens']:,}"
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    data_dir = args.data_dir.expanduser().resolve()
    fasttext_path = args.fasttext_model.expanduser().resolve()
    device = choose_device(args.device)

    print(f"设备：{device}")
    print(f"随机种子：{args.seed}")
    print()

    train, validation, tests = discover_data(data_dir)
    label_to_id, id_to_label, label_counts = build_label_vocab(train)

    print(f"训练集 UPOS 标签（{len(id_to_label)}）：")
    print(", ".join(id_to_label))
    print("标签频数：")
    print("  " + "  ".join(f"{x}={label_counts[x]:,}" for x in id_to_label))
    print()

    check_unseen_labels(validation, label_to_id, "validation")
    check_unseen_labels(
        (sentence for group in tests.values() for sentence in group),
        label_to_id,
        "test",
    )

    # Include all input token types so that each unseen test word can still use
    # its own fixed fastText subword vector. No validation/test labels are used.
    all_sentences = list(train) + list(validation)
    for group in tests.values():
        all_sentences.extend(group)

    token_to_id, id_to_token = build_token_vocab(all_sentences)
    embedding_matrix = build_embedding_matrix(fasttext_path, id_to_token)

    train_ids = index_sentences(train, token_to_id, label_to_id)
    validation_ids = index_sentences(validation, token_to_id, label_to_id)
    test_ids = {
        name: index_sentences(group, token_to_id, label_to_id)
        for name, group in tests.items()
    }

    del train, validation, tests, all_sentences
    gc.collect()

    train_loader = make_loader(
        train_ids,
        args.batch_size,
        True,
        args.num_workers,
    )
    validation_loader = make_loader(
        validation_ids,
        args.eval_batch_size,
        False,
        args.num_workers,
    )

    model = PosTagger(
        embedding_matrix=embedding_matrix,
        num_labels=len(id_to_label),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        input_dropout=args.input_dropout,
        lstm_dropout=args.lstm_dropout,
        classifier_dropout=args.classifier_dropout,
    ).to(device)

    del embedding_matrix
    gc.collect()

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(
        ignore_index=IGNORE_INDEX,
        label_smoothing=args.label_smoothing,
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in trainable)

    print("模型：冻结 fastText → LayerNorm → 2-layer BiLSTM → Linear → UPOS")
    print(f"总参数：{total_parameters:,}")
    print(f"可训练参数：{trainable_parameters:,}")
    print(f"冻结参数：{total_parameters - trainable_parameters:,}")
    print(f"训练 batch：{len(train_loader):,}")
    print(f"验证 batch：{len(validation_loader):,}")
    print()

    best_f1 = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    no_improvement = 0

    for epoch in range(1, args.epochs + 1):
        started = time.time()

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            len(id_to_label),
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
            len(id_to_label),
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d}  "
            f"lr={current_lr:.2e}  "
            f"time={time.time() - started:.1f}s"
        )
        print(f"  train: {metric_text(train_metrics)}")
        print(f"  valid: {metric_text(validation_metrics)}")

        if validation_metrics["macro_f1"] > best_f1 + args.min_delta:
            best_f1 = float(validation_metrics["macro_f1"])
            best_epoch = epoch
            no_improvement = 0
            # The frozen embedding never changes, so do not duplicate the large
            # embedding matrix in the in-memory best-state copy.
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
                if name != "embedding.weight"
            }
            print("  验证集提升；最佳参数仅保存在内存中。")
        else:
            no_improvement += 1
            print(f"  未提升：{no_improvement}/{args.patience}")

        print()

        if no_improvement >= args.patience:
            print("触发 early stopping。")
            print()
            break

    if best_state is None:
        raise RuntimeError("No best in-memory model was produced.")

    model.load_state_dict(best_state, strict=False)
    del best_state
    gc.collect()

    print(
        f"使用第 {best_epoch} 轮的最佳内存参数；"
        f"validation macro-F1={best_f1 * 100:.2f}%"
    )
    print()

    print("各树库 test 结果")
    print("=" * 100)
    print(
        f"{'树库':<24}"
        f"{'句子':>12}"
        f"{'token':>12}"
        f"{'loss':>12}"
        f"{'accuracy':>14}"
        f"{'macro-F1':>14}"
    )
    print("-" * 100)

    combined_confusion = torch.zeros(
        (len(id_to_label), len(id_to_label)),
        dtype=torch.long,
    )
    combined_loss_sum = 0.0
    combined_tokens = 0

    for treebank in sorted(test_ids):
        loader = make_loader(
            test_ids[treebank],
            args.eval_batch_size,
            False,
            args.num_workers,
        )
        metrics = run_epoch(
            model,
            loader,
            criterion,
            device,
            len(id_to_label),
        )

        print(
            f"{treebank:<24}"
            f"{len(test_ids[treebank]):>12,}"
            f"{metrics['tokens']:>12,}"
            f"{metrics['loss']:>12.4f}"
            f"{metrics['accuracy'] * 100:>13.2f}%"
            f"{metrics['macro_f1'] * 100:>13.2f}%"
        )

        combined_confusion += metrics["confusion"]
        combined_loss_sum += metrics["loss"] * metrics["tokens"]
        combined_tokens += metrics["tokens"]

    combined = summarize_metrics(
        combined_loss_sum,
        combined_tokens,
        combined_confusion,
    )

    print("-" * 100)
    print(
        f"{'全部 test 混合':<24}"
        f"{'-':>12}"
        f"{combined['tokens']:>12,}"
        f"{combined['loss']:>12.4f}"
        f"{combined['accuracy'] * 100:>13.2f}%"
        f"{combined['macro_f1'] * 100:>13.2f}%"
    )
    print("=" * 100)
    print()
    print("完成：未写入 checkpoint、预测结果或日志文件。")


if __name__ == "__main__":
    main()



















