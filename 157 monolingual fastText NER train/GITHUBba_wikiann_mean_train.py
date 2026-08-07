import ast
import gc
import random
import time
from pathlib import Path

import fasttext
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import (
    pad_sequence,
    pack_padded_sequence,
    pad_packed_sequence,
)
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. 路径
# ============================================================

base_dir = Path(YOUR PATH)

wikiann_data_dir = base_dir

fasttext_model_path = (
    base_dir / "cc.ba.300.bin"
)


# ============================================================
# 2. 明确选择 train / validation / test 使用哪些语言
# ============================================================

all_lang_codes = [
    # "az",
    "ba",
    # "cv",
    # "kk",
    # "ky",
    # "tk",
    # "tr",
    # "tt",
    # "ug",
    # "uz",
]


# 例子一：10 种语言共同训练
train_lang_codes = [
    # "az",
    "ba",
    # "cv",
    # "kk",
    # "ky",
    # "tk",
    # "tr",
    # "tt",
    # "ug",
    # "uz",
]


# 可以改成：
# validation_lang_codes = ["tk"]
validation_lang_codes = [
    # "az",
    "ba",
    # "cv",
    # "kk",
    # "ky",
    # "tk",
    # "tr",
    # "tt",
    # "ug",
    # "uz",
]


# 可以改成：
# test_lang_codes = ["ug"]
test_lang_codes = [
    # "az",
    "ba",
    # "cv",
    # "kk",
    # "ky",
    # "tk",
    # "tr",
    # "tt",
    # "ug",
    # "uz",
]


# ============================================================
# 3. 超参数
# ============================================================

random_seed =  #41,42,43

fasttext_dimension = 300

lstm_hidden_size = 32
lstm_num_layers = 1

input_dropout = 0.25
lstm_dropout = 0.35

batch_size = 8

learning_rate = 3e-4
weight_decay = 0

maximum_epochs = 500
maximum_gradient_norm = 5.0

# 学习率调整
lr_scheduler_patience = maximum_epochs + 1
lr_scheduler_factor = 0.5
minimum_learning_rate = 1e-5

# 早停
early_stopping_patience = 50
early_stopping_min_delta = 1e-4

# 为避免把巨大的 fastText 模型复制给多个进程
number_of_data_loader_workers = 0


# ============================================================
# 4. WikiANN 标签
# ============================================================

id_to_tag = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-ORG",
    4: "I-ORG",
    5: "B-LOC",
    6: "I-LOC",
}

number_of_tags = len(id_to_tag)


# ============================================================
# 5. 固定随机种子
# ============================================================

random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)


# ============================================================
# 6. 选择设备
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda")

elif (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
):
    device = torch.device("mps")

else:
    device = torch.device("cpu")


print("=" * 70)
print("BiLSTM + CRF WikiANN 实验")
print("=" * 70)

print("运行设备：", device)
print("train 语言：", train_lang_codes)
print("validation 语言：", validation_lang_codes)
print("test 语言：", test_lang_codes)


# ============================================================
# 7. 检查语言代码
# ============================================================

for selected_lang_codes in [
    train_lang_codes,
    validation_lang_codes,
    test_lang_codes,
]:
    for lang_code in selected_lang_codes:
        if lang_code not in all_lang_codes:
            raise ValueError(
                f"不支持的语言代码：{lang_code}"
            )


# ============================================================
# 8. 读取五行一组的数据
# ============================================================

def load_one_wikiann_file(
    lang_code,
    split_name,
):
    input_path = (
        wikiann_data_dir
        / f"wikiann_{lang_code}_{split_name}.txt"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"找不到数据文件：{input_path}"
        )

    nonempty_lines = []

    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as input_file:

        for raw_line in input_file:

            # 读取之后首先 strip
            line = raw_line.strip()

            if line != "":
                nonempty_lines.append(line)

    if len(nonempty_lines) % 5 != 0:
        raise ValueError(
            f"{input_path} 的非空行数不是 5 的倍数："
            f"{len(nonempty_lines)}"
        )

    expected_header = (
        f"{lang_code}_{split_name}"
    )

    examples = []

    for start_index in range(
        0,
        len(nonempty_lines),
        5,
    ):
        first_header = nonempty_lines[start_index]
        original_tokens_line = (
            nonempty_lines[start_index + 1]
        )
        unified_groups_line = (
            nonempty_lines[start_index + 2]
        )
        ner_tags_line = (
            nonempty_lines[start_index + 3]
        )
        second_header = (
            nonempty_lines[start_index + 4]
        )

        example_number = start_index // 5

        if first_header != second_header:
            raise ValueError(
                f"{input_path} 第 {example_number} 组："
                f"首尾标记不一致"
            )

        if first_header != expected_header:
            raise ValueError(
                f"{input_path} 第 {example_number} 组："
                f"预期标记 {expected_header!r}，"
                f"实际是 {first_header!r}"
            )

        original_tokens = ast.literal_eval(
            original_tokens_line
        )

        unified_groups = ast.literal_eval(
            unified_groups_line
        )

        ner_tags = ast.literal_eval(
            ner_tags_line
        )

        if (
            len(original_tokens)
            != len(unified_groups)
            or len(original_tokens)
            != len(ner_tags)
        ):
            raise ValueError(
                f"{input_path} 第 {example_number} 组长度不一致："
                f"original_tokens={len(original_tokens)}，"
                f"unified_groups={len(unified_groups)}，"
                f"ner_tags={len(ner_tags)}"
            )

        if len(original_tokens) == 0:
            raise ValueError(
                f"{input_path} 第 {example_number} 组为空"
            )

        for token_index, unified_group in enumerate(
            unified_groups
        ):
            if not isinstance(unified_group, list):
                raise TypeError(
                    f"{input_path} 第 {example_number} 组，"
                    f"token {token_index} 的统一书写形式"
                    f"不是列表"
                )

            if len(unified_group) == 0:
                raise ValueError(
                    f"{input_path} 第 {example_number} 组，"
                    f"原 token "
                    f"{original_tokens[token_index]!r} "
                    f"没有统一书写 subtoken"
                )

            for subtoken in unified_group:
                if not isinstance(subtoken, str):
                    raise TypeError(
                        f"统一书写 subtoken 不是字符串："
                        f"{subtoken!r}"
                    )

        for tag_id in ner_tags:
            if tag_id not in id_to_tag:
                raise ValueError(
                    f"发现未知 NER 标签 ID：{tag_id}"
                )

        examples.append(
            {
                "language": lang_code,
                "original_tokens": original_tokens,
                "unified_groups": unified_groups,
                "tags": ner_tags,
            }
        )

    print(
        f"已读取 {lang_code} {split_name}："
        f"{len(examples):,} 句"
    )

    return examples


def load_mixed_split(
    lang_codes,
    split_name,
):
    mixed_examples = []

    for lang_code in lang_codes:
        language_examples = load_one_wikiann_file(
            lang_code,
            split_name,
        )

        mixed_examples.extend(language_examples)

    return mixed_examples


print()
print("开始读取数据……")

train_examples = load_mixed_split(
    train_lang_codes,
    "train",
)

validation_examples = load_mixed_split(
    validation_lang_codes,
    "validation",
)

test_examples = load_mixed_split(
    test_lang_codes,
    "test",
)


print()
print("混合后的数据规模：")
print("train：", f"{len(train_examples):,}")
print(
    "validation：",
    f"{len(validation_examples):,}",
)
print("test：", f"{len(test_examples):,}")


# ============================================================
# 9. 建立统一书写 subtoken 词表
# ============================================================

def build_subtoken_vocabulary(
    example_collections,
):
    # 0 留给理论上的空输入占位符。
    # 正常情况下不会被使用。
    subtoken_to_id = {
        "<EMPTY>": 0,
    }

    for examples in example_collections:
        for example in examples:
            unified_groups = example[
                "unified_groups"
            ]

            for unified_group in unified_groups:
                for subtoken in unified_group:
                    if subtoken not in subtoken_to_id:
                        subtoken_to_id[subtoken] = (
                            len(subtoken_to_id)
                        )

    return subtoken_to_id


print()
print("开始建立统一书写 subtoken 词表……")

subtoken_to_id = build_subtoken_vocabulary(
    [
        train_examples,
        validation_examples,
        test_examples,
    ]
)

print(
    "统一书写 subtoken 词表大小：",
    f"{len(subtoken_to_id):,}",
)


# ============================================================
# 10. 从 frozen fastText 模型中提取向量
# ============================================================

if not fasttext_model_path.exists():
    raise FileNotFoundError(
        f"找不到 fastText 模型："
        f"{fasttext_model_path}"
    )

print()
print("开始加载 fastText 模型……")

fasttext_model = fasttext.load_model(
    str(fasttext_model_path)
)

actual_fasttext_dimension = (
    fasttext_model.get_dimension()
)

if actual_fasttext_dimension != fasttext_dimension:
    raise ValueError(
        f"预期 fastText 维度为 "
        f"{fasttext_dimension}，"
        f"实际为 {actual_fasttext_dimension}"
    )

print("fastText 模型加载完成")
print(
    "fastText 维度：",
    actual_fasttext_dimension,
)


embedding_matrix = np.zeros(
    (
        len(subtoken_to_id),
        fasttext_dimension,
    ),
    dtype=np.float32,
)


print()
print("开始提取统一书写 subtoken 向量……")

for subtoken, subtoken_id in subtoken_to_id.items():

    if subtoken_id == 0:
        continue

    vector = fasttext_model.get_word_vector(
        subtoken
    )

    embedding_matrix[subtoken_id] = vector

    if subtoken_id % 50_000 == 0:
        print(
            f"已提取 {subtoken_id:,} 个向量",
            flush=True,
        )


print(
    "向量矩阵形状：",
    embedding_matrix.shape,
)


# fastText .bin 很大。
# 向量提取完成后释放它，给 BiLSTM 训练腾出内存。
del fasttext_model
gc.collect()

print("已经从内存释放 fastText 完整模型")


# ============================================================
# 11. 将嵌套 subtoken 转成 ID
# ============================================================

def encode_examples(
    examples,
    subtoken_to_id,
):
    for example in examples:
        encoded_groups = []

        for unified_group in example[
            "unified_groups"
        ]:
            encoded_group = []

            for subtoken in unified_group:
                encoded_group.append(
                    subtoken_to_id[subtoken]
                )

            encoded_groups.append(
                encoded_group
            )

        example["group_ids"] = encoded_groups

        # 后面的模型只需要 ID。
        # 删除字符串嵌套列表以减少内存占用。
        del example["unified_groups"]


encode_examples(
    train_examples,
    subtoken_to_id,
)

encode_examples(
    validation_examples,
    subtoken_to_id,
)

encode_examples(
    test_examples,
    subtoken_to_id,
)


del subtoken_to_id
gc.collect()


# ============================================================
# 12. Dataset
# ============================================================

class WikiAnnNERDataset(Dataset):

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


# ============================================================
# 13. Collate：保留原 token 层级
# ============================================================

def collate_wikiann_batch(batch):
    sentence_lengths = []

    flat_subtoken_ids = []
    group_offsets = []

    maximum_sentence_length = 0

    for example in batch:
        sentence_length = len(
            example["group_ids"]
        )

        sentence_lengths.append(
            sentence_length
        )

        if sentence_length > maximum_sentence_length:
            maximum_sentence_length = (
                sentence_length
            )

        # 每一个 unified_group 对应一个原 token。
        for group_ids in example["group_ids"]:

            group_offsets.append(
                len(flat_subtoken_ids)
            )

            flat_subtoken_ids.extend(
                group_ids
            )

    batch_size_here = len(batch)

    tags_tensor = torch.zeros(
        (
            batch_size_here,
            maximum_sentence_length,
        ),
        dtype=torch.long,
    )

    mask_tensor = torch.zeros(
        (
            batch_size_here,
            maximum_sentence_length,
        ),
        dtype=torch.bool,
    )

    languages = []

    for batch_index, example in enumerate(batch):
        sentence_length = sentence_lengths[
            batch_index
        ]

        tags_tensor[
            batch_index,
            :sentence_length,
        ] = torch.tensor(
            example["tags"],
            dtype=torch.long,
        )

        mask_tensor[
            batch_index,
            :sentence_length,
        ] = True

        languages.append(
            example["language"]
        )

    return {
        "flat_subtoken_ids": torch.tensor(
            flat_subtoken_ids,
            dtype=torch.long,
        ),
        "group_offsets": torch.tensor(
            group_offsets,
            dtype=torch.long,
        ),
        "lengths": torch.tensor(
            sentence_lengths,
            dtype=torch.long,
        ),
        "tags": tags_tensor,
        "mask": mask_tensor,
        "languages": languages,
    }


# ============================================================
# 14. 线性链 CRF
# ============================================================

class LinearChainCRF(nn.Module):

    def __init__(self, number_of_tags):
        super().__init__()

        self.number_of_tags = number_of_tags

        self.start_transitions = nn.Parameter(
            torch.empty(number_of_tags)
        )

        self.end_transitions = nn.Parameter(
            torch.empty(number_of_tags)
        )

        # transitions[from_tag, to_tag]
        self.transitions = nn.Parameter(
            torch.empty(
                number_of_tags,
                number_of_tags,
            )
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(
            self.start_transitions,
            -0.1,
            0.1,
        )

        nn.init.uniform_(
            self.end_transitions,
            -0.1,
            0.1,
        )

        nn.init.uniform_(
            self.transitions,
            -0.1,
            0.1,
        )

    def negative_log_likelihood(
        self,
        emissions,
        tags,
        mask,
    ):
        gold_score = self._score_gold_sequence(
            emissions,
            tags,
            mask,
        )

        log_partition = self._compute_log_partition(
            emissions,
            mask,
        )

        negative_log_likelihood = (
            log_partition - gold_score
        )

        return negative_log_likelihood.mean()

    def _score_gold_sequence(
        self,
        emissions,
        tags,
        mask,
    ):
        batch_size_here = emissions.size(0)
        sequence_length = emissions.size(1)

        first_tags = tags[:, 0]

        score = self.start_transitions[
            first_tags
        ]

        first_emission_score = (
            emissions[:, 0]
            .gather(
                1,
                first_tags.unsqueeze(1),
            )
            .squeeze(1)
        )

        score = score + first_emission_score

        for timestep in range(
            1,
            sequence_length,
        ):
            previous_tags = tags[
                :,
                timestep - 1,
            ]

            current_tags = tags[
                :,
                timestep,
            ]

            transition_score = self.transitions[
                previous_tags,
                current_tags,
            ]

            emission_score = (
                emissions[:, timestep]
                .gather(
                    1,
                    current_tags.unsqueeze(1),
                )
                .squeeze(1)
            )

            timestep_mask = mask[
                :,
                timestep,
            ].to(emissions.dtype)

            score = score + (
                transition_score
                + emission_score
            ) * timestep_mask

        sequence_lengths = (
            mask.long().sum(dim=1)
        )

        last_tag_indices = (
            sequence_lengths - 1
        ).unsqueeze(1)

        last_tags = (
            tags.gather(
                1,
                last_tag_indices,
            )
            .squeeze(1)
        )

        score = score + self.end_transitions[
            last_tags
        ]

        return score

    def _compute_log_partition(
        self,
        emissions,
        mask,
    ):
        sequence_length = emissions.size(1)

        score = (
            self.start_transitions
            + emissions[:, 0]
        )

        for timestep in range(
            1,
            sequence_length,
        ):
            # 当前形状：
            # batch × previous_tag × current_tag
            next_score = (
                score.unsqueeze(2)
                + self.transitions.unsqueeze(0)
                + emissions[
                    :,
                    timestep,
                ].unsqueeze(1)
            )

            next_score = torch.logsumexp(
                next_score,
                dim=1,
            )

            timestep_mask = mask[
                :,
                timestep,
            ].unsqueeze(1)

            score = torch.where(
                timestep_mask,
                next_score,
                score,
            )

        score = score + self.end_transitions

        return torch.logsumexp(
            score,
            dim=1,
        )

    def decode(
        self,
        emissions,
        mask,
    ):
        batch_size_here = emissions.size(0)
        sequence_length = emissions.size(1)

        score = (
            self.start_transitions
            + emissions[:, 0]
        )

        history = []

        for timestep in range(
            1,
            sequence_length,
        ):
            next_score = (
                score.unsqueeze(2)
                + self.transitions.unsqueeze(0)
            )

            best_score, best_previous_tag = (
                next_score.max(dim=1)
            )

            best_score = (
                best_score
                + emissions[:, timestep]
            )

            timestep_mask = mask[
                :,
                timestep,
            ].unsqueeze(1)

            score = torch.where(
                timestep_mask,
                best_score,
                score,
            )

            history.append(
                best_previous_tag
            )

        score = score + self.end_transitions

        best_last_tags = score.argmax(dim=1)

        sequence_lengths = (
            mask.long()
            .sum(dim=1)
            .tolist()
        )

        decoded_paths = []

        for batch_index in range(
            batch_size_here
        ):
            actual_length = sequence_lengths[
                batch_index
            ]

            current_tag = int(
                best_last_tags[
                    batch_index
                ].item()
            )

            path = [current_tag]

            relevant_history = history[
                :actual_length - 1
            ]

            for history_tensor in reversed(
                relevant_history
            ):
                current_tag = int(
                    history_tensor[
                        batch_index,
                        current_tag,
                    ].item()
                )

                path.append(current_tag)

            path.reverse()
            decoded_paths.append(path)

        return decoded_paths


# ============================================================
# 15. BiLSTM + CRF
# ============================================================

class BiLSTMCRF(nn.Module):

    def __init__(
        self,
        embedding_matrix,
        input_dimension,
        hidden_size,
        number_of_layers,
        number_of_tags,
        input_dropout,
        lstm_dropout,
    ):
        super().__init__()

        frozen_embedding_tensor = (
            torch.from_numpy(
                embedding_matrix
            )
        )

        # 每一个 bag 就是一个原 token 内部的
        # 若干统一书写 subtoken。
        # mode="mean" 会直接取平均。
        self.embedding_bag = (
            nn.EmbeddingBag.from_pretrained(
                frozen_embedding_tensor,
                freeze=True,
                mode="mean",
            )
        )

        self.input_dropout_layer = nn.Dropout(
            input_dropout
        )

        self.lstm = nn.LSTM(
            input_size=input_dimension,
            hidden_size=hidden_size,
            num_layers=number_of_layers,
            batch_first=True,
            bidirectional=True,
            dropout=(
                lstm_dropout
                if number_of_layers > 1
                else 0.0
            ),
        )

        self.output_dropout_layer = nn.Dropout(
            lstm_dropout
        )

        self.hidden_to_tags = nn.Linear(
            hidden_size * 2,
            number_of_tags,
        )

        self.crf = LinearChainCRF(
            number_of_tags
        )

    def forward(
        self,
        flat_subtoken_ids,
        group_offsets,
        lengths,
    ):
        # 输出：
        # sum(sentence_lengths) × 300
        group_vectors = self.embedding_bag(
            flat_subtoken_ids,
            group_offsets,
        )

        sentence_length_list = (
            lengths.tolist()
        )

        sentence_vectors = torch.split(
            group_vectors,
            sentence_length_list,
            dim=0,
        )

        # batch × max_sentence_length × 300
        padded_sentence_vectors = pad_sequence(
            sentence_vectors,
            batch_first=True,
            padding_value=0.0,
        )

        padded_sentence_vectors = (
            self.input_dropout_layer(
                padded_sentence_vectors
            )
        )

        packed_input = pack_padded_sequence(
            padded_sentence_vectors,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        packed_output, _ = self.lstm(
            packed_input
        )

        lstm_output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=(
                padded_sentence_vectors.size(1)
            ),
        )

        lstm_output = (
            self.output_dropout_layer(
                lstm_output
            )
        )

        emissions = self.hidden_to_tags(
            lstm_output
        )

        return emissions


# ============================================================
# 16. BIO 实体级指标
# ============================================================

def extract_bio_entities(tag_id_sequence):
    tag_sequence = []

    for tag_id in tag_id_sequence:
        tag_sequence.append(
            id_to_tag[tag_id]
        )

    entities = set()

    current_entity_type = None
    current_start = None

    # 末尾补一个 O，方便关闭最后一个实体
    extended_tags = tag_sequence + ["O"]

    for token_index, tag_name in enumerate(
        extended_tags
    ):
        if tag_name == "O":
            if current_entity_type is not None:
                entities.add(
                    (
                        current_entity_type,
                        current_start,
                        token_index,
                    )
                )

                current_entity_type = None
                current_start = None

            continue

        prefix, entity_type = tag_name.split(
            "-",
            1,
        )

        if prefix == "B":
            if current_entity_type is not None:
                entities.add(
                    (
                        current_entity_type,
                        current_start,
                        token_index,
                    )
                )

            current_entity_type = entity_type
            current_start = token_index

        elif prefix == "I":
            # 非法 I-X：
            # 如果前面不是同类型实体，
            # 就把它当作新实体的开始。
            if current_entity_type != entity_type:
                if current_entity_type is not None:
                    entities.add(
                        (
                            current_entity_type,
                            current_start,
                            token_index,
                        )
                    )

                current_entity_type = entity_type
                current_start = token_index

    return entities


def calculate_ner_metrics(
    gold_sequences,
    predicted_sequences,
):
    correct_token_count = 0
    total_token_count = 0

    true_positive = 0
    false_positive = 0
    false_negative = 0

    for gold_tags, predicted_tags in zip(
        gold_sequences,
        predicted_sequences,
    ):
        if len(gold_tags) != len(predicted_tags):
            raise ValueError(
                "gold 与 prediction 长度不一致"
            )

        for gold_tag, predicted_tag in zip(
            gold_tags,
            predicted_tags,
        ):
            if gold_tag == predicted_tag:
                correct_token_count += 1

            total_token_count += 1

        gold_entities = extract_bio_entities(
            gold_tags
        )

        predicted_entities = extract_bio_entities(
            predicted_tags
        )

        true_positive += len(
            gold_entities & predicted_entities
        )

        false_positive += len(
            predicted_entities - gold_entities
        )

        false_negative += len(
            gold_entities - predicted_entities
        )

    token_accuracy = (
        correct_token_count
        / total_token_count
        if total_token_count > 0
        else 0.0
    )

    precision_denominator = (
        true_positive + false_positive
    )

    recall_denominator = (
        true_positive + false_negative
    )

    precision = (
        true_positive / precision_denominator
        if precision_denominator > 0
        else 0.0
    )

    recall = (
        true_positive / recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    f1_denominator = precision + recall

    f1 = (
        2.0 * precision * recall
        / f1_denominator
        if f1_denominator > 0
        else 0.0
    )

    return {
        "token_accuracy": token_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


# ============================================================
# 17. DataLoader
# ============================================================

train_dataset = WikiAnnNERDataset(
    train_examples
)

validation_dataset = WikiAnnNERDataset(
    validation_examples
)

test_dataset = WikiAnnNERDataset(
    test_examples
)


data_loader_generator = torch.Generator()
data_loader_generator.manual_seed(
    random_seed
)


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=number_of_data_loader_workers,
    collate_fn=collate_wikiann_batch,
    generator=data_loader_generator,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=number_of_data_loader_workers,
    collate_fn=collate_wikiann_batch,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=number_of_data_loader_workers,
    collate_fn=collate_wikiann_batch,
)


# ============================================================
# 18. 建立模型
# ============================================================

model = BiLSTMCRF(
    embedding_matrix=embedding_matrix,
    input_dimension=fasttext_dimension,
    hidden_size=lstm_hidden_size,
    number_of_layers=lstm_num_layers,
    number_of_tags=number_of_tags,
    input_dropout=input_dropout,
    lstm_dropout=lstm_dropout,
)

model = model.to(device)


# 模型已经拥有自己的 embedding tensor。
del embedding_matrix
gc.collect()


trainable_parameter_count = 0

for parameter in model.parameters():
    if parameter.requires_grad:
        trainable_parameter_count += (
            parameter.numel()
        )


print()
print("模型建立完成")
print(
    "可训练参数数量：",
    f"{trainable_parameter_count:,}",
)
print(
    "fastText 向量是否冻结：",
    not model.embedding_bag.weight.requires_grad,
)


# ============================================================
# 19. 优化器与 scheduler
# ============================================================

trainable_parameters = []

for parameter in model.parameters():
    if parameter.requires_grad:
        trainable_parameters.append(parameter)


optimizer = torch.optim.AdamW(
    trainable_parameters,
    lr=learning_rate,
    weight_decay=weight_decay,
)


scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
        threshold=early_stopping_min_delta,
        min_lr=minimum_learning_rate,
    )
)


# ============================================================
# 20. 单轮训练
# ============================================================

def train_one_epoch(
    model,
    data_loader,
    optimizer,
):
    model.train()

    total_loss = 0.0
    total_sentence_count = 0

    for batch_number, batch in enumerate(
        data_loader,
        start=1,
    ):
        flat_subtoken_ids = batch[
            "flat_subtoken_ids"
        ].to(device)

        group_offsets = batch[
            "group_offsets"
        ].to(device)

        # pack_padded_sequence 要求 lengths 在 CPU 上
        lengths = batch["lengths"]

        tags = batch["tags"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        emissions = model(
            flat_subtoken_ids,
            group_offsets,
            lengths,
        )

        loss = model.crf.negative_log_likelihood(
            emissions,
            tags,
            mask,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            maximum_gradient_norm,
        )

        optimizer.step()

        batch_sentence_count = tags.size(0)

        total_loss += (
            loss.item()
            * batch_sentence_count
        )

        total_sentence_count += (
            batch_sentence_count
        )

        if batch_number % 200 == 0:
            print(
                f"  已训练 {batch_number:,} batches，"
                f"当前 loss={loss.item():.4f}",
                flush=True,
            )

    average_loss = (
        total_loss / total_sentence_count
    )

    return average_loss


# ============================================================
# 21. validation / test
# ============================================================

@torch.no_grad()
def evaluate_model(
    model,
    data_loader,
):
    model.eval()

    total_loss = 0.0
    total_sentence_count = 0

    all_gold_sequences = []
    all_predicted_sequences = []

    for batch in data_loader:
        flat_subtoken_ids = batch[
            "flat_subtoken_ids"
        ].to(device)

        group_offsets = batch[
            "group_offsets"
        ].to(device)

        lengths = batch["lengths"]

        tags = batch["tags"].to(device)
        mask = batch["mask"].to(device)

        emissions = model(
            flat_subtoken_ids,
            group_offsets,
            lengths,
        )

        loss = model.crf.negative_log_likelihood(
            emissions,
            tags,
            mask,
        )

        predicted_paths = model.crf.decode(
            emissions,
            mask,
        )

        batch_sentence_count = tags.size(0)

        total_loss += (
            loss.item()
            * batch_sentence_count
        )

        total_sentence_count += (
            batch_sentence_count
        )

        lengths_list = lengths.tolist()
        tags_cpu = tags.cpu()

        for batch_index, actual_length in enumerate(
            lengths_list
        ):
            gold_path = (
                tags_cpu[
                    batch_index,
                    :actual_length,
                ]
                .tolist()
            )

            predicted_path = predicted_paths[
                batch_index
            ]

            all_gold_sequences.append(
                gold_path
            )

            all_predicted_sequences.append(
                predicted_path
            )

    average_loss = (
        total_loss / total_sentence_count
    )

    metrics = calculate_ner_metrics(
        all_gold_sequences,
        all_predicted_sequences,
    )

    metrics["loss"] = average_loss

    return metrics


# ============================================================
# 22. 只保存可训练部分的最佳参数到内存
# ============================================================

def copy_trainable_state_to_cpu(model):
    copied_state = {}

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            copied_state[name] = (
                parameter
                .detach()
                .cpu()
                .clone()
            )

    return copied_state


def restore_trainable_state(
    model,
    copied_state,
):
    named_parameters = dict(
        model.named_parameters()
    )

    with torch.no_grad():
        for name, saved_parameter in (
            copied_state.items()
        ):
            target_parameter = named_parameters[
                name
            ]

            target_parameter.copy_(
                saved_parameter.to(
                    target_parameter.device
                )
            )


# ============================================================
# 23. 正式训练与早停
# ============================================================

print()
print("=" * 70)
print("开始训练")
print("=" * 70)

training_start_time = time.time()

best_validation_f1 = -1.0
best_epoch = 0
best_trainable_state = None

epochs_without_improvement = 0


for epoch_number in range(
    1,
    maximum_epochs + 1,
):
    epoch_start_time = time.time()

    print()
    print(
        f"Epoch {epoch_number}/"
        f"{maximum_epochs}"
    )

    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
    )

    validation_metrics = evaluate_model(
        model,
        validation_loader,
    )

    validation_f1 = validation_metrics[
        "f1"
    ]

    scheduler.step(validation_f1)

    current_learning_rate = (
        optimizer.param_groups[0]["lr"]
    )

    epoch_seconds = (
        time.time() - epoch_start_time
    )

    print(
        f"train loss：{train_loss:.6f}"
    )

    print(
        f"validation loss："
        f"{validation_metrics['loss']:.6f}"
    )

    print(
        f"validation token accuracy："
        f"{validation_metrics['token_accuracy']:.4f}"
    )

    print(
        f"validation precision："
        f"{validation_metrics['precision']:.4f}"
    )

    print(
        f"validation recall："
        f"{validation_metrics['recall']:.4f}"
    )

    print(
        f"validation entity F1："
        f"{validation_f1:.4f}"
    )

    print(
        f"learning rate："
        f"{current_learning_rate:.8f}"
    )

    print(
        f"本轮用时："
        f"{epoch_seconds / 60:.2f} 分钟"
    )

    if (
        validation_f1
        > best_validation_f1
        + early_stopping_min_delta
    ):
        best_validation_f1 = validation_f1
        best_epoch = epoch_number

        best_trainable_state = (
            copy_trainable_state_to_cpu(
                model
            )
        )

        epochs_without_improvement = 0

        print(
            "validation F1 提升，"
            "已在内存中记录当前最佳参数"
        )

    else:
        epochs_without_improvement += 1

        print(
            "validation F1 未达到新的最佳值，"
            f"连续 {epochs_without_improvement}/"
            f"{early_stopping_patience} 轮"
        )

    if (
        epochs_without_improvement
        >= early_stopping_patience
    ):
        print()
        print(
            "触发早停：validation entity F1 "
            f"连续 {early_stopping_patience} "
            "轮没有改善"
        )

        break


if best_trainable_state is None:
    raise RuntimeError(
        "训练过程中没有记录到最佳参数"
    )


training_seconds = (
    time.time() - training_start_time
)


# ============================================================
# 24. 恢复 validation 最好的模型
# ============================================================

restore_trainable_state(
    model,
    best_trainable_state,
)

del best_trainable_state
gc.collect()


print()
print("=" * 70)
print("训练完成")
print("=" * 70)

print("最佳 epoch：", best_epoch)
print(
    "最佳 validation entity F1：",
    f"{best_validation_f1:.4f}",
)
print(
    "总训练时间：",
    f"{training_seconds / 3600:.2f} 小时",
)

print(
    "下游模型未写入磁盘，"
    "现在直接进行 test"
)


# ============================================================
# 25. 混合 test 集总体测试
# ============================================================

print()
print("=" * 70)
print("混合 test 集结果")
print("=" * 70)

test_metrics = evaluate_model(
    model,
    test_loader,
)

print(
    "test loss：",
    f"{test_metrics['loss']:.6f}",
)

print(
    "test token accuracy：",
    f"{test_metrics['token_accuracy']:.4f}",
)

print(
    "test entity precision：",
    f"{test_metrics['precision']:.4f}",
)

print(
    "test entity recall：",
    f"{test_metrics['recall']:.4f}",
)

print(
    "test entity F1：",
    f"{test_metrics['f1']:.4f}",
)


# ============================================================
# 26. 每种 test 语言分别测试
# ============================================================

print()
print("=" * 70)
print("各 test 语言分别测试")
print("=" * 70)


for lang_code in test_lang_codes:
    language_test_examples = []

    for example in test_examples:
        if example["language"] == lang_code:
            language_test_examples.append(
                example
            )

    if len(language_test_examples) == 0:
        continue

    language_test_dataset = (
        WikiAnnNERDataset(
            language_test_examples
        )
    )

    language_test_loader = DataLoader(
        language_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=number_of_data_loader_workers,
        collate_fn=collate_wikiann_batch,
    )

    language_metrics = evaluate_model(
        model,
        language_test_loader,
    )

    print()
    print(
        f"{lang_code}："
        f"{len(language_test_examples):,} 句"
    )

    print(
        f"  token accuracy："
        f"{language_metrics['token_accuracy']:.4f}"
    )

    print(
        f"  entity precision："
        f"{language_metrics['precision']:.4f}"
    )

    print(
        f"  entity recall："
        f"{language_metrics['recall']:.4f}"
    )

    print(
        f"  entity F1："
        f"{language_metrics['f1']:.4f}"
    )


print()
print("=" * 70)
print("全部完成")
print("=" * 70)












