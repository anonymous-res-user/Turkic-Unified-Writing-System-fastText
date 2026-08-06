from pathlib import Path
import unicodedata


input_dir = Path(".")
output_dir = Path("uroman")

output_dir.mkdir(parents=True, exist_ok=True)


def token_has_l(token):
    """
    判断一个 token 是否至少包含一个 Unicode L 类字符。
    """

    for char in token:
        if unicodedata.category(char).startswith("L"):
            return True

    return False


def token_has_l_or_n(token):
    """
    判断一个 token 是否至少包含一个 Unicode L 或 N 类字符。
    """

    for char in token:
        unicode_category = unicodedata.category(char)

        if (
            unicode_category.startswith("L")
            or unicode_category.startswith("N")
        ):
            return True

    return False


def count_l_tokens(line):
    """
    统计一行中至少包含一个 Unicode L 字符的 token 数量。
    """

    token_count = 0

    for token in line.split():
        if token_has_l(token):
            token_count += 1

    return token_count


def merge_paragraph_lines(paragraph_lines):
    """
    处理一个没有空白行的段落。

    如果某行含 L 的 token 数量小于 5，
    就继续拼接后面的行，直到含 L 的 token 总数大于等于 20。

    如果到段落末尾仍然不足 20，则保留能够得到的最长结果。
    """

    merged_lines = []
    merged_group_count = 0

    line_index = 0

    while line_index < len(paragraph_lines):
        current_line = paragraph_lines[line_index]
        current_l_token_count = count_l_tokens(current_line)

        # 这一行本身不属于短行，直接保留
        if current_l_token_count >= 5:
            merged_lines.append(current_line)
            line_index += 1
            continue

        # 当前行少于 5 个含 L 的 token，开始向后拼接
        lines_to_merge = [current_line]
        total_l_token_count = current_l_token_count

        line_index += 1

        while (
            total_l_token_count < 20
            and line_index < len(paragraph_lines)
        ):
            next_line = paragraph_lines[line_index]

            lines_to_merge.append(next_line)
            total_l_token_count += count_l_tokens(next_line)

            line_index += 1

        # 行与行之间使用一个普通空格连接
        merged_line = " ".join(lines_to_merge)

        if merged_line != "":
            merged_lines.append(merged_line)

        if len(lines_to_merge) > 1:
            merged_group_count += 1

    return merged_lines, merged_group_count


def write_punc_file(input_path, output_path):
    """
    读取原始统一书写系统文件，
    合并短行，并生成保留全部标点的训练文件。
    """

    input_nonempty_line_count = 0
    output_line_count = 0
    merged_group_count = 0

    paragraph_lines = []

    with open(input_path, "r", encoding="utf-8") as input_file:
        with open(output_path, "w", encoding="utf-8") as output_file:

            for raw_line in input_file:

                # 必须是读取一行之后进行的第一个操作
                line = raw_line.strip()

                # 空白行是段落边界
                if line == "":
                    if paragraph_lines:
                        processed_lines, paragraph_merged_count = (
                            merge_paragraph_lines(paragraph_lines)
                        )

                        for processed_line in processed_lines:
                            # 最终文件不写入任何空白行
                            if processed_line != "":
                                output_file.write(processed_line + "\n")
                                output_line_count += 1

                        merged_group_count += paragraph_merged_count
                        paragraph_lines = []

                    continue

                input_nonempty_line_count += 1
                paragraph_lines.append(line)

            # 处理文件末尾没有空白行的最后一个段落
            if paragraph_lines:
                processed_lines, paragraph_merged_count = (
                    merge_paragraph_lines(paragraph_lines)
                )

                for processed_line in processed_lines:
                    if processed_line != "":
                        output_file.write(processed_line + "\n")
                        output_line_count += 1

                merged_group_count += paragraph_merged_count

    return {
        "input_nonempty_line_count": input_nonempty_line_count,
        "output_line_count": output_line_count,
        "merged_group_count": merged_group_count,
    }


def write_nopunc_file(punc_path, nopunc_path):
    """
    从保留标点的训练文件生成去标点版本。

    如果一个 token 既不含 Unicode L，也不含 Unicode N，
    就删除这个 token。

    清理后变成空行的内容不会写入文件。
    """

    input_line_count = 0
    output_line_count = 0
    deleted_token_count = 0

    with open(punc_path, "r", encoding="utf-8") as input_file:
        with open(nopunc_path, "w", encoding="utf-8") as output_file:

            for raw_line in input_file:

                # 读取一行之后首先 strip
                line = raw_line.strip()

                if line == "":
                    continue

                input_line_count += 1

                kept_tokens = []

                for token in line.split():
                    if token_has_l_or_n(token):
                        kept_tokens.append(token)
                    else:
                        deleted_token_count += 1

                # 确保最终文件没有空白行
                if len(kept_tokens) == 0:
                    continue

                cleaned_line = " ".join(kept_tokens)

                output_file.write(cleaned_line + "\n")
                output_line_count += 1

    return {
        "input_line_count": input_line_count,
        "output_line_count": output_line_count,
        "deleted_token_count": deleted_token_count,
    }


def process_all_files():
    input_paths = sorted(input_dir.glob("*uroman*.txt"))

    if len(input_paths) == 0:
        raise FileNotFoundError(
            f"在 {input_dir} 中没有找到任何 .txt 文件"
        )

    for input_path in input_paths:
        lang_code = input_path.stem

        punc_path = output_dir / f"{lang_code}_punc.txt"
        nopunc_path = output_dir / f"{lang_code}_nopunc.txt"

        print()
        print("=" * 70)
        print(f"开始处理语言：{lang_code}")
        print(f"输入文件：{input_path}")
        print("=" * 70)

        punc_statistics = write_punc_file(
            input_path=input_path,
            output_path=punc_path,
        )

        nopunc_statistics = write_nopunc_file(
            punc_path=punc_path,
            nopunc_path=nopunc_path,
        )

        print(f"保留标点文件：{punc_path}")
        print(
            "输入非空行数：",
            f"{punc_statistics['input_nonempty_line_count']:,}",
        )
        print(
            "输出行数：",
            f"{punc_statistics['output_line_count']:,}",
        )
        print(
            "发生拼接的短行组数：",
            f"{punc_statistics['merged_group_count']:,}",
        )

        print()

        print(f"去标点文件：{nopunc_path}")
        print(
            "输入行数：",
            f"{nopunc_statistics['input_line_count']:,}",
        )
        print(
            "输出行数：",
            f"{nopunc_statistics['output_line_count']:,}",
        )
        print(
            "删除的 token 数量：",
            f"{nopunc_statistics['deleted_token_count']:,}",
        )


if __name__ == "__main__":
    process_all_files()