from pathlib import Path
import os
import random
import subprocess


input_dir = Path(".")
output_path = input_dir / "uroman_mergeshuffledata.txt"

# 固定随机种子，方便论文实验复现
random_seed = 20260101
random_generator = random.Random(random_seed)

# *punc.txt 会同时匹配：
# az_punc.txt
# az_nopunc.txt
input_paths = sorted(input_dir.glob("*punc.txt"))

if len(input_paths) != 22:
    raise ValueError(
        f"预期找到 22 个文件，实际找到 {len(input_paths)} 个：\n"
        + "\n".join(str(path) for path in input_paths)
    )

print("将合并以下文件：")

for input_path in input_paths:
    print("  ", input_path.name)


# 临时文件：
# 每一行格式是“随机键\t原始文本”
random_key_path = input_dir / "_merge_shuffle_random_keys.tmp"

total_line_count = 0

print("\n第一步：合并文件并生成随机键")

with open(
    random_key_path,
    "w",
    encoding="utf-8",
) as random_key_file:

    for input_path in input_paths:
        print("正在读取：", input_path.name)

        with open(
            input_path,
            "r",
            encoding="utf-8",
        ) as input_file:

            for raw_line in input_file:

                # 读取之后首先 strip
                line = raw_line.strip()

                # 不把空白行写入最终训练数据
                if line == "":
                    continue

                # 128 位随机数，写成固定长度十六进制字符串
                random_key = (
                    f"{random_generator.getrandbits(128):032x}"
                )

                random_key_file.write(
                    random_key + "\t" + line + "\n"
                )

                total_line_count += 1

                if total_line_count % 1_000_000 == 0:
                    print(
                        f"已读取 {total_line_count:,} 行",
                        flush=True,
                    )


print(
    f"\n随机键文件生成完成，共 {total_line_count:,} 行"
)
print("第二步：根据随机键进行外部排序")


# LC_ALL=C 让 sort 按字节顺序排序，更快也更稳定
environment = os.environ.copy()
environment["LC_ALL"] = "C"

# 让系统 sort 负责外部排序。
# 结果通过 stdout 流式读取，不再生成第二个巨大临时文件。
sort_process = subprocess.Popen(
    [
        "sort",
        str(random_key_path),
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    encoding="utf-8",
    env=environment,
)


written_line_count = 0

with open(
    output_path,
    "w",
    encoding="utf-8",
) as output_file:

    assert sort_process.stdout is not None

    for sorted_line in sort_process.stdout:
        random_key, separator, original_line = (
            sorted_line.partition("\t")
        )

        if separator == "":
            raise ValueError(
                "临时文件中发现没有制表符的异常行："
                + repr(sorted_line[:200])
            )

        # original_line 已经包含行末换行符
        output_file.write(original_line)

        written_line_count += 1

        if written_line_count % 1_000_000 == 0:
            print(
                f"已写入 {written_line_count:,} 行",
                flush=True,
            )


stderr_text = ""

if sort_process.stderr is not None:
    stderr_text = sort_process.stderr.read()

return_code = sort_process.wait()

if return_code != 0:
    raise RuntimeError(
        "sort 执行失败：\n" + stderr_text
    )


# 排序成功后删除带随机键的临时文件
random_key_path.unlink()


print("\n全部完成")
print("输入文件数量：", len(input_paths))
print("最终行数：", f"{written_line_count:,}")
print("输出文件：", output_path)