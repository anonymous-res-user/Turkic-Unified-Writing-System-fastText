#!/usr/bin/env bash

set -u
set -o pipefail

PYTHON="/opt/anaconda3/envs/pymlling/bin/python"
DATE="20260101"
JOBS="${JOBS:-4}"
KEEP_DUMP="${KEEP_DUMP:-0}"
ROOT="${ROOT:-turkic_wikipedia_${DATE}}"

CODES=(
    az
    ba
    cv
    kaa
    kk
    ky
    tk
    tr
    tt
    ug
    uz
)

mkdir -p "$ROOT/dumps"
mkdir -p "$ROOT/extracted"
mkdir -p "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/rawwikitext"

if [[ ! -x "$PYTHON" ]]; then
    echo "错误：找不到 Python：$PYTHON"
    exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "错误：系统中找不到 curl"
    exit 1
fi

if ! command -v bzip2 >/dev/null 2>&1; then
    echo "错误：系统中找不到 bzip2"
    exit 1
fi

if ! "$PYTHON" -m wikiextractor.WikiExtractor --help >/dev/null 2>&1; then
    echo "错误：WikiExtractor 无法在 pymlling 环境中运行"
    exit 1
fi

for CODE in "${CODES[@]}"; do
    PROJECT="${CODE}wiki"
    DUMP="${PROJECT}-${DATE}-pages-articles.xml.bz2"
    URL="https://dumps.wikimedia.org/${PROJECT}/${DATE}/${DUMP}"

    DUMP_PATH="$ROOT/dumps/$DUMP"
    EXTRACT_DIR="$ROOT/extracted/$CODE"
    TXT_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/rawwikitext/${CODE}_${DATE}.txt"
    TEMP_TXT="${TXT_PATH}.tmp"

    echo
    echo "============================================================"
    echo "语言代码：$CODE"
    echo "项目名称：$PROJECT"
    echo "固定日期：$DATE"
    echo "============================================================"

    if [[ -s "$TXT_PATH" ]]; then
        echo "已经完成，跳过：$TXT_PATH"
        continue
    fi

    rm -rf "$EXTRACT_DIR"
    rm -f "$TEMP_TXT"
    mkdir -p "$EXTRACT_DIR"

    if [[ -f "$DUMP_PATH" ]] && bzip2 -t "$DUMP_PATH" >/dev/null 2>&1; then
        echo "[1/3] dump 已完整存在，跳过下载："
        echo "$DUMP_PATH"
    else
        echo "[1/3] 下载或断点续传："
        echo "$URL"

        if ! curl \
            --fail \
            --location \
            --retry 5 \
            --retry-delay 5 \
            --connect-timeout 30 \
            --continue-at - \
            --output "$DUMP_PATH" \
            "$URL"; then

            echo "下载失败：$CODE"
            rm -rf "$EXTRACT_DIR"
            continue
        fi

        echo "正在检查 dump 完整性……"

        if ! bzip2 -t "$DUMP_PATH"; then
            echo "dump 文件不完整或损坏：$DUMP_PATH"
            rm -rf "$EXTRACT_DIR"
            continue
        fi
    fi

    echo "[2/3] WikiExtractor 正在解析：$CODE"

    if [[ "$CODE" == "krc" || "$CODE" == "tr" ]]; then
        echo "提示：$CODE 使用 --no-templates，跳过模板展开"

        if ! "$PYTHON" -m wikiextractor.WikiExtractor \
            "$DUMP_PATH" \
            --json \
            -ns 0 \
            --no-templates \
            --processes "$JOBS" \
            --output "$EXTRACT_DIR"; then

            echo "WikiExtractor 解析失败：$CODE"
            rm -rf "$EXTRACT_DIR"
            continue
        fi
    else
        if ! "$PYTHON" -m wikiextractor.WikiExtractor \
            "$DUMP_PATH" \
            --json \
            -ns 0 \
            --processes "$JOBS" \
            --output "$EXTRACT_DIR"; then

            echo "WikiExtractor 解析失败：$CODE"
            rm -rf "$EXTRACT_DIR"
            continue
        fi
    fi

    echo "[3/3] 正在生成：$TXT_PATH"

    if ! EXTRACT_DIR="$EXTRACT_DIR" \
         TEMP_TXT="$TEMP_TXT" \
         "$PYTHON" <<'PYTHON_SCRIPT'
import html
import json
import os
import re
from pathlib import Path

extract_dir = Path(os.environ["EXTRACT_DIR"])
output_path = Path(os.environ["TEMP_TXT"])

input_files = sorted(
    path
    for path in extract_dir.rglob("wiki_*")
    if path.is_file()
)

if not input_files:
    raise SystemExit("错误：没有找到 WikiExtractor 输出文件")

article_count = 0
invalid_json_count = 0

with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
    for input_path in input_files:
        with input_path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as input_file:
            for line_number, line in enumerate(input_file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    article = json.loads(line)
                except json.JSONDecodeError:
                    invalid_json_count += 1
                    print(
                        f"警告：跳过无效 JSON："
                        f"{input_path}:{line_number}"
                    )
                    continue

                title = str(article.get("title") or "").strip()
                text = str(article.get("text") or "").strip()

                if not text:
                    continue

                text = html.unescape(text)

                text = re.sub(
                    r"<templatestyles\b[^>]*?/?>",
                    "",
                    text,
                    flags=re.IGNORECASE,
                )

                text = re.sub(
                    r"__[A-Z0-9_]+__",
                    "",
                    text,
                )

                text = re.sub(
                    r"[ \t]+\n",
                    "\n",
                    text,
                )

                text = re.sub(
                    r"\n{3,}",
                    "\n\n",
                    text,
                ).strip()

                if not text:
                    continue

                output_file.write(title)
                output_file.write("\n")
                output_file.write(text)
                output_file.write("\n\n")

                article_count += 1

if article_count == 0:
    if output_path.exists():
        output_path.unlink()
    raise SystemExit("错误：没有输出任何有效文章")

print(f"文章数量：{article_count:,}")
print(f"无效 JSON：{invalid_json_count:,}")
print(f"临时文件：{output_path}")
PYTHON_SCRIPT
    then
        echo "合并 TXT 失败：$CODE"
        rm -f "$TEMP_TXT"
        rm -rf "$EXTRACT_DIR"
        continue
    fi

    mv "$TEMP_TXT" "$TXT_PATH"
    rm -rf "$EXTRACT_DIR"

    if [[ "$KEEP_DUMP" != "1" ]]; then
        rm -f "$DUMP_PATH"
    fi

    echo "完成：$TXT_PATH"
done

echo
echo "============================================================"
echo "全部突厥语处理结束"
echo "固定日期：$DATE"
echo "TXT 目录：$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/rawwikitext"
echo "============================================================"
