from pathlib import Path
import time

import fasttext


input_path = Path("uroman_mergeshuffledata.txt")
output_path = Path("turkic_uroman.bin")


if not input_path.exists():
    raise FileNotFoundError(
        f"找不到训练文件：{input_path.resolve()}"
    )


print("=" * 70)
print("开始训练 fastText CBOW 模型")
print("=" * 70)

print("训练文件：", input_path.resolve())
print("输出模型：", output_path.resolve())

print()
print("训练参数：")
print("  model     = cbow")
print("  dim       = 300")
print("  minCount  = 5")
print("  minn      = 2")
print("  maxn      = 5")
print("  bucket    = 15,000,000")
print("  epoch     = 10")
print("  lr        = 0.05")
print("  ws        = 5")
print("  neg       = 10")
print("  loss      = ns")
print("  thread    = 18")

print()

start_time = time.time()


model = fasttext.train_unsupervised(
    input=str(input_path),
    model="cbow",

    dim=300,
    minCount=5,

    minn=2,
    maxn=5,
    bucket=15_000_000,

    epoch=10,
    lr=0.05,
    ws=5,

    neg=10,
    loss="ns",

    thread=18,
    verbose=2,
)


training_seconds = time.time() - start_time

print()
print("训练完成，开始保存模型……")

model.save_model(str(output_path))


print()
print("=" * 70)
print("训练及保存完成")
print("=" * 70)

print("模型文件：", output_path.resolve())
print("词表大小：", len(model.get_words()))
print("向量维度：", model.get_dimension())

training_hours = training_seconds / 3600

print(
    "训练用时：",
    f"{training_seconds:,.2f} 秒",
)

print(
    "训练用时：",
    f"{training_hours:,.2f} 小时",
)