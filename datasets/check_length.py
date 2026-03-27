from .data_generator import Dataset
# -*- coding: utf-8 -*-
import os
import csv
import math
import collections
import numpy as np
import matplotlib.pyplot as plt

# 改成你的实际文件名（不要带 .py）
from .data_generator import Dataset

RNA_SS_data = collections.namedtuple('RNA_SS_data', 'contact data_fcn_2 seq_raw length name')

def main():
    # ====== 1. 改成你的数据目录 ======
    data_root = [
        "/storage/student2/xiao/lql/RNADiffFold/data/bpRNA/TR0"
    ]

    # ====== 2. 创建数据集 ======
    dataset = Dataset(data_root=data_root, upsampling=False)


    # dataset.index 中每个元素都是 (file_idx, local_idx, length)
    all_info = dataset.index
    all_lengths = [int(L) for _, _, L in all_info]

    if len(all_lengths) == 0:
        print("没有读到任何样本。")
        return

    # ====== 3. 基本统计 ======
    print("=" * 60)
    print(f"总样本数: {len(all_lengths)}")
    print(f"最短长度: {min(all_lengths)}")
    print(f"最长长度: {max(all_lengths)}")
    print(f"平均长度: {np.mean(all_lengths):.2f}")
    print(f"中位数长度: {np.median(all_lengths):.2f}")
    print(f"长度标准差: {np.std(all_lengths):.2f}")
    print("=" * 60)

    # ====== 4. 打印前 20 个样本长度 ======
    print("前 20 个样本长度：")
    for i, (fi, li, L) in enumerate(all_info[:20]):
        print(f"global_idx={i:5d}, file_idx={fi:3d}, local_idx={li:5d}, length={L}")

    print("=" * 60)

    # ====== 5. 打印最常见长度 ======
    counter = collections.Counter(all_lengths)
    print("最常见的 20 个长度：")
    for length, cnt in counter.most_common(20):
        print(f"length={length:4d}, count={cnt}")

    print("=" * 60)

    # ====== 6. 导出每个样本的长度到 CSV ======
    csv_path = "sample_lengths.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["global_idx", "file_idx", "local_idx", "length"])

        for global_idx, (fi, li, L) in enumerate(all_info):
            writer.writerow([global_idx, fi, li, int(L)])

    print(f"每个样本的长度已经保存到: {csv_path}")

    # ====== 7. 如果你还想把样本名也导出来，可以取消下面这段注释 ======
    # 注意：这会真的逐个访问样本，速度会慢一些
    """
    csv_with_name_path = "sample_lengths_with_name.csv"
    with open(csv_with_name_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["global_idx", "file_idx", "local_idx", "length", "name", "seq_len"])

        for global_idx, (fi, li, L) in enumerate(all_info):
            contact, data_fcn_2, seq_raw, length, name = dataset[global_idx]
            writer.writerow([global_idx, fi, li, int(L), str(name), len(seq_raw)])

    print(f"带样本名的长度表已经保存到: {csv_with_name_path}")
    """

    # ====== 8. 画长度分布直方图 ======
    plt.figure(figsize=(10, 6))
    plt.hist(all_lengths, bins=50)
    plt.xlabel("Sequence Length")
    plt.ylabel("Count")
    plt.title("Length Distribution")
    plt.tight_layout()
    hist_path = "length_hist.png"
    plt.savefig(hist_path, dpi=200)
    plt.close()

    print(f"长度分布直方图已经保存到: {hist_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()