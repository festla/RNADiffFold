import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


def dot_bracket_to_contact_map(sequence: str, dot_bracket: str) -> np.ndarray:
    """
    将 RNA 序列和点括号结构转换为二维接触图矩阵。
    配对位置记为1，非配对位置记为0。
    """
    if len(sequence) != len(dot_bracket):
        raise ValueError("sequence 和 dot_bracket 的长度必须一致")

    length = len(sequence)
    contact_map = np.zeros((length, length), dtype=np.int32)

    bracket_pairs = {
        "(": ")",
        "[": "]",
        "{": "}",
        "<": ">"
    }
    stacks = {left: [] for left in bracket_pairs}
    reverse_pairs = {right: left for left, right in bracket_pairs.items()}

    for i, ch in enumerate(dot_bracket):
        if ch in bracket_pairs:
            stacks[ch].append(i)
        elif ch in reverse_pairs:
            left_bracket = reverse_pairs[ch]
            if not stacks[left_bracket]:
                raise ValueError(f"位置 {i} 处括号 {ch} 不匹配")
            j = stacks[left_bracket].pop()
            contact_map[i, j] = 1
            contact_map[j, i] = 1
        elif ch == ".":
            continue
        else:
            raise ValueError(f"发现不支持的结构符号: {ch}")

    for left_bracket, stack in stacks.items():
        if stack:
            raise ValueError(f"括号 {left_bracket} 未闭合，位置: {stack}")

    return contact_map


def plot_contact_map(contact_map: np.ndarray,
                     sequence: str = None,
                     title: str = "RNA Secondary Structure Contact Map",
                     save_path: str = None,
                     show: bool = True):
    """
    将接触图绘制为热图形式：
    - 0: 空白/浅色
    - 1: 填充方格
    """
    length = contact_map.shape[0]

    # 两色离散 colormap：0=白色, 1=蓝色
    cmap = ListedColormap(["white", "#4A90E2"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)

    im = ax.imshow(
        contact_map,
        cmap=cmap,
        norm=norm,
        origin="lower",
        interpolation="none"
    )

    # 网格线，让每个格子更清楚
    ax.set_xticks(np.arange(-0.5, length, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, length, 1), minor=True)
    ax.grid(which="minor", color="#D9D9D9", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Nucleotide Position")
    ax.set_ylabel("Nucleotide Position")

    # 如果序列较短，可以直接显示碱基标签
    if sequence is not None and length <= 40:
        ax.set_xticks(np.arange(length))
        ax.set_yticks(np.arange(length))
        ax.set_xticklabels(list(sequence), fontsize=8)
        ax.set_yticklabels(list(sequence), fontsize=8)
    else:
        ax.set_xticks(np.arange(0, length, max(1, length // 10)))
        ax.set_yticks(np.arange(0, length, max(1, length // 10)))

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    # 示例
    sequence = "GGUGGCCCCGUCGGUCCCUCGCGACGCUAGAUCGAAAAUCCCGCCAGGGCCGGAAGGCAGCAACGGUAUCGAUUGAUGCGGGCGCCGAGGUCAACCGGCGGGGGCACCACCC"
    dot_bracket = "((((.((((((((((.((((((..(((..((((((.....(((.....(((....))).....)))..))))))...))).)...)))))...)))))))))).))))...."

    contact_map = dot_bracket_to_contact_map(sequence, dot_bracket)

    print("Contact map matrix:")
    print(contact_map)

    # 显示热图
    plot_contact_map(
        contact_map,
        sequence=sequence,
        title="RNA Contact Map",
        save_path="rna_contact_map.png",
        show=True
    )