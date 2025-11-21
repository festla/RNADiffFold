# -*- coding: utf-8 -*-
"""
Convert RNA .ct file (columns: index, base, ..., pair_index, ...) 
to sequence + dot-bracket notation.
"""

def parse_ct(ct_path):
    """
    读取 .ct 文件，提取序列和配对信息。
    """
    indices, bases, pairs = [], [], []
    with open(ct_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            cols = line.split()
            if len(cols) < 5:
                continue
            i = int(cols[0])
            base = cols[1].upper()
            pair = int(cols[4])
            indices.append(i)
            bases.append(base)
            pairs.append(pair)
    return indices, bases, pairs


def ct_to_dotbracket(pairs):
    """
    将配对信息（list）转换为 dot-bracket 表示。
    """
    L = len(pairs)
    dot = ['.'] * L
    for i, j in enumerate(pairs, start=1):
        if j > i:
            dot[i - 1] = '('
            dot[j - 1] = ')'
    return ''.join(dot)


def write_dbn(output_path, seq, dot):
    """
    保存为 .dbn 格式：三行 [name] [sequence] [structure]
    """
    with open(output_path, 'w') as f:
        f.write(">converted_from_ct\n")
        f.write(seq + "\n")
        f.write(dot + "\n")
    print(f"[OK] dot-bracket saved to {output_path}")


if __name__ == "__main__":
    ct_file = "/storage/student2/xiao/lql/RNADiffFold/predict_results/ct_files/bpRNA_CRW_54624(6).ct"         # 你的输入文件路径
    output_dbn = "/storage/student2/xiao/lql/RNADiffFold/predict_results/ct_files/6.dbn"     # 输出文件路径

    indices, bases, pairs = parse_ct(ct_file)
    sequence = ''.join(bases)
    dotbracket = ct_to_dotbracket(pairs)

    print("Sequence:\n", sequence)
    print("\nDot-bracket:\n", dotbracket)
    print("\nLength:", len(sequence))

    write_dbn(output_dbn, sequence, dotbracket)
