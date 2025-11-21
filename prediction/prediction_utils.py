# -*- coding: utf-8 -*-
import os
from os.path import join
import munch
import random
import numpy as np
import pandas as pd
import torch
import math
from typing import List, Sequence, Tuple
import collections
from itertools import product
from collections import defaultdict
from models.model import DiffusionRNA2dPrediction
import json

seq_to_onehot_dict = {
    'A': np.array([1, 0, 0, 0]),
    'U': np.array([0, 1, 0, 0]),  # T or U
    'C': np.array([0, 0, 1, 0]),
    'G': np.array([0, 0, 0, 1]),
    'N': np.array([0, 0, 0, 0]),

    'R': np.array([1, 0, 0, 1]),
    'Y': np.array([0, 1, 1, 0]),
    'K': np.array([0, 1, 0, 1]),
    'M': np.array([1, 0, 1, 0]),
    'S': np.array([0, 0, 1, 1]),
    'W': np.array([1, 1, 0, 0]),
    'B': np.array([0, 1, 1, 1]),
    'D': np.array([1, 1, 0, 1]),
    'H': np.array([1, 1, 1, 0]),
    'V': np.array([1, 0, 1, 1]),
    '_': np.array([0, 0, 0, 0]),
    '~': np.array([0, 0, 0, 0]),
    '.': np.array([0, 0, 0, 0]),
    'P': np.array([0, 0, 0, 0]),
    'I': np.array([0, 0, 0, 0]),
    'X': np.array([0, 0, 0, 0])
}


char_dict = {
    0: 'A',
    1: 'U',
    2: 'C',
    3: 'G'
}


PARENTHESES = [
    ("(", ")"),
    ("[", "]"),
    ("<", ">"),
    ("{", "}")
]


def process_config(jsonfile):
    with open(jsonfile, 'r') as f:
        config_dict = json.load(f)
    config = munch.Munch(config_dict)
    config.model = munch.Munch(config.model)
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_fasta(fasta_file_path):
    sequences = dict()
    current_id = None
    current_seq = []
    with open(fasta_file_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)
                current_id = line.strip()[1:]
                current_seq = []
            else:
                current_seq.append(line.strip().upper())

        if current_id is not None:
            sequences[current_id] = ''.join(current_seq)

    return sequences

import numpy as np
import pandas as pd
from typing import Iterable, Tuple, Set, List, Union

def ct_to_dotbracket(
    ct: Union[pd.DataFrame, str],
    *,
    enforce_mutual: bool = True,
    enforce_noncrossing: bool = True
) -> str:
    """
    将 CT 表（或文件路径）转换为 dot-bracket 字符串。
    
    参数：
        ct: pandas DataFrame（包含列 'pair_index'）或 CT 文件路径（无表头，tab 分隔）
        enforce_mutual: 仅保留互相指向一致的配对（推荐 True）
        enforce_noncrossing: 使用 Nussinov 在允许配对集合上求无伪结的最大匹配（推荐 True）
        
    返回：
        dot-bracket 字符串（长度 = N）
    """
    # 1) 读取 CT
    if isinstance(ct, str):
        df = pd.read_csv(ct, sep=r'\s+|\t', header=None, engine='python')
        # 兼容常见 6 列 CT：idx, base, i-1, i+1, pair, n
        if df.shape[1] < 5:
            raise ValueError("CT 文件列数不足，至少应含第5列 pair_index。")
        pair_col = df.iloc[:, 4].to_numpy()  # 1-based partner, 0 if unpaired
    else:
        df = ct
        if 'pair_index' not in df.columns:
            raise ValueError("DataFrame 缺少列 'pair_index'。")
        pair_col = df['pair_index'].to_numpy()

    N = len(pair_col)
    # 将 1-based 的配对索引转换为 0-based；未配对（0） 仍为 -1
    partner = np.array([p-1 if p > 0 else -1 for p in pair_col], dtype=int)

    # 2) 仅保留互配（可选）
    if enforce_mutual:
        for i in range(N):
            j = partner[i]
            if j < 0: 
                continue
            if j >= N or partner[j] != i:
                partner[i] = -1
        # 再次清理不对称残留（对称置空）
        for i in range(N):
            j = partner[i]
            if j >= 0 and partner[j] != i:
                partner[i] = -1

    # 生成允许配对集合（i<j）
    allowed_pairs: Set[Tuple[int, int]] = set()
    for i in range(N):
        j = partner[i]
        if j > i:
            allowed_pairs.add((i, j))

    # 如果不强制无伪结，直接根据互配输出括号（可能不是合法括号序列）
    if not enforce_noncrossing:
        out = ['.'] * N
        for i, j in allowed_pairs:
            out[i] = '('
            out[j] = ')'
        return ''.join(out)

    # 3) Nussinov 动态规划：只允许 allowed_pairs 的配对，求最大无交叉匹配
    # S[i][j] = 区间 [i, j] 中最多可配对数
    S = np.zeros((N, N), dtype=int)

    # 预先做一个快速查询矩阵 allowed[i][j]
    allowed = np.zeros((N, N), dtype=np.int8)
    for i, j in allowed_pairs:
        allowed[i, j] = 1

    # 递推（按区间长度增加）
    for length in range(1, N):  # length = j - i
        for i in range(0, N - length):
            j = i + length
            best = S[i+1, j] if i+1 <= j else 0   # i 不配
            best = max(best, S[i, j-1] if i <= j-1 else 0)  # j 不配
            # i 与 j 配对（若允许）
            if allowed[i, j]:
                val = (S[i+1, j-1] if i+1 <= j-1 else 0) + 1
                best = max(best, val)
            # 枚举分割点
            for k in range(i, j):
                cand = S[i, k] + S[k+1, j]
                if cand > best:
                    best = cand
            S[i, j] = best

    # 回溯获取选中的配对
    selected: Set[Tuple[int, int]] = set()
    def traceback(i: int, j: int):
        if i >= j:
            return
        if S[i, j] == (S[i+1, j] if i+1 <= j else 0):
            traceback(i+1, j)
            return
        if S[i, j] == (S[i, j-1] if i <= j-1 else 0):
            traceback(i, j-1)
            return
        if allowed[i, j] and S[i, j] == ((S[i+1, j-1] if i+1 <= j-1 else 0) + 1):
            selected.add((i, j))
            traceback(i+1, j-1)
            return
        # 分割
        for k in range(i, j):
            if S[i, j] == S[i, k] + S[k+1, j]:
                traceback(i, k)
                traceback(k+1, j)
                return

    traceback(0, N-1)

    # 4) 生成 dot-bracket
    out = ['.'] * N
    for i, j in selected:
        out[i] = '('
        out[j] = ')'
    return ''.join(out)


def encoding2seq(arr):
    seq = list()
    for arr_row in list(arr):
        if sum(arr_row) == 0:
            seq.append('N')   # replace '.' to 'N'
        else:
            seq.append(char_dict[np.argmax(arr_row)])
    return ''.join(seq)


def seq2encoding(seq):
    encoding = list()
    for char in seq:
        encoding.append(seq_to_onehot_dict[char])
    return np.array(encoding)


def contact2ct(contact, seq, seq_len):
    contact = contact[:seq_len, :seq_len]
    structure = np.where(contact)
    pair_dict = dict()
    for i in range(seq_len):
        pair_dict[i] = -1
    for i in range(len(structure[0])):
        pair_dict[structure[0][i]] = structure[1][i]
    first_col = list(range(1, seq_len + 1))
    second_col = list(seq)
    third_col = list(range(seq_len))
    fourth_col = list(range(2, seq_len + 2))
    fifth_col = [pair_dict[i] + 1 for i in range(seq_len)]
    last_col = list(range(1, seq_len + 1))
    df = pd.DataFrame()
    df['index'] = first_col
    df['base'] = second_col
    df['index-1'] = third_col
    df['index+1'] = fourth_col
    df['pair_index'] = fifth_col
    df['n'] = last_col
    return df


def extract_pseudoknot(pairs):
    pseudo_pairs = list()
    for (i1, j1) in pairs:
        for (i2, j2) in pairs:
            if i1 < i2 < j1 < j2:
                pseudo_pairs.append((i2, j2))
    return pseudo_pairs


def contact2dbn(contact, seq_len):
    contact = contact[:,:seq_len, :seq_len]
    structure = np.where(contact)[1:]
    pairs = list(map(lambda i: (structure[0][i], structure[1][i]), range(len(structure[0]))))
    pairs_0 = [(i, j) for (i, j) in pairs if (j,i) in set(pairs) and i < j]
    pairs_dict = defaultdict(list)
    pk_pairs_1 = extract_pseudoknot(pairs_0)
    pk_pairs_2 = extract_pseudoknot(pk_pairs_1)
    pk_pairs_3 = extract_pseudoknot(pk_pairs_2)
    for index, pairs in enumerate([pairs_0, pk_pairs_1, pk_pairs_2, pk_pairs_3]):
        if len(pairs) != 0:
            pairs_dict[index] = pairs

    dbn = np.array(['.'] * seq_len)
    for index, pairs in pairs_dict.items():
        for (i, j) in pairs:
            dbn[i] = PARENTHESES[index][0]
            dbn[j] = PARENTHESES[index][1]
    dbn = ''.join(dbn)
    return dbn


def ct2dbn(ctfile):
    seq = ''.join(list(ctfile.loc[:, 1])).upper()
    seq_len = len(seq)
    rnadata1 = list(ctfile.loc[:, 0].values)
    rnadata2 = list(ctfile.loc[:, 4].values)
    rna_pairs = list(zip(rnadata1, rnadata2))
    rna_pairs = list(filter(lambda x: x[1] > 0, rna_pairs))
    pairs_0 = (np.array(rna_pairs) - 1).tolist()

    pairs_dict = defaultdict(list)
    pk_pairs_1 = extract_pseudoknot(pairs_0)
    pk_pairs_2 = extract_pseudoknot(pk_pairs_1)
    pk_pairs_3 = extract_pseudoknot(pk_pairs_2)
    for index, pairs in enumerate([pairs_0, pk_pairs_1, pk_pairs_2, pk_pairs_3]):
        if len(pairs) != 0:
            pairs_dict[index] = pairs

    dbn = np.array(['.'] * seq_len)
    for index, pairs in pairs_dict.items():
        for [i, j] in pairs:
            if i < j and [j, i] in pairs:
                dbn[i] = PARENTHESES[index][0]
                dbn[j] = PARENTHESES[index][1]
    dbn = ''.join(dbn)
    return (seq, dbn)


def get_data(file_path, alphabet):
    data_dict = parse_fasta(file_path)
    name_list = list()
    seq_list = list()
    seq_len_list = list()
    for i, (k, v) in enumerate(data_dict.items()):
        name_list.append(k)
        seq_list.append(v)
        seq_len_list.append(len(v))

    seq_max_len = max(seq_len_list)
    set_max_len = (seq_max_len // 80 + int(seq_max_len % 80 != 0)) * 80
    seq_encoding_list = list(map(lambda x: seq2encoding(x), seq_list))
    seq_encoding_pad_list = list(map(lambda x: padding(x,set_max_len), seq_encoding_list))
    data_fcn_2 = list(map(lambda x: get_data_fcn(x[0], x[1], set_max_len),
                          zip(seq_encoding_pad_list, seq_len_list)))

    seq_encoding_pad = torch.tensor(np.stack(seq_encoding_pad_list, axis=0)).float()
    data_fcn_2 = torch.tensor(np.stack(data_fcn_2, axis=0)).float()
    seq_length = torch.tensor(seq_len_list).long()
    tokens = generate_token_batch(alphabet, seq_list)
    return data_fcn_2, tokens, seq_encoding_pad, seq_length, name_list, set_max_len, seq_list, seq_len_list


def get_data_fcn(data_seq, data_length, set_length):
    perm = list(product(np.arange(4), np.arange(4)))
    data_fcn = np.zeros((16, set_length, set_length))
    for n, cord in enumerate(perm):
        i, j = cord
        data_fcn[n, :data_length, :data_length] = np.matmul(
            data_seq[:data_length, i].reshape(-1, 1),
            data_seq[:data_length, j].reshape(1, -1)
        )
    data_fcn_1 = np.zeros((1, set_length, set_length))
    data_fcn_1[0, :data_length, :data_length] = creatmat(data_seq[:data_length, :])
    data_fcn_2 = np.concatenate((data_fcn, data_fcn_1), axis=0)

    return data_fcn_2


def Gaussian(x):
    return math.exp(-0.5 * (x * x))


def paired(x, y):
    if x == [1, 0, 0, 0] and y == [0, 1, 0, 0]:
        return 2
    elif x == [0, 0, 0, 1] and y == [0, 0, 1, 0]:
        return 3
    elif x == [0, 0, 0, 1] and y == [0, 1, 0, 0]:
        return 0.8
    elif x == [0, 1, 0, 0] and y == [1, 0, 0, 0]:
        return 2
    elif x == [0, 0, 1, 0] and y == [0, 0, 0, 1]:
        return 3
    elif x == [0, 1, 0, 0] and y == [0, 0, 0, 1]:
        return 0.8
    else:
        return 0


# 产生RNA二级结构pair probability的算法
def creatmat(data):
    mat = np.zeros([len(data), len(data)])
    for i in range(len(data)):
        for j in range(len(data)):
            coefficient = 0
            for add in range(30):
                if i - add >= 0 and j + add < len(data):
                    score = paired(list(data[i - add]), list(data[j + add]))
                    if score == 0:
                        break
                    else:
                        coefficient = coefficient + score * Gaussian(add)
                else:
                    break
            if coefficient > 0:
                for add in range(1, 30):
                    if i + add < len(data) and j - add >= 0:
                        score = paired(list(data[i + add]), list(data[j - add]))
                        if score == 0:
                            break
                        else:
                            coefficient = coefficient + score * Gaussian(add)
                    else:
                        break
            mat[[i], [j]] = coefficient
    return mat


def generate_token_batch(alphabet, seq_strs):
    batch_size = len(seq_strs)
    max_len = max(len(seq_str) for seq_str in seq_strs)
    tokens = torch.empty(
        (
            batch_size,
            max_len
            + int(alphabet.prepend_bos)
            + int(alphabet.append_eos),
        ),
        dtype=torch.int64,
    )
    tokens.fill_(alphabet.padding_idx)
    for i, seq_str in enumerate(seq_strs):
        if alphabet.prepend_bos:
            tokens[i, 0] = alphabet.cls_idx
        seq = torch.tensor([alphabet.get_idx(s) for s in seq_str], dtype=torch.int64)
        tokens[i, int(alphabet.prepend_bos): len(seq_str) + int(alphabet.prepend_bos), ] = seq
        if alphabet.append_eos:
            tokens[i, len(seq_str) + int(alphabet.prepend_bos)] = alphabet.eos_idx
    return tokens


def padding(data_array, maxlen):
    a, b = data_array.shape
    # np.pad(array, ((before_1,after_1),……,(before_n,after_n),module)
    return np.pad(data_array, ((0, maxlen - a), (0, 0)), 'constant')


def contact_map_masks(data_lens, matrix_rep):
    n_seq = len(data_lens)
    assert matrix_rep.shape[0] == n_seq
    for i in range(n_seq):
        l = int(data_lens[i].cpu().numpy())
        matrix_rep[i, :, :l, :l] = 1
    return matrix_rep


def get_model_prediction(args):
    model = DiffusionRNA2dPrediction(
        num_classes=args.num_classes,
        diffusion_dim=args.diffusion_dim,
        cond_dim=args.cond_dim,
        diffusion_steps=args.diffusion_steps,
        dp_rate=args.dp_rate,
        u_ckpt=args.u_conditioner_ckpt
    )
    alphabet = model.get_alphabet()
    return model, alphabet


def vote4struct(struc_list: List[np.ndarray]) -> np.ndarray:
    """
    Vote for the structure with the most votes.
    Args:
        struc_list: a list of predicted structures.

    Returns:
        The structure with the most votes.
    """
    id_struc_dict = dict()
    vote_dict = collections.defaultdict(int)

    for index, pred in enumerate(struc_list):
        id_loc = pred.argmax(axis=0)
        id_loc = list(id_loc)
        id_loc = ''.join(str(i) for i in id_loc)
        id_struc_dict[(index, id_loc)] = pred
        vote_dict[id_loc] += 1

    vote_id = max(vote_dict, key=vote_dict.get)

    for k, v in id_struc_dict.items():
        if k[1] == vote_id:
            return v


if __name__ == '__main__':
    ROOT_PATH = os.getcwd()
    config = process_config(join(ROOT_PATH, 'config.json'))
    model, alphabet = get_model_prediction(config.model)
    data_fcn_2, tokens, seq_encoding_pad_list, seq_length, name_list, set_max_len = \
        get_data(join(ROOT_PATH + '/predict_data/' + config.predict_data), alphabet)

