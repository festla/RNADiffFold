# -*- coding: utf-8 -*-
import collections
import os
import pdb
import pickle as cPickle
from os.path import join
from random import shuffle
from torch.utils import data
from itertools import product
from typing import List, Tuple
from common.data_utils import *


perm = list(product(np.arange(4), np.arange(4)))
perm2 = [[1, 3], [3, 1]]
perm_nc = [[0, 0], [0, 2], [0, 3], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 3]]


def make_dataset(
        directory: str
) -> List[str]:
    instances = []                                         # 初始化一个空列表，用来收集符合条件的文件的“完整路径”
    directory = os.path.expanduser(directory)              # 把路径里的 ~ 展开成用户主目录，如 "~/.data" -> "/home/user/.data"
    for root, _, fnames in sorted(os.walk(directory)):
        for fname in sorted(fnames):    # 遍历当前目录下的文件名列表，同样用 sorted 保证顺序稳定；上面是按字典序排序root;
            if fname.endswith('.cPickle') or fname.endswith('.Pickle'):
                path = os.path.join(root, fname)    # 把目录 root 和文件名 fname 拼成该文件的“绝对/规范路径”
                instances.append(path)              # 收集进结果列表

    return instances


class ParserData(object):
    def __init__(self, path):
        self.path = path
        self.data = self.load_data(self.path)
        self.len = len(self.data)
        self.seq_max_len = max([x.seq_raw for x in self.data])
        self.set_max_len = (self.seq_max_len // 80 + int(self.seq_max_len % 80 != 0)) * 80

    def load_data(self, path):
        RNA_SS_data = collections.namedtuple('RNA_SS_data', 'contact data_fcn_2 seq_raw length name')
        with open(path, 'rb') as f:
            load_data = cPickle.load(f)
        return load_data

    def padding(self, data_array, maxlen):
        a, b = data_array.shape
        return np.pad(data_array, ((0, maxlen - a), (0, 0)), 'constant')

    def pairs2map(self, pairs, seq_len):
        contact = np.zeros([seq_len, seq_len])
        for pair in pairs:
            contact[pair[0], pair[1]] = 1
        return contact

    def preprocess_data(self):
        shuffle(self.data)
        contact_list = [item.name for item in self.data]
        data_fcn_2_list = [item.contact for item in self.data]
        data_seq_raw_list = [item.data_fcn_2 for item in self.data]
        data_length_list = [item.seq_raw for item in self.data]    # 这是一个存储数据集中所有样本真实长度的列表
        data_name_list = [item.length for item in self.data]
    
        T = max(data_length_list)  # 或者更省内存：T = max(data_length_list)

        def pairs2map_padded(self, pairs, L, T):
            M = self.pairs2map(pairs, L)
            if L < T:
                M = np.pad(M, ((0, T-L), (0, T-L)), 'constant')
            return M

        def pad_L4(a, T):
            out = np.zeros((T,4), dtype=np.float32)
            out[:a.shape[0]] = a.astype(np.float32)
            return out

        contact_array    = np.stack([pairs2map_padded(self, pairs=p, L=L, T=T) for p, L in zip(contact_list, data_length_list)], axis=0)   # (10789,1,L,2)->(10789,1,T,T)
        data_fcn_2_array = np.stack([pad_L4(f, T) for f in data_fcn_2_list], axis=0)  


        data_seq_encode_list = list(map(lambda x: seq_encoding(x), data_seq_raw_list))    # 这一步就是把碱基字符串转为one-hot编码
        data_seq_encode_pad_list = list(map(lambda x: self.padding(x, T), data_seq_encode_list))
        data_seq_encode_pad_array = np.stack(data_seq_encode_pad_list, axis=0)
        '''
        print(f"contact_array.shape:{contact_array.shape}")             # contact_array.shape:(10794, 498, 498)堆叠在一起的contact
        print(f"data_fcn_2_array.shape:{data_fcn_2_array.shape}")       # data_fcn_2_array.shape:(10794, 498, 4)堆叠在一起的one-hot
        print(f"data_seq_raw_list.len:{len(data_seq_raw_list)}")        # data_seq_raw_list.len:10794 序列个数（字符串列表）
        print(f"data_seq_raw_list[0]:{data_seq_raw_list[0]}")           # data_seq_raw_list[0]:GCAUAAAAAAAGCCACGGUUCUCACCGUGGCAAAAUCCAACAUAGCUAAAUUAAAAAUAAUCAGGAGGGCUGCCCGCCG
        print(f"data_seq_raw_list[0].len:{len(data_seq_raw_list[0])}")  # data_seq_raw_list[0].len:79 第一个样本序列长度
        print(f"data_length_list.len:{len(data_length_list)}")          # data_length_list.len:10794  所有样本长度的列表
        print(f"data_length_list[0]:{data_length_list[0]}")             # data_length_list[0]:79      样本长度的列表的第一个元素
        print(f"data_seq_encode_list.len: {len(data_seq_encode_list)}") # 10794
        print(f"data_seq_encode_list[0]: {data_seq_encode_list[0]}")
        print(f"data_seq_encode_list[0].len: {len(data_seq_encode_list[0])}")    # data_seq_encode_list[0]: [[0 0 0 1] [0 0 1 0] ... [0 0 0 1]]
        print(f"data_fcn_2_array[0]: {data_fcn_2_array[0]}")            # data_fcn_2_array[0]: [[0. 0. 0. 1.]  [0. 0. 1. 0.] ... [0. 0. 0. 0.]]这里是pad了
        print(f"data_fcn_2_array[0].len: {len(data_fcn_2_array[0])}")
        #print(f"data_seq_encode_pad_list[0]: {data_seq_encode_pad_list[0]}")
        #print(f"data_seq_encode_pad_array[0]: {data_seq_encode_pad_array[0]}")
        import pdb;pdb.set_trace()'''
        return contact_array, data_fcn_2_array, data_seq_raw_list, data_length_list, data_name_list, T, data_seq_encode_pad_array


class Dataset(data.Dataset):

    def __init__(
            self,
            data_root: List[str],
            upsampling: bool = False
    ) -> None:
        self.data_root = data_root
        self.upsampling = upsampling
        if len(self.data_root) == 1:
            samples = self.make_dataset(self.data_root[0])
        elif len(self.data_root) > 1:
            samples = []
            for root in self.data_root:
                samples += self.make_dataset(root)
        else:
            raise ValueError('data_root is empty')

        self.samples = samples    # 这里的samples就是文件目录的字符串列表

        self.index = []        # [(file_idx, local_idx, length),...]
        self.sample_sizes = [] # 每个文件的样本个数（大小）
        # print(f"samples.size: {len(samples)}") samples.size: 1
        for fi, p in enumerate(self.samples):
            with open(p, 'rb') as f:
                load_data = cPickle.load(f)
            lens = [item.seq_raw for item in load_data]
            self.sample_sizes.append(len(load_data))
            # print(f"self.sample_sizes: {self.sample_sizes}") self.sample_sizes: [10794]
            # import pdb;pdb.set_trace()
            for li, L in enumerate(lens):
                self.index.append((fi, li, int(L)))

        if self.upsampling:
            self.upsampling_data()
        
        self._cache_file_idx = None
        self._cache_data_list = None

    @staticmethod
    def make_dataset(
            directory: str
    ) -> List[str]:
        return make_dataset(directory)

    # for data balance, 4 times for 160~320 & 320~640
    def upsampling_data(self):
        import random
        new_index = []
        for fi, li, L in self.index:
            k = 4 if L in (320, 640) else 1
            new_index.extend([(fi, li, L)] * k)
        random.shuffle(new_index)
        self.index = new_index

    def __len__(self) -> int:
        'Denotes the total number of samples'
        return len(self.index)

    def __getitem__(self, index: int):
        fi, li, L = self.index[index]

        if self._cache_file_idx != fi:
            with open(self.samples[fi], 'rb') as f:
                self._cache_data_list = cPickle.load(f)
            self._cache_file_idx = fi

        data_list = self._cache_data_list

        item = data_list[li]
        length = item.seq_raw
        contact = item.name
        data_fcn_2 = item.contact
        seq_raw = item.data_fcn_2
        name = item.length

        '''print(f"length: {length}")
        print(f"contact: {contact}")
        print(f"data_fcn_2: {data_fcn_2}")
        print(f"seq_raw: {seq_raw}")
        print(f"name: {name}")
        pdb.set_trace()'''
        '''batch_data_path = self.samples[index]
        batch_data = ParserData(batch_data_path)

        contact_array, data_fcn_2_array, data_seq_raw_list, data_length_list, data_name_list, set_max_len, \
        data_seq_encode_pad_array = batch_data.preprocess_data()

        contact = torch.tensor(contact_array).unsqueeze(1).long()
        data_fcn_2 = torch.tensor(data_fcn_2_array).float()
        data_length = torch.tensor(data_length_list).long()
        data_seq_encode_pad = torch.tensor(data_seq_encode_pad_array).float()'''

        return contact, data_fcn_2, seq_raw, length, name



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

def diff_collate_fn(batch, alphabet):
    """
    batch: List[tuple]，每个元素对应 __getitem__ 的返回：
           (contact_pairs, data_fcn_2_onehot[L,4], seq_raw_str, length_int, name)
    返回：
      contact: torch.bool   [B,1,T,T]
      data_fcn_2: torch.f32 [B,T,4]
      tokens: torch.int64   [B, T'(+BOS/EOS)]
      data_length: torch.i64[B]
      data_name: list(len=B)
      set_max_len: int (=T)
      data_seq_encode_pad: torch.f32 [B,T,4]
    """
    contact_pairs_list, fcn2_list_np, seq_list, length_list, name_list = zip(*batch)
    lengths = list(map(int, length_list))
    T = max(lengths)

    contact_dense = []
    for pairs, L in zip(contact_pairs_list, lengths):
        M = pairs2map(pairs, L)
        if L<T:
            M = padding(pairs, T)
        
        contact_dense.append(torch.from_numpy(M))
    contact = torch.stack(contact_dense, dim=0).unsqueeze(1).bool()    # [B, 1, T, T]


'''
def diff_collate_fn(batch, alphabet):
    contact, data_fcn_2, data_seq_raw_list, data_length, data_name_list, set_max_len, data_seq_encode_pad = zip(*batch)
    if len(contact) == 1:
        contact = contact[0]
        data_fcn_2 = data_fcn_2[0]
        data_seq_raw = data_seq_raw_list[0]
        data_length = data_length[0]
        data_name = data_name_list[0]
        set_max_len = set_max_len[0]
        data_seq_encode_pad = data_seq_encode_pad[0]

    else:
        set_max_len = max(set_max_len) if isinstance(set_max_len, tuple) else set_max_len

        contact_list = list()
        for item in contact:
            if item.shape[-1] < set_max_len:
                item = F.pad(item, (0, set_max_len - item.shape[-1], 0, set_max_len - item.shape[-1]), 'constant', 0)
                contact_list.append(item)
            else:
                contact_list.append(item)

        data_fcn_2_list = list()
        for item in data_fcn_2:
            if item.shape[-1] < set_max_len:
                item = F.pad(item, (0, set_max_len - item.shape[-1], 0, set_max_len - item.shape[-1]), 'constant', 0)
                data_fcn_2_list.append(item)
            else:
                data_fcn_2_list.append(item)

        data_seq_encode_pad_list = list()
        for item in data_seq_encode_pad:
            if item.shape[-1] < set_max_len:
                item = F.pad(item, (0, set_max_len - item.shape[-1], 0, set_max_len - item.shape[-1]), 'constant', 0)
                data_seq_encode_pad_list.append(item)
            else:
                data_seq_encode_pad_list.append(item)

        contact = torch.cat(contact_list, dim=0)
        data_fcn_2 = torch.cat(data_fcn_2_list, dim=0)
        data_seq_encode_pad = torch.cat(data_seq_encode_pad_list, dim=0)

        data_seq_raw = list()
        for item in data_seq_raw_list:
            data_seq_raw.extend(item)

        data_length = torch.cat(data_length, dim=0)

        data_name = list()
        for item in data_name_list:
            data_name.extend(item)

    tokens = generate_token_batch(alphabet, data_seq_raw)

    return contact, data_fcn_2, tokens, data_length, data_name, set_max_len, data_seq_encode_pad
'''

def padding(data_array, maxlen):
    a, b = data_array.shape
    return np.pad(data_array, ((0, maxlen - a), (0, 0)), 'constant')


def pairs2map(pairs, seq_len):
    contact = np.zeros([seq_len, seq_len])
    for pair in pairs:
        contact[pair[0], pair[1]] = 1
    return contact


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

# 这个函数就是在构造 Pairing features, 也就是 Unet 的输入的 17 通道的第 17 个通道
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

def build_17ch_from_L4(data_fcn_2_L4) -> torch.Tensor:
    """
    输入：单样本的 one-hot，形状 [L,4]（可为 list/np.ndarray/torch.Tensor）
    返回：torch.float32，形状 [17, L, L]
      - 前 16 通道：克罗内科外积（4x4 组合）
      - 第 17 通道：creatmat(data) 的结果
    """
    # 统一为 numpy float32 [L,4]
    if isinstance(data_fcn_2_L4, torch.Tensor):
        x = data_fcn_2_L4.detach().cpu().numpy()
    else:
        x = np.asarray(data_fcn_2_L4)
    x = x.astype(np.float32, copy=False)
    assert x.ndim == 2 and x.shape[1] == 4, f"[L,4] expected, got {x.shape}"
    L = x.shape[0]

    # 16 通道外积：[4,L] -> [4,4,L,L] -> [16,L,L]
    xT = torch.from_numpy(x.T.copy())                              # [4,L]
    pair16 = (xT[:, None, :, None] * xT[None, :, None, :])         # [4,4,L,L]
    pair16 = pair16.reshape(16, L, L).to(torch.float32)            # [16,L,L]

    # 第 17 通道：creatmat（用你已有的函数）
    mat17 = torch.from_numpy(creatmat(x).astype(np.float32)).unsqueeze(0)  # [1,L,L]

    return torch.cat([pair16, mat17], dim=0)                       # [17,L,L]