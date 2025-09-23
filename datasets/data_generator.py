# -*- coding: utf-8 -*-
import collections
import os
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
        '''for x in self.data:
            print(f"data_fcn_2: {x.contact}")
            print(f"seq_raw: {x.data_fcn_2}")
            print(f"length: {x.seq_raw}")
            print(f"name: {x.length}")
            print(f"contact: {x.name}")
            import pdb; pdb.set_trace()'''
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
        # print(data_fcn_2_list[0].shape)   (90, 4)
        # import pdb;pdb.set_trace()
        contact_array    = np.stack([pairs2map_padded(self, pairs=p, L=L, T=T) for p, L in zip(contact_list, data_length_list)], axis=0)   # (10789,1,L,2)->(10789,1,T,T)
        # print(contact_array.shape)    (10794, 498, 498)
        # import pdb;pdb.set_trace()
        data_fcn_2_array = np.stack([pad_L4(f, T) for f in data_fcn_2_list], axis=0)  
        # print(data_fcn_2_array.shape)    # (10794, T=498, 4)
        # import pdb;pdb.set_trace()

        data_seq_encode_list = list(map(lambda x: seq_encoding(x), data_seq_raw_list))    # 这一步就是把碱基字符串转为one-hot编码
        data_seq_encode_pad_list = list(map(lambda x: self.padding(x, T), data_seq_encode_list))
        data_seq_encode_pad_array = np.stack(data_seq_encode_pad_list, axis=0)

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

        self.samples = samples
        if self.upsampling:
            self.samples = self.upsampling_data()
        
        self.file_paths = self.samples
        # 加一个扁平索引
        self.index = []
        for p in self.file_paths:
            with open(p, 'rb') as f:
                arr = cPickle.load(f)
            for i in range(len(arr)):
                self.index.append((p, i))
        
        self._cache_path = None
        self._cache_data = None

    @staticmethod    # 逻辑上归属于类，但不依赖实例和类本身的函数。
    def make_dataset(
            directory: str
    ) -> List[str]:
        return make_dataset(directory)

    # for data balance, 4 times for 160~320 & 320~640
    def upsampling_data(self):
        RNA_SS_data = collections.namedtuple('RNA_SS_data', ' contact data_fcn_2 seq_raw length name')
        augment_data_list = list()
        final_data_list = self.samples
        for data_path in final_data_list:
            with open(data_path, 'rb') as f:
                load_data = cPickle.load(f)
            max_len = max([x.seq_raw for x in load_data])
            if max_len == 160:
                continue
            elif max_len == 320:
                augment_data_list.append(data_path)
            elif max_len == 640:
                augment_data_list.append(data_path)

        augment_data_list = list(np.random.choice(augment_data_list, 3 * len(augment_data_list)))
        final_data_list.extend(augment_data_list)
        shuffle(final_data_list)
        return final_data_list

    def __len__(self) -> int:
        'Denotes the total number of samples'
        # return len(self.samples)
        return len(self.index)

    # 增加读缓存
    def _load_file(self, path):
        if self._cache_path != path:
            with open(path, 'rb') as f:
                self._cache_data = cPickle.load(f)
            self._cache_path = path
        return self._cache_data

    def __getitem__(self, index: int):
        try:
            path, inner_idx = self.index[index]
            data_list = self._load_file(path)
            item = data_list[inner_idx]
            # 字段含义对齐你之前的注释：
            # item.name       -> contact 的配对列表 (pairs)
            # item.contact    -> data_fcn_2 的 (L,4) one-hot
            # item.data_fcn_2 -> 原始序列字符串
            # item.seq_raw    -> 序列长度 L
            # item.length     -> 名字
            L = int(item.seq_raw)
            contact = item.name
            data_fcn_2 = np.asarray(item.contact, dtype=np.float32)  # (L,4)
            seq_raw = item.data_fcn_2
            name = item.length
            #print(f"contact_before(L*L): {contact}")
            # contact -> (1, L, L)
            M = pairs2map(contact, L)  # 用下面的 pairs2map 实现
            contact = torch.from_numpy(M).unsqueeze(0).long()
            #torch.set_printoptions(profile="full")   
            #print(f"contact_after(L*L): {contact}")
            #import pdb;pdb.set_trace()
            
            data_fcn_2 = torch.from_numpy(data_fcn_2).float()    # data_fcn_2 保持 (L,4)，pad 交给 collate_fn
            # 额外返回一个 seq_encoding 版（也是 (L,4)），供你后续需要
            seq_enc = torch.from_numpy(seq_encoding(seq_raw)).float()
            data_length = 

            # 注意：按你原来的接口，这里把字符串和名字放到 list，便于 collate_fn extend
            return contact, data_fcn_2, seq_raw, data_length, name, L, seq_enc

        except Exception as e:
            print(f"Error in __getitem__ at index {index}: {e}")
            raise

    '''
    def __getitem__(self, index: int):
        try:
            batch_data_path = self.samples[index]
            batch_data = ParserData(batch_data_path)

            contact_array, data_fcn_2_array, data_seq_raw_list, data_length_list, data_name_list, set_max_len, data_seq_encode_pad_array = batch_data.preprocess_data()

            contact = torch.tensor(contact_array).unsqueeze(1).long()
            data_fcn_2 = torch.tensor(data_fcn_2_array).float()
            # print(f"contact:{contact}")
            # print(f"contact_shape:{contact.shape}")          # contact_shape:torch.Size([10794, 1, 498, 498])
            # print(f"data_fcn_2:{data_fcn_2}")
            # print(f"data_fcn_2_shape:{data_fcn_2.shape}")    # data_fcn_2_shape:torch.Size([10794, 498, 4])
            # import pdb;pdb.set_trace()
            data_length = torch.tensor(data_length_list).long()

            data_seq_encode_pad = torch.tensor(data_seq_encode_pad_array).float()

            return contact, data_fcn_2, data_seq_raw_list, data_length, data_name_list, set_max_len, data_seq_encode_pad
        except Exception as e:
            print(f"Error in __getitem__ at index {index}: {e}")
            raise
    '''

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
