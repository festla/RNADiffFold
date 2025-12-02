# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader
from .data_generator import Dataset, diff_collate_fn, RNA_Augmentation
from os.path import join
from functools import partial

dataset_choices = ['RNAStrAlign', 'archiveII', 'bpRNA', 'bpRNAnew', 'pdbnew', 'all']

ROOT_PATH = './data'


def add_data_args(parser):
    # Data params
    parser.add_argument('--dataset', type=str, default='bpRNA', choices=dataset_choices)
    parser.add_argument('--seq_len', type=str, default='160', choices={'160', '600', '640', 'all'})
    parser.add_argument('--upsampling', type=eval, default=False)

    # Train params
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=64)
    parser.add_argument('--pin_memory', type=eval, default=False)

    # === 新增：数据增强 & label smoothing 的可选参数 ===
    parser.add_argument('--use_aug', type=eval, default=False, help='whether to use RNA covariation augmentation')
    parser.add_argument('--aug_select', type=float, default=0.5, help='probability of selecting a sample for aug')
    parser.add_argument('--aug_replace', type=float, default=0.2, help='probability of mutating one base pair')
    parser.add_argument('--aug_mode', type=str, default='cov', choices=['cov', 'cg'], help='augmentation mode')
    parser.add_argument('--smooth', type=float, default=0.0, help='label smoothing strength for contact map (0 = off)')


def get_data_id(args):
    return '{}_{}'.format(args.dataset, args.seq_len)


def get_data(args, alphabet):
    assert args.dataset in dataset_choices

    if args.dataset == 'RNAStrAlign':
        train = Dataset([join(ROOT_PATH, args.dataset, 'train')], upsampling=True)
        val = Dataset([join(ROOT_PATH, args.dataset, 'val')])
        test = Dataset([join(ROOT_PATH, args.dataset, 'test'),
                        join(ROOT_PATH,'archiveII')])

    elif args.dataset == 'bpRNA':
        train = Dataset([join(ROOT_PATH, args.dataset, 'TR0')], upsampling=True)
        val = Dataset([join(ROOT_PATH, args.dataset, 'VL0')])
        test = Dataset([join(ROOT_PATH, args.dataset, 'TS0'),
                        join(ROOT_PATH, 'bpRNAnew')])

    elif args.dataset == 'bpRNAnew':
        train = Dataset([join(ROOT_PATH, args.dataset, 'mutate')], upsampling=True)
        val = Dataset([join(ROOT_PATH, 'bpRNA', 'VL0')])
        test = Dataset([join(ROOT_PATH, args.dataset, 'bpRNAnew')])
    elif args.dataset == 'pdbnew':
        train = Dataset([join(ROOT_PATH, args.dataset, 'TR1')], upsampling=True)
        val = Dataset([join(ROOT_PATH, args.dataset, 'VL1')])
        test = Dataset([join(ROOT_PATH, args.dataset, 'TS1'),
                        join(ROOT_PATH, args.dataset, 'TS2'),
                        join(ROOT_PATH, args.dataset, 'TS3')
                        ])

    elif args.dataset == 'all':
        train = Dataset([join(ROOT_PATH, 'RNAStrAlign/train'),
                         join(ROOT_PATH, 'bpRNA/TR0/'),
                         join(ROOT_PATH, 'bpRNAnew/mutate')], upsampling=True)
        val = Dataset([join(ROOT_PATH, 'bpRNA/VL0/'),
                       join(ROOT_PATH, 'RNAStrAlign/val')])
        test = Dataset([join(ROOT_PATH, 'bpRNA/TS0/'),
                        join(ROOT_PATH, 'RNAStrAlign/test'),
                        join(ROOT_PATH, 'bpRNAnew/bpRNAnew')])

    else:
        raise NotImplementedError

    '''
    partial(diff_collate_fn, alphabet=alphabet) 的作用是：把 diff_collate_fn(batch, alphabet) 这个二参函数，
    包装成只需要一个参数 batch 的新函数，并把 alphabet 这个实参预先固定住。
    也就是做了“柯里化 / 预填参数”。等 DataLoader 调用时，它只会传入 batch, alphabet 会用你这里固定好的那个
    '''
    # ====== 关键新逻辑：构造增强器 & smoothing 参数 ======
    # label smoothing：<=0 当成不用
    smooth = None if (not hasattr(args, 'smooth') or args.smooth <= 0.0) else args.smooth

    # 训练阶段的数据增强（只对 train_loader 生效）
    if hasattr(args, 'use_aug') and bool(args.use_aug):
        train_aug = RNA_Augmentation(
            select=getattr(args, 'aug_select', 0.5),
            replace=getattr(args, 'aug_replace', 0.2),
            seed=42,
            mode=getattr(args, 'aug_mode', 'cov')
        )
    else:
        train_aug = None

    # === collate_fn 定义 ===
    # 训练用：可以带 aug / smooth
    train_collate_fn = partial(
        diff_collate_fn,
        alphabet=alphabet,
        aug=train_aug,
        smooth=smooth
    )

    # 验证 / 测试：默认不增强、不 smoothing（跟以前完全一致）
    eval_collate_fn = partial(
        diff_collate_fn,
        alphabet=alphabet
    )

    train_loader = DataLoader(train,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              persistent_workers=True,
                              collate_fn=train_collate_fn,
                              pin_memory=args.pin_memory,
                              drop_last=True)

    val_loader = DataLoader(val,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            persistent_workers=True,
                            collate_fn=eval_collate_fn,
                            pin_memory=args.pin_memory,
                            drop_last=False)

    test_loader = DataLoader(test,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             persistent_workers=True,
                             collate_fn=eval_collate_fn,
                             pin_memory=args.pin_memory,
                             drop_last=False)

    return train_loader, val_loader, test_loader
