# -*- coding: utf-8 -*-
import time

T0 = time.perf_counter()

def log_time(msg):
    print(f"[TIME] {msg}: {time.perf_counter() - T0:.2f}s", flush=True)

log_time("enter train.py")

import torch
log_time("import torch done")

import argparse
import collections
log_time("import argparse / collections done")

from common.utils import add_parent_path, set_seeds
log_time("import common.utils done")

add_parent_path(level=1)
log_time("add_parent_path done")

from models.model import get_model, get_model_id, add_model_args
log_time("import models.model done")

from optim.multistep import get_optim, get_optim_id, add_optim_args, LinearWarmupScheduler
log_time("import optim.multistep done")

from datasets.data import get_data_id, add_data_args, get_data
log_time("import datasets.data done")

from experiment import Experiment, add_exp_args
log_time("import experiment done")

from torch.optim.lr_scheduler import MultiStepLR
log_time("import MultiStepLR done")


# ================== Setup ==================
log_time("before argparse setup")

parser = argparse.ArgumentParser()

add_model_args(parser)
log_time("add_model_args done")

add_data_args(parser)
log_time("add_data_args done")

add_optim_args(parser)
log_time("add_optim_args done")

add_exp_args(parser)
log_time("add_exp_args done")

args = parser.parse_args()
log_time("args parsed")

set_seeds(args.seed)
log_time("set_seeds done")

print("[ARGS]", args, flush=True)


# ================== model ==================
log_time("before get_model_id")
model_id = get_model_id(args)
log_time("after get_model_id")

log_time("before get_model")
model, alphabet = get_model(args)
log_time("after get_model")


# ================== data ==================
log_time("before get_data_id")
data_id = get_data_id(args)
log_time("after get_data_id")

RNA_SS_data = collections.namedtuple(
    'RNA_SS_data',
    'contact data_fcn_2 seq_raw length name'
)

log_time("before get_data")
train_loader, val_loader, test_loader = get_data(args, alphabet)
log_time("after get_data")


# ================== Finetune: 加载预训练权重 ==================
if args.finetune:
    assert args.pretrained_ckpt is not None, "使用 --finetune 时必须指定 --pretrained_ckpt"
    print(f"[Finetune] 从预训练 checkpoint 加载参数: {args.pretrained_ckpt}", flush=True)

    log_time("before finetune torch.load")
    ckpt = torch.load(args.pretrained_ckpt, map_location=args.device)
    log_time("after finetune torch.load")

    # 兼容几种不同格式的 ckpt
    log_time("before parse checkpoint state_dict")
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt
    log_time("after parse checkpoint state_dict")

    log_time("before model.load_state_dict")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    log_time("after model.load_state_dict")

    print(f"[Finetune] load_state_dict: missing={missing}, unexpected={unexpected}", flush=True)

    # ============= 阶段一：只调 head（+ 可选 mask token） ============
    if getattr(args, "head_only", False):
        print("[Finetune] Head-only 模式：只训练 contact_head / lm_head (+ mask_token 可选)", flush=True)

        log_time("before freeze all params")
        # 1) 先冻结所有参数
        for name, p in model.named_parameters():
            p.requires_grad = False
        log_time("after freeze all params")

        log_time("before unfreeze head params")
        # 2) 再只解冻我们关心的 head / mask token
        for name, p in model.named_parameters():
            # contact / lm head
            if name.startswith("fm_conditioner.contact_head") \
               or name.startswith("fm_conditioner.lm_head"):
                p.requires_grad = True
                print("[Unfreeze HEAD]", name, flush=True)

            # 可选：如果你模型里有 mask_token，可以顺带调一调
            if "mask_token" in name:
                p.requires_grad = True
                print("[Unfreeze MASK]", name, flush=True)
        log_time("after unfreeze head params")


# ================== 统计可训练参数 ==================
log_time("before count trainable params")

num_trainable = 0
for name, p in model.named_parameters():
    if p.requires_grad:
        num_trainable += 1
        # print("[Trainable]", name)

print("Total trainable params:", num_trainable, flush=True)
log_time("after count trainable params")


# ================== optimizer ==================
log_time("before get_optim_id")
optim_id = get_optim_id(args)
log_time("after get_optim_id")


# ====== 情况 A：阶段一 head-only finetune ======
if args.finetune and getattr(args, "head_only", False):
    log_time("enter optimizer branch A: head-only finetune")
    print("[Finetune] 使用 head-only 优化器设置", flush=True)

    # head_only 场景，用一个 lr 就够了：用 head_lr，否则回退到 lr
    head_lr = args.head_lr if getattr(args, "head_lr", None) is not None else args.lr

    log_time("before collect head_params")
    head_params = [p for p in model.parameters() if p.requires_grad]
    log_time("after collect head_params")

    print(f"[Finetune] head_params = {len(head_params)}, head_lr = {head_lr}", flush=True)

    log_time("before AdamW init for head-only")
    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": head_lr, "weight_decay": args.weight_decay}]
    )
    log_time("after AdamW init for head-only")

    # warmup（按 step 调整）
    log_time("before scheduler_iter init")
    if args.warmup is not None:
        scheduler_iter = LinearWarmupScheduler(optimizer, total_epoch=args.warmup)
    else:
        scheduler_iter = None
    log_time("after scheduler_iter init")

    # 多步下降（按 epoch 调整）
    log_time("before scheduler_epoch init")
    if len(args.milestones) > 0:
        scheduler_epoch = MultiStepLR(
            optimizer, milestones=args.milestones, gamma=args.gamma
        )
    else:
        scheduler_epoch = None
    log_time("after scheduler_epoch init")


# ====== 情况 B：全参微调 + 分组学习率（你之前的方案） ======
elif args.finetune and getattr(args, "separate_lr", False):
    log_time("enter optimizer branch B: separate lr finetune")
    print("[Finetune] 使用分组学习率：fm / diffusion / head", flush=True)

    base_lr = args.lr
    fm_lr   = args.fm_lr   if getattr(args, "fm_lr", None)   is not None else base_lr * 0.01
    diff_lr = args.backbone_lr if args.backbone_lr is not None else base_lr * 0.1
    head_lr = args.head_lr if args.head_lr is not None else base_lr

    fm_params   = []
    diff_params = []
    head_params = []

    log_time("before split parameter groups")
    for name, p in model.named_parameters():
        # 已经在 head_only 阶段被手动 freeze 的就跳过
        if not p.requires_grad:
            continue

        # ===== 1) FM 编码器：只放开最后几层（示例：layers.9/10/11），其余冻结 =====
        if name.startswith("fm_conditioner.layers"):
            # 根据你之前打印的层数，这里假设一共有 12 层：0~11
            # 只让 9,10,11 参与训练，其余直接 freeze
            if any(f".layers.{i}." in name for i in [9, 10, 11]):
                fm_params.append(p)   # 用极小 fm_lr
            else:
                p.requires_grad = False   # 完全冻结前面的 FM 层
            continue

        # ===== 2) head：contact_head / lm_head 始终参与训练 =====
        if ("fm_conditioner.contact_head" in name) or ("fm_conditioner.lm_head" in name):
            head_params.append(p)
            continue

        # ===== 3) 其它：denoise_layer / u_conditioner / 其余 fm_cond，归到 diff_params =====
        diff_params.append(p)
    log_time("after split parameter groups")

    print(f"[Finetune] fm={len(fm_params)}, diff={len(diff_params)}, head={len(head_params)}", flush=True)
    print(f"[Finetune] fm_lr={fm_lr}, diff_lr={diff_lr}, head_lr={head_lr}", flush=True)

    log_time("before AdamW init for separate lr")
    optimizer = torch.optim.AdamW(
        [
            {"params": fm_params,   "lr": fm_lr,   "weight_decay": 0.0},              # 尽量保护 FM
            {"params": diff_params, "lr": diff_lr, "weight_decay": args.weight_decay},
            {"params": head_params, "lr": head_lr, "weight_decay": args.weight_decay},
        ]
    )
    log_time("after AdamW init for separate lr")

    # warmup（按 step 调整）
    log_time("before scheduler_iter init")
    if args.warmup is not None:
        scheduler_iter = LinearWarmupScheduler(optimizer, total_epoch=args.warmup)
    else:
        scheduler_iter = None
    log_time("after scheduler_iter init")

    # 多步下降（按 epoch 调整）
    log_time("before scheduler_epoch init")
    if len(args.milestones) > 0:
        scheduler_epoch = MultiStepLR(
            optimizer, milestones=args.milestones, gamma=args.gamma
        )
    else:
        scheduler_epoch = None
    log_time("after scheduler_epoch init")


# ====== 情况 C：原始训练逻辑（不分组 / 不微调） ======
else:
    log_time("enter optimizer branch C: default get_optim")
    log_time("before get_optim")
    optimizer, scheduler_iter, scheduler_epoch = get_optim(args, model)
    log_time("after get_optim")


# ================== training ==================
log_time("before Experiment init")

exp = Experiment(args=args,
                 data_id=data_id,
                 model_id=model_id,
                 optim_id=optim_id,
                 train_loader=train_loader,
                 val_loader=val_loader,
                 test_loader=test_loader,
                 model=model,
                 optimizer=optimizer,
                 scheduler_iter=scheduler_iter,
                 scheduler_epoch=scheduler_epoch)

log_time("after Experiment init")

log_time("before exp.run")
exp.run()
log_time("after exp.run")