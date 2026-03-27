# -*- coding: utf-8 -*-

import time
import os
import sys
from os.path import join
import torch
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
from .eval_utils import parse_config, get_data_test, get_model_test, vote4struct, clean_dict, log_eval_metrics, \
    save_metrics
from common.data_utils import contact_map_masks
from common.loss_utils import rna_evaluation
import collections

def build_prefix_mask_2d(prefix_len: torch.Tensor, Lmax: int, device) -> torch.Tensor:
    B = prefix_len.shape[0]
    idx = torch.arange(Lmax, device=device).view(1, Lmax)  # [1, L]
    vis = (idx < prefix_len.view(B, 1)).float()            # [B, L]
    mask2d = (vis.unsqueeze(2) * vis.unsqueeze(1)).unsqueeze(1)
    return mask2d

def evaluation_prefix_curve(args, eval_model, dataloader, prefix_ratios):
    """
    返回：
        summary: dict,
        curve_df: 每个ratio的平均指标
        detail_df: 每个样本、每个ratio的明细
    """
    eval_model.eval()
    device = args.device

    rows = []    # 逐样本、逐ratio记录

    with torch.no_grad():
        pbar = tqdm(
            dataloader,
            total=len(dataloader),
            desc="Evaluating Prefix Curve",
            unit="batch",
            leave = True,
            file = sys.stdout
        )
        for _, batch in enumerate(pbar):
            try:
                (contact, data_fcn_2, data_seq_raw, data_length, data_name, set_max_len, data_seq_encoding) = batch
            except TypeError as e:
                print(f"Skipping batch due to error: {e}")
                continue
            
            # move to device
            data_fcn_2 = data_fcn_2.to(device)
            data_seq_raw = data_seq_raw.to(device)
            data_seq_encoding = data_seq_encoding.to(device)
            data_length = data_length.to(device)

            B = contact.shape[0]
            Lmax = int(set_max_len)

            matrix_rep = torch.zeros_like(contact)
            length_mask = contact_map_masks(data_length, matrix_rep).to(device)

            gt_contact = contact.float().cpu()

            # 逐ratio推理
            for ratio in prefix_ratios:
                prefix_len = torch.clamp((data_length.float() * float(ratio)).long(), min=1)
                prefix_len = torch.minimum(prefix_len, data_length)

                prefix_mask = build_prefix_mask_2d(prefix_len, Lmax, device=device)
                effective_mask = length_mask * prefix_mask

                pred_x0_copy_dict = {}
                best_pred_list = []

                select_seeds = list(range(args.num_samples))

                for seed_ind in select_seeds:
                    torch.manual_seed(seed_ind)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed_ind)

                    pred_x0, _ = eval_model.sample(
                        B, data_fcn_2, data_seq_raw, set_max_len, effective_mask, data_seq_encoding
                    )
                    pred_x0_copy_dict[seed_ind] = pred_x0
                
                # vote 每个样本
                for i in range(B):
                    pred_i_list = [
                        pred_x0_copy_dict[s][i].squeeze().detach().cpu().numpy()
                        for s in select_seeds
                    ]
                    best_pred_i = torch.tensor(vote4struct(pred_i_list))
                    best_pred_list.append(best_pred_i)

                pred_x0 = torch.stack(best_pred_list,dim=0).float()
                if pred_x0.dim() == 3:
                    pred_x0 = pred_x0.unsqueeze(1)
                pred_x0 = pred_x0.cpu()

                # prefix 裁剪评测
                prefix_mask_cpu = prefix_mask.detach().cpu()

                for i in range(B):
                    name_i = data_name[i] if isinstance(data_name, (list, tuple)) else str(data_name)
                    name_i = str(name_i)
                    len_i = int(data_length[i].item())
                    p_i = int(prefix_len[i].item())

                    pred_i = pred_x0[i].squeeze()
                    gt_i = gt_contact[i].squeeze()
                    m_i = prefix_mask_cpu[i].squeeze()

                    pred_pref = pred_i * m_i
                    gt_pref = gt_i * m_i

                    acc, prec, rec, sens, spec, f1, mcc = rna_evaluation(pred_pref, gt_pref)

                    rows.append({
                        "name": name_i,
                        "length": len_i,
                        "ratio":float(ratio),
                        "prefix_len": p_i,
                        "accuracy": float(acc),
                        "precision": float(prec),
                        "recall": float(rec),
                        "sensitivity": float(sens),
                        "specificity": float(spec),
                        "f1": float(f1),
                        "mcc": float(mcc),
                    })
    
    # 汇总 curve + AUC
    detail_df = pd.DataFrame(rows)

    # 每个ratio下对样本均值
    metric_cols = ["f1", "precision", "recall", "sensitivity", "specificity", "accuracy", "mcc"]
    curve_df = (
        detail_df.groupby("ratio")[metric_cols]
        .mean()
        .reset_index()
        .sort_values("ratio")
    )

    # AUC (F1-curve)
    auc_f1 = float(np.trapz(curve_df["f1"].values, curve_df["ratio"].values))

    summary = {
        "auc_f1": auc_f1,
    }
    if(curve_df["ratio"] == 1.0).any():
        summary["f1@1.0"] = float(curve_df.loc[curve_df["ratio"] == 1.0, "f1"].values[0])

    return summary, curve_df, detail_df






if __name__ == "__main__":
    start = time.time()
    # config
    config = parse_config('config.json')
    torch.manual_seed(config.seed)
    print('#'*10, f'Start prefix_evaluate {config.data.dataset}', '#'*10)
    save_root_path = config.save_root_path    # 这里的保存路径我之前都没设置，导致一直在更新
    name = f'{config.project_name}.round_{config.round}.dataset_{config.data.dataset}.num_sample_{config.num_samples}'
    save_path = join(config.save_root_path, 'results', f'dataset_{config.data.dataset}', f'round_{config.round}')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    model, alphabet = get_model_test(config.model)    # 这里先只负责搭起模型结构，还没加载权重
    RNA_SS_data = collections.namedtuple('RNA_SS_data', 'contact data_fcn_2 seq_raw length name')
    test_loader = get_data_test(config.data, alphabet)    # 构建 test 的 DataLoader

    # model load checkpoint
    print(f"Load model checkpoint from: {config.model_ckpt_path}")
    checkpoint = torch.load(config.model_ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(config.device)

    # config_dict = clean_dict(vars(config.toDict()), keys=no_log_keys)

    if not config.dry_run:
        wandb.init(project=config.project_name, name=name, config=config.toDict(), dir=save_path)

    prefix_ratios = [0.2, 0.4, 0.6, 0.8, 1.0]  # 先跑通，后面再加密
    summary, curve_df, detail_df = evaluation_prefix_curve(config, model, test_loader, prefix_ratios)

    curve_df.to_csv(join(save_path, f'{name}.prefix_curve.csv'), index=False)
    detail_df.to_csv(join(save_path, f'{name}.prefix_detail.csv'), index=False)

    if not config.dry_run:
        wandb.log({"prefix_auc_f1": summary["auc_f1"]})
        table = wandb.Table(dataframe=curve_df)  # 直接把曲线df扔进去
        wandb.log({"prefix_curve": table})

    stop_time = time.time()
    print(f'Finished in {(stop_time - start) / 60:.2f} minutes')
    print(f'Finish time: {time.asctime()}')
