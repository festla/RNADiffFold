# -*- coding: utf-8 -*-
import torch
import pdb
import numpy as np
import pandas as pd
from common.utils import add_parent_path
from common.experiment import add_exp_args as add_exp_args_parent
from common.experiment import DiffusionExperiment
from common.data_utils import contact_map_masks
from common.loss_utils import bce_loss, evaluate_f1_precision_recall
from common.loss_utils import calculate_auc, calculate_mattews_correlation_coefficient,rna_evaluation
add_parent_path(level=2)

from tqdm.auto import tqdm
import time

def add_exp_args(parser):
    add_exp_args_parent(parser)


class Experiment(DiffusionExperiment):

    def train_fn(self, epoch):
        self.model.train()
        loss_sum = 0.0
        loss_count = 0
        device = self.args.device
        pbar = tqdm(self.train_loader,
            total=len(self.train_loader),           # 需 Dataset.__len__
            desc=f"Epoch {epoch+1}/{self.args.epochs}",
            unit="batch",
            leave=True)
        """
        print("model device:", next(self.model.parameters()).device) cuda:1
        pdb.set_trace()
        """
        for _, (contact, data_fcn_2, data_seq_raw, data_length, _, set_max_len, data_seq_encoding) in enumerate(pbar):    # self.train_loader
            self.optimizer.zero_grad()
            contact = contact.to(device)
            data_fcn_2 = data_fcn_2.to(device)
            matrix_rep = torch.zeros_like(contact)
            data_length = data_length.to(device)
            data_seq_raw = data_seq_raw.to(device)
            data_seq_encoding = data_seq_encoding.to(device)
            contact_masks = contact_map_masks(data_length, matrix_rep).to(device)
            '''print(f"contact.shape: {contact.shape}")
            print(f"data_fcn_2.shape: {data_fcn_2.shape}")
            print(f"matrix_rep.shape: {matrix_rep.shape}")
            print(f"data_length.shape: {data_length.shape}")
            print(f"data_seq_raw.shape: {data_seq_raw.shape}")
            print(f"data_seq_encoding.shape: {data_seq_encoding.shape}")
            print(f"contact_masks.shape: {contact_masks.shape}")
            pdb.set_trace()
            contact.shape: torch.Size([4, 1, 384, 384])
            data_fcn_2.shape: torch.Size([4, 17, 384, 384])
            matrix_rep.shape: torch.Size([4, 1, 384, 384])
            data_length.shape: torch.Size([4])
            data_seq_raw.shape: torch.Size([4, 372])
            data_seq_encoding.shape: torch.Size([4, 384, 4])
            contact_masks.shape: torch.Size([4, 1, 384, 384])'''
            loss = self.model(contact, data_fcn_2, data_seq_raw, contact_masks, set_max_len, data_seq_encoding)
            loss.backward()

            self.optimizer.step()
            if self.scheduler_iter:
                self.scheduler_iter.step()
            bs = len(contact)
            loss_sum += loss.detach().cpu().item() * bs
            loss_count += bs
            bpd = loss_sum / loss_count

            # 在进度条尾部显示指标
            lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix(bpd=f"{bpd:.5f}", lr=f"{lr:.2e}")
            '''loss_sum += loss.detach().cpu().item() * len(contact)
            loss_count += len(contact)
            print('Training. Epoch: {}/{}, Bits/dim: {:.5f}'.
                  format(epoch + 1, self.args.epochs, loss_sum / loss_count), end='\r')'''
        print('')
        if self.scheduler_epoch: self.scheduler_epoch.step()
        return {'bpd': loss_sum / loss_count}

    def val_fn(self, epoch):
        self.model.eval()

        device = self.args.device
        with torch.no_grad():
            loss_count = 0
            val_loss_sum = 0.0
            auc_score = 0.0
            auc_count = 0
            val_no_train = list()
            mcc_no_train = list()

            for _, (contact, data_fcn_2, data_seq_raw, data_length, _, set_max_len, data_seq_encoding) in enumerate(self.val_loader):
                data_fcn_2 = data_fcn_2.to(device)
                matrix_rep = torch.zeros_like(contact)
                data_length = data_length.to(device)
                data_seq_raw = data_seq_raw.to(device)
                data_seq_encoding = data_seq_encoding.to(device)
                contact_masks = contact_map_masks(data_length, matrix_rep).to(device)

                # calculate contact loss
                batch_size = contact.shape[0]
                pred_x0, _ = self.model.sample(batch_size, data_fcn_2, data_seq_raw, set_max_len, contact_masks, data_seq_encoding)

                pred_x0 = pred_x0.cpu().float()
                val_loss_sum += bce_loss(pred_x0.float(), contact.float()).cpu().item()
                loss_count += len(contact)
                auc_score += calculate_auc(contact.float(), pred_x0)
                auc_count += 1
                val_no_train_tmp = list(map(lambda i: evaluate_f1_precision_recall(
                    pred_x0[i].squeeze(), contact.float()[i].squeeze()), range(pred_x0.shape[0])))
                val_no_train += val_no_train_tmp

                mcc_no_train_tmp = list(map(lambda i: calculate_mattews_correlation_coefficient(
                    pred_x0[i].squeeze(), contact.float()[i].squeeze()), range(pred_x0.shape[0])))
                mcc_no_train += mcc_no_train_tmp

            val_precision, val_recall, val_f1 = zip(*val_no_train)

            val_precision = np.average(np.nan_to_num(np.array(val_precision)))
            val_recall = np.average(np.nan_to_num(np.array(val_recall)))
            val_f1 = np.average(np.nan_to_num(np.array(val_f1)))

            mcc_final = np.average(np.nan_to_num(np.array(mcc_no_train)))

            print('#' * 80)
            print('Average val F1 score: ', round(val_f1, 3))
            print('Average val precision: ', round(val_precision, 3))
            print('Average val recall: ', round(val_recall, 3))
            print('#' * 80)
            print('Average val MCC', round(mcc_final, 3))
            print('#' * 80)
            print('')
        return {'f1': val_f1, 'precision': val_precision, 'recall': val_recall,
                'auc_score': auc_score / auc_count, 'mcc': mcc_final, 'bce_loss': val_loss_sum / loss_count}

    def test_fn(self, epoch):
        self.model.eval()
        device = self.args.device
        with torch.no_grad():
            test_no_train = list()
            total_name_list = list()
            total_length_list = list()

            for _, (contact, data_fcn_2, data_seq_raw, data_length, data_name, set_max_len, data_seq_encoding) in enumerate(
                    self.test_loader):
                total_name_list += [item for item in data_name]
                total_length_list += [item.item() for item in data_length]

                data_fcn_2 = data_fcn_2.to(device)
                matrix_rep = torch.zeros_like(contact)
                data_length = data_length.to(device)
                data_seq_raw = data_seq_raw.to(device)
                data_seq_encoding = data_seq_encoding.to(device)
                contact_masks = contact_map_masks(data_length, matrix_rep).to(device)

                # calculate contact loss
                batch_size = contact.shape[0]
                pred_x0, _ = self.model.sample(batch_size, data_fcn_2, data_seq_raw, set_max_len, contact_masks, data_seq_encoding)

                pred_x0 = pred_x0.cpu().float()

                test_no_train_tmp = list(map(lambda i: rna_evaluation(
                    pred_x0[i].squeeze(), contact.float()[i].squeeze()), range(pred_x0.shape[0])))
                test_no_train += test_no_train_tmp

            accuracy, prec, recall, sens, spec, F1, MCC = zip(*test_no_train)

            f1_pre_rec_df = pd.DataFrame({'name': total_name_list,
                                          'length': total_length_list,
                                          'accuracy': list(np.array(accuracy)),
                                          'precision': list(np.array(prec)),
                                          'recall': list(np.array(recall)),
                                          'sensitivity': list(np.array(sens)),
                                          'specificity': list(np.array(spec)),
                                          'f1': list(np.array(F1)),
                                          'mcc': list(np.array(MCC))})

            accuracy = np.average(np.nan_to_num(np.array(accuracy)))
            precision = np.average(np.nan_to_num(np.array(prec)))
            recall = np.average(np.nan_to_num(np.array(recall)))
            sensitivity = np.average(np.nan_to_num(np.array(sens)))
            specificity = np.average(np.nan_to_num(np.array(spec)))
            F1 = np.average(np.nan_to_num(np.array(F1)))
            MCC = np.average(np.nan_to_num(np.array(MCC)))

            print('#' * 40)
            print('Average testing accuracy: ', round(accuracy, 3))
            print('Average testing F1 score: ', round(F1, 3))
            print('Average testing precision: ', round(precision, 3))
            print('Average testing recall: ', round(recall, 3))
            print('Average testing sensitivity: ', round(sensitivity, 3))
            print('Average testing specificity: ', round(specificity, 3))
            print('#' * 40)
            print('Average testing MCC', round(MCC, 3))
            print('#' * 40)
            print('')
        return {'f1': F1, 'precision': precision, 'recall': recall,
                'sensitivity': sensitivity, 'specificity': specificity, 'accuracy': accuracy, 'mcc': MCC}, f1_pre_rec_df

'''
print(f"contact       -> shape={tuple(contact.shape)}, dtype={contact.dtype}, device={contact.device}")
print( contact[0, :, :min(5, contact.shape[-2]), :min(5, contact.shape[-1])] )

print(f"data_fcn_2 -> shape={tuple(data_fcn_2.shape)}, dtype={data_fcn_2.dtype}, device={data_fcn_2.device}")
C, H, W = data_fcn_2.shape
print(data_fcn_2[:min(3, C), :min(5, H), :min(5, W)])

print(f"matrix_rep    -> shape={tuple(matrix_rep.shape)}, dtype={matrix_rep.dtype}, device={matrix_rep.device}")
print( matrix_rep[0, :, :min(5, matrix_rep.shape[-2]), :min(5, matrix_rep.shape[-1])] )

print(f"data_length   -> shape={tuple(data_length.shape)}, dtype={data_length.dtype}, device={data_length.device}")
print( data_length )

# data_seq_raw 既可能是 Tensor 也可能是 list[str]，两种都处理
if torch.is_tensor(data_seq_raw):
    print(f"data_seq_raw  -> shape={tuple(data_seq_raw.shape)}, dtype={data_seq_raw.dtype}, device={data_seq_raw.device}")
    print( data_seq_raw if data_seq_raw.numel() <= 32 else data_seq_raw.view(-1)[:32] )
else:
    print(f"data_seq_raw  -> type={type(data_seq_raw).__name__}, len={len(data_seq_raw)}")
    print( data_seq_raw[:2] )

print(f"data_seq_encoding -> shape={tuple(data_seq_encoding.shape)}, dtype={data_seq_encoding.dtype}, device={data_seq_encoding.device}")
if data_seq_encoding.dim() == 3:
    # [B, L, 4] 或 [B, 4, L]，都打印前 5 行
    if data_seq_encoding.shape[-1] == 4:   # [B, L, 4]
        print( data_seq_encoding[0, :min(5, data_seq_encoding.shape[1]), :] )
    elif data_seq_encoding.shape[1] == 4:  # [B, 4, L]
        print( data_seq_encoding[0, :, :min(5, data_seq_encoding.shape[2])] )
    else:
        print( data_seq_encoding[0, :min(5, data_seq_encoding.shape[1]), :min(5, data_seq_encoding.shape[2])] )
else:
    print( data_seq_encoding )

print(f"contact_masks -> shape={tuple(contact_masks.shape)}, dtype={contact_masks.dtype}, device={contact_masks.device}")
if contact_masks.dim() == 4:
    print( contact_masks[0, :, :min(5, contact_masks.shape[-2]), :min(5, contact_masks.shape[-1])] )
elif contact_masks.dim() == 3:
    print( contact_masks[0, :min(5, contact_masks.shape[-2]), :min(5, contact_masks.shape[-1])] )
else:
    print( contact_masks )
import pdb;pdb.set_trace()'''