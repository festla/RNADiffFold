
# -*- coding: utf-8 -*-
import math
import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
from os.path import join
from models.diffusion_multinomial import MultinomialDiffusion
from models.layers import SegmentationUnet2DCondition
from models.condition.u_conditioner import Unet_conditioner
from models.condition.fm_conditioner.pretrained import load_model_and_alphabet_local
from models.condition.fm_conditioner import fm
import lightning.pytorch as pl

CH_FOLD = 1
cond_ckpt_path = './ckpt/cond_ckpt'


def add_model_args(parser):
    # Model params
    parser.add_argument('--diffusion_steps', type=int, default=20)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--diffusion_dim', type=int, default=8)
    parser.add_argument('--use_interval_guidance', action='store_true')
    parser.add_argument('--growth_mode', type=str, default='forward', choices=['forward', 'reverse', 'random'], 
        help='growth mode for interval guidance: forward, reverse, or random'
    )
    parser.add_argument('--random_growth_seed', type=int, default=1234)
    parser.add_argument('--no_use_interval_guidance', dest='use_interval_guidance', action='store_false')
    parser.set_defaults(use_interval_guidance=True)
    parser.add_argument('--cond_dim', type=int, default=8)
    parser.add_argument('--dp_rate', type=float, default=0.1)
    parser.add_argument('--u_conditioner_ckpt', type=str, default='ufold_train_alldata.pt')


def get_model_id(args):
    return 'multinomial_diffusion'


class DiffusionRNA2dPrediction(nn.Module):
    def __init__(self,
                 num_classes,
                 diffusion_dim,
                 cond_dim,
                 diffusion_steps,
                 dp_rate,
                 u_ckpt,
                 use_interval_guidance,
                 growth_mode='forward',
                 random_growth_seed=1234
                 ):
        super(DiffusionRNA2dPrediction, self).__init__()

        self.num_classes = num_classes
        self.diffusion_dim = diffusion_dim
        self.cond_dim = cond_dim
        self.diffusion_steps = diffusion_steps
        self.dp_rate = dp_rate
        self.u_ckpt = u_ckpt
        self.use_interval_guidance = use_interval_guidance
        self.growth_mode = growth_mode
        self.random_growth_seed = random_growth_seed


        # condition
        self.fm_conditioner, self.alphabet = load_model_and_alphabet_local(
            join(cond_ckpt_path, 'RNA-FM_pretrained.pth'))
        self.u_conditioner = None
        self.load_u_conditioner()

        self.denoise_layer = SegmentationUnet2DCondition(
            num_classes=self.num_classes,
            dim=self.diffusion_dim,
            cond_dim=self.cond_dim,
            num_steps=self.diffusion_steps,
            dim_mults=(1, 2, 4, 8),
            dropout=self.dp_rate
        )

        self.diffusion = MultinomialDiffusion(
            self.num_classes,
            self.diffusion_steps,
            self.denoise_layer,
            self.use_interval_guidance,
            growth_mode=self.growth_mode,
            random_growth_seed=self.random_growth_seed,
        )

    def load_u_conditioner(self):
        self.u_conditioner = Unet_conditioner(img_ch=17, output_ch=1)
        self.u_conditioner.load_state_dict(torch.load(join(cond_ckpt_path, self.u_ckpt), map_location='cpu'))
        condition_out = nn.Conv2d(int(32 * CH_FOLD), self.cond_dim, kernel_size=1, stride=1, padding=0)
        self.u_conditioner.Conv_1x1 = condition_out
        self.u_conditioner.requires_grad_(True)

    def get_alphabet(self):
        return self.alphabet

    # @torch.no_grad()
    def get_fm_embedding(self, data_seq_raw, set_max_len):
        self.fm_conditioner.eval()

        device = data_seq_raw.device

        fm_condition = dict()

        with torch.no_grad():
            backbone_result = self.fm_conditioner(data_seq_raw, need_head_weights=False, repr_layers=[12],
                                                  return_contacts=True)
            fm_embedding = backbone_result['representations'][12]
            fm_embedding = fm_embedding[:, 1:-1, :]

            fm_attention_map = backbone_result['attentions']
            b, l, n, l1, l2 = fm_attention_map.shape
            fm_attention_map = fm_attention_map.reshape(b, l*n, l1, l2)[:, :, 1:-1, 1:-1]

            padding_value = 0
            padding_size = (0, set_max_len - fm_attention_map.shape[-2], 0, set_max_len - fm_attention_map.shape[-1])
            fm_embedding_pad = torch.zeros(fm_embedding.shape[0], set_max_len - fm_embedding.shape[1],
                                           fm_embedding.shape[2]).to(device)
            fm_embedding = torch.cat([fm_embedding, fm_embedding_pad], dim=1)

            fm_attention_map = F.pad(fm_attention_map, padding_size, 'constant', value=padding_value)

            fm_condition['fm_embedding'] = fm_embedding
            fm_condition['fm_attention_map'] = fm_attention_map

        return fm_condition
    
    # add positional factor matrix from PriFold 2026/3/28
    def get_prior_factor(self, data_seq_raw, set_max_len, scale=0.0001):
        device = data_seq_raw.device
        prior_condition = dict()

        seq_token = data_seq_raw[:, 1:-1]   # remove cls only and eos
        B, L = seq_token.shape

        token_A = self.alphabet.get_idx("A")
        token_C = self.alphabet.get_idx("C")
        token_G = self.alphabet.get_idx("G")
        token_U = self.alphabet.get_idx("U")
        token_pad = self.alphabet.padding_idx
        token_eos = self.alphabet.eos_idx

        # 先构造 raw score，默认全 1
        raw_score = torch.ones(B, L, L, device=device, dtype=torch.float32)

        A_mask = (seq_token == token_A)
        C_mask = (seq_token == token_C)
        G_mask = (seq_token == token_G)
        U_mask = (seq_token == token_U)

        invalid_mask = (seq_token == token_pad) | (seq_token == token_eos)

        AU = A_mask.unsqueeze(2) & U_mask.unsqueeze(1)
        UA = U_mask.unsqueeze(2) & A_mask.unsqueeze(1)
        GC = G_mask.unsqueeze(2) & C_mask.unsqueeze(1)
        CG = C_mask.unsqueeze(2) & G_mask.unsqueeze(1)
        GU = G_mask.unsqueeze(2) & U_mask.unsqueeze(1)
        UG = U_mask.unsqueeze(2) & G_mask.unsqueeze(1)

        raw_score[GU | UG] = 1.0
        raw_score[AU | UA] = 3.0
        raw_score[GC | CG] = 6.0

        prior_factor = 1.0 + scale * raw_score

        # eos / pad 恢复中性
        invalid_pair = invalid_mask.unsqueeze(2) | invalid_mask.unsqueeze(1)
        prior_factor[invalid_pair] = 1.0

        # |i-j| < 4 恢复中性
        idx = torch.arange(L, device=device)
        dist = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        sharp_loop_mask = dist < 4
        prior_factor[:, sharp_loop_mask] = 1.0

        prior_factor = prior_factor.unsqueeze(1)  # [B,1,L,L]

        if L < set_max_len:
            padding_size = (0, set_max_len - L, 0, set_max_len - L)
            prior_factor = F.pad(prior_factor, padding_size, mode='constant', value=1.0)

        prior_condition['prior_factor'] = prior_factor
        return prior_condition

    def get_ufold_condition(self, data_fcn_2):

        u_condition = self.u_conditioner(data_fcn_2)

        return u_condition

    def forward(self,
                x_0,
                data_fcn_2,
                data_seq_raw,
                contact_masks,
                set_max_len,
                data_seq_encoding,
                ):

        fm_condition = self.get_fm_embedding(data_seq_raw, set_max_len)

        u_condition = self.get_ufold_condition(data_fcn_2)

        # pri_condition = self.get_prior_factor(data_seq_raw, set_max_len)

        # fm_condition.update(pri_condition)

        """print(f"x_0.shape: {x_0.shape}")
        print(f"data_fcn_2.shape: {data_fcn_2.shape}")
        print(f"data_seq_raw.shape: {data_seq_raw.shape}")
        print(f"data_seq_encoding.shape: {data_seq_encoding.shape}")
        print(f"contact_masks.shape: {contact_masks.shape}")
        print(f"fm_condition.shape: {fm_condition.shape}")
        print(f"fm_embedding: {fm_condition['fm_embedding'].shape}")
        print(f"fm_attention_map: {fm_condition['fm_attention_map'].shape}")
        print(f"u_condition: {u_condition.shape}")
        pdb.set_trace()"""
        '''
        x_0.shape: torch.Size([4, 1, 384, 384])
        data_fcn_2.shape: torch.Size([4, 17, 384, 384])
        data_seq_raw.shape: torch.Size([4, 372])
        data_seq_encoding.shape: torch.Size([4, 384, 4])
        contact_masks.shape: torch.Size([4, 1, 384, 384])
        fm_embedding: torch.Size([4, 384, 640])
        fm_attention_map: torch.Size([4, 240, 384, 384])
        u_condition: torch.Size([4, 8, 384, 384]) 这就是由data_fcn_2的17通道经过Unet得到的'''
        loss = self.diffusion(x_0, fm_condition, u_condition, contact_masks, data_seq_encoding)  # 训练过程在这里输入：

        loglik_bpd = -loss.sum()/(math.log(2) * x_0.shape.numel())
        return loglik_bpd

    @torch.no_grad()
    def sample(self,
               num_samples,
               data_fcn_2,
               data_seq_raw,
               set_max_len,
               contact_masks,
               seq_encoding,
               show_progress: bool = False
               ):
        fm_condition = self.get_fm_embedding(data_seq_raw, set_max_len)

        u_condition = self.get_ufold_condition(data_fcn_2)

        pred_x_0, model_prob = self.diffusion.sample(
            num_samples, fm_condition, u_condition, contact_masks, set_max_len, seq_encoding, show_progress
        )

        return pred_x_0, model_prob

    @torch.no_grad()
    def sample_chain(self,
                     num_samples,
                     data_fcn_2,
                     data_seq_raw,
                     set_max_len,
                     contact_masks,
                     seq_encoding
                     ):
        fm_condition = self.get_fm_embedding(data_seq_raw, set_max_len)

        u_condition = self.get_ufold_condition(data_fcn_2)

        pred_x_0_chain, model_prob_chain, pred_x_0, model_prob = self.diffusion.sample_chain(
            num_samples, fm_condition, u_condition, contact_masks, set_max_len, seq_encoding  
        )
        return pred_x_0_chain, model_prob_chain, pred_x_0, model_prob


def get_model(args):
    model = DiffusionRNA2dPrediction(
        num_classes=args.num_classes,
        diffusion_dim=args.diffusion_dim,
        cond_dim=args.cond_dim,
        diffusion_steps=args.diffusion_steps,
        dp_rate=args.dp_rate,
        u_ckpt=args.u_conditioner_ckpt,
        use_interval_guidance=args.use_interval_guidance,
        growth_mode=args.growth_mode,
        random_growth_seed=args.random_growth_seed
    )
    alphabet = model.get_alphabet()
    return model, alphabet