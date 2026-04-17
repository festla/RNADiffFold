# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from inspect import isfunction
from tqdm import tqdm
import math
import pdb


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def exists(x):
    return x is not None


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def log_onehot_to_index(log_x):
    return log_x.argmax(1)


def index_to_log_onehot(x, K):
    assert x.max().item() < K, f'Error: {x.max().item()} >= {K}'

    x_onehot = F.one_hot(x, K)

    permute_order = (0, -1) + tuple(range(1, len(x.size())))

    x_onehot = x_onehot.permute(permute_order)

    log_x = torch.log(x_onehot.float().clamp(min=1e-30))

    return log_x


def sum_except_batch(x, num_dims=1):
    '''
    Sums all dimensions except the first.

    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)

    Returns:
        x_sum: Tensor, shape (batch_size,)
    '''
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def log_categorical(log_x_0, log_prob):
    return (log_x_0.exp() * log_prob).sum(dim=1)


def beta_schedule(num_steps, schedule_name='cosine', s=0.01):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    # t = np.linspace(0, num_steps, steps)
    t = torch.arange(0, num_steps + 1, dtype=torch.float64)
    if schedule_name == 'cosine':
        # f_t = np.cos(((t / num_steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        f_t = torch.cos(((t / num_steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    elif schedule_name == 'sqrt':
        # f_t = 1 - np.sqrt(t / num_steps + 0.0001)
        f_t = 1 - torch.sqrt(t / num_steps + 0.0001)
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")
    alpha_bars = f_t / f_t[0]
    alphas = (alpha_bars[1:] / alpha_bars[:-1])
    # alphas = np.clip(alphas, a_min=0.001, a_max=1.)
    alphas = torch.clamp(alphas, min=0.001, max=0.999)

    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.
    alphas = torch.sqrt(alphas)
    return alphas


class MultinomialDiffusion(nn.Module):
    def __init__(
            self,
            num_classes,
            time_steps,    # 就是diffusion_steps, 去噪步数T=20
            denoise_fn,
    ):
        super(MultinomialDiffusion, self).__init__()
        self.K = num_classes
        self.time_steps = time_steps
        self._denoise_fn = denoise_fn
        # self.fm_conditioner = fm_conditioner
        # self.u_conditioner = u_conditioner

        self.seq_mask_token = nn.Parameter(torch.zeros(4))
        self.fm_mask_token  = nn.Parameter(torch.zeros(640))

        alphas = beta_schedule(time_steps, schedule_name='cosine', s=0.01)
        log_alphas = torch.log(alphas)
        log_alpha_bars = torch.cumsum(log_alphas, dim=0)
        log_1_minus_alphas = torch.log(1 - torch.exp(log_alphas) + 1e-40)
        log_1_minus_alpha_bars = torch.log(1 - torch.exp(log_alpha_bars) + 1e-40)

        assert log_add_exp(log_alphas, log_1_minus_alphas).abs().sum().item() < 1e-5
        assert log_add_exp(log_alpha_bars, log_1_minus_alpha_bars).abs().sum().item() < 1e-5
        assert (torch.cumsum(log_alphas, dim=0) - log_alpha_bars).abs().sum().item() < 1e-5

        # Convert to float32 and register buffers.
        self.register_buffer('log_alphas', log_alphas.float())
        self.register_buffer('log_alpha_bars', log_alpha_bars.float())
        self.register_buffer('log_1_minus_alphas', log_1_minus_alphas.float())
        self.register_buffer('log_1_minus_alpha_bars', log_1_minus_alpha_bars.float())

        self.register_buffer('Lt_history', torch.zeros(self.time_steps))
        self.register_buffer('Lt_count', torch.zeros(self.time_steps))

    # KL divergence
    def multinomial_kl(self, log_prob1, log_prob2):
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    # p(x)--> x
    def log_sample_categorical(self, logits):
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.K)
        return log_sample

    # q(xt|xt-1)
    def q_pred_one_step(self, log_x_t, t):
        log_alphas_t = extract(self.log_alphas, t, log_x_t.shape)
        log_1_minus_alphas_t = extract(self.log_1_minus_alphas, t, log_x_t.shape)

        # Eq.11, alpha_t * E[xt] + (1 - alpha_t) / K
        log_probs = log_add_exp(log_x_t + log_alphas_t, log_1_minus_alphas_t - np.log(self.K))
        return log_probs

    # q(xt|x0)
    def q_pred(self, log_x_0, t):
        log_alpha_bars_t = extract(self.log_alpha_bars, t, log_x_0.shape)
        log_1_minus_alpha_bars_t = extract(self.log_1_minus_alpha_bars, t, log_x_0.shape)

        # Eq.12, alpha_bar_t * E[x0] + (1 - alpha_bar_t) / K
        log_probs = log_add_exp(log_x_0 + log_alpha_bars_t, log_1_minus_alpha_bars_t - np.log(self.K))
        return log_probs

    # q(xt-1|xt, x0)
    def q_posterior(self, log_x_t, log_x_0, t):
        # q(xt-1 | xt, x0) = q(xt | xt-1, x0) * q(xt-1 | x0) / q(xt | x0)
        # where q(xt | xt-1, x0) = q(xt | xt-1).
        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self.q_pred(log_x_0, t_minus_1)

        num_axes = (1,) * (len(log_x_0.size()) - 1)
        t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_x_0)
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_0, log_EV_qxtmin_x0)

        unnormed_logprobs = log_EV_qxtmin_x0 + self.q_pred_one_step(log_x_t, t)

        log_EV_xtmin_given_xt_given_xstart = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=1, keepdim=True)

        return log_EV_xtmin_given_xt_given_xstart

    # x_0_hat
    def predict_x_0(self, log_x_t, t, fm_condition, u_condition, seq_encoding, use_interval_guidance=False):
        x_t = log_onehot_to_index(log_x_t)

        B, L = log_x_t.size(0), log_x_t.size(-1)
        device = log_x_t.device

        if use_interval_guidance:
            mask_1d, mask_2d = self._build_prefix_masks(t, L, device)   # [B,L], [B,1,L,L]
        else:
            mask_1d = torch.ones((B, L), device=device, dtype=torch.float32)
            mask_2d = torch.ones((B, 1, L, L), device=device, dtype=torch.float32)

        g_seq = mask_1d.unsqueeze(-1)
        g_2d = mask_2d

        # ========= 1) seq_encoding：early 全 masked，中期前缀生长，late 全长 =========
        if seq_encoding is None:
            seq_enc_masked = None
        else:
            D_seq = seq_encoding.size(-1)
            assert self.seq_mask_token.numel() == D_seq, \
                f"seq_mask_token dim mismatch: {self.seq_mask_token.numel()} vs {D_seq}"

            seq_mask_token = self.seq_mask_token.to(dtype=seq_encoding.dtype, device=seq_encoding.device)

            seq_enc_masked = (
                seq_encoding * g_seq
                + (1.0 - g_seq) * seq_mask_token.view(1, 1, -1)
            )
        # ========= 2) fm_embedding：同样三段式 =========
        fm_embed = fm_condition['fm_embedding']   # [B, L, D_fm]
        D_fm = fm_embed.size(-1)
        assert self.fm_mask_token.numel() == D_fm, \
            f"fm_mask_token dim mismatch: {self.fm_mask_token.numel()} vs {D_fm}"

        fm_mask_token = self.fm_mask_token.to(dtype=fm_embed.dtype, device=fm_embed.device)

        fm_embed_masked = (
            fm_embed * g_seq
            + (1.0 - g_seq) * fm_mask_token.view(1, 1, -1)
        )

        fm_cond_masked = dict(fm_condition)
        fm_cond_masked['fm_embedding'] = fm_embed_masked


        # ========= 3) 全局信息：early / middle / late 都保留 =========
        # 你现在的设定是：
        # 前期不给序列，只给全局信息
        # 所以这些 attention_map/global 条件先保持原样
        if 'fm_attention_map' in fm_condition:
            fm_cond_masked['fm_attention_map'] = fm_condition['fm_attention_map']

        # u_condition 作为全局/2D信息，保持原样
        u_cond_masked = u_condition

        # 4) 调用去噪网络
        out = self._denoise_fn(t, x_t, fm_cond_masked, u_cond_masked, seq_enc_masked)

        # 5) 形状校验 + softmax
        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.K
        assert out.size()[2:] == x_t.size()[1:]

        log_pred = F.log_softmax(out, dim=1)
        return log_pred

    # p(xt-1|xt)
    def p_pred(self, log_x_t, t, fm_condition, u_condition, seq_encoding, use_interval_guidance=False):
        log_x_0_hat = self.predict_x_0(log_x_t, t, fm_condition, u_condition, seq_encoding, use_interval_guidance=use_interval_guidance)   # [B, 2, 1, L, L]
        log_probs = self.q_posterior(log_x_t, log_x_0_hat, t)    # log_x_t.shape=[B, 1, L, L]
        return log_probs

    # q(xt|x0) -> xt
    def q_sample(self, log_x_0, t):
        log_EV_qxt_x0 = self.q_pred(log_x_0, t)
        log_x_t = self.log_sample_categorical(log_EV_qxt_x0)
        return log_x_t

    # p(xt-1|xt) -> xt-1
    @torch.no_grad()
    def p_sample(self, log_x_t, t, fm_condition, u_condition, seq_encoding, use_interval_guidance=False):
        log_probs = self.p_pred(log_x_t, t, fm_condition, u_condition, seq_encoding,use_interval_guidance=use_interval_guidance)
        x_t_minus_1 = self.log_sample_categorical(log_probs)
        return x_t_minus_1, log_probs

    # L_T,prior matching term
    def kl_prior(self, log_x_0):
        b = log_x_0.size(0)
        device = log_x_0.device
        ones = torch.ones(b, device=device).long()

        log_qxT_prob = self.q_pred(log_x_0, t=(self.time_steps - 1) * ones)
        log_half_prob = -torch.log(self.K * torch.ones_like(log_qxT_prob))
        kl_prior = self.multinomial_kl(log_qxT_prob, log_half_prob)
        return sum_except_batch(kl_prior)

    # compute L_{t-1} and L_0
    def compute_Lt(self, log_x_0, log_x_t, fm_condition, u_condition, seq_encoding, t, contact_masks, detach_mean=False):
        log_true_prob = self.q_posterior(log_x_t=log_x_t, log_x_0=log_x_0, t=t)

        log_model_prob = self.p_pred(
            log_x_t=log_x_t,
            t=t,
            fm_condition=fm_condition,
            u_condition=u_condition * contact_masks,
            seq_encoding=seq_encoding,
            use_interval_guidance=False
        )

        if detach_mean:
            log_model_prob = log_model_prob.detach()
        
        # contact_masks: [B,1,L,L]
        valid_mask = contact_masks.to(dtype=log_model_prob.dtype)

        # L_{t-1}, denoising matching term
        kl_map = self.multinomial_kl(log_true_prob, log_model_prob)
        kl = sum_except_batch(kl_map * valid_mask)

        # L_0,reconstruction term,decoder
        decoder_nll_map = -log_categorical(log_x_0, log_model_prob)
        decoder_nll = sum_except_batch(decoder_nll_map * valid_mask)

        mask = (t == torch.zeros_like(t)).float()
        loss = mask * decoder_nll + (1. - mask) * kl

        return loss

    # sample t and pt
    def sample_time(self, b, device, method='uniform'):
        # Importance sampling
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.time_steps, (b,), device=device).long()

            pt = torch.ones_like(t).float() / self.time_steps
            return t, pt
        else:
            raise ValueError('Unknown method: {}'.format(method))


    def forward(self, x_0, fm_condition, u_condition, contact_masks, seq_encoding): # 
        batch, device = x_0.size(0), x_0.device

        t, pt = self.sample_time(batch, device, 'importance')    # 每个样本随机抽一个扩散时间步 t 。
        log_x_0 = index_to_log_onehot(x_0, self.K)    # 把离散标签 x_0 变成 log-onehot，即[B, 1, T, T] -> [B, k(2), 1, T, T]
        """print(f"x_0.shape: {x_0.shape}") x_0.shape: torch.Size([1, 1, 112, 112])
        print(f"fm_attention_map.shape: {fm_condition['fm_attention_map'].shape}") fm_attention_map.shape: torch.Size([1, 240, 112, 112])
        print(f"fm_embedding.shape: {fm_condition['fm_embedding'].shape}") fm_embedding.shape: torch.Size([1, 112, 640])
        print(f"u_condition.shape: {u_condition.shape}") u_condition.shape: torch.Size([1, 8, 112, 112])
        print(f"seq_encoding.shape: {seq_encoding.shape}") seq_encoding.shape: torch.Size([1, 112, 4])
        pdb.set_trace()"""
        kl = self.compute_Lt(    # 计算训练损失
            log_x_0=log_x_0,
            log_x_t=self.q_sample(log_x_0, t),    # 对 log_x_0 加噪得到 log_x_t
            fm_condition=fm_condition,
            u_condition=u_condition,
            seq_encoding=seq_encoding,
            t=t,
            contact_masks=contact_masks
        )

        Lt2 = kl.pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

        kl_prior = self.kl_prior(log_x_0)

        # Upweigh loss term of the kl
        vb_loss = kl / pt + kl_prior

        return -vb_loss

    @torch.no_grad()
    def sample(self, num_samples, fm_condition, u_condition, contact_masks, set_max_len, seq_encoding, show_progress: bool = False):
        b = num_samples
        data_shape = (1, int(set_max_len), int(set_max_len))

        device = self.log_alphas.device
        uniform_logits = torch.zeros((b, self.K) + data_shape, device=device)
        log_z = self.log_sample_categorical(uniform_logits)

        iterator = reversed(range(0, self.time_steps))    # lql added
        if show_progress and sys.stdout.isatty():         # lql added
            iterator = tqdm(iterator, total=self.time_steps, desc='sampling loop time step', leave=False, dynamic_ncols=True)
        for i in iterator:
            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z, model_log_prob = self.p_sample(log_x_t=log_z,
                                                  t=t,
                                                  fm_condition=fm_condition,
                                                  u_condition=u_condition * contact_masks,
                                                  seq_encoding=seq_encoding,
                                                  use_interval_guidance=False)

        model_prob = torch.exp(model_log_prob)[:, 1, :, :, :] * contact_masks
        return log_onehot_to_index(log_z) * contact_masks, model_prob

    @torch.no_grad()
    def sample_chain(self, num_samples, fm_condition, u_condition, contact_masks, set_max_len, seq_encoding, show_progress: bool = False):
        b = num_samples
        data_shape = (1, int(set_max_len), int(set_max_len))

        device = self.log_alphas.device
        uniform_logits = torch.zeros((b, self.K) + data_shape, device=device)

        zs = torch.zeros((self.time_steps, b) + data_shape).long()
        z_probs = torch.zeros((self.time_steps, b) + data_shape).float()

        log_z = self.log_sample_categorical(uniform_logits)

        iterator = reversed(range(0, self.time_steps))
        if show_progress and sys.stdout.isatty():
            iterator = tqdm(iterator, total=self.time_steps, desc='sampling loop time step', leave=False, dynamic_ncols=True)
        for i in iterator:
            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z, model_log_prob = self.p_sample(log_x_t=log_z,
                                                  t=t,
                                                  fm_condition=fm_condition,
                                                  u_condition=u_condition * contact_masks,
                                                  seq_encoding=seq_encoding,
                                                  use_interval_guidance=False)

            z_probs[i] = torch.exp(model_log_prob)[:, 1, :, :, :] * contact_masks
            zs[i] = log_onehot_to_index(log_z) * contact_masks

        return zs, z_probs, zs[-1], z_probs[-1]

    def _build_prefix_masks(self, t: torch.Tensor, L: int, device):
        P = self._time_to_prefix_len_three_stage(
            t, L=L, low_frac=0.20, high_frac=0.65, mid_start_ratio=0.10, update_every=2
        )  # [B]

        idx = torch.arange(L, device=device).view(1, L)
        mask_1d = (idx < P.view(-1, 1)).float()  # [B, L]
        mask_2d = (mask_1d.unsqueeze(2) * mask_1d.unsqueeze(1)).unsqueeze(1)  # [B,1,L,L]
        return mask_1d, mask_2d

    def stage_masks(self, t: torch.Tensor, low_frac=0.20, high_frac=0.65):

        Tm1 = max(self.time_steps-1, 1)
        s = t.float() / float(Tm1)

        early_mask = s > high_frac
        middle_mask = (s >= low_frac) & (s <= high_frac)
        late_mask = s < low_frac

        return early_mask, middle_mask, late_mask

    def _time_to_prefix_len_three_stage(
        self,
        t: torch.Tensor,
        L: int,
        low_frac: float = 0.20,
        high_frac: float = 0.65,
        mid_start_ratio: float = 0.10,
        update_every: int = 2,
    ):
        """
        三段式：
        - early:  P = 0
        - middle: P 从 mid_start_ratio * L 逐步增长到 L
        - late:   P = L
        """
        device = t.device
        B = t.size(0)
        Tm1 = max(self.time_steps - 1, 1)
        s = t.float() / float(Tm1)   # 0..1, 小=late, 大=early

        P = torch.zeros_like(t, dtype=torch.long, device=device)

        # late: full length
        late_mask = s < low_frac
        P = torch.where(late_mask, torch.full_like(P, L), P)

        # middle: prefix growth
        middle_mask = (s >= low_frac) & (s <= high_frac)
        if middle_mask.any():
            prog = ((high_frac - s) / (high_frac - low_frac)).clamp(0.0, 1.0)  # 0->1

            # interval 化：阶梯式更新
            n_mid_steps = max(int(round((high_frac - low_frac) * Tm1)), 1)
            n_updates = max(n_mid_steps // max(update_every, 1) + 1, 1)

            if n_updates > 1:
                prog = torch.floor(prog * (n_updates - 1 + 1e-8)) / float(n_updates - 1)
            else:
                prog = torch.zeros_like(prog)

            P_mid = torch.round((mid_start_ratio * L) + prog * (L - mid_start_ratio * L)).long()
            P_mid = P_mid.clamp(min=1, max=L)

            P = torch.where(middle_mask, P_mid, P)

        # early 默认保持 0
        return P