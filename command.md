# data augmentation

```bash
WANDB_MODE=offline python train.py --device cuda:1 --dataset bpRNA --batch_size 4 --seed 2025 --log_wandb True --eval_every 5 --check_every 5 --use_aug True --aug_select 0.1 --aug_replace 0.3 --aug_mode cov --name only_cov_data_aug --diffusion_dim 8 --diffusion_steps 20 --cond_dim 8 --dp_rate 0.1 --lr 0.0001 --warmup 5 --epochs 400

WANDB_MODE=offline python train.py --device cuda:0 --dataset RNAStrAlign --batch_size 2 --seed 2026 --log_wandb True --eval_every 5 --check_every 5 --use_aug True --aug_select 0.1 --aug_replace 0.3 --aug_mode cov --name rnastralign_only_cov_data_aug_3_21 --diffusion_dim 8 --diffusion_steps 20 --cond_dim 8 --dp_rate 0.1 --lr 0.0001 --warmup 5 --epochs 400