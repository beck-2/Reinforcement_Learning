from ssr_config import SSRConfig
from train_ssr import train

cfg = SSRConfig()
cfg.use_last_choice  = True
cfg.use_start_flag   = True
cfg.use_stem_sector  = True
cfg.sr_loss_coef     = 0.0
cfg.entropy_coef     = 0.005
cfg.lr               = 3e-4
cfg.grad_clip        = 0.5
cfg.rollout_length   = 256
cfg.gamma            = 0.97
cfg.num_train_steps  = 200_000
cfg.output_dir       = "results/ceiling"

train(cfg)