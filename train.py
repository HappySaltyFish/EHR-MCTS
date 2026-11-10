import os
import time
from datetime import datetime
import yaml
import argparse
from contextlib import nullcontext

import numpy as np
import torch

from model import DelphiSMURF, DelphiSMURFConfig
from dataset.dataset import get_dataloader_smurf
from src.dataset.nhp_dataset import get_dataloader_nhp_thp


def thp_batch_to_smurf_batch(batch, device):
    event_type, event_time, time_gap = batch

    event_time = event_time.to(device)
    time_gap = time_gap.to(device)
    event_type = event_type.to(device)

    X = event_type[:, :-1].contiguous()
    Y = event_type[:, 1:].contiguous()
    A = event_time[:, :-1].contiguous()
    B = event_time[:, 1:].contiguous()

    tg = time_gap[:, :max(X.size(1) - 1, 0)].contiguous()

    return X, A, Y, B, tg


class Config:
    """Simple flat config wrapper for dot access."""
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def load_yaml_config(config_path: str):
    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    if raw_cfg is None:
        raise ValueError(f"Empty YAML config: {config_path}")
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"YAML config must be a flat mapping/dict: {config_path}")

    yaml_cfg = Config(raw_cfg)

    required_fields = [
        # basic / logging
        "out_dir",
        "eval_interval",
        "eval_iters",
        "log_interval",
        "always_save_checkpoint",
        "wandb_log",
        "wandb_project",

        # training flow
        # "seed",
        "eval_only",
        "init_from",
        "num_epochs",

        # data
        "dataset",
        "data_fraction",
        "batch_size",

        # model
        "n_layer",
        "n_head",
        "n_embd",
        "dropout",
        "bias",
        "vocab_size",
        "ignore_tokens",
        "t_min",
        "token_dropout",
        "mask_ties",

        # optimizer / lr
        "weight_decay",
        "learning_rate",
        "eta_min",
        "beta1",
        "beta2",
        "warmup_iters",
        "grad_clip",

        # system
        "device",
        "dtype",
        "compile",

        # smurf-specific
        "use_score_matching",
        "parametrize",
        "add_noise",
        "noise_type",
        "var_noise",
        "num_noise",
        "langevin_step",
        "n_langevin_steps",
        "n_samples",
        "denoise_step",
        "loss_lambda",

        "config_type"
    ]

    missing = [k for k in required_fields if not hasattr(yaml_cfg, k)]
    if missing:
        raise ValueError(
            "Missing required config fields in YAML:\n"
            + "\n".join(f"  - {k}" for k in missing)
        )

    if not hasattr(yaml_cfg, "data_format"):
        yaml_cfg.data_format = "bin"
        raw_cfg["data_format"] = "bin"
    if not hasattr(yaml_cfg, "normalize"):
        yaml_cfg.normalize = "normal"
        raw_cfg["normalize"] = "normal"

    return yaml_cfg, raw_cfg


parser = argparse.ArgumentParser(description="Train DelphiSMURF with YAML config")
parser.add_argument("--config", type=str, required=True, help="path to the yaml config file")
args = parser.parse_args()

yaml_cfg, config = load_yaml_config(args.config)

print(f"Loaded config from {args.config}")
print("Running on Configuration=")
print(config)

# -----------------------------------------------------------------------------
# setup
# -----------------------------------------------------------------------------

os.makedirs(yaml_cfg.out_dir, exist_ok=True)
torch.set_float32_matmul_precision("high")

device_type = "cuda" if "cuda" in yaml_cfg.device else "cpu"
ptdtype = {
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[yaml_cfg.dtype]

ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

torch.set_default_dtype(ptdtype)

data_dir = os.path.join("data", yaml_cfg.dataset)
train_split_name = "train"

if yaml_cfg.data_format == "nhp":
    train_loader, num_types = get_dataloader_nhp_thp(
        data_dir=data_dir,
        batch_size=yaml_cfg.batch_size,
        split=train_split_name,
        normalize=yaml_cfg.normalize,
        shuffle=True,
    )

    time_mean = train_loader.dataset.time_mean
    if yaml_cfg.normalize == "log":
        log_mean = train_loader.dataset.log_mean
        log_std = train_loader.dataset.log_std

        val_loader, _ = get_dataloader_nhp_thp(
            data_dir=data_dir,
            batch_size=yaml_cfg.batch_size,
            split="dev",
            normalize=yaml_cfg.normalize,
            shuffle=False,
            time_mean=time_mean,
            log_mean=log_mean,
            log_std=log_std
        )
    else:   # normal
        val_loader, _ = get_dataloader_nhp_thp(
            data_dir=data_dir,
            batch_size=yaml_cfg.batch_size,
            split="dev",
            normalize=yaml_cfg.normalize,
            shuffle=False,
            time_mean=time_mean
        )

    if yaml_cfg.vocab_size is None or yaml_cfg.vocab_size <= 0:
        yaml_cfg.vocab_size = num_types + 1
        config["vocab_size"] = yaml_cfg.vocab_size
        print(f"[Info] Auto-detected vocab_size = {yaml_cfg.vocab_size} "
              f"(dim_process={num_types} + 1 for padding)")

    time_mean = train_loader.dataset.time_mean
    time_std = train_loader.dataset.time_std
    time_quantile_05 = train_loader.dataset.time_quantile_05
    time_quantile_95 = train_loader.dataset.time_quantile_95

else:
    # Default: binary format (.bin files)
    train_loader = get_dataloader_smurf(
        data_dir=data_dir,
        batch_size=yaml_cfg.batch_size,
        split=train_split_name,
        shuffle=True,
        normalize="normal"
    )

    time_mean = train_loader.dataset.time_mean

    val_loader = get_dataloader_smurf(
        data_dir=data_dir,
        batch_size=yaml_cfg.batch_size,
        split="val",
        shuffle=True,
        normalize="normal",
        time_mean=time_mean,
    )

    time_mean = train_loader.dataset.time_mean
    time_std = train_loader.dataset.time_std
    time_quantile_05 = train_loader.dataset.time_quantile_05
    time_quantile_95 = train_loader.dataset.time_quantile_95

print("[Info] Time statistics from DatasetSmurf:")
print(f"  - time_mean: {time_mean:.2f}, time_std: {time_std:.2f}")
print(f"  - time_quantile_05: {time_quantile_05:.2f}, time_quantile_95: {time_quantile_95:.2f}")

steps_per_epoch = len(train_loader)
max_iters = yaml_cfg.num_epochs * steps_per_epoch

iter_num = 0
best_val_loss = 1e9

# -----------------------------------------------------------------------------
# model init
# -----------------------------------------------------------------------------

model_args = dict(
    n_layer=yaml_cfg.n_layer,
    n_head=yaml_cfg.n_head,
    n_embd=yaml_cfg.n_embd,
    bias=yaml_cfg.bias,
    vocab_size=yaml_cfg.vocab_size,
    dropout=yaml_cfg.dropout,
    token_dropout=yaml_cfg.token_dropout,
    t_min=yaml_cfg.t_min,
    mask_ties=yaml_cfg.mask_ties,
    ignore_tokens=yaml_cfg.ignore_tokens,
    use_score_matching=yaml_cfg.use_score_matching,
    parametrize=yaml_cfg.parametrize,
    add_noise=yaml_cfg.add_noise,
    noise_type=yaml_cfg.noise_type,
    var_noise=yaml_cfg.var_noise,
    num_noise=yaml_cfg.num_noise,
    langevin_step=yaml_cfg.langevin_step,
    n_langevin_steps=yaml_cfg.n_langevin_steps,
    n_samples=yaml_cfg.n_samples,
    denoise_step=yaml_cfg.denoise_step,
    loss_lambda=yaml_cfg.loss_lambda,
    time_mean=time_mean,
    time_std=time_std,
    time_quantile_05=time_quantile_05,
    time_quantile_95=time_quantile_95,
)

if yaml_cfg.init_from == "scratch":
    print("Initializing a new DelphiSMURF model from scratch")
    gptconf = DelphiSMURFConfig(**model_args)
    model = DelphiSMURF(gptconf)

elif yaml_cfg.init_from == "resume":
    print(f"Resuming training from {yaml_cfg.out_dir}")
    ckpt_path = os.path.join(yaml_cfg.out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=yaml_cfg.device)

    checkpoint_model_args = checkpoint["model_args"]
    for k in ["n_layer", "n_head", "n_embd", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]
    for k in [
        "use_score_matching", "parametrize", "add_noise", "noise_type",
        "var_noise", "num_noise", "loss_lambda"
    ]:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]

    gptconf = DelphiSMURFConfig(**model_args)
    model = DelphiSMURF(gptconf)
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)

    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
else:
    raise ValueError(f"Unsupported init_from: {yaml_cfg.init_from}")

model.to(yaml_cfg.device)

scaler = torch.amp.GradScaler("cuda", enabled=(yaml_cfg.dtype == "float16"))

optimizer = model.configure_optimizers(
    yaml_cfg.weight_decay,
    yaml_cfg.learning_rate,
    (yaml_cfg.beta1, yaml_cfg.beta2),
    device_type,
)

if yaml_cfg.init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])

# -----------------------------------------------------------------------------
# lr scheduler: linear warmup → cosine annealing (per-step)
# -----------------------------------------------------------------------------

warmup_iters = yaml_cfg.warmup_iters
cosine_iters = max(max_iters - warmup_iters, 1)

warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=1e-8 / yaml_cfg.learning_rate,  # start from ~0
    end_factor=1.0,
    total_iters=warmup_iters,
)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=cosine_iters,
    eta_min=yaml_cfg.eta_min,
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_iters],
)

if yaml_cfg.init_from == "resume" and "scheduler" in checkpoint:
    scheduler.load_state_dict(checkpoint["scheduler"])
    # fast-forward scheduler to the correct step
elif yaml_cfg.init_from == "resume":
    # legacy checkpoint without scheduler state — fast-forward
    for _ in range(iter_num):
        scheduler.step()

print(f"[Info] LR schedule: warmup {warmup_iters} iters → cosine decay over "
      f"{cosine_iters} iters (total {max_iters}), eta_min={yaml_cfg.eta_min}")

if yaml_cfg.compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)


def estimate_loss():
    """Estimate loss on train/val splits (training losses only)."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        loader = train_loader if split == "train" else val_loader
        losses = torch.zeros(yaml_cfg.eval_iters, 2)

        loader_iter = iter(loader)
        for i in range(yaml_cfg.eval_iters):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)

            X, A, Y, B, time_gap = thp_batch_to_smurf_batch(batch, yaml_cfg.device)
            X = X.to(yaml_cfg.device)
            A = A.to(yaml_cfg.device)
            Y = Y.to(yaml_cfg.device)
            B = B.to(yaml_cfg.device)
            time_gap = time_gap.to(yaml_cfg.device)

            with torch.enable_grad(), ctx:
                logits, loss, _ = model(X, A, Y, B, time_gap=time_gap, validation_loss_mode=True)

            losses[i] = torch.stack([
                loss["loss_ce"].detach(),
                loss["loss_score"].detach(),
            ])
            del logits, loss

        out[split] = losses.mean(0)

    model.train()
    if "cuda" in yaml_cfg.device:
        torch.cuda.empty_cache()
    return out


# -----------------------------------------------------------------------------
# logging
# -----------------------------------------------------------------------------

if yaml_cfg.wandb_log:
    import wandb

    wandb_token = os.environ.get("WANDB_TOKEN")
    if wandb_token:
        wandb.login(key=wandb_token)
    else:
        print("[Warning] WANDB_TOKEN not found in environment. "
              "Falling back to cached credentials or interactive login.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb_run_name = f"{yaml_cfg.config_type}-{yaml_cfg.dataset}-{yaml_cfg.batch_size}-{yaml_cfg.num_epochs}-{yaml_cfg.learning_rate}-{timestamp}"
    wandb.init(
        project=yaml_cfg.wandb_project,
        name=wandb_run_name,
        config=config,
    )
    # Define custom x-axis for step-based metrics
    wandb.define_metric("iter")
    wandb.define_metric("train/*", step_metric="iter")
    wandb.define_metric("val/*", step_metric="iter")
    wandb.define_metric("lr", step_metric="iter")
    wandb.define_metric("epoch", step_metric="iter")
    wandb.define_metric("timing/*", step_metric="iter")

# -----------------------------------------------------------------------------
# initial batch validation
# -----------------------------------------------------------------------------

data_iter = iter(train_loader)
tokens, ages, time_gap = next(data_iter)
tokens = tokens.to(yaml_cfg.device)
ages = ages.to(yaml_cfg.device)
time_gap = time_gap.to(yaml_cfg.device)

t0 = time.time()
local_iter_num = 0
val_loss = None

log_loss_all = []
log_loss_ce = []
log_loss_sc = []

# -----------------------------------------------------------------------------
# training loop (epoch-based, kept as requested)
# -----------------------------------------------------------------------------

for epoch in range(yaml_cfg.num_epochs):
    print(f"--- Epoch {epoch + 1}/{yaml_cfg.num_epochs} ---")

    for batch_idx, batch in enumerate(train_loader):
        X, A, Y, B, time_gap = thp_batch_to_smurf_batch(batch, yaml_cfg.device)
        # X = X.to(yaml_cfg.device)
        # A = A.to(yaml_cfg.device)
        # Y = Y.to(yaml_cfg.device)
        # B = B.to(yaml_cfg.device)
        # time_gap = time_gap.to(yaml_cfg.device)

        lr = optimizer.param_groups[0]["lr"]

        if iter_num > 0 and iter_num % yaml_cfg.eval_interval == 0:
            losses = estimate_loss()

            if val_loss is None:
                val_loss_unpooled = losses["val"]
            val_loss_unpooled = 0.1 * losses["val"] + 0.9 * val_loss_unpooled

            val_loss = (
                val_loss_unpooled[0].item()
                + yaml_cfg.loss_lambda * val_loss_unpooled[1].item()
            )

            train_loss_ce = losses["train"][0].item()
            train_loss_score = losses["train"][1].item()
            train_loss_backward = train_loss_ce + yaml_cfg.loss_lambda * train_loss_score

            val_loss_ce = val_loss_unpooled[0].item()
            val_loss_score = val_loss_unpooled[1].item()
            val_loss_backward = val_loss_ce + yaml_cfg.loss_lambda * val_loss_score

            print(f"step {iter_num}: train loss {train_loss_backward:.4f}, val loss {val_loss_backward:.4f}")
            print(f"  - val loss_ce: {val_loss_ce:.4f}, val loss_score: {val_loss_score:.4f}")

            if yaml_cfg.wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "epoch": epoch + 1,
                    # Validation losses (smoothed)
                    "val/loss_total": val_loss_backward,
                    "val/loss_ce": val_loss_ce,
                    "val/loss_score": val_loss_score,
                    "val/best_loss": best_val_loss,
                    # Training eval losses
                    "train/eval_loss_total": train_loss_backward,
                    "train/eval_loss_ce": train_loss_ce,
                    "train/eval_loss_score": train_loss_score,
                })

            if yaml_cfg.always_save_checkpoint or val_loss < best_val_loss:
                best_val_loss = val_loss
                if iter_num > 0:
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": val_loss,
                        "config": config,
                    }
                    print(f"saving checkpoint to {yaml_cfg.out_dir} ckpt.pt")
                    torch.save(checkpoint, os.path.join(yaml_cfg.out_dir, "ckpt.pt"))

            if yaml_cfg.always_save_checkpoint:
                if iter_num > 0:
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": val_loss,
                        "config": config,
                    }
                    print(f"saving last checkpoint to {yaml_cfg.out_dir} ckpt_last.pt")
                    torch.save(checkpoint, os.path.join(yaml_cfg.out_dir, "ckpt_last.pt"))

        if iter_num == 0 and yaml_cfg.eval_only:
            break

        with ctx:
            logits, loss, att = model(X, A, Y, B, time_gap=time_gap)
            loss_backward = loss["loss_total"]
            scaler.scale(loss_backward).backward()

        # loss step forward
        if yaml_cfg.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), yaml_cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        iter_num += 1
        local_iter_num += 1

        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        loss_backward_f = loss_backward.item()
        loss_ce_f = loss["loss_ce"].item()
        loss_score_f = loss["loss_score"].item()

        log_loss_all.append(loss_backward_f)
        log_loss_ce.append(loss_ce_f)
        log_loss_sc.append(loss_score_f)

        if iter_num % yaml_cfg.log_interval == 0:
            avg_loss_backward = np.mean(log_loss_all)
            avg_loss_ce = np.mean(log_loss_ce)
            avg_loss_sc = np.mean(log_loss_sc)

            log_loss_all.clear()
            log_loss_ce.clear()
            log_loss_sc.clear()

            print(
                f"iter {iter_num}/{max_iters}: "
                f"loss(latest-avg) {loss_backward_f:.4f}-{avg_loss_backward:.4f} "
                f"(ce(latest-avg): {loss_ce_f:.4f}-{avg_loss_ce:.4f}, "
                f"score(latest-avg): {loss_score_f:.4f}-{avg_loss_sc:.4f}), "
                f"time {dt * 1000:.2f}ms"
            )

            if yaml_cfg.wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "epoch": epoch + 1,
                    # Per-step training losses (latest)
                    "train/loss_total": loss_backward_f,
                    "train/loss_ce": loss_ce_f,
                    "train/loss_score": loss_score_f,
                    # Averaged training losses (over log_interval)
                    "train/avg_loss_total": avg_loss_backward,
                    "train/avg_loss_ce": avg_loss_ce,
                    "train/avg_loss_score": avg_loss_sc,
                    # Learning rate & timing
                    "lr": lr,
                    "timing/step_ms": dt * 1000,
                })

    if iter_num == 0 and yaml_cfg.eval_only:
        break

if yaml_cfg.wandb_log:
    wandb.finish()
