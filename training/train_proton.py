#!/usr/bin/env python3
"""Train or continue the final ProtonFiLM-Lite model from a prepared BEV cache."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter

from doserad2026.cache import ProtonCacheDataset
from doserad2026.losses import Level1Loss
from doserad2026.model import ProtonFiLMLiteResUNet3D, count_parameters


class EpochShuffleSampler(Sampler[int]):
    def __init__(self, dataset, seed: int) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        yield from torch.randperm(len(self.dataset), generator=generator).tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--min-lr", type=float, default=2.5e-5)
    parser.add_argument("--weight-decay", type=float, default=5.0e-4)
    parser.add_argument("--lateral-flip-prob", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", choices=("bf16", "fp16", "off"), default="bf16")
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--checkpoint-every", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--continue-phase", type=Path)
    parser.add_argument("--init-weights", type=Path)
    parser.add_argument("--max-train-batches", type=int, help="Smoke-test helper")
    parser.add_argument("--max-val-batches", type=int, help="Smoke-test helper")
    args = parser.parse_args()
    selected = sum(path is not None for path in (args.resume, args.continue_phase, args.init_weights))
    if selected > 1:
        parser.error("--resume, --continue-phase and --init-weights are mutually exclusive")
    if args.epochs < 1 or args.batch_size < 1 or args.accumulation_steps < 1:
        parser.error("epochs, batch-size and accumulation-steps must be positive")
    if not 0.0 <= args.lateral_flip_prob <= 1.0:
        parser.error("--lateral-flip-prob must be in [0, 1]")
    if not 0.0 <= args.min_lr < args.lr:
        parser.error("--min-lr must be non-negative and smaller than --lr")
    return args


def make_loader(dataset, batch_size: int, workers: int, sampler=None) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        drop_last=sampler is not None and len(dataset) >= batch_size,
    )


def make_optimizer(model, lr: float, weight_decay: float) -> AdamW:
    decay, no_decay = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return AdamW(
        (
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ),
        lr=lr,
        weight_decay=0.0,
    )


def amp_context(device: torch.device, mode: str):
    if device.type != "cuda" or mode == "off":
        return nullcontext()
    dtype = torch.bfloat16 if mode == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].detach().cpu().to(torch.uint8))
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(
            [item.detach().cpu().to(torch.uint8) for item in state["cuda"]]
        )


def atomic_save(state: dict, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    torch.save(state, temporary)
    os.replace(temporary, destination)


def load_state(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint {path} is not a dictionary")
    return state


def load_model_state(model: torch.nn.Module, state: dict) -> None:
    model_state = state.get("model", state)
    model.load_state_dict(model_state, strict=True)


def train_or_validate(
    model,
    loader,
    criterion,
    device,
    amp_mode,
    *,
    optimizer=None,
    scaler=None,
    accumulation_steps=1,
    grad_clip=0.0,
    max_batches=None,
    log_every=100,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "optimization_loss": 0.0, "masked_mae": 0.0, "idd": 0.0}
    seen = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    limit = len(loader) if max_batches is None else min(len(loader), max_batches)

    context = torch.enable_grad if training else torch.no_grad
    with context():
        for step, batch in enumerate(loader):
            if step >= limit:
                break
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            energy = batch["energy"].to(device, non_blocking=True)
            with amp_context(device, amp_mode):
                outputs = model(inputs, energy)
                prediction, aux_dec2, aux_dec3 = outputs
                terms = criterion(prediction, targets)
                optimization_loss = terms["loss"]
                if training:
                    for weight, auxiliary in ((0.20, aux_dec2), (0.10, aux_dec3)):
                        auxiliary_target = F.interpolate(
                            targets,
                            size=auxiliary.shape[2:],
                            mode="trilinear",
                            align_corners=False,
                        )
                        optimization_loss = (
                            optimization_loss + weight * criterion(auxiliary, auxiliary_target)["loss"]
                        )

            if training:
                backward_loss = optimization_loss / accumulation_steps
                if scaler is None:
                    backward_loss.backward()
                else:
                    scaler.scale(backward_loss).backward()
                update = (step + 1) % accumulation_steps == 0 or step + 1 == limit
                if update:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    if grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    if scaler is None:
                        optimizer.step()
                    else:
                        scaler.step(optimizer)
                        scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            batch_size = inputs.shape[0]
            for name in ("loss", "masked_mae", "idd"):
                totals[name] += float(terms[name].detach()) * batch_size
            totals["optimization_loss"] += float(optimization_loss.detach()) * batch_size
            seen += batch_size
            if log_every and ((step + 1) % log_every == 0 or step + 1 == limit):
                phase = "train" if training else "validation"
                print(
                    f"{phase} {step + 1}/{limit}: loss={totals['loss'] / seen:.7f}",
                    flush=True,
                )
    if seen == 0:
        raise RuntimeError("No batch was processed")
    return {name: value / seen for name, value in totals.items()}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.amp == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16; use --amp fp16")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    existing = list(args.out_dir.iterdir()) if args.out_dir.exists() else []
    if existing and args.resume is None:
        raise ValueError(f"Refusing to use non-empty output directory {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = ProtonCacheDataset(
        args.cache_dir / "train", lateral_flip_probability=args.lateral_flip_prob
    )
    validation_dataset = ProtonCacheDataset(args.cache_dir / "validation")
    sampler = EpochShuffleSampler(train_dataset, args.seed)
    train_loader = make_loader(train_dataset, args.batch_size, args.num_workers, sampler)
    validation_loader = make_loader(validation_dataset, args.batch_size, args.num_workers)

    model = ProtonFiLMLiteResUNet3D(
        base_channels=6, deep_dropout=0.05, deep_supervision=True
    ).to(device)
    criterion = Level1Loss()
    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp == "fp16")
    start_epoch = 0
    best = float("inf")

    if args.init_weights:
        load_model_state(model, load_state(args.init_weights))
    elif args.continue_phase:
        checkpoint = load_state(args.continue_phase)
        load_model_state(model, checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
            group["initial_lr"] = args.lr
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        if "rng_state" in checkpoint:
            restore_rng_state(checkpoint["rng_state"])
        best = float(checkpoint.get("best", best))
    elif args.resume:
        checkpoint = load_state(args.resume)
        load_model_state(model, checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        if "rng_state" in checkpoint:
            restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint["best"])

    config = vars(args).copy()
    config.update(
        {
            "architecture": "ProtonFiLMLiteResUNet3D",
            "base_channels": 6,
            "deep_supervision": True,
            "loss": "masked_beam_mae + idd",
            "scheduler": "cosine",
            "weight_decay_mode": "weights-only",
        }
    )
    (args.out_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8"
    )
    writer = SummaryWriter(str(args.out_dir / "tensorboard"), flush_secs=30)
    metrics_file = args.out_dir / "metrics.jsonl"
    print(
        f"device={device} parameters={count_parameters(model):,} "
        f"effective_batch={args.batch_size * args.accumulation_steps}",
        flush=True,
    )

    try:
        for epoch in range(start_epoch, args.epochs):
            sampler.set_epoch(epoch)
            started = time.perf_counter()
            learning_rate = optimizer.param_groups[0]["lr"]
            train_metrics = train_or_validate(
                model,
                train_loader,
                criterion,
                device,
                args.amp,
                optimizer=optimizer,
                scaler=scaler if scaler.is_enabled() else None,
                accumulation_steps=args.accumulation_steps,
                grad_clip=args.grad_clip,
                max_batches=args.max_train_batches,
                log_every=args.log_every,
            )
            validation_metrics = train_or_validate(
                model,
                validation_loader,
                criterion,
                device,
                args.amp,
                max_batches=args.max_val_batches,
                log_every=args.log_every,
            )
            scheduler.step()
            record = {
                "epoch": epoch,
                "seconds": time.perf_counter() - started,
                "lr": learning_rate,
                "next_lr": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "validation": validation_metrics,
            }
            print(json.dumps(record), flush=True)
            with metrics_file.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            for phase, values in (("train", train_metrics), ("validation", validation_metrics)):
                for name, value in values.items():
                    writer.add_scalar(f"epoch/{phase}_{name}", value, epoch + 1)
            writer.add_scalar("epoch/learning_rate", learning_rate, epoch + 1)
            writer.flush()

            state = {
                "epoch": epoch,
                "best": min(best, validation_metrics["loss"]),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "rng_state": capture_rng_state(),
                "config": config,
            }
            if validation_metrics["loss"] < best:
                best = validation_metrics["loss"]
                atomic_save(state, args.out_dir / "best.pt")
            atomic_save(state, args.out_dir / "last.pt")
            if args.checkpoint_every and (epoch + 1) % args.checkpoint_every == 0:
                atomic_save(state, args.out_dir / f"epoch_{epoch + 1:03d}.pt")
    finally:
        writer.close()


if __name__ == "__main__":
    main()
