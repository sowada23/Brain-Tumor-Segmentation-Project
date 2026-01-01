from __future__ import annotations
import datetime
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..utils.seed import set_seed
from ..utils.io import make_output_dirs, save_splits, list_case_ids_under, save_final_plots
from ..utils.device import get_device            # ← use your helper
from ..utils.db import ExperimentDB
from ..data.brats_slice_dataset import build_loaders
from ..models.unet2d import UNet2D
from ..losses.combined import CombinedLoss
from .train import train_one_epoch, evaluate


def main(config_path: str):
    # ---- load config ----
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = make_output_dirs(run_id, root=cfg.get("output_root", "Output"))
    db_path = Path(cfg.get("output_root", "Output")) / "experiment.sqlite"

    # ---- seed & device ----
    set_seed(int(cfg.get("seed", 42)))
    # cfg["device"] can be "auto" | "cuda" | "mps" | "cpu"
    device = get_device(cfg.get("device", "auto"))
    print(f"Using device: {device}")

    # ---- discover cases & split ----
    modalities = tuple(cfg.get("input_modalities", ["flair", "t1ce"]))
    all_ids = list_case_ids_under(cfg["data_root"], modalities, require_seg=True)
    assert len(all_ids) > 0, "No training cases with all modalities + seg found."

    val_split = float(cfg.get("val_split", 0.15))
    test_split = float(cfg.get("test_split", 0.15))
    train_ids, temp_ids = train_test_split(
        all_ids,
        test_size=val_split + test_split,
        shuffle=True,
        random_state=int(cfg.get("seed", 42))
    )
    rel_test = test_split / (val_split + test_split) if (val_split + test_split) > 0 else 0.0
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=rel_test,
        shuffle=True,
        random_state=int(cfg.get("seed", 42))
    )

    def _limit_split(ids: list[str], max_n: int | None, split_name: str):
        if max_n is None:
            return ids
        if max_n <= 0:
            raise ValueError(f"{split_name}: max_n must be postive when provided(got {max_n})")
        return ids[:max_n]
        
    train_ids = _limit_split(train_ids, cfg.get("max_train_cases"), "train")
    val_ids = _limit_split(val_ids, cfg.get("max_val_cases"), "val")
    test_ids = _limit_split(test_ids, cfg.get("max_test_cases"), "test")


    print(f"Split → train {len(train_ids)} | val {len(val_ids)} | test {len(test_ids)} (total {len(all_ids)})")
    save_splits(out["logs"], run_id, {"train": train_ids, "val": val_ids, "test": test_ids})

    # ---- data loaders ----
    roots = [cfg["data_root"]]
    ds_args = dict(
        img_size=int(cfg.get("img_size", 128)),
        volume_slices=int(cfg.get("volume_slices", 100)),
        volume_start_at=int(cfg.get("volume_start_at", 22)),
        cache_size=int(cfg.get("cache_size", 6)),
        modalities=modalities,
    )
    loaders = build_loaders(
        roots=roots,
        train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
        batch_size=int(cfg.get("batch_size", 64)),
        num_workers=int(cfg.get("num_workers", 16)),
        # If your build_loaders supports pin_memory, you can pass:
        # pin_memory=(device.type == "cuda"),
        **ds_args
    )
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = loaders

    # ---- model & loss ----
    model = UNet2D(in_ch=len(modalities), n_classes=4, dropout=float(cfg.get("dropout", 0.35))).to(device)
    loss_fn = CombinedLoss(alpha=float(cfg.get("alpha", 0.5)), beta=float(cfg.get("beta", 0.5)))
    optimizer = AdamW(model.parameters(), lr=float(cfg.get("lr", 2e-4)),
                      weight_decay=float(cfg.get("weight_decay", 1e-4)))
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                  patience=int(cfg.get("lr_patience", 5)),
                                  min_lr=float(cfg.get("lr_min", 1e-6)))

    """
    ExperimentDB keeps a single SQLite file under Output/ and exposes tiny helpers that the loop
    can call to record what happened. Using is inside a context manager ensures the connections closes cleanly
    when training ends.

    """ 

    with ExperimentDB(db_path) as db:
        # Store the run metadata + config once so future inserts can reference it.
        db.log_run_start(run_id, out["base"], cfg)

        # ---- train loop ----
        from collections import defaultdict
        history = defaultdict(list)
        best_val = float("-inf")
        best_path = out["weights"] / "best_model.pt"
        epochs = int(cfg.get("epochs", 40))

        for epoch in range(1, epochs + 1):
            train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            val_metrics   = evaluate(model, val_loader, loss_fn, device)
            scheduler.step(val_metrics['dice_no_bg'])

            for k in ["loss", "accuracy", "dice_no_bg", "mean_iou"]:
                history[k].append(float(train_metrics.get(k, float('nan'))))
                history[f"val_{k}"].append(float(val_metrics.get(k, float('nan'))))

            # Mirror the existing history tracking in SQLite for later queries.
            db.log_epoch(
                run_id=run_id,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                lr=optimizer.param_groups[0]["lr"],
            )

            print(
                f"# ------ Epoch {epoch:03d} ------ # \n"
                f"train_loss {train_metrics['loss']:.4f} | "
                f"(CE {train_metrics['ce_loss']:.4f} + Dice {train_metrics['dice_loss']:.4f}) | \n"
                f"val_loss {val_metrics['loss']:.4f} | "
                f"(CE {val_metrics['ce_loss']:.4f} + Dice {val_metrics['dice_loss']:.4f}) | \n"
                f"DiceNB {train_metrics['dice_no_bg']:.4f} | \n"
                f"val_DiceNB {val_metrics['dice_no_bg']:.4f} | \n"
                f"mIoU {train_metrics['mean_iou']:.4f} | \n"
                f"val_mIoU {val_metrics['mean_iou']:.4f} | \n"
                f"lr {optimizer.param_groups[0]['lr']:.2e} "
            )

            if val_metrics['dice_no_bg'] > best_val:
                best_val = val_metrics['dice_no_bg']
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'dice_no_bg': best_val,
                    'settings': {
                        'img_size': ds_args["img_size"],
                        'volume_slices': ds_args["volume_slices"],
                        'volume_start_at': ds_args["volume_start_at"],
                        'dropout': float(cfg.get("dropout", 0.35)),
                        'lr': float(cfg.get("lr", 2e-4)),
                        'weight_decay': float(cfg.get("weight_decay", 1e-4)),
                        'modalities': modalities,
                    },
                    'splits': {"train": train_ids, "val": val_ids, "test": test_ids},
                }, best_path)

                # Keep the best checkpoint metadata alongside the run row.
                db.log_best_checkpoint(run_id, epoch, best_path, best_val)
                print(f"Saved best model → {best_path} (dice_no_bg={best_val:.6f})")

        print("Training complete.")

        # ----- Plots ----- #
        save_final_plots(history, out["logs"], run_id = run_id)
        print(f"Saved final plots to {out['logs']}")

        # ----- Test ----- #
        test_metrics = evaluate(model, test_loader, loss_fn, device)
        
        # Capture the final held-out test metrics in the database as well.
        db.log_test_results(run_id, test_metrics)
        out_path = out["logs"] / f"test_results_{run_id}.txt"
        with open(out_path, "w") as f:
            for k in ["loss", "accuracy", "mean_iou", "dice_no_bg"]:
                v = float(test_metrics.get(k, float('nan')))
                f.write(f"{k}: {v:.6f}\n")
        print(f"Saved test results to {out_path}")

        # ----- Final weights ----- #
        final_state_path = out["model"] / "BrainTumSeg_state_dict.pt"
        torch.save(model.state_dict(), final_state_path)
        print(f"Saved final state dict to {final_state_path}")