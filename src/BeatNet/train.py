# Author: Mojtaba Heydari <mheydari@ur.rochester.edu>
# Main training script for BeatNet.
#
# Usage:
#   python -m BeatNet.train --config src/BeatNet/configs/default.yaml
#   python -m BeatNet.train --config configs/default.yaml learning_rate=0.001 batch_size=128
#   python -m BeatNet.train --config configs/default.yaml --resume output/checkpoint_epoch_100.pt

import argparse
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from BeatNet.dataset import build_datasets
from BeatNet.model import BDA

logger = logging.getLogger('BeatNet.train')


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path, overrides=None):
    """Load YAML config and apply key=value CLI overrides."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if overrides:
        for ov in overrides:
            if '=' not in ov:
                continue
            key, val = ov.split('=', 1)
            # Try to parse as int, float, bool, or leave as string
            for parser in (int, float):
                try:
                    val = parser(val)
                    break
                except ValueError:
                    continue
            else:
                if val.lower() in ('true', 'false'):
                    val = val.lower() == 'true'
            config[key] = val

    return config


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(model, val_loader, inference_type, device):
    """Run validation: model predictions -> inference decoding -> F-measure evaluation.

    Returns
    -------
    avg_beat_f : float
        Average beat F-measure across validation tracks.
    avg_down_f : float
        Average downbeat F-measure across validation tracks.
    """
    from madmom.evaluation import BeatEvaluation
    from madmom.features import DBNDownBeatTrackingProcessor

    model.eval()
    beat_fmeasures = []
    down_fmeasures = []

    with torch.no_grad():
        for batch in val_loader:
            # batch size is 1 for validation
            feats = batch['feats'].transpose(1, 2).to(device)   # (1, T, 272)
            gt = batch['ground_truth']                           # (1, 3, T)

            # Get model activations
            preds = model.train_forward(feats)[0]                # (3, T)
            preds = F.softmax(preds, dim=0)
            preds = preds.cpu().numpy()
            preds = np.transpose(preds[:2, :])                   # (T, 2): [beat_act, down_act]

            # Ground truth times
            gt_np = gt[0].numpy()                                # (3, T)
            beats_g = np.argwhere(gt_np[0] == 1) * 0.02         # frame to time at 50fps
            downs_g = np.argwhere(gt_np[1] == 1) * 0.02
            # Beats include downbeats for evaluation
            beats_g = np.sort(np.append(downs_g, beats_g)).flatten()
            downs_g = downs_g.flatten()

            if len(beats_g) == 0:
                continue

            # Inference decoding
            try:
                if inference_type == 'DBN':
                    meter = max(2, round(len(beats_g) / max(1, len(downs_g))))
                    meter = min(meter, 4)
                    dbn = DBNDownBeatTrackingProcessor(
                        beats_per_bar=[meter], fps=50, observation_lambda=16
                    )
                    decoded = dbn(preds)
                    if len(decoded) == 0:
                        continue
                    pred_downs = decoded[:, 0][decoded[:, 1] == 1]
                    pred_beats = decoded[:, 0]
                elif inference_type == 'PF':
                    from BeatNet.particle_filtering_cascade import particle_filter_cascade
                    pf = particle_filter_cascade(beats_per_bar=[], fps=50, plot=[], mode='online')
                    output = pf.process(preds)
                    if output is None or len(output) == 0:
                        continue
                    pred_beats = output[:, 0]
                    pred_downs = output[:, 0][output[:, 1] == 1]
                else:
                    raise ValueError(f"Unknown inference type: {inference_type}")
            except (ValueError, IndexError):
                # DBN/PF can fail on very short or degenerate activations
                continue

            # Evaluate
            if len(pred_beats) > 0 and len(beats_g) > 0:
                beat_eval = BeatEvaluation(pred_beats, beats_g, skip=5)
                beat_fmeasures.append(beat_eval.fmeasure)

            if len(pred_downs) > 0 and len(downs_g) > 0:
                down_eval = BeatEvaluation(pred_downs, downs_g, skip=5)
                down_fmeasures.append(down_eval.fmeasure)

    avg_beat_f = np.mean(beat_fmeasures) if beat_fmeasures else 0.0
    avg_down_f = np.mean(down_fmeasures) if down_fmeasures else 0.0
    return avg_beat_f, avg_down_f


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config):
    """Main training procedure."""
    set_seed(config.get('seed', 42))
    device = torch.device(config.get('device', 'cpu'))

    # Output directory
    output_dir = config.get('output_dir', './output')
    os.makedirs(output_dir, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))

    # Build datasets
    logger.info("Building datasets...")
    train_dataset, val_dataset, test_dataset = build_datasets(config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 200),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=min(config.get('num_workers', 4), 2),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=min(config.get('num_workers', 4), 2),
        drop_last=False,
    )

    # Initialize model
    feature_dim = config.get('feature_dim', 272)
    num_cells = config.get('num_cells', 150)
    num_layers = config.get('num_layers', 2)
    model = BDA(feature_dim, num_cells, num_layers, device)
    model.to(device)
    logger.info(f"Model: BDA({feature_dim}, {num_cells}, {num_layers}) on {device}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    lr = config.get('learning_rate', 5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Class weights for cross-entropy
    cw = config.get('class_weights', [50, 400, 5])
    class_weights = torch.FloatTensor(cw).to(device)

    # Training state
    start_epoch = 0
    best_val_f = 0.0
    patience_counter = 0
    max_epochs = config.get('max_epochs', 10000)
    patience = config.get('patience', 20)
    checkpoint_every = config.get('checkpoint_every', 10)
    val_inference = config.get('val_inference', 'DBN')

    # Resume from checkpoint
    resume_path = config.get('resume', None)
    if resume_path and os.path.exists(resume_path):
        logger.info(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_val_f = ckpt.get('best_val_f', 0.0)
        patience_counter = ckpt.get('patience_counter', 0)
        logger.info(f"Resumed at epoch {start_epoch}, best_val_f={best_val_f:.4f}")

    # Training loop
    logger.info(f"Starting training: epochs={max_epochs}, batch_size={config.get('batch_size', 200)}, "
                f"lr={lr}, patience={patience}")

    for epoch in range(start_epoch, max_epochs):
        model.train()
        epoch_losses = []
        t_start = time.time()

        for batch in train_loader:
            optimizer.zero_grad()

            feats = batch['feats'].transpose(1, 2).to(device)    # (B, T, 272)
            gt = batch['ground_truth'].to(device)                  # (B, 3, T)

            preds = model.train_forward(feats)                     # (B, 3, T)

            # Cross-entropy expects (B, C, T) predictions and (B, T) integer targets
            targets = torch.argmax(gt, dim=1)                      # (B, T)
            loss = F.cross_entropy(preds, targets, weight=class_weights)

            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        elapsed = time.time() - t_start
        writer.add_scalar('train/loss', avg_loss, global_step=epoch + 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch + 1}/{max_epochs} | loss={avg_loss:.4f} | time={elapsed:.1f}s")

        # Checkpoint and validation
        if (epoch + 1) % checkpoint_every == 0:
            # Validate
            logger.info(f"Validating (epoch {epoch + 1})...")
            beat_f, down_f = validate(model, val_loader, val_inference, device)
            writer.add_scalar('val/beat_fmeasure', beat_f, global_step=epoch + 1)
            writer.add_scalar('val/downbeat_fmeasure', down_f, global_step=epoch + 1)
            logger.info(f"  Val beat F={beat_f:.4f}, downbeat F={down_f:.4f}")

            # Test set evaluation
            if len(test_dataset) > 0:
                test_beat_f, test_down_f = validate(model, test_loader, val_inference, device)
                writer.add_scalar('test/beat_fmeasure', test_beat_f, global_step=epoch + 1)
                writer.add_scalar('test/downbeat_fmeasure', test_down_f, global_step=epoch + 1)
                logger.info(f"  Test beat F={test_beat_f:.4f}, downbeat F={test_down_f:.4f}")

            # Save checkpoint
            ckpt_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch + 1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_f': best_val_f,
                'patience_counter': patience_counter,
                'config': config,
            }, ckpt_path)

            # Also save weights-only file compatible with inference code
            weights_path = os.path.join(output_dir, f'model_weights_epoch_{epoch + 1}.pt')
            torch.save(model.state_dict(), weights_path)

            # Early stopping check
            combined_f = beat_f  # track based on beat F-measure
            if combined_f > best_val_f:
                best_val_f = combined_f
                patience_counter = 0
                # Save best model
                best_path = os.path.join(output_dir, 'best_model_weights.pt')
                torch.save(model.state_dict(), best_path)
                logger.info(f"  New best model saved (beat F={best_val_f:.4f})")
            else:
                patience_counter += 1
                logger.info(f"  No improvement ({patience_counter}/{patience})")

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    # Save final model
    final_path = os.path.join(output_dir, 'final_model_weights.pt')
    torch.save(model.state_dict(), final_path)
    logger.info(f"Training complete. Final weights saved to {final_path}")
    logger.info(f"Best validation beat F-measure: {best_val_f:.4f}")

    writer.close()
    return model


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train BeatNet CRNN model')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('overrides', nargs='*', help='Config overrides as key=value pairs')
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    if args.resume:
        config['resume'] = args.resume

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    train(config)


if __name__ == '__main__':
    main()
