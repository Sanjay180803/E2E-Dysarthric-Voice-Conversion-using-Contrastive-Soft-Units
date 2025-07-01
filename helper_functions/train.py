import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from acoustic import AcousticModel
from acoustic.dataset import MelDataset
from acoustic.utils import Metric, save_checkpoint, load_checkpoint, plot_spectrogram

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

########################################################################################
# Define hyperparameters for training:
########################################################################################

BATCH_SIZE = 32
LEARNING_RATE = 4e-4
BETAS = (0.8, 0.99)
WEIGHT_DECAY = 1e-5
STEPS = 80000        # total training steps
LOG_INTERVAL = 5
VALIDATION_INTERVAL = 1000
CHECKPOINT_INTERVAL = 1000

def train(args):
    ####################################################################################
    # Setup logging utilities:
    ####################################################################################
    log_dir = args.checkpoint_dir / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)

    # File logging
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_dir / f"{args.checkpoint_dir.stem}.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%m/%d/%Y %I:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # TensorBoard writer
    writer = SummaryWriter(log_dir)

    # Choose CPU explicitly (change to 'cuda' if you want GPU instead)
    device = torch.device("cpu")

    ####################################################################################
    # Initialize model and optimizer
    ####################################################################################
    acoustic = AcousticModel(discrete=args.discrete).to(device)

    optimizer = optim.AdamW(
        acoustic.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )

    ####################################################################################
    # Initialize datasets and dataloaders
    ####################################################################################
    train_dataset = MelDataset(
        root=args.dataset_dir,
        train=True,
        discrete=args.discrete,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,         # shuffle in single-process mode
        collate_fn=train_dataset.pad_collate,
        num_workers=0,        # for CPU, typically set to 0 or 1
        pin_memory=False,
        drop_last=True,
    )

    validation_dataset = MelDataset(
        root=args.dataset_dir,
        train=False,
        discrete=args.discrete,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=validation_dataset.pad_collate,
        num_workers=0,
        pin_memory=False,
    )

    ####################################################################################
    # Load checkpoint if args.resume is set
    ####################################################################################
    if args.resume is not None:
        global_step, best_loss = load_checkpoint(
            load_path=args.resume,
            acoustic=acoustic,
            optimizer=optimizer,
            rank=0,         # dummy rank
            logger=logger,
        )
    else:
        global_step, best_loss = 0, float("inf")

    # Compute how many epochs to run based on total steps
    steps_per_epoch = len(train_loader)
    n_epochs = STEPS // steps_per_epoch + 1
    start_epoch = global_step // steps_per_epoch + 1

    logger.info("**" * 40)
    logger.info(f"Running on device: {device}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"batch size: {BATCH_SIZE}")
    logger.info(f"iterations per epoch: {steps_per_epoch}")
    logger.info(f"max # of epochs (based on STEPS): {n_epochs}")
    logger.info(f"starting at epoch: {start_epoch}")
    logger.info("**" * 40 + "\n")

    average_loss = Metric()
    epoch_loss = Metric()
    validation_loss = Metric()

    ####################################################################################
    # Training loop
    ####################################################################################
    for epoch in range(start_epoch, n_epochs + 1):
        acoustic.train()
        epoch_loss.reset()

        for mels, mels_lengths, units, units_lengths in train_loader:
            # Move data to CPU
            mels, mels_lengths = mels.to(device), mels_lengths.to(device)
            units, units_lengths = units.to(device), units_lengths.to(device)

            # Forward pass
            optimizer.zero_grad()
            # mels shape: [batch_size, seq_len, mel_dim]
            # We predict the next frame from the previous:
            # mels[:, :-1, :] are inputs, mels[:, 1:, :] are targets
            mels_pred = acoustic(units, mels[:, :-1, :])

            # Compute L1 loss between predicted frames and ground truth
            loss = F.l1_loss(mels_pred, mels[:, 1:, :], reduction="none")
            # Weighted by sequence length to handle padding:
            # (mels_pred.size(-1) = mel_dim, mels_lengths = actual # of frames - 1)
            loss = torch.sum(loss, dim=(1, 2)) / (mels_pred.size(-1) * mels_lengths)
            loss = torch.mean(loss)

            # Backprop
            loss.backward()
            optimizer.step()

            global_step += 1

            # Track training metrics
            average_loss.update(loss.item())
            epoch_loss.update(loss.item())

            # Logging
            if (global_step % LOG_INTERVAL) == 0:
                writer.add_scalar("train/loss", average_loss.value, global_step)
                logger.info(f"[Step {global_step}] train loss: {average_loss.value:.4f}")
                average_loss.reset()

            # Validation
            if (global_step % VALIDATION_INTERVAL) == 0:
                acoustic.eval()
                validation_loss.reset()

                for i, (mels_val, units_val, _, _) in enumerate(validation_loader, start=1):
                    mels_val, units_val = mels_val.to(device), units_val.to(device)

                    with torch.no_grad():
                        mels_pred_val = acoustic(units_val, mels_val[:, :-1, :])
                        loss_val = F.l1_loss(mels_pred_val, mels_val[:, 1:, :])

                    validation_loss.update(loss_val.item())

                    # Save a sample mel plot
                    if i <= 3:  # limit how many you log
                        fig = plot_spectrogram(
                            mels_pred_val.squeeze().transpose(0, 1).cpu().numpy()
                        )
                        writer.add_figure(f"generated/mel_{i}", fig, global_step)

                # Log validation
                avg_val_loss = validation_loss.value
                writer.add_scalar("validation/loss", avg_val_loss, global_step)
                logger.info(f"[Validation] epoch: {epoch}, loss: {avg_val_loss:.4f}")

                # Checkpoint if improved or at intervals
                new_best = best_loss > avg_val_loss
                if new_best or (global_step % CHECKPOINT_INTERVAL) == 0:
                    if new_best:
                        logger.info("-------- new best model found!")
                        best_loss = avg_val_loss

                    save_checkpoint(
                        checkpoint_dir=args.checkpoint_dir,
                        acoustic=acoustic,
                        optimizer=optimizer,
                        step=global_step,
                        loss=avg_val_loss,
                        best=new_best,
                        logger=logger,
                    )

                acoustic.train()

        # End of epoch logging
        logger.info(f"train -- epoch: {epoch}, loss: {epoch_loss.value:.4f}")

    logger.info("Training complete!")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the acoustic model (CPU-only).")
    parser.add_argument(
        "dataset_dir",
        metavar="dataset-dir",
        help="path to the data directory.",
        type=Path,
    )
    parser.add_argument(
        "checkpoint_dir",
        metavar="checkpoint-dir",
        help="path to the checkpoint directory.",
        type=Path,
    )
    parser.add_argument(
        "--resume",
        help="path to the checkpoint to resume from.",
        type=Path,
    )
    parser.add_argument(
        "--discrete",
        action="store_true",
        help="use discrete units.",
    )
    args = parser.parse_args()

    # Simply call train(args) in single process
    train(args)
