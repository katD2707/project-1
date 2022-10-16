import os
import time
import math
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

import utils


def save_checkpoint(
        epoch,
        checkpoints_path,
        model,
        optimizer,
        lr_scheduler=None,
):
    """
    Save the current state of the model, optimizer and learning rate scheduler,
    both locally and on wandb (if available and enabled)
    """
    # Create state dictionary
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": (
            lr_scheduler.state_dict() if lr_scheduler is not None else dict()
        ),
        "epoch": epoch,
    }

    # Save state dictionary
    checkpoint_file = os.path.join(checkpoints_path, f"titanet_epoch_{epoch}.pth")
    torch.save(state_dict, checkpoint_file)

def training_loop(
        # run_name,
        epochs,
        model,
        optimizer,
        train_dataloader,
        checkpoints_path,
        test_dataset=None,
        val_dataloader=None,
        val_every=None,
        lr_scheduler=None,
        checkpoints_frequency=None,
        mindcf_p_target=0.01,
        mindcf_c_fa=1,
        mindcf_c_miss=1,
        device="cpu",
):
    """
    Standard training loop function: train and evaluate
    after each training epoch
    """
    # Create checkpoints directory
    # checkpoints_path = os.path.join(checkpoints_path, run_name)
    if os.path.exists(checkpoints_path) is False:
        os.makedirs(checkpoints_path)

    # For each epoch
    for epoch in range(1, epochs + 1):

        # Train for one epoch
        train_one_epoch(
            epoch,
            epochs,
            model,
            optimizer,
            train_dataloader,
            lr_scheduler=lr_scheduler,
            device=device,
        )

        # Decay the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Save checkpoints once in a while
        if checkpoints_frequency is not None and epoch % checkpoints_frequency == 0:
            save_checkpoint(
                epoch,
                checkpoints_path,
                model,
                optimizer,
                lr_scheduler=lr_scheduler,
            )

        # Evaluate once in a while (always evaluate at the first and last epochs)
        if (
                val_dataloader is not None
                and val_every is not None
                and (epoch % val_every == 0 or epoch == 1 or epoch == epochs)
        ):
            evaluate(
                model,
                val_dataloader,
                current_epoch=epoch,
                total_epochs=epochs,
                device=device,
            )

    # Always save the last checkpoint
    save_checkpoint(
        epochs,
        checkpoints_path,
        model,
        optimizer,
        lr_scheduler=lr_scheduler,
    )

    # Final test
    if test_dataset is not None:
        test(
            model,
            test_dataset,
            mindcf_p_target=mindcf_p_target,
            mindcf_c_fa=mindcf_c_fa,
            mindcf_c_miss=mindcf_c_miss,
            device=device,
        )

def train_one_epoch(
        current_epoch,
        total_epochs,
        model,
        optimizer,
        dataloader,
        lr_scheduler=None,
        device="cpu",
):
    """
    Train the given model for one epoch with the given dataloader and optimizer
    """
    # Put the model in training mode
    model.train()

    # For each batch
    step = 1
    epoch_loss, epoch_data_time, epoch_model_time, epoch_opt_time = 0, 0, 0, 0
    epoch_preds, epoch_targets, epoch_embeddings = [], [], []
    data_time = time.time()
    for spectrograms, _, speakers in dataloader:

        # Get data loading time
        data_time = time.time() - data_time

        # Get model outputs
        model_time = time.time()

        optimizer.zero_grad()
        embeddings, preds, loss = model(spectrograms.to(device), speakers.to(device))

        model_time = time.time() - model_time

        # Store epoch info
        epoch_loss += loss
        epoch_data_time += data_time
        epoch_model_time += model_time
        epoch_embeddings += embeddings
        epoch_targets += speakers.detach().cpu().tolist()
        if preds is not None:
            epoch_preds += preds.detach().cpu().tolist()

        # Stop if loss is not finite
        if not math.isfinite(loss):
            print("Loss is {}, stopping training".format(loss))
            sys.exit(1)

        # Perform backpropagation
        opt_time = time.time()

        loss.backward()
        optimizer.step()
        opt_time = time.time() - opt_time
        epoch_opt_time += opt_time

        if step % 312 == 0:
            print(f'===> Training: Epoch [{current_epoch}/{total_epochs}]({step}/{len(dataloader)}): Loss: {loss:.2f} '
                  f'in {model_time + data_time + opt_time:.1f} sec')

        # Empty CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Increment step and re-initialize time counter
        step += 1
        data_time = time.time()

    # Get metrics
    metrics = dict()
    if len(epoch_preds) > 0:
        metrics = utils.get_train_val_metrics(
            epoch_targets, epoch_preds, prefix="train"
        )

    print(f'++++++++  TRAINING: Epoch [{current_epoch}/{total_epochs}]: AVGLoss: {(epoch_loss / len(dataloader)):.2f} '
          f'in {epoch_model_time + epoch_data_time + epoch_opt_time:.2f} sec '
          f'with LR: {lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else optimizer.param_groups[0]["lr"]}  '
          f'++++++++')
    print(f'++++++++  Metrics: '
          f'Accuracy: {metrics["train/accuracy"]:.2f}   '
          f'Precision: {metrics["train/precision"]:.2f}   '
          f'Recall: {metrics["train/recall"]:.2f}   '
          f'F1: {metrics["train/f1"]:.2f}  ++++++++')


@torch.no_grad()
def evaluate(
        model,
        dataloader,
        current_epoch=None,
        total_epochs=None,
        device="cpu",
):
    """
    Evaluate the given model for one epoch with the given dataloader
    """
    # Put the model in evaluation mode
    model.eval()

    # For each batch
    step = 1
    epoch_loss, epoch_data_time, epoch_model_time = 0, 0, 0
    epoch_preds, epoch_targets, epoch_embeddings = [], [], []
    data_time = time.time()
    for spectrograms, _, speakers in dataloader:

        # Get data loading time
        data_time = time.time() - data_time

        # Get model outputs
        model_time = time.time()
        embeddings, preds, loss = model(
            spectrograms.to(device), speakers=speakers.to(device)
        )
        model_time = time.time() - model_time

        # Store epoch info
        epoch_loss += loss
        epoch_data_time += data_time
        epoch_model_time += model_time
        epoch_embeddings += embeddings
        epoch_targets += speakers.detach().cpu().tolist()
        if preds is not None:
            epoch_preds += preds.detach().cpu().tolist()

        # Empty CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if step % 312 == 0:
            print(f'===> Validate: Epoch [{current_epoch}/{total_epochs}]({step}/{len(dataloader)}): Loss: {loss:.2f} in '
                  f'{model_time + data_time:.1f} sec')

        # Increment step and re-initialize time counter
        step += 1
        data_time = time.time()

    # Get metrics and return them
    metrics = dict()
    if len(epoch_preds) > 0:
        metrics = utils.get_train_val_metrics(epoch_targets, epoch_preds, prefix="val")

    print(f'++++++++  VALIDATE: Epoch [{current_epoch}/{total_epochs}]: '
          f'AVGLoss: {(epoch_loss / len(dataloader)):.2f} '
          f'in {epoch_model_time + epoch_data_time:.2f} sec  ++++++++')
    print(f'++++++++  Metrics: '
          f'Accuracy: {metrics["val/accuracy"]:.2f}   '
          f'Precision: {metrics["val/precision"]:.2f}   '
          f'Recall: {metrics["val/recall"]:.2f}   '
          f'F1: {metrics["val/f1"]:.2f}  ++++++++')


@torch.no_grad()
def test(
        model,
        test_dataset,
        indices=None,
        mindcf_p_target=0.01,
        mindcf_c_fa=1,
        mindcf_c_miss=1,
        device="cpu",
):
    """
    Test the given model and store EER and minDCF metrics
    """
    # Put the model in evaluation mode
    model.eval()

    # Get cosine similarity scores and labels
    samples = (
        test_dataset.get_sample_pairs(indices=indices, device=device)
        if not isinstance(test_dataset, torch.utils.data.Subset)
        else test_dataset.dataset.get_sample_pairs(
            indices=test_dataset.indices, device=device
        )
    )
    scores, labels = [], []
    for s1, s2, label in tqdm(samples, desc="Building scores and labels"):
        e1, e2 = model(s1), model(s2)
        scores += [F.cosine_similarity(e1, e2).item()]
        labels += [int(label)]

    # Get test metrics (EER and minDCF)
    metrics = utils.get_test_metrics(
        scores,
        labels,
        mindcf_p_target=mindcf_p_target,
        mindcf_c_fa=mindcf_c_fa,
        mindcf_c_miss=mindcf_c_miss,
        prefix="test",
    )

    print(f'++++++++  TESTING: EER: {metrics["test/eer"]} / Precision: {metrics["test/mindcf"]}  ++++++++')

    return metrics


def infer(
        model,
        utterances,
        dataset,
        device="cpu",
):
    """
    Compute embeddings for the given utterances and plot them
    """
    # Put the model in evaluation mode
    model.eval()

    # Compute embeddings
    all_embeddings = []
    for utterance in tqdm(utterances):
        data = dataset[utterance]
        embeddings = model(data["spectrogram"].to(device))
        all_embeddings += embeddings

    return all_embeddings
