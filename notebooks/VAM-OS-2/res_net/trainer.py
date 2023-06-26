import os
from datetime import datetime
import time

import torch
import pandas as pd
import matplotlib.pyplot as plt

from utils import format_time
from config import EPOCHS, LEARNING_RATE, RESULTS_DIR


def fit(model, train_loader, val_loader, learning_rate=LEARNING_RATE, epochs=EPOCHS):
    """
    Trains a given model using the provided training data and validates it using the validation data.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
        epochs (int, optional): Number of epochs for training. Defaults to 80.

    Returns:
        None
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    losses = pd.DataFrame(
        columns=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

    os.makedirs(RESULTS_DIR, exist_ok=True)

    for epoch in range(epochs):
        start_time = time.time()

        train_loss, train_acc = _train_epoch(
            model, train_loader, optimizer, loss_fn, epoch, epochs)
        val_loss, val_acc = _validate(model, val_loader, loss_fn, epoch, epochs)

        row = pd.DataFrame({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }, index=[0])

        losses = pd.concat([losses, row], ignore_index=True)

        print(losses)
        _save_losses_plot(losses, epoch)
        print_epoch_info(epoch, epochs, start_time)


def _save_losses_plot(losses, epoch):
    """
    Saves a plot of the training and validation losses.

    Args:
        losses (pd.DataFrame): DataFrame containing the loss values.
        epoch (int): Current epoch number.

    Returns:
        None
    """
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
    losses.plot(x='epoch', y=['train_loss', 'val_loss'], ax=ax1)
    losses.plot(x='epoch', y=['train_acc', 'val_acc'], ax=ax2)
    ax1.set_title('Loss')
    ax2.set_title('Accuracy')
    plt.savefig(
        f'{RESULTS_DIR}/losses_{datetime.now().strftime("%Y%m%d-%H%M%S")}_epoch_{epoch}.png')


def print_epoch_info(epoch, epochs, epoch_start_time):
    """
    Prints information about the current epoch.

    Args:
        epoch (int): Current epoch number.
        epochs (int): Total number of epochs.
        epoch_start_time (float): Start time of the current epoch.

    Returns:
        None
    """
    epoch_elapsed_time = time.time() - epoch_start_time
    epoch_remaining_time = (epochs - epoch - 1) * epoch_elapsed_time
    print(f'Epoch {epoch + 1}/{epochs} | Elapsed Time: {format_time(epoch_elapsed_time)} | '
          f'Remaining Time - Total: {format_time(epoch_remaining_time)}')


def _train_epoch(model, train_loader, optimizer, loss_fn, epoch, epochs):
    """
    Performs one epoch of training on the given model using the provided training data.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): The optimizer used for parameter optimization.
        loss_fn (torch.nn.Module): The loss function used for calculating the loss.
        epoch (int): The current epoch number.
        epochs (int): The total number of epochs.

    Returns:
        tuple: A tuple containing the average training loss and the training accuracy.
    """
    model.train()
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_loss = 0.0
    total_correct_guesses = 0

    for i, (spectrograms, labels) in enumerate(train_loader):
        spectrograms, labels = spectrograms.to(device), labels.to(device)
        optimizer.zero_grad()

        results = model(spectrograms)
        loss_result = loss_fn(results, labels)
        loss_result.backward()
        optimizer.step()

        total_loss += loss_result.item()
        total_correct_guesses += _count_correct_guesses(results, labels)

        _print_progress(epoch, epochs, i, len(
            train_loader), loss_result, start_time)

        torch.cuda.empty_cache()

    avg_train_loss = total_loss / len(train_loader)
    train_acc = (total_correct_guesses.double() / len(train_loader.dataset)).item()

    return avg_train_loss, train_acc


def _count_correct_guesses(results, labels):
    """
    Counts the number of correct predictions given the model's output results and target labels.

    Args:
        results (torch.Tensor): The output results from the model.
        labels (torch.Tensor): The target labels.

    Returns:
        torch.Tensor: The number of correct predictions.
    """
    _, preds = torch.max(results, 1)
    return torch.sum(preds == labels.data)


def _print_progress(epoch, epochs, step, total_steps, loss_result, start_time):
    """
    Prints the progress information during training.

    Args:
        epoch (int): Current epoch number.
        epochs (int): Total number of epochs.
        step (int): Current step number.
        total_steps (int): Total number of steps.
        loss_result (torch.Tensor): Loss value for the current step.
        start_time (float): Start time of the epoch.

    Returns:
        None
    """
    if step % 1 == 0:
        elapsed_time = time.time() - start_time
        remaining_time_epoch = (total_steps - step) * elapsed_time / (step + 1)
        remaining_time_train = (epochs - epoch) * total_steps * \
            elapsed_time / ((epoch + 1) * (step + 1))
        print(f'Epoch {epoch + 1}/{epochs} | Step {step}/{total_steps} | Loss: {loss_result.item():.4f} | '
              f'Elapsed Time: {format_time(elapsed_time)} | Remaining Time - Epoch: '
              f'{format_time(remaining_time_epoch)} | Remaining Time - Total: {format_time(remaining_time_train)}')


def _validate(model, val_loader, loss_fn, epoch, epochs):
    """
    Validates the model using the validation data.

    Args:
        model (torch.nn.Module): The model to be validated.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
        loss_fn (torch.nn.Module): Loss function for calculating the loss.

    Returns:
        Tuple[float, float]: Average validation loss and validation accuracy.
    """
    current_loss = 0.0
    correct_guesses = 0
    model.eval()
    start_time = time.time()

    with torch.no_grad():
        for i, (spectrograms, labels) in enumerate(val_loader):
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)

            results = model(spectrograms)
            loss_result = loss_fn(results, labels)
            current_loss += loss_result.item()
            _, preds = torch.max(results, 1)
            correct_guesses += torch.sum(preds == labels.data)

            if i % 10 == 0:
                elapsed_time = time.time() - start_time
                remaining_time_epoch = (
                    len(val_loader) - i) * elapsed_time / (i + 1)
                remaining_time_train = (
                    epochs - epoch) * len(val_loader) * elapsed_time / ((epoch + 1) * (i + 1))
                print(f'Epoch {epoch + 1}/{epochs} | Step{format_time(i)}/{len(val_loader)} | '
                      f'Loss: {loss_result.item():.4f} | Elapsed Time: {format_time(elapsed_time)} | '
                      f'Remaining Time - Epoch: {format_time(remaining_time_epoch)} | '
                      f'Remaining Time - Total: {format_time(remaining_time_train)}')

    avg_val_loss = current_loss / len(val_loader)
    val_acc = (correct_guesses.double() / len(val_loader.dataset)).item()

    return avg_val_loss, val_acc
