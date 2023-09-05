import os
from datetime import datetime
import time
import random
import datetime
import os
import json

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import format_time
from res_net.res_block import ResBlock
from config import CHECKPOINTS_DIR, EARLY_STOPPING_PATIENCE, EPOCHS, LEARNING_RATE, LOG_DIR, RESULTS_DIR


def fine_tune(model_class, train_loader, val_loader, hyperparameters_range, num_trials, learning_rate=0.001, epochs=80):
    # Create a directory to store the logs
    os.makedirs(LOG_DIR, exist_ok=True)

    # Initialize a results dictionary to store the hyperparameters and performance metrics
    results = []

    for trial in range(num_trials):
        # Randomly sample hyperparameters from the range
        hyperparameters = {param: random.choice(
            values) for param, values in hyperparameters_range.items()}

        print(f"Trial {trial + 1}/{num_trials} - Hyperparameters:")
        print(json.dumps(hyperparameters, indent=4))

        # Create a new instance of the model with the sampled hyperparameters
        model = model_class(**hyperparameters, in_channels=1, resblock=ResBlock, repeat=[2, 2, 2, 2], useBottleneck=False, outputs=10)

        # model to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Call the fit() function to train the model
        best_stats = fit(model, train_loader, val_loader,
            learning_rate=learning_rate, epochs=epochs)

        performance_metrics = {
            "hyperparameters": hyperparameters,
            "best_train_loss": best_stats.train_loss,
            "best_train_acc": best_stats.train_acc,
            "best_val_loss": best_stats.val_loss,
            "best_val_acc": best_stats.val_acc
        }

        # Append the performance metrics to the results list
        results.append(performance_metrics)

        # Save the results to a JSON file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.json"
        filepath = os.path.join(LOG_DIR, filename)
        with open(filepath, "w") as file:
            json.dump(results, file)

        # Print the current trial's results
        print(f"Trial {trial + 1}/{num_trials} - Results:")
        print(json.dumps(performance_metrics, indent=4))
        print("-" * 40)

    # Return the results for further analysis if needed
    return results


def fit(model,
        train_loader,
        val_loader,
        save_checkpoint=10,
        load_checkpoint=False,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        ):
    """
    Trains a given model using the provided training data and validates it using the validation data.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
        save_checkpoint (int, optional): Number of epochs after which a checkpoint is saved. Defaults to None.
        load_checkpoint (bool, optional): Whether to load a checkpoint or not. Defaults to False.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
        epochs (int, optional): Number of epochs for training. Defaults to 80.
        early_stopping_patience (int, optional): Number of epochs to wait before stopping training if no improvement is made. Defaults to 5.

    Returns:
        dict: Dictionary containing the training and validation losses and accuracies.
    """
    checkpoint_path, optimizer, loss_fn, epoch, best_stats, losses = _init_settings(
        model, learning_rate, save_checkpoint, load_checkpoint)

    # epochs go [1, epochs], not [0, epochs[
    epochs = epochs + 1
    no_improvement = 0

    current_experiment = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for epoch in range(epoch, epochs):
        # early stopping
        if no_improvement >= early_stopping_patience:
            print("Early stopping triggered. No improvement in the last 5 epochs.")
            break

        start_time = time.time()

        train_loss, train_acc = _train_epoch(
            model, train_loader, optimizer, loss_fn, epoch, epochs)
        val_loss, val_acc = _validate(
            model, val_loader, loss_fn, epoch, epochs)

        if val_acc > best_stats.val_acc:
            no_improvement = 0
        else:
            no_improvement += 1

        losses, best_stats = _update_stats(
            losses, best_stats, epoch, train_loss, train_acc, val_loss, val_acc)

        if save_checkpoint is not None:
            _update_checkpoint(epoch, save_checkpoint, best_stats, checkpoint_path,
                               model, optimizer, loss_fn, losses)

        _display_info(losses, best_stats, epoch, epochs,
                      start_time, current_experiment)
        
    return best_stats


def _display_info(losses, best_stats, epoch, epochs, start_time, current_experiment):
    """
    Displays information about the current epoch.

    Args:
        losses (pd.DataFrame): The DataFrame to hold the loss and accuracy values.
        best_stats (Stats): The best statistics for the model.
        epoch (int): The epoch number.
        epochs (int): The total number of epochs.
        start_time (float): The time at which the current epoch started.
        current_experiment (str): The name of the current experiment.

    Returns:
        None
    """
    show_epoch_info(losses, best_stats, epoch)
    _save_results_plot(losses, epoch, current_experiment)
    print_epoch_info(epoch, epochs, start_time)


def _update_checkpoint(epoch, save_checkpoint, best_stats, checkpoint_path, model, optimizer, loss_fn, losses):
    """
    Updates the checkpoint for the model.

    Args:
        epoch (int): The epoch number.
        save_checkpoint (int): Number of epochs after which a checkpoint is saved.
        best_stats (Stats): The best statistics for the model.
        checkpoint_path (str): The path to save the checkpoint.
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Adam): The optimizer for the model.
        loss_fn (torch.nn.CrossEntropyLoss): The loss function for the model.
        losses (pd.DataFrame): The DataFrame to hold the loss and accuracy values.

    Returns:
        None
    """
    if epoch % save_checkpoint == 0 \
            and best_stats.last_checkpoint_acc < best_stats.val_acc:
        _save_checkpoint(model, optimizer, loss_fn, epoch,
                         best_stats, losses, checkpoint_path)


def _update_stats(losses, best_stats, epoch, train_loss, train_acc, val_loss, val_acc):
    """
    Updates the statistics for the model.

    Args:
        losses (pd.DataFrame): The DataFrame to hold the loss and accuracy values.
        best_stats (dict): The best statistics for the model.
        epoch (int): The epoch number.
        train_loss (float): The loss for the training set.
        train_acc (float): The accuracy for the training set.
        val_loss (float): The loss for the validation set.
        val_acc (float): The accuracy for the validation set.

    Returns:
        losses (pd.DataFrame): The DataFrame to hold the loss and accuracy values.
        best_stats (dict): The best statistics for the model.
    """
    losses = _update_losses_df(
        losses, epoch, train_loss, train_acc, val_loss, val_acc)

    if val_acc > best_stats.val_acc:
        best_stats.update(epoch, train_loss, train_acc, val_loss, val_acc)

    return losses, best_stats


def _init_settings(model, learning_rate, save_checkpoint, load_checkpoint):
    """
    Initializes settings for training.

    Args:
        model (torch.nn.Module): The model to be trained.
        learning_rate (float): Learning rate for the optimizer.
        save_checkpoint (int): Number of epochs after which a checkpoint is saved.
        load_checkpoint (bool): Whether to load a checkpoint or not.

    Returns:
        checkpoint_path (str): The path to save the checkpoint.
        optimizer (torch.optim.Adam): The optimizer for the model.
        loss_fn (torch.nn.CrossEntropyLoss): The loss function for the model.
        epoch (int): The epoch number.
        best_stats (Stats): The best statistics for the model.
        losses (pd.DataFrame): The DataFrame to hold the loss and accuracy values.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    epoch = 0

    losses = _create_losses_df()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    best_stats = Stats()

    checkpoint_path = f'{CHECKPOINTS_DIR}/checkpoint_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.pt'
    if load_checkpoint:
        # if checkpoints dir exists
        if os.path.isdir(CHECKPOINTS_DIR) and len(os.listdir(CHECKPOINTS_DIR)) > 0:
            checkpoint_path = f'{CHECKPOINTS_DIR}/{os.listdir(CHECKPOINTS_DIR)[-1]}'
            checkpoint = torch.load(checkpoint_path)

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            loss_fn = checkpoint['loss_fn']
            epoch = checkpoint['epoch']
            best_stats = checkpoint['best_stats']
            losses = checkpoint['losses']
            print(f'Loaded checkpoint from {checkpoint_path}')
        else:
            os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    # if there is no checkpoint, start from epoch 1
    # if there is a checkpoint, start from the next epoch
    epoch += 1

    return checkpoint_path, optimizer, loss_fn, epoch, best_stats, losses


def _create_losses_df():
    """
    Creates a DataFrame to hold the loss and accuracy values for the training and validation sets.

    Returns:
        losses (pd.DataFrame): The DataFrame to hold the loss and accuracy values.
    """
    losses = pd.DataFrame(
        columns=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
    return losses


def _save_checkpoint(model, optimizer, loss_fn, epoch, best_stats, losses, checkpoint_path):
    """
    Saves a checkpoint for the current epoch.

    Args:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Adam): The optimizer for the model.
        loss_fn (torch.nn.CrossEntropyLoss): The loss function for the model.
        epoch (int): The epoch number.
        best_stats (Stats): Dictionary to hold the best statistics.
        losses (pd.DataFrame): The DataFrame to hold the loss and accuracy values.
        checkpoint_path (str): The path to save the checkpoint.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_fn': loss_fn,
        'epoch': epoch,
        'best_stats': best_stats,
        'losses': losses
    }

    torch.save(checkpoint, checkpoint_path)


def _update_losses_df(losses, epoch, train_loss, train_acc, val_loss, val_acc):
    """
    Updates the DataFrame to hold the loss and accuracy values for the current epoch.

    Args:
        losses (pd.DataFrame): The DataFrame to hold the lossand accuracy values.
        epoch (int): The epoch number.
        train_loss (float): The training loss for the epoch.
        train_acc (float): The training accuracy for the epoch.
        val_loss (float): The validation loss for the epoch.
        val_acc (float): The validation accuracy for the epoch.

    Returns:
        losses (pd.DataFrame): The updated DataFrame to hold the loss and accuracy values.
    """
    row = pd.DataFrame({
        'epoch': epoch,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc
    }, index=[0])

    losses = pd.concat([losses, row], ignore_index=True)
    return losses


def show_epoch_info(losses, best_stats, epoch):
    """
    Shows the information for the epoch.

    Args:
        losses (pd.DataFrame): The DataFrame to hold the loss and accuracy values.
        best_stats (Stats): Dictionary to hold the best statistics.
        epoch (int): The epoch number.
    """
    print(losses)
    print(f'Best stats until now: {best_stats}')


def _save_results_plot(losses, epoch, current_experiment):
    """
    Saves the plot of loss and accuracy values for all epochs.

    Args:
        losses (pd.DataFrame): The DataFrame to hold the loss and accuracy values.
        epoch (int): The epoch number.
        current_experiment (str): The name of the current experiment.
    """
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))

    ax[0].plot(losses['epoch'], losses['train_loss'], label='Train Loss')
    ax[0].plot(losses['epoch'], losses['val_loss'], label='Validation Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Loss vs Epochs')
    ax[0].legend()

    ax[1].plot(losses['epoch'], losses['train_acc'], label='Train Accuracy')
    ax[1].plot(losses['epoch'], losses['val_acc'], label='Validation Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Accuracy vs Epochs')
    ax[1].legend()

    # save the figure on a folder that has the current date and time
    os.makedirs(f'{RESULTS_DIR}/{current_experiment}', exist_ok=True)
    plt.savefig(f'{RESULTS_DIR}/{current_experiment}/results_{epoch}.png')


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
    print(f'Epoch {epoch}/{epochs} | Elapsed Time: {format_time(epoch_elapsed_time)} | '
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
    train_acc = (total_correct_guesses.double() /
                 len(train_loader.dataset)).item()

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
            elapsed_time / ((epoch) * (step + 1))
        print(f'Epoch {epoch}/{epochs} | Step {step}/{total_steps} | Loss: {loss_result.item():.4f} | '
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
                    epochs - epoch) * len(val_loader) * elapsed_time / ((epoch) * (i + 1))
                print(f'Epoch {epoch}/{epochs} | Step{format_time(i)}/{len(val_loader)} | '
                      f'Loss: {loss_result.item():.4f} | Elapsed Time: {format_time(elapsed_time)} | '
                      f'Remaining Time - Epoch: {format_time(remaining_time_epoch)} | '
                      f'Remaining Time - Total: {format_time(remaining_time_train)}')

    avg_val_loss = current_loss / len(val_loader)
    val_acc = (correct_guesses.double() / len(val_loader.dataset)).item()

    return avg_val_loss, val_acc

class Stats:
    def __init__(self):
        self.epoch = 0
        self.train_loss = 0.0
        self.train_acc = 0.0
        self.val_loss = 0.0
        self.val_acc = 0.0
        self.last_checkpoint_acc = 0.0

    def update(self, epoch, train_loss, train_acc, val_loss, val_acc):
        self.epoch = epoch
        self.train_loss = train_loss
        self.train_acc = train_acc
        self.val_loss = val_loss
        self.val_acc = val_acc
        self.last_checkpoint_acc = val_acc

    def __str__(self):
        return f'Epoch: {self.epoch} | Train Loss: {self.train_loss:.4f} | Train Acc: {self.train_acc:.4f} | ' \
               f'Val Loss: {self.val_loss:.4f} | Val Acc: {self.val_acc:.4f} | Last Checkpoint Acc: {self.last_checkpoint_acc:.4f}'