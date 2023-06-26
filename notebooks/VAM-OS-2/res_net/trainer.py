import os
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchsummary import summary
import time
from datetime import datetime

from config import RESULTS_DIR


def format_time(time_in_seconds):
    if time_in_seconds < 60:
        return f"{time_in_seconds:.2f}s"
    elif time_in_seconds < 3600:
        minutes = int(time_in_seconds // 60)
        seconds = int(time_in_seconds % 60)
        return f"{minutes}m {seconds}s"
    else:
        hours = int(time_in_seconds // 3600)
        minutes = int((time_in_seconds % 3600) // 60)
        seconds = int(time_in_seconds % 60)
        return f"{hours}h {minutes}m {seconds}s"


def fit(model, train_loader, val_loader, learning_rate=0.001, epochs=80):
    # definition of optimizer function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # TODO try this one
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)

    # definition of loss function
    loss = torch.nn.CrossEntropyLoss()

    # save the loss and accuracy per epoch
    losses = pd.DataFrame(
        columns=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
    
    # if the results file does not exist, create it
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # training loop
    for epoch in range(epochs):
        # start training
        current_loss = 0.0
        correct_guesses = 0
        model.train()

        start_time = time.time() # record start time
        epoch_start_time = time.time() # record epoch start time

        for i, (spectrograms, labels) in enumerate(train_loader):
            # send the input to the gpu
            spectrograms = spectrograms.to(
                torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            labels = labels.to(
                torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            
            # forward pass the input through the model
            results = model(spectrograms)

            # compute the loss
            loss_result = loss(results, labels)

            # reset the gradients
            optimizer.zero_grad()

            # backpropagate to compute the gradients
            loss_result.backward()

            # update the parameters
            optimizer.step()

            # udpate loss
            current_loss += loss_result.item()

            # update the accuracy
            _, preds = torch.max(results, 1)
            correct_guesses += torch.sum(preds == labels.data)

            # print very useful information with elapsed time and remaining time for end of epoch and end of training
            if i % 1 == 0:
                elapsed_time = time.time() - epoch_start_time
                remaining_time_epoch = (len(train_loader) - i) * elapsed_time / (i + 1)
                remaining_time_train = (epochs - epoch) * len(train_loader) * elapsed_time / ((epoch + 1) * (i + 1))
                print(f'Epoch {epoch + 1}/{epochs} | Step {i}/{len(train_loader)} | Loss: {loss_result.item():.4f} | Elapsed Time: {format_time(elapsed_time)} | Remaining Time - Epoch: {format_time(remaining_time_epoch)} | Remaining Time - Total: {format_time(remaining_time_train)}')

            torch.cuda.empty_cache() # free up some memory

        # compute the loss and accuracy
        avg_train_loss = current_loss / len(train_loader)
        train_acc = (correct_guesses.double() / len(train_loader.dataset)).item()

        # start validation
        current_loss = 0.0
        correct_guesses = 0
        model.eval()

        start_time = time.time() # record start time
        
        with torch.no_grad():
            for i, (spectrograms, labels) in enumerate(val_loader):
                # to gpu
                spectrograms = spectrograms.to(
                    torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
                labels = labels.to(
                    torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
                
                # forward pass the input through the model
                results = model(spectrograms)

                # compute the loss
                loss_result = loss(results, labels)

                # add the loss to the current epoch loss
                current_loss += loss_result.item()

                # update the accuracy
                _, preds = torch.max(results, 1)
                correct_guesses += torch.sum(preds == labels.data)

                # print very useful information with elapsed time and remaining time for end of epoch and end of training
                if i % 10 == 0:
                    elapsed_time = time.time() - start_time
                    remaining_time_epoch = (len(val_loader) - i) * elapsed_time / (i + 1)
                    remaining_time_train = (epochs - epoch) * len(val_loader) * elapsed_time / ((epoch + 1) * (i + 1))
                    print(f'Epoch {epoch + 1}/{epochs} | Step{format_time(i)}/{len(val_loader)} | Loss: {loss_result.item():.4f} | Elapsed Time: {format_time(elapsed_time)} | Remaining Time - Epoch: {format_time(remaining_time_epoch)} | Remaining Time - Total: {format_time(remaining_time_train)}')

        # compute the loss and accuracy
        avg_val_loss = current_loss / len(val_loader)
        val_acc = (correct_guesses.double() / len(val_loader.dataset)).item()

        # log the metrics a graph
        new_row = pd.DataFrame({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'train_acc': train_acc,
            'val_loss': avg_val_loss,
            'val_acc': val_acc
        }, index=[0])

        losses = pd.concat([losses, new_row], ignore_index=True)

        print(losses)

        # display in the same graph both losses and accuracies
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
        losses.plot(x='epoch', y=['train_loss', 'val_loss'], ax=ax1)
        losses.plot(x='epoch', y=['train_acc', 'val_acc'], ax=ax2)
        ax1.set_title('Loss')
        ax2.set_title('Accuracy')
        # save the picture with current date and epoch
        plt.savefig(f'{RESULTS_DIR}/losses_{datetime.now().strftime("%Y%m%d-%H%M%S")}_epoch_{epoch}.png')
        
        # print epoch end information with elapsed time and remaining time for end of training
        epoch_elapsed_time = time.time() - epoch_start_time
        epoch_remaining_time = (epochs - epoch - 1) * epoch_elapsed_time
        print(f'Epoch {epoch + 1}/{epochs} | Elapsed Time: {format_time(epoch_elapsed_time)} | Remaining Time - Total: {format_time(epoch_remaining_time)}')