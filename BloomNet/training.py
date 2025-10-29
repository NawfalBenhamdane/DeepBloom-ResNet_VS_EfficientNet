import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn

# Assuming DataLoader, device, and train_loader are available from data_processing.py
# from data_processing import train_loader, device

def train_model(model, criterion, optimizer, scheduler, num_epochs, train_loader, device):
    """
    Trains a given PyTorch model.

    Args:
        model (nn.Module): The model to train.
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimizer.
        scheduler (optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        num_epochs (int): The number of epochs to train for.
        train_loader (DataLoader): DataLoader for the training data.
        device (torch.device): The device to train on (e.g., 'cuda' or 'cpu').

    Returns:
        list: A list of average losses for each epoch.
    """
    losses_per_epoch = []
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the parameter gradients

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        losses_per_epoch.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Step the scheduler if it exists
        if scheduler:
            scheduler.step()

    return losses_per_epoch

# Example usage (assuming model, criterion, optimizer, scheduler, device, train_loader are defined elsewhere):
#
# from data_processing import train_loader, device
# from models import FlowerCNN # or other models
#
# # --- Example for FlowerCNN ---
# model_cnn = FlowerCNN().to(device)
# criterion_cnn = nn.CrossEntropyLoss()
# optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=0.001)
# # No scheduler used for this simple CNN in the notebook example
# scheduler_cnn = None
# epochs_cnn = 5
# print("--- Training FlowerCNN ---")
# losses_cnn = train_model(model_cnn, criterion_cnn, optimizer_cnn, scheduler_cnn, epochs_cnn, train_loader, device)
#
# # --- Example for MiniResNet ---
# # Assuming MiniResNet is imported from models.py
# # model_resnet = MiniResNet(num_classes=102).to(device)
# # criterion_resnet = nn.CrossEntropyLoss()
# # optimizer_resnet = optim.Adam(model_resnet.parameters(), lr=0.001)
# # scheduler_resnet = None # No scheduler used for this in the notebook example
# # epochs_resnet = 20
# # print("\n--- Training MiniResNet ---")
# # losses_resnet = train_model(model_resnet, criterion_resnet, optimizer_resnet, scheduler_resnet, epochs_resnet, train_loader, device)
#
# # --- Example for Fine-tuning with Scheduler ---
# # Assuming model_ft is loaded and its parameters are set up for training
# # criterion_ft = nn.CrossEntropyLoss()
# # optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=0.0003)
# # scheduler_ft = StepLR(optimizer_ft, step_size=5, gamma=0.5)
# # epochs_ft = 20
# # print("\n--- Training Fine-tuned Model ---")
# # losses_ft = train_model(model_ft, criterion_ft, optimizer_ft, scheduler_ft, epochs_ft, train_loader, device)
