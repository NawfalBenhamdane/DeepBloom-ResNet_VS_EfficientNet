import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming DataLoader, device are available from data_processing.py
# from data_processing import test_loader, device

def evaluate_model(model, test_loader, device):
    """
    Evaluates a PyTorch model on the test set and prints classification report and confusion matrix.

    Args:
        model (nn.Module): The model to evaluate.
        test_loader (DataLoader): DataLoader for the test data.
        device (torch.device): The device to evaluate on (e.g., 'cuda' or 'cpu').

    Returns:
        tuple: A tuple containing (y_true, y_pred) lists.
    """
    model.eval()  # Set model to evaluation mode
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Print global report
    print("--- Classification Report ---")
    print(classification_report(y_true, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap='Blues', annot=False)
    plt.xlabel("Predictions")
    plt.ylabel("Ground Truth")
    plt.title("Confusion Matrix")
    plt.show()

    return y_true, y_pred

# Example usage (assuming model, device, test_loader are defined elsewhere):
#
# from data_processing import test_loader, device
#
# # Assuming 'model_ft' is your trained model
# # y_true, y_pred = evaluate_model(model_ft, test_loader, device)
