import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# List of flower names
flower_names = [
    'pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'english marigold',
    'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon', 'colt’s foot',
    'king protea', 'spear thistle', 'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily',
    'balloon flower', 'giant white arum lily', 'fire lily', 'pincushion flower', 'fritillary', 'red ginger',
    'grape hyacinth', 'corn poppy', 'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william',
    'carnation', 'garden phlox', 'love in the mist', 'mexican aster', 'alpine sea holly', 'ruby-lipped cattleya',
    'cape flower', 'great masterwort', 'siam tulip', 'lenten rose', 'barbeton daisy', 'daffodil', 'sword lily',
    'poinsettia', 'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'oxeye daisy', 'common dandelion',
    'petunia', 'wild pansy', 'primula', 'sunflower', 'pelargonium', 'bishop of llandaff', 'gaura', 'geranium',
    'orange dahlia', 'pink-yellow dahlia', 'cautleya spicata', 'japanese anemone', 'black-eyed susan',
    'silverbush', 'californian poppy', 'osteospermum', 'spring crocus', 'bearded iris', 'windflower',
    'tree poppy', 'gazania', 'azalea', 'water lily', 'rose', 'thorn apple', 'morning glory', 'passion flower',
    'lotus', 'toad lily', 'anthurium', 'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert-rose',
    'tree mallow', 'magnolia', 'cyclamen', 'watercress', 'canna lily', 'hippeastrum', 'bee balm', 'ball moss',
    'foxglove', 'bougainvillea', 'camellia', 'mallow', 'mexican petunia', 'bromelia', 'blanket flower',
    'trumpet creeper', 'blackberry lily'
]

def plot_loss(losses, title="Loss Curve", xlabel="Epoch", ylabel="Average Loss"):
    """
    Plots the training loss curve.

    Args:
        losses (list): A list of average losses per epoch.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def visualize_grad_cam(model, img_tensor, target_layers, device, title="Grad-CAM Visualization", pred_class=None, true_label=None):
    """
    Generates and displays a Grad-CAM visualization for a given image and model.

    Args:
        model (nn.Module): The PyTorch model.
        img_tensor (torch.Tensor): The input image tensor (single image, batched).
        target_layers (list): A list of target layers for Grad-CAM.
        device (torch.device): The device to perform computation on.
        title (str): The title for the plot.
        pred_class (int, optional): The predicted class index. Defaults to None.
        true_label (int, optional): The true label index. Defaults to None.
    """
    # Ensure model is in evaluation mode
    model.eval()

    # Get the image for visualization (remove batch dimension and move to CPU)
    # Assumes img_tensor is a single image batch (e.g., shape [1, C, H, W])
    img_show = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # Normalize image for display if it's not already in [0, 1] range
    img_show = (img_show - img_show.min()) / (img_show.max() - img_show.min())

    # Determine the target class for Grad-CAM
    targets = []
    if pred_class is not None:
        targets.append(ClassifierOutputTarget(pred_class))
    else:
        # If pred_class is not provided, try to infer it or use a default
        with torch.no_grad():
            output = model(img_tensor)
            pred_class = torch.argmax(output).item()
            targets.append(ClassifierOutputTarget(pred_class))

    # Initialize Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)

    # Generate grayscale CAM
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0, :]

    # Show CAM on image
    visualization = show_cam_on_image(img_show, grayscale_cam, use_rgb=True)

    # Display the visualization
    plt.figure(figsize=(8, 6))
    plt.imshow(visualization)
    plot_title = title
    if true_label is not None and pred_class is not None:
        plot_title += f" (True: {true_label}, Pred: {pred_class})"
    elif pred_class is not None:
        plot_title += f" (Pred: {pred_class})"
    plt.title(plot_title)
    plt.axis('off')
    plt.show()

# Example usage for Grad-CAM (requires model, image tensor, target_layers, device, and potentially true_label):
#
# from data_processing import test_loader, device
# from models import MiniResNet # or other models
#
# # Load a trained model (example)
# # model_eff = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
# # num_features = model_eff.classifier[1].in_features
# # model_eff.classifier[1] = nn.Linear(num_features, 102)
# # model_eff = model_eff.to(device)
# # Load weights if available
#
# # Get a sample image
# # images, labels = next(iter(test_loader))
# # img_tensor_sample = images[0].unsqueeze(0).to(device)
# # true_label_sample = labels[0].item()
#
# # Define target layers (example for EfficientNet-B0)
# # target_layers_eff = [model_eff.features[-1]]
#
# # Visualize
# # visualize_grad_cam(model_eff, img_tensor_sample, target_layers_eff, device,
# #                    title="Grad-CAM - EfficientNet-B0", true_label=true_label_sample)
