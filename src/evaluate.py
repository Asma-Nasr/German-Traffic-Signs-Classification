import torch
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import math

def evaluate_model_with_images(model, csv_file, num_images,test_folder='Test', batch_size=16):
    # Set the model to evaluation mode
    model.eval()

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images
        transforms.ToTensor(),            # Convert images to tensors
    ])

    # Load true classes from CSV
    data = pd.read_csv(csv_file)

    image_files = data['Path'].tolist()  # Get the list of image paths
    correct_labels = data['ClassId'].tolist()  # Get the list of correct class IDs

    # Load and process images
    images = []
    for image_file in image_files:
        image = Image.open(image_file).convert('RGB')
        image_tensor = transform(image).to(device)  # Move to the same device
        images.append(image_tensor)  # Store only the image tensor

    # Convert list of images to a tensor
    images_tensor = torch.stack(images)

    # Make predictions in batches
    predictions = []
    with torch.no_grad():
        for i in range(0, images_tensor.size(0), batch_size):
            batch = images_tensor[i:i + batch_size]
            outputs = model(batch)
            _, predicted = torch.max(outputs, 1)
            predictions.append(predicted)

    # Flatten the list of predictions
    predictions = torch.cat(predictions)

    # Plotting results
    plt.figure(figsize=(15, 8))
    num_images = min(num_images, len(images))  # Display only up to available images

    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)

    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i].permute(1, 2, 0).cpu())  # Move to CPU for plotting
        title_color = 'green' if predictions[i] == correct_labels[i] else 'red'
        plt.title(f'Correct: {correct_labels[i]}\n Predicted: {predictions[i].item()}', color=title_color, fontsize=10)
        plt.axis('off')


def eval(model,test_loader,device):
 
    model.eval()
    test_iter = iter(test_loader)
    test_imgs, test_labels = next(test_iter)
    test_imgs = test_imgs.to(device)
    total = 0
    correct = 0  
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the test set:  { (100 * correct / total):.2f}%')
