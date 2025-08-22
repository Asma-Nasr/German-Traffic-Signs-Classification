import torch
import torchvision.models as models

def load_pretrained_model(model_name,num_classes):
    if model_name == 'vgg16':
        model = models.vgg16(weights='DEFAULT')#pretrained=True)
        model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=num_classes)
    elif model_name == 'vgg19':
        model = models.vgg19(weights='DEFAULT')#pretrained=True)
        model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=num_classes)
    elif model_name == 'mobilenet':
        model = models.mobilenet_v2(weights='DEFAULT')#pretrained=True)
        model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=num_classes)
    else:
        raise ValueError("Model not supported.")
    return model


def freeze_layers(model, freeze_until_layer):
    for name, param in model.named_parameters():
        if freeze_until_layer in name:
            break
        param.requires_grad = False

def eval_model(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    return accuracy
