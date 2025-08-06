import torch
from torchvision import models, transforms, datasets


def get_runtime_options():
    should_quantize_dynamically = True
    should_quantize_statically = True

    q1 = str(input('Deseja quantizar o modelo(S/N)? ')).strip().upper()
    if q1 == 'N':
        should_quantize_statically = False
        should_quantize_dynamically = False
    else:
        q2 = str(input('Deseja quantizar o modelo de forma dinâmica(S/N)? ')).strip().upper()
        if q2 == 'N':
            should_quantize_dynamically = False
        q3 = str(input('Deseja quantizar o modelo de forma estatica(S/N)? ')).strip().upper()
        if q3 == 'N':
            should_quantize_statically = False


    return should_quantize_dynamically, should_quantize_statically

def evaluate_model(model, data_loader, device):
    """Avalia o modelo e retorna métricas"""
    model.eval()
    correct = total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    return accuracy, all_preds, all_labels
