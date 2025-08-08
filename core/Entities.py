import os
import sys
from importlib.util import spec_from_file_location, module_from_spec

import torch
from torchvision import models, transforms, datasets


def get_model_options():
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


def _get_models():
    models_dir = "scripts/models"
    models = {}
    if os.path.exists(models_dir):
        for model_file in os.listdir(models_dir):
            if model_file.endswith('.py'):
                module_name = model_file[:-3]  # .py
                if module_name.endswith('Impl'):
                    module_name = module_name[:-4]
                try:
                    spec = spec_from_file_location(module_name, os.path.join(models_dir, model_file))
                    module = module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if hasattr(module, 'display_name'):
                        # New API: display_name() takes no args and returns the human name
                        models[module.display_name()] = module

                except Exception as e:
                    print(f"Error loading {model_file}: {e}")

    return models


def request_model():
    models = _get_models()
    if not models:
        print(f'Não há nenhum modelo implementado.')
        sys.exit(1)

    print('\nModelos disponiveis:')
    models_names = list(models.keys())
    for i, model in enumerate(models_names):
        print(f'{i + 1}. {model}')
    model_opt = int(input('Escolha o modelo: '))

    while model_opt not in range(1, len(models_names) + 1):
        print('Opção inválida.')
        model_opt = int(input('Escolha o modelo: '))

    model_name = models_names[model_opt - 1]
    module = models[model_name]
    model, criterion, optimizer = module.init()
    train_fn = getattr(module, 'train', None)

    return model, model_name, criterion, optimizer, train_fn


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
