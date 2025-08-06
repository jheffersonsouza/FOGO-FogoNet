import sys
from pathlib import Path

import torch
from torchvision import models, transforms, datasets
from torchvision.models import Inception_V3_Weights
from torch import nn, optim
from torch.utils.data import DataLoader
import os
import time
import numpy as np

from core.Entities import evaluate_model, get_runtime_options
from core.DatasetsLoader import request_dataset
from core.ModelQuantizer import ModelQuantizer
from core.ReportGenerator import ReportGenerator

DATASETS_ROOT_PATH = os.path.join('.', "datasets")
if __name__ == "__main__":
    print('Iniciando sistema')
    reportGenerator = ReportGenerator(output_dir="results")



    should_quantize_dynamically, should_quantize_statically = get_runtime_options()

    dataset = request_dataset(DATASETS_ROOT_PATH)
    if not dataset:
        print("Dê uma olhada no arquivo 'SETUP' na pasta", Path(DATASETS_ROOT_PATH).absolute())
        sys.exit(1)

    train_data, val_data = dataset.get(transform_model='inception-format')
    print('Classes do dataset:', train_data.class_to_idx)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=2, pin_memory=False)

    model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, aux_logits=True)
    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, 2)
    model.fc = nn.Linear(model.fc.in_features, 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    print('-'*50)
    print('Treinando modelo...')
    max_epochs = 1
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs, aux_outputs = model(inputs)
            loss = criterion(outputs, labels) + 0.4 * criterion(aux_outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        # Fiz assim pra mensagem aparecer logo de uma vez, já que evaluate_model tem um delayzinho.
        epoch_summary = f"Epoch {epoch + 1}/{max_epochs} - Loss: {avg_loss:.4f},"
        accuracy, all_preds, all_labels = evaluate_model(model, val_loader, device)
        epoch_summary += f" Accuracy: {accuracy:.2f}%"
        print(epoch_summary)
    print('Gerando os resultados...')
    reportGenerator.summary("Original", model, val_loader, device, save_model=True)

    # Quantização
    quantizer = ModelQuantizer(model, val_loader, reportGenerator)
    if should_quantize_dynamically:
        quantizer.dynamic()
    if should_quantize_statically:
        quantizer.static()
"""
print('-' * 60)
print("COMPARAÇÃO FINAL DOS MODELOS")
print("-" * 60)

# Temporario
static_accuracy = 0
static_size = 0
static_time = 0
model_static_quantized = None

if static_accuracy > 0:
    comparison_data = {
        'Modelo': ['Original', 'Quantizado Dinâmico', 'Quantizado Estático'],
        'Acurácia (%)': [original_accuracy, dynamic_accuracy, static_accuracy],
        'Tamanho (MB)': [original_size, dynamic_size, static_size],
        'Tempo/Batch (s)': [original_time, dynamic_time, static_time]
    }
else:
    comparison_data = {
        'Modelo': ['Original', 'Quantizado Dinâmico'],
        'Acurácia (%)': [original_accuracy, dynamic_accuracy],
        'Tamanho (MB)': [original_size, dynamic_size],
        'Tempo/Batch (s)': [original_time, dynamic_time]
    }

print(f"{'Modelo':<20} {'Acurácia (%)':<12} {'Tamanho (MB)':<12} {'Tempo/Batch (s)':<15}")
print("-" * 60)
for i in range(len(comparison_data['Modelo'])):
    print(f"{comparison_data['Modelo'][i]:<20} "
          f"{comparison_data['Acurácia (%)'][i]:<12.2f} "
          f"{comparison_data['Tamanho (MB)'][i]:<12.2f} "
          f"{comparison_data['Tempo/Batch (s)'][i]:<15.4f}")

print(f"\nREDUÇÃO DE TAMANHO:")
print(f"Quantização Dinâmica: {((original_size - dynamic_size) / original_size * 100):.1f}%")
if static_size > 0:
    print(f"Quantização Estática: {((original_size - static_size) / original_size * 100):.1f}%")

print(f"\nMELHORIA DE VELOCIDADE:")
print(f"Quantização Dinâmica: {((original_time - dynamic_time) / original_time * 100):.1f}%")
if static_time > 0:
    print(f"Quantização Estática: {((original_time - static_time) / original_time * 100):.1f}%")

with open(os.path.join(output_dir, "comparacao_modelos.txt"), "w") as f:
    f.write("COMPARAÇÃO DOS MODELOS\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"{'Modelo':<20} {'Acurácia (%)':<12} {'Tamanho (MB)':<12} {'Tempo/Batch (s)':<15}\n")
    f.write("-" * 60 + "\n")
    for i in range(len(comparison_data['Modelo'])):
        f.write(f"{comparison_data['Modelo'][i]:<20} "
                f"{comparison_data['Acurácia (%)'][i]:<12.2f} "
                f"{comparison_data['Tamanho (MB)'][i]:<12.2f} "
                f"{comparison_data['Tempo/Batch (s)'][i]:<15.4f}\n")

    f.write(f"\nREDUÇÃO DE TAMANHO:\n")
    f.write(f"Quantização Dinâmica: {((original_size - dynamic_size) / original_size * 100):.1f}%\n")
    if static_size > 0:
        f.write(f"Quantização Estática: {((original_size - static_size) / original_size * 100):.1f}%\n")

    f.write(f"\nMELHORIA DE VELOCIDADE:\n")
    f.write(f"Quantização Dinâmica: {((original_time - dynamic_time) / original_time * 100):.1f}%\n")
    if static_time > 0:
        f.write(f"Quantização Estática: {((original_time - static_time) / original_time * 100):.1f}%\n")

torch.save(model_dynamic_quantized.state_dict(), os.path.join(output_dir, "modelo_quantizado_dinamico.pth"))
if static_accuracy > 0:
    torch.save(model_static_quantized.state_dict(), os.path.join(output_dir, "modelo_quantizado_estatico.pth"))

print(f"\nTodos os resultados salvos em: {output_dir}/")
print("Arquivos gerados:")
print("- Matrizes de confusão (.png)")
print("- Relatórios de classificação (.txt)")
print("- Matrizes de confusão (.csv)")
print("- Comparação dos modelos (comparacao_modelos.txt)")
print("- Modelos quantizados (.pth)")

print("\n=== QUANTIZAÇÃO CONCLUÍDA ===")
if static_accuracy == 0:
    print("NOTA: Quantização estática não funcionou neste ambiente.")
    print("Isso é comum em algumas instalações do PyTorch.")
    print("A quantização dinâmica ainda oferece bons resultados!")
"""