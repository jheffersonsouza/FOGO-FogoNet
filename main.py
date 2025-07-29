import sys
from pathlib import Path

import torch
from torchvision import models, transforms, datasets
from torchvision.models import Inception_V3_Weights
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.quantization as quantization
import os
import time
import numpy as np

from utils.Entities import evaluate_model
from utils.DatasetsLoader import request_dataset
from utils.ReportGenerator import ReportGenerator

# Responsavel por calcular tempo de inferencia, tamanho e matriz de confusão do modelo.
reportGenerator = ReportGenerator(output_dir="results")

DATASETS_ROOT_PATH = os.path.join('.', "datasets")

dataset = request_dataset(DATASETS_ROOT_PATH)
if not dataset:
    print("Dê uma olhada no arquivo 'SETUP' na pasta", Path(DATASETS_ROOT_PATH).absolute())
    sys.exit(1)

train_data, val_data = dataset.get()
print('Classes do dataset:', train_data.class_to_idx)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, aux_logits=True)
model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, 2)
model.fc = nn.Linear(model.fc.in_features, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

print("=== TREINAMENTO DO MODELO ORIGINAL ===")
max_epochs = 50
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
    print(f"Epoch {epoch + 1}/{max_epochs}, Loss: {avg_loss:.4f}")
    accuracy = evaluate_model(model, val_loader, device)
    print(f"Epoch {epoch + 1}/{max_epochs}, Acurácia: {accuracy:.2f}%")

reportGenerator.summary("Original", model, val_loader, device, save_model=True)
"""

print("\n=== QUANTIZAÇÃO DINÂMICA ===")
model_cpu = model.to('cpu')
model_dynamic_quantized = torch.quantization.quantize_dynamic(
    model_cpu,
    {torch.nn.Linear},
    dtype=torch.qint8
)
print("Quantização dinâmica aplicada!")
device_quant = torch.device("cpu")
reportGenerator.summary("Quantizado Dinâmico", model_dynamic_quantized, val_loader, device_quant)

print("\n=== QUANTIZAÇÃO ESTÁTICA (PÓS-TREINAMENTO) ===")

try:
    model_static = models.googlenet(pretrained=True)
    model_static.fc = nn.Linear(model_static.fc.in_features, 2)
    model_static.load_state_dict(model_cpu.state_dict())

    model_static.eval()

    backends = ['fbgemm', 'qnnpack']
    model_static_quantized = None

    for backend in backends:
        try:
            print(f"Tentando quantização com backend: {backend}")
            model_static.qconfig = quantization.get_default_qconfig(backend)

            model_static_prepared = quantization.prepare(model_static, inplace=False)

            print("Calibrando modelo com dados de validação...")
            with torch.no_grad():
                for i, (inputs, _) in enumerate(val_loader):
                    if i >= 10:
                        break
                    model_static_prepared(inputs)
                    if i % 5 == 0:
                        print(f"Calibração: {i + 1}/10 batches processados")

            model_static_quantized = quantization.convert(model_static_prepared, inplace=False)
            print(f"Quantização estática aplicada com sucesso usando backend: {backend}!")
            break

        except Exception as e:
            print(f"Erro com backend {backend}: {e}")
            continue

    if model_static_quantized is not None:
        static_accuracy, static_preds, static_labels = evaluate_model(
            model_static_quantized, val_loader, device_quant, "Modelo Quantizado Estático"
        )
        static_time = measure_inference_time(model_static_quantized, val_loader, device_quant,
                                             "Modelo Quantizado Estático")
        static_size = get_model_size(model_static_quantized, "Modelo Quantizado Estático")

        save_confusion_matrix(static_labels, static_preds, class_names, output_dir, "Modelo Quantizado Estático")
    else:
        print("ERRO: Não foi possível aplicar quantização estática com nenhum backend disponível.")
        print("Continuando apenas com quantização dinâmica...")
        static_accuracy = 0
        static_time = 0
        static_size = 0

except Exception as e:
    print(f"ERRO na quantização estática: {e}")
    print("Continuando apenas com quantização dinâmica...")
    static_accuracy = 0
    static_time = 0
    static_size = 0

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
