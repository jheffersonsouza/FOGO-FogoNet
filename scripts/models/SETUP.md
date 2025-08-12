# Guia de Criação de Modelos

Este guia explica como implementar e integrar um novo modelo ao FOGO-FogoNet usando PyTorch.

## Requisitos e Limitações

- Framework obrigatório: PyTorch (torch e torchvision).
- Dispositivo: suporta CPU e, quando disponível, CUDA (GPU).
- Quantização: opcional. Se implementar, siga a API mostrada abaixo.
- Dependências extras: evite frameworks alternativos (ex.: TensorFlow, JAX). O pipeline assume PyTorch.

## Estrutura esperada do arquivo do modelo

Crie um arquivo em `scripts/models/` com o padrão `NomeDoModeloImpl.py` expondo as seguintes funções:

```python
# scripts/models/MeuModeloImpl.py
import torch
from torch import nn
from torchvision import models  # se for usar pré-treinados do torchvision
from core.Entities import evaluate_model
import torch.quantization as quantization  # opcional, apenas se for quantizar


def display_name():
    """Nome legível do modelo para logs e relatórios."""
    return "MeuModelo"


def init():
    """Inicializa o modelo, a função de perda e o otimizador.

    Returns:
        model (nn.Module): modelo PyTorch pronto para treino.
        criterion (nn.Module): função de perda (ex.: CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): otimizador (ex.: AdamW).
    """
    # Exemplo usando um backbone pré-treinado do torchvision (opcional)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Ajuste da última camada para 2 classes
    model.fc = nn.Linear(model.fc.in_features, 2)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    return model, criterion, optimizer


def train(max_epochs, device, model, optimizer, criterion, train_loader, val_loader):
    """Loop de treino. Deve exibir métricas por época e retornar o modelo treinado."""
    print('-' * 50)
    print('Treinando modelo...')
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            # Se o seu modelo tiver saídas auxiliares (ex.: Inception/GoogLeNet), some a perda auxiliar aqui
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_loader))
        # Avaliação de validação usando utilitário do projeto
        epoch_summary = f"Epoch {epoch + 1}/{max_epochs} - Loss: {avg_loss:.4f},"
        accuracy, all_preds, all_labels = evaluate_model(model, val_loader, device)
        epoch_summary += f" Accuracy: {accuracy:.2f}%"
        print(epoch_summary)

    print('-' * 50)
    return model


# Opcional, mas recomendado: quantização estática
# Implemente somente se souber que a arquitetura é compatível com quantização.
# É necessário rodar em CPU e calibrar passando pelo dataset de novo.
def static_quantize(model_cpu, val_loader, report_generator):
   ...
```

## Passo a passo para adicionar um novo modelo

1. Crie `scripts/models/NomeDoModeloImpl.py` seguindo o template acima. Use nomes claros.
2. No `init()`, ajuste a última camada para o número de classes (padrão = 2).
3. Garanta que o `train()` receba exatamente: `(max_epochs, device, model, optimizer, criterion, train_loader, val_loader)` e retorne o `model` ao final.
4. Se seu modelo tiver saídas auxiliares (ex.: Inception/GoogLeNet), some a perda auxiliar no `train()` (veja os exemplos existentes no repositório).
5. (Opcional) Implemente `static_quantize(...)` apenas se a arquitetura for compatível. Teste em CPU.

## Exemplos de referência

Consulte implementações prontas:
- `scripts/models/GooglenetImpl.py`
- `scripts/models/InceptionV3Impl.py`
