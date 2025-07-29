## 🔗 Links para Download dos Datasets
1. [Dataset do Otavio](google.com)
2. [Dataset do Jhefferson](google.com)
3. ...

> ⚠️ O espaço de armazenamento para baixar todos é de X GB.

---

## 📁 Adicionar manualmente

Este projeto utiliza a estrutura de pastas compatível com o `ImageFolder` do **PyTorch**. Para garantir o funcionamento correto, organize seus dados da seguinte forma:

```
Nome do dataset/
├── train/
│   ├── classe_1/
│   └── classe_2/
└── test/
    ├── classe_1/
    └── classe_2/
```

Cada subpasta dentro de `train/` e `test/` deve conter imagens pertencentes à respectiva classe.

> 🔍 Para mais informações sobre o `ImageFolder`, consulte a [documentação oficial do PyTorch](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html).

---
