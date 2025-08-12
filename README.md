# FOGO-FogoNet
Sistema responsável pelo treinamento e avaliação de modelos de IA para reconhecimento de fogo em imagens capturadas por drones.
## Equipe responsável
[Time de Dados](https://github.com/orgs/pelican-program/teams/time-de-dados)

## Funcionalidades

- Treinamento de modelos de IA para detecção de fogo em imagens
- Avaliação de desempenho dos modelos (acurácia, tempo de inferência, tamanho)
- Quantização dos modelos (dinâmica e estática)
- Geração de relatórios detalhados sobre o desempenho e matriz de confusão

## Estrutura do Projeto

```
FOGO-FogoNet/
├── datasets/
│   └── SETUP.md
├── core/
│   ├── DatasetsLoader.py
│   ├── Entities.py
│   ├── ModelQuantizer.py
│   └── ReportGenerator.py
├── scripts/
│   └── models/
│       ├── GooglenetImpl.py
│       ├── InceptionV3Impl.py
│       └── SETUP.md
├── results/
├── dependencies.txt
├── main.py
└── README.md
```

Para aprender a criar/registrar um novo modelo, consulte: [scripts/models/SETUP.md](scripts/models/SETUP.md)

## Dependências

Leia os arquivo de [dependências](dependencies.txt).


## Instalação

1. Clone este repositório.
2. Instale as dependências.
3. [Baixe um Dataset](/datasets/SETUP.md#-links-para-download-dos-datasets) ou [adicione um](/datasets/SETUP.md#-adicionar-manualmente).
4. Rode o arquivo principal [main.py](main.py).

## Resultados

Os resultados são salvos no diretório `results/` e incluem:

- Matrizes de confusão (.png e .csv)
- Relatórios de classificação (.txt)
- Comparação de desempenho entre modelos (comparacao_modelos.txt)
- Modelos salvos (.pth)

