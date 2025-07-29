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
├── datasets/                  - Diretório para armazenamento dos datasets
│   └── SETUP.md               
├── results/                   - Diretório onde são salvos os resultados da avaliação
├── utils/                     
│   ├── DatasetsLoader.py      - Carregamento e pré-processamento dos datasets
│   ├── Entities.py            - Funções referente ao modelo a ser utilizado e outras coisas ae
│   └── ReportGenerator.py     - Geração de relatórios e visualizações
├── dependencies.txt           
├── main.py                    - Script principal para treinamento e avaliação
└── README.md                  
```

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

