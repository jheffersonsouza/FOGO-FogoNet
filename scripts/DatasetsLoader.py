import os
from pathlib import Path

from torchvision import transforms, datasets


class Dataset:
    def __init__(self, name, root_path):
        self.name = name
        self.root_path = Path(root_path)

        self.train_tfms = None
        self.test_tfms = None

    def define_train_transform(self, transform_model):
        self.train_tfms = transform_model

    def define_test_transform(self, transform_model):
        self.test_tfms = transform_model

    def get(self, transform_model=None):
        if transform_model == "inception-format":
            self.train_tfms = transforms.Compose([
                transforms.RandomResizedCrop(299, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.RandomRotation(10),  # Rotaciona a imagem em relação o chao
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

            self.test_tfms = transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        elif transform_model == "googlenet-format":
            pass

        train_data = datasets.ImageFolder(self.root_path / self.name / 'train', transform=self.train_tfms)
        val_data = datasets.ImageFolder(self.root_path / self.name / 'test', transform=self.test_tfms)
        return train_data, val_data


def _get_datasets(datasets_root_path):
    datasets = []
    for folder in os.listdir(datasets_root_path):
        full_path = os.path.join(datasets_root_path, folder)
        if os.path.isdir(full_path):
            datasets.append(Dataset(folder, datasets_root_path))
    return datasets


def request_dataset(datasets_path):
    datasets = _get_datasets(datasets_path)
    if len(datasets) == 0:
        print('Nenhum dataset encontrado!')
        return None

    print('\nDatasets disponiveis:')
    for i, dataset in enumerate(datasets):
        print(f'{i + 1}. {dataset.name}')
    datasetOpt = int(input('Escolha o seu dataset: '))
    while datasetOpt not in range(1, len(datasets) + 1):
        print('Opção inválida.')
        datasetOpt = int(input('Escolha o seu dataset: '))
    return datasets[datasetOpt - 1]
