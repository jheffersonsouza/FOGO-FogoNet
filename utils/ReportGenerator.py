import os
import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from utils.Entities import evaluate_model

class ReportGenerator:
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def summary(self, name, model, val_loader, device, save_model=False):
        print(f'{'-'*10} Avaliação do modelo {name}{'-'*10}')
        # TODO: Talvez tirar acentuação?
        modelNameDir = name.strip().lower().replace(' ', '_')
        output_dir = os.path.join(self.output_dir, modelNameDir)

        accuracy, preds, labels = evaluate_model(model, val_loader, device)
        print(f"Acurácia: {accuracy:.2f}%")
        print(f"Tempo médio por batch: {self._measure_inference_time(model, val_loader, device):.4f}s")
        print(f"Tamanho: {self._get_model_size(model, name):.2f} MB")
        print('Gerando matrix de confusão...')
        self._save_confusion_matrix(name, output_dir, labels, preds, val_loader.dataset.classes)

        if save_model:
            torch.save(model.state_dict(), os.path.join(output_dir, modelNameDir+ ".pth"))

    def _measure_inference_time(self, model, data_loader, device):
        model.eval()
        times = []

        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(device)
                start = time.time()
                _ = model(inputs)[0]
                end = time.time()
                times.append(end - start)

        avg_time = np.mean(times)
        return avg_time

    def _get_model_size(self, model, model_name="Sem Nome"):
        safe_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        temp_file = f"temp_{safe_name}.pth"

        try:
            torch.save(model.state_dict(), temp_file)
            size = os.path.getsize(temp_file) / 1024 / 1024  # MB
            os.remove(temp_file)
            return size
        except Exception as e:
            print(f"Erro ao calcular tamanho do modelo '{model_name}': {e}")
            return 0

    def _save_confusion_matrix(self, model_name, output_dir, all_labels, all_preds, class_names):
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

        plt.figure(figsize=(6, 6))
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title(f"Matriz de Confusão - {model_name}")
        plt.savefig(os.path.join(output_dir, f"matriz_confusao_{model_name.lower().replace(' ', '_')}.png"))
        plt.close()

        report = classification_report(all_labels, all_preds, target_names=class_names)
        with open(os.path.join(output_dir, f"relatorio_{model_name.lower().replace(' ', '_')}.txt"), "w") as f:
            f.write(f"=== {model_name} ===\n")
            f.write(report)

        np.savetxt(os.path.join(output_dir, f"matriz_confusao_{model_name.lower().replace(' ', '_')}.csv"),
                   cm, delimiter=",", fmt="%d")

