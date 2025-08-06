import torch
from torchvision import models

from scripts.ReportGenerator import ReportGenerator
import torch.quantization as quantization



class ModelQuantizer:
    def __init__(self, model_info: list, val_loader, report_generator: ReportGenerator):
        self.modelName = model_info[0]
        self.model = model_info[1]
        self.val_loader = val_loader
        self.reportGenerator = report_generator
    def dynamic(self):
        print('Aplicando quantização dinâmica...')
        model_cpu = self.model.to('cpu')
        model_dynamic_quantized = torch.quantization.quantize_dynamic(
            model_cpu,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        # Não da pra mandar o 'model_cpu" como devic?
        self.reportGenerator.summary("Quantizado Dinâmico", model_dynamic_quantized, val_loader, torch.device("cpu"))
    def static(self):
        print('Aplicando quantização estatica')
        model_cpu = self.model.to('cpu')
        if self.modelName == "Inception-V3":
            pass
        elif self.modelName == "GoogLeNet":
            pass
        else:
            print('Modelo escolhido não possui uma implementação de quantização estatica.')



class StaticQuantizerAbstract:
    def __init__(self):
        pass
    def inceptionv3(self, model_cpu, val_loader):
        try:
            model_static = models.googlenet(pretrained=True)
            model_static.fc = torch.nn.Linear(model_static.fc.in_features, 2)
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

                save_confusion_matrix(static_labels, static_preds, class_names, output_dir,
                                      "Modelo Quantizado Estático")
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
