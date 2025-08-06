import torch
from torchvision import models
import torch.quantization as quantization

def handles_model(model_name):
    return model_name == "inceptionv3"

def quantize(model_cpu, val_loader, report_generator):
    try:
        model_static = models.inception_v3(pretrained=True, aux_logits=True)
        model_static.AuxLogits.fc = torch.nn.Linear(model_static.AuxLogits.fc.in_features, 2)
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
                
                report_generator.summary("Quantizado Estático", model_static_quantized, val_loader, torch.device("cpu"))
                
                return model_static_quantized

            except Exception as e:
                print(f"Erro com backend {backend}: {e}")
                continue
    except Exception as e:
        print(f"ERRO na quantização estática: {e}")
        print("Continuando apenas com quantização dinâmica...")
    
    return None