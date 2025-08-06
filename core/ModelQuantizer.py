import torch
from torchvision import models
import os
import importlib.util

from core.ReportGenerator import ReportGenerator
import torch.quantization as quantization

from importlib.util import spec_from_file_location
from importlib.util import module_from_spec


class ModelQuantizer:
    def __init__(self, model, val_loader, report_generator: ReportGenerator):
        self.modelName = model[0]
        self.model = model[1]
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
        self.reportGenerator.summary("Quantizado Dinâmico", model_dynamic_quantized, self.val_loader, torch.device("cpu"))
    def static(self):
        print('Aplicando quantização estatica...')
        model_cpu = self.model.to('cpu')
        
        quantizer_dir = "scripts/static_quantizers"
        model_static_quantized = None        
        if os.path.exists(quantizer_dir):
            for filename in os.listdir(quantizer_dir):
                if filename.endswith('.py'):
                    module_name = filename[:-3]  #.py
                    try:
                        spec = spec_from_file_location(module_name, os.path.join(quantizer_dir, filename))
                        module = module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        if hasattr(module, 'handles_model') and module.handles_model(self.modelName):
                            print(f"Using quantizer from {filename}")
                            model_static_quantized = module.quantize(model_cpu, self.val_loader, self.reportGenerator)
                            break
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")
        
        if not model_static_quantized:
            print(f'Modelo {self.modelName} não possui uma implementação de quantização estatica.')


