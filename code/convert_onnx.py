import torch
import torch.nn as nn
from torchvision import models

# 1. Настройки (должны совпадать с тем, что было при обучении)
NUM_CLASSES = 34 # Количество твоих классов
IMG_SIZE = 224
PT_FILE = "bukva_mobilenet_v2_8fps_hands_correct.pt"
ONNX_FILE = "bukva_mobilenet_v2_8fps_hands_correct.onnx"

print("Создаем архитектуру модели...")
# 2. Воссоздаем пустую архитектуру MobileNetV2
model = models.mobilenet_v2()
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)

print(f"Загружаем веса из {PT_FILE}...")
# 3. Загружаем твои обученные веса (на CPU, так как для конвертации GPU не обязателен)
checkpoint = torch.load(PT_FILE, map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["model"])
model.eval()

print("Конвертируем в ONNX...")
# 4. Выполняем экспорт
dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

torch.onnx.export(
    model, 
    dummy_input, 
    ONNX_FILE,
    export_params=True,        # Вшиваем веса внутрь файла
    input_names=["input"], 
    output_names=["output"],
    opset_version=14,          # Оптимально для RKNN
    do_constant_folding=True
)

print(f"Успех! Файл {ONNX_FILE} создан и готов к скармливанию в RKNN-Toolkit2.")
