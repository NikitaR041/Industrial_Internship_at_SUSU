# Руководство по конвертации YOLO11 в RKNN (Orange Pi 5B)

Данный документ содержит пошаговую инструкцию и технические требования для конвертации обученной модели YOLO11 в формат `.rknn`, оптимизированный для запуска на NPU процессора Rockchip RK3588S.

---

## 1. Системные требования и зависимости

Для обеспечения точной воспроизводимости процесса используйте следующие версии библиотек, установленные в рабочей среде:

| Компонент | Версия |
| :--- | :--- |
| **PyTorch** | 2.4.0 |
| **rknn-toolkit2** | 2.3.2 |
| **ONNX** | 1.18.0 |
| **OpenCV-Python** | 4.11.0.86 |

**Содержимое файла `requirements.txt`:**

```text
onnx==1.18.0
onnx-ir==0.2.1
onnxruntime==1.26.0
onnxscript==0.7.1
onnxslim==0.1.34
rknn-toolkit2==2.3.2
torch==2.4.0
torchvision==0.28.0
-e git+https://github.com/airockchip/ultralytics_yolo11.git@0692e9297670acf4cc6d0cec773d7a9493cb8a5f#egg=ultralytics
ultralytics-thop==2.0.20
numpy==1.26.4
pandas==3.0.3
opencv-python==4.11.0.86
pillow==10.2.0
scipy==1.17.1
tqdm==4.68.4
PyYAML==6.0.1
requests==2.31.0
matplotlib==3.11.0
seaborn==0.13.2
```

Также для проведения конвертации и последующего пост-процессинга необходимо клонировать официальный репозиторий rknn_model_zoo в корневую директорию вашего проекта:

```
git clone [https://github.com/airockchip/rknn_model_zoo.git](https://github.com/airockchip/rknn_model_zoo.git)
```

## 2. Установка зависимостей

### 2.1. Создать виртуальное окружение

```
python3 -m venv venv
source venv/bin/activate
```

### 2.2. Установка

Установите пакеты из сформированного списка. Если часть зависимостей не скачивается с основного индекса PyPI, используйте зеркало университета Цинхуа, передав ключ -i:

```
pip install -r requirements.txt -i [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple)
```

## 3. Инструкция по конвертации

### 3.1. Экспорт в ONNX

Установленный форк airockchip/ultralytics_yolo11 отсекает подграфы за свёрточными слоями головы и оставляет девять «сырых» тензоров, что позволяет корректно отобразить модель на аппаратуру NPU. Выполните экспорт:

```
yolo export model=best.pt format=onnx opset=12 simplify=True
```

### 3.2. Калибровка и квантизация

Сформируйте файл dataset.txt, содержащий абсолютные пути к калибровочным изображениям (по одному пути в строке). Затем используйте следующий скрипт для компиляции (Перед запуском скрипта убедитесь, что вы указали корректные относительные пути к вашей обученной модели (PT_MODEL_NAME) и папке с калибровочными изображениями (CALIBRATION_DIR)):

```
import os
import sys
import glob
import re
import subprocess
from ultralytics import YOLO

# Путь к вашей обученной модели (например, внутри папки проекта)
PT_MODEL_NAME = "best.pt"

# Папка с калибровочными изображениями для квантования INT8
CALIBRATION_DIR = "calibration_samples"

# Имя итогового файла RKNN (сохранится в корневую папку)
FINAL_RKNN_NAME = "yolo11n_rk3588.rknn"
# ==============================================

# Автоматически определяем путь для ONNX на основе модели
ONNX_MODEL_NAME = os.path.splitext(PT_MODEL_NAME)[0] + ".onnx"

DATASET_TXT = "dataset.txt"
TARGET_PLATFORM = "rk3588"

def log_step(step_num, title):
    print(f"\n\n{'='*70}\n[ЭТАП {step_num}] {title}\n{'='*70}")

# === ЭТАП 1: Поиск скрипта convert.py в Model Zoo ===
log_step(1, "Поиск скрипта конвертера в rknn_model_zoo")
zoo_convert_script = "rknn_model_zoo/examples/yolo11/python/convert.py"

if not os.path.exists(zoo_convert_script):
    print(f"[ОШИБКА] Не найден скрипт по пути: {zoo_convert_script}")
    sys.exit(1)
print(f"[ОК] Найден официальный конвертер: {zoo_convert_script}")

# === ЭТАП 2: Подготовка калибровочного списка ===
log_step(2, f"Сканирование папки {CALIBRATION_DIR}")
if not os.path.exists(CALIBRATION_DIR):
    print(f"[ОШИБКА] Папка {CALIBRATION_DIR} не найдена в текущей директории!")
    sys.exit(1)

extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
img_paths = []
for ext in extensions:
    img_paths.extend(glob.glob(os.path.join(CALIBRATION_DIR, ext)))

if not img_paths:
    print(f"[ОШИБКА] В папке {CALIBRATION_DIR} нет изображений.")
    sys.exit(1)

with open(DATASET_TXT, "w") as f:
    for path in img_paths:
        f.write(f"{os.path.abspath(path)}\n")
print(f"[ОК] Создан {DATASET_TXT}. Найдено кадров для калибровки: {len(img_paths)}")

# === ЭТАП 3: Экспорт в ONNX через форк ===
log_step(3, f"Экспорт {PT_MODEL_NAME} -> ONNX")
if not os.path.exists(PT_MODEL_NAME):
    print(f"[ОШИБКА] Исходный файл {PT_MODEL_NAME} не найден. Проверьте правильность пути.")
    sys.exit(1)

print("Запуск экспортера...")
model = YOLO(PT_MODEL_NAME)
model.export(format="onnx", opset=12, simplify=True)

if not os.path.exists(ONNX_MODEL_NAME):
    print(f"[ОШИБКА] Экспорт не удался. Ожидаемый файл не найден по пути: {ONNX_MODEL_NAME}")
    sys.exit(1)
print(f"[ОК] Оптимизированный ONNX сохранен: {ONNX_MODEL_NAME}")

# === ЭТАП 4: Конвертация через Model Zoo ===
log_step(4, "Квантование INT8 и сборка RKNN")

with open(zoo_convert_script, 'r') as file:
    code = file.read()

code = re.sub(r"DATASET_PATH\s*=\s*['\"].*?['\"]", f"DATASET_PATH = '{os.path.abspath(DATASET_TXT)}'", code)

with open(zoo_convert_script, 'w') as file:
    file.write(code)

cmd = [
    sys.executable,
    zoo_convert_script,
    os.path.abspath(ONNX_MODEL_NAME),
    TARGET_PLATFORM,
    "i8",
    os.path.abspath(FINAL_RKNN_NAME)
]

print(f"Запуск команды: {' '.join(cmd)}\n")
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

while True:
    output = process.stdout.readline()
    if output == '' and process.poll() is not None:
        break
    if output:
        print(output.strip())

if process.poll() == 0 and os.path.exists(FINAL_RKNN_NAME):
    print(f"\n[УСПЕХ] Сборка завершена. Модель для Orange Pi 5B: {os.path.abspath(FINAL_RKNN_NAME)}")
else:
    print(f"\n[ОШИБКА] Конвертация завершилась неудачей с кодом {process.poll()}")
```

## 4. Версии ПО на целевом устройстве

Версии инструмента конвертации (на ПК) и библиотеки исполнения (на устройстве) должны совпадать, иначе загрузка модели завершится ошибкой или выдаст некорректные результаты. Для корректного инференса на целевом устройстве Orange Pi 5B (RK3588S) требуются:

**Инструмент инференса:** rknn-toolkit-lite2 версии 2.3.2

**Библиотека исполнения:** librknnrt 2.3.2

**Драйвер NPU:** 0.9.6
