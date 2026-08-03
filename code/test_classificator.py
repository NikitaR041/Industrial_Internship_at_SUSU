# Запуск: python3 test_classifier.py bukva_mobilenet_v2_8fps_hands_correct.rknn test_video.mp4 result.mp4
import sys
import cv2
import numpy as np
import time
from rknnlite.api import RKNNLite

# Твои классы
CLASSES = ["no_event", "Ё", "А", "Б", "В", "Г", "Д", "Е", "Ж", "З", "И", "Й",
           "К", "Л", "М", "Н", "О", "П", "Р", "С", "Т", "У", "Ф", "Х", "Ц",
           "Ч", "Ш", "Щ", "Ъ", "Ы", "Ь", "Э", "Ю", "Я"]

# MobileNetV2 требует размер 224x224
IMG_SIZE = (224, 224)

def main():
    model_path = sys.argv[1]
    video_path = sys.argv[2]
    out_path = sys.argv[3]

    # Инициализация RKNN на 0-м ядре NPU
    rknn = RKNNLite()
    print("Загрузка модели...")
    rknn.load_rknn(model_path)
    rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frames = 0
    t0 = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Подготовка кадра
        # Переводим в RGB, так как OpenCV читает в BGR
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Классификатору нужен жесткий ресайз (без сохранения пропорций, в отличие от YOLO)
        img = cv2.resize(img, IMG_SIZE)
        # Добавляем размерность батча -> (1, 224, 224, 3)
        img = np.expand_dims(img, 0)

        # 2. Инференс (отправляем в NPU)
        outputs = rknn.inference(inputs=[img])
        
        # 3. Постобработка
        # Выход имеет форму (1, 34), берем первый элемент
        probabilities = outputs[0][0]
        
        # Ищем индекс класса с максимальной вероятностью
        class_id = np.argmax(probabilities)
        score = probabilities[class_id]

        # 4. Отрисовка
        label = f"{CLASSES[class_id]}: {score:.2f}"
        
        # Рисуем подложку для текста, чтобы было лучше видно
        cv2.rectangle(frame, (10, 10), (400, 70), (0, 0, 0), -1)
        # Пишем текст в левом верхнем углу
        cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        writer.write(frame)
        frames += 1
        print(f"Обработано кадров: {frames}", end="\r")

    dt = time.time() - t0
    print(f"\nГотово! {frames} кадров за {dt:.1f} сек. Средний FPS: {frames/dt:.1f}")
    
    cap.release()
    writer.release()
    rknn.release()

if __name__ == "__main__":
    main()
