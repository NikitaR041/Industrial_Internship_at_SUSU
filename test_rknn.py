# main.py — видео-детекция транспорта на Orange Pi 5B, пул на 3 ядра NPU (rknn-toolkit-lite2)
# Запуск: python3 main.py yolov8_vehicles.rknn traffic.mp4 output.mp4
#         python3 main.py yolov8_vehicles.rknn 0            output.mp4   # камера (Ctrl+C для выхода)
import sys, time, threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from rknnlite.api import RKNNLite

# --- ВПИШИТЕ ВАШИ КЛАССЫ В ТОМ ЖЕ ПОРЯДКЕ, что и в data.yaml ---
CLASSES = ("Car", "Number Plate", "Blur Number Plate", "Two Wheeler", "Auto", "Bus", "Truck")

IMG_SIZE   = (640, 640)   # (w, h)
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
NUM_CORES  = 3            # RK3588S: 3 ядра NPU

# ==================== препроцессинг ====================
def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    h0, w0 = im.shape[:2]
    r = min(new_shape[1] / h0, new_shape[0] / w0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    dw, dh = (new_shape[0] - new_w) / 2, (new_shape[1] - new_h) / 2
    im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, dw, dh

def scale_boxes(boxes, r, dw, dh):
    boxes = boxes.copy().astype(np.float32)
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes /= r
    return boxes

# ==================== постобработка (numpy, без torch) ====================
def dfl(position):
    n, c, h, w = position.shape
    mc = c // 4
    y = position.reshape(n, 4, mc, h, w)
    e = np.exp(y - np.max(y, axis=2, keepdims=True))
    y = e / np.sum(e, axis=2, keepdims=True)
    acc = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    return (y * acc).sum(2)

def box_process(position):
    gh, gw = position.shape[2:4]
    col, row = np.meshgrid(np.arange(gw), np.arange(gh))
    col = col.reshape(1, 1, gh, gw)
    row = row.reshape(1, 1, gh, gw)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1] // gh, IMG_SIZE[0] // gw]).reshape(1, 2, 1, 1)
    position = dfl(position)
    box_xy  = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    return np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

def filter_boxes(boxes, box_confidences, box_class_probs):
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    return boxes[pos], classes[pos], (class_max_score * box_confidences)[pos]

def nms_boxes(boxes, scores):
    x, y = boxes[:, 0], boxes[:, 1]
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    areas = w * h
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
        w1 = np.maximum(0.0, xx2 - xx1 + 1e-5)
        h1 = np.maximum(0.0, yy2 - yy1 + 1e-5)
        inter = w1 * h1
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= NMS_THRESH)[0] + 1]
    return np.array(keep)

def post_process(outputs):
    boxes, scores, classes_conf = [], [], []
    per_branch = len(outputs) // 3
    for i in range(3):
        boxes.append(box_process(outputs[per_branch * i]))
        classes_conf.append(outputs[per_branch * i + 1])
        scores.append(np.ones_like(outputs[per_branch * i + 1][:, :1, :, :], dtype=np.float32))

    def flat(v):
        ch = v.shape[1]
        return v.transpose(0, 2, 3, 1).reshape(-1, ch)

    boxes = np.concatenate([flat(b) for b in boxes])
    classes_conf = np.concatenate([flat(c) for c in classes_conf])
    scores = np.concatenate([flat(s) for s in scores])
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    nb, ncl, ns = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b, cc, s = boxes[inds], classes[inds], scores[inds]
        keep = nms_boxes(b, s)
        if len(keep):
            nb.append(b[keep]); ncl.append(cc[keep]); ns.append(s[keep])
    if not ncl:
        return None, None, None
    return np.concatenate(nb), np.concatenate(ncl), np.concatenate(ns)

# ==================== обработка одного кадра ====================
def infer_frame(rknn, frame):
    img, r, dw, dh = letterbox(frame, IMG_SIZE, (0, 0, 0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # модель обучалась на RGB
    img = np.expand_dims(img, 0)                        # ← ДОБАВИТЬ: (1, 640, 640, 3), RKNN требует 4D
    outputs = rknn.inference(inputs=[img])
    boxes, classes, scores = post_process(outputs)
    if boxes is not None:
        boxes = scale_boxes(boxes, r, dw, dh)
        for box, cl, sc in zip(boxes, classes, scores):
            x1, y1, x2, y2 = [int(v) for v in box]
            name = CLASSES[cl] if cl < len(CLASSES) else f"id{cl}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} {sc:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame

# ==================== пул на 3 ядра NPU ====================
class RKNNPool:
    """N экземпляров модели, привязанных к ядрам 0/1/2. Порядок кадров сохраняется (FIFO),
       каждый экземпляр защищён своим мьютексом от одновременного доступа из двух потоков."""
    _CORES = [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2]

    def __init__(self, model_path, func, n=3):
        self.func = func
        self.n = n
        self.q = Queue()
        self.pool = ThreadPoolExecutor(max_workers=n)
        self.models, self.locks = [], []
        for i in range(n):
            m = RKNNLite()
            assert m.load_rknn(model_path) == 0, "Не удалось загрузить .rknn"
            assert m.init_runtime(core_mask=self._CORES[i % 3]) == 0, \
                "init_runtime failed — проверьте версию librknnrt.so (должна быть 2.3.2)"
            self.models.append(m)
            self.locks.append(threading.Lock())
        self.i = 0

    def _work(self, idx, frame):
        with self.locks[idx]:
            return self.func(self.models[idx], frame)

    def put(self, frame):
        idx = self.i % self.n
        self.q.put(self.pool.submit(self._work, idx, frame))
        self.i += 1

    def get(self):
        if self.q.empty():
            return None, False
        return self.q.get().result(), True

    def release(self):
        self.pool.shutdown()
        for m in self.models:
            m.release()

# ==================== main ====================
def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "yolov8_vehicles.rknn"
    source     = sys.argv[2] if len(sys.argv) > 2 else "traffic.mp4"
    out_path   = sys.argv[3] if len(sys.argv) > 3 else "output.mp4"

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    assert cap.isOpened(), f"Не удалось открыть источник: {source}"
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # если mp4v недоступен — замените на cv2.VideoWriter_fourcc(*"XVID") и out.avi
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))

    pool = RKNNPool(model_path, infer_frame, n=NUM_CORES)

    # заполняем конвейер, чтобы все 3 ядра сразу заработали
    for _ in range(NUM_CORES):
        ret, frame = cap.read()
        if ret:
            pool.put(frame)

    running, frames, t0 = True, 0, time.time()
    try:
        while True:
            if running:
                ret, frame = cap.read()
                if ret:
                    pool.put(frame)
                else:
                    running = False
            result, ok = pool.get()
            if not ok:
                if not running:
                    break
                continue
            writer.write(result)
            frames += 1
            if frames % 30 == 0:
                fps = frames / (time.time() - t0)
                print(f"Кадров: {frames}, средний FPS: {fps:.1f}", end="\r")
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.")

    dt = time.time() - t0
    if dt > 0:
        print(f"\nГотово: {frames} кадров за {dt:.1f} c, средний FPS: {frames/dt:.1f}")
    print(f"Результат сохранён: {out_path}")
    cap.release()
    writer.release()
    pool.release()

if __name__ == "__main__":
    main()
