#!/usr/bin/env python3
# Инференс YOLO11 (.rknn) на Orange Pi 5 / RK3588S.
#
# Примеры:
#   python3 detect_video.py best_fp16.rknn video.mp4                  # -> out.mp4
#   python3 detect_video.py best_fp16.rknn video.mp4 --save res.mp4
#   python3 detect_video.py best_fp16.rknn 0 --save cam.mp4           # USB-камера
#   python3 detect_video.py best_fp16.rknn video.mp4 --imgsz 640 --conf 0.25

import sys
import time
import argparse
import queue
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from rknnlite.api import RKNNLite

# Имена классов. ДОЛЖНЫ совпадать по количеству и порядку с моделью --
# проверь командой:
#   python -c "import torch; print(torch.load('best.pt', map_location='cpu',
#              weights_only=False)['model'].names)"
# Если длина не совпадёт с моделью, скрипт подпишет боксы как class_N.
CLASSES = (
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
)

_rng = np.random.default_rng(7)
COLORS = _rng.integers(60, 255, size=(64, 3)).tolist()

NUM_WORKERS = 3  # по числу ядер NPU RK3588
CORE_MASKS = None  # заполняется в main


# ---------------- препроцессинг ----------------
def letterbox(img, new_shape, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape[0] / w, new_shape[1] / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    pad_w, pad_h = new_shape[0] - nw, new_shape[1] - nh
    pad_left, pad_top = pad_w // 2, pad_h // 2
    if (w, h) != (nw, nh):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(
        img, pad_top, pad_h - pad_top, pad_left, pad_w - pad_left,
        cv2.BORDER_CONSTANT, value=color,
    )
    return img, r, (pad_left, pad_top)


# ---------------- постпроцессинг ----------------
def softmax(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def dfl(position):
    n, c, h, w = position.shape
    mc = c // 4
    y = softmax(position.reshape(n, 4, mc, h, w), 2)
    acc = np.arange(mc).reshape(1, 1, mc, 1, 1).astype(np.float32)
    return (y * acc).sum(2)


def box_process(position, img_size):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
    grid = np.concatenate(
        (col.reshape(1, 1, grid_h, grid_w), row.reshape(1, 1, grid_h, grid_w)),
        axis=1,
    ).astype(np.float32)
    stride = np.array(
        [img_size[1] // grid_h, img_size[0] // grid_w], dtype=np.float32
    ).reshape(1, 2, 1, 1)
    position = dfl(position)
    xy1 = grid + 0.5 - position[:, 0:2, :, :]
    xy2 = grid + 0.5 + position[:, 2:4, :, :]
    return np.concatenate((xy1 * stride, xy2 * stride), axis=1)


def nms(boxes, scores, thresh):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[np.where(iou <= thresh)[0] + 1]
    return keep


def detect_layout(outputs):
    per_branch = len(outputs) // 3
    if per_branch not in (2, 3):
        raise RuntimeError(
            f"Неожиданное число выходов: {len(outputs)} (жду 6 или 9). "
            "Проверь, что экспорт делался через форк airockchip."
        )
    nc = outputs[1].shape[1]
    return per_branch, nc


def post_process(outputs, img_size, nc, per_branch, obj_thresh, nms_thresh):
    boxes, scores = [], []
    for i in range(3):
        box_t = outputs[per_branch * i]
        cls_t = outputs[per_branch * i + 1]
        b = box_process(box_t, img_size)
        boxes.append(b.transpose(0, 2, 3, 1).reshape(-1, 4))
        scores.append(cls_t.transpose(0, 2, 3, 1).reshape(-1, nc))

    boxes = np.concatenate(boxes)
    scores = np.concatenate(scores)

    class_ids = scores.argmax(1)
    confidences = scores.max(1)
    mask = confidences >= obj_thresh
    boxes, confidences, class_ids = boxes[mask], confidences[mask], class_ids[mask]
    if len(boxes) == 0:
        return None, None, None

    if nc == 1:
        keep = nms(boxes, confidences, nms_thresh)
    else:  # NMS внутри каждого класса (сдвиг боксов по class_id)
        offset = class_ids.astype(np.float32).reshape(-1, 1) * 4096.0
        keep = nms(boxes + offset, confidences, nms_thresh)

    return boxes[keep], confidences[keep], class_ids[keep]


def scale_boxes(boxes, ratio, pad):
    boxes = boxes.copy()
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes /= ratio
    return boxes


def class_name(k, nc):
    if nc == len(CLASSES):
        return CLASSES[k]
    if nc == 1:
        return "person"
    return f"class_{k}"


# ---------------- пул NPU-инстансов (по одному на ядро) ----------------
class NpuPool:
    def __init__(self, model_path, n=NUM_WORKERS):
        masks = [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2]
        self.q = queue.Queue()
        self.instances = []
        for i in range(n):
            r = RKNNLite()
            if r.load_rknn(model_path) != 0:
                sys.exit("Не удалось загрузить модель")
            if r.init_runtime(core_mask=masks[i % 3]) != 0:
                sys.exit(f"Не удалось инициализировать NPU (ядро {i % 3})")
            self.instances.append(r)
            self.q.put(r)

    def infer(self, inp):
        r = self.q.get()          # берём свободное ядро
        try:
            return r.inference(inputs=[inp])
        finally:
            self.q.put(r)         # возвращаем в пул

    def release(self):
        for r in self.instances:
            r.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="путь к .rknn")
    ap.add_argument("source", help="видеофайл или индекс камеры")
    ap.add_argument("--save", default="out.mp4", help="выходной mp4 (по умолчанию out.mp4)")
    ap.add_argument("--imgsz", type=int, default=640, help="вход модели (как при экспорте)")
    ap.add_argument("--conf", type=float, default=0.25, help="порог уверенности")
    ap.add_argument("--nms", type=float, default=0.45, help="порог NMS (IoU)")
    args = ap.parse_args()

    img_size = (args.imgsz, args.imgsz)

    print(f"Инициализирую {NUM_WORKERS} инстанса NPU...")
    pool = NpuPool(args.model)

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"Не удалось открыть источник: {args.source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 0/отриц. для камер
    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))

    state = {"nc": None, "per_branch": None}
    lock = threading.Lock()

    def process(frame):
        img, ratio, pad = letterbox(frame, img_size)
        inp = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
        outputs = pool.infer(inp)

        with lock:
            if state["nc"] is None:
                state["per_branch"], state["nc"] = detect_layout(outputs)
                grid = outputs[0].shape[2]
                if grid * 8 != img_size[1]:
                    print(f"[!] ВНИМАНИЕ: модель ждёт вход {grid*8}, "
                          f"а --imgsz {args.imgsz}. Перезапусти с --imgsz {grid*8}")
                print(f"Модель: {state['nc']} класс(ов), выходов {len(outputs)}, "
                      f"вход {grid*8}")

        boxes, confs, ids = post_process(
            outputs, img_size, state["nc"], state["per_branch"], args.conf, args.nms
        )
        if boxes is not None:
            boxes = scale_boxes(boxes, ratio, pad)
            for (x1, y1, x2, y2), c, k in zip(boxes, confs, ids):
                color = tuple(COLORS[int(k) % len(COLORS)])
                p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                cv2.rectangle(frame, p1, p2, color, 2)
                cv2.putText(frame, f"{class_name(int(k), state['nc'])} {c:.2f}",
                            (p1[0], max(0, p1[1] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return frame

    # Конвейер: держим "окно" из нескольких кадров в обработке одновременно,
    # чтобы все 3 ядра NPU были заняты, но запись в файл шла строго по порядку.
    WINDOW = NUM_WORKERS * 2
    frames_done = 0
    t0 = time.time()

    print(f"Работаю (headless). Результат -> {args.save}. Ctrl+C — остановить.")
    try:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
            pending = deque()
            eof = False
            while True:
                while not eof and len(pending) < WINDOW:
                    ok, frame = cap.read()
                    if not ok:
                        eof = True
                        break
                    pending.append(ex.submit(process, frame))

                if not pending:
                    break

                writer.write(pending.popleft().result())
                frames_done += 1
                if frames_done % 50 == 0:
                    fps = frames_done / (time.time() - t0)
                    prog = f"{frames_done}/{total}" if total > 0 else str(frames_done)
                    print(f"  кадров: {prog}, FPS: {fps:.1f}")
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.")
    finally:
        cap.release()
        writer.release()
        pool.release()
        dt = time.time() - t0
        print(f"Готово: {frames_done} кадров за {dt:.1f} c "
              f"({frames_done / max(dt, 1e-6):.1f} FPS) -> {args.save}")


if __name__ == "__main__":
    main()
