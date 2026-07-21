# Detect_video

Инференс YOLO11 (.rknn) на Orange Pi 5 / RK3588S.

# Примеры вызова. 

python3 detect_video.py best_fp16.rknn video.mp4                  # -> out.mp4
python3 detect_video.py best_fp16.rknn video.mp4 --save res.mp4
python3 detect_video.py best_fp16.rknn 0 --save cam.mp4           # USB-камера
python3 detect_video.py best_fp16.rknn video.mp4 --imgsz 640 --conf 0.25

# Изменение число ядер NPU в самом скрипте - NUM_WORKERS



