# Rknn-converter
В папке dataset находятся фото для квантизации, в models исходная и получившиеся модели.

# Сборка (один раз)

cd ВАШ ПУТЬ\rknn-converter
docker build -t rknn-converter .

# Использование — одна команда, без входа в контейнер
FP16 (imgsz определяется автоматически):

docker run --rm -v ${PWD}/models:/workspace/models -v ${PWD}/dataset:/workspace/dataset rknn-converter convert models/best.pt

FP16 + INT8 (PTQ; в dataset/ должны лежать 200+ картинок из train-части датасета):

docker run --rm -v ${PWD}/models:/workspace/models -v ${PWD}/dataset:/workspace/dataset rknn-converter convert models/best.pt --int8

Результат: models\best_fp16.rknn (и models\best_int8.rknn) на хосте.

# Опции

--imgsz N вход модели; по умолчанию читается (train_args)
--int8 (дополнительно) сборка INT8 с пост-обучающей квантизацией 
--calib ПУТЬ папка калибровки (по умолчанию /workspace/dataset) 
--ncalib N сколько картинок брать (по умолчанию 400) 
--platform целевая платформа (по умолчанию rk3588; для RK3588S — то же)




