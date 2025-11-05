Plant-Health Infer Service

FastAPI-сервис для анализа фотографий зелёных насаждений: сегментация растений и дефектов, расчёт эвристик (наклон, доля сухих ветвей), генерация оверлея и машиночитаемого отчёта.

Структура проекта
docker-compose.cpu.yml
docker-compose.gpu.yml
Dockerfile.cpu
Dockerfile.gpu
requirements.txt
app/
  main.py               # FastAPI приложение, монтирует статику /artifacts
  config.py             # переменные окружения и дефолты
  pipeline.py           # основной ML-пайплайн
  schemas.py            # Pydantic-схемы ответа
  models/
    loaders.py          # ленивые загрузчики весов
    species_infer.py    # классификация видов
  routers/
    infer.py            # POST /infer
    debug.py            # /debug/* служебные маршруты
  utils/
    enc.py, visualize.py
rules/
tools/
.env.example           # шаблон конфигурации

Быстрый старт

Создать .env (можно на базе .env.example):

cp .env.example .env


Ключевые переменные:

DEVICE — cuda:0 или cpu

THRESH_PLANT — порог детекции растений, по умолчанию 0.25

THRESH_DEFECT — порог детекции дефектов, по умолчанию 0.01

PLANT_SEG_WEIGHTS — путь к весам сегментации растений (внутри контейнера), дефолт /srv/app/weights/plant/plant_seg.pt

DEFECT_SEG_WEIGHTS — путь к весам сегментации дефектов, дефолт /srv/app/weights/defect/defect_seg.pt

SPECIES_TS, SPECIES_CLASSES — файлы классификатора видов

SEVERITY_RULES_CSV — CSV с правилами тяжести дефектов

OUT_DIR — каталог артефактов, дефолт /data/out

Подготовить директории на хосте:

./weights/plant/plant_seg.pt
./weights/defect/defect_seg.pt
./models/species/model_ts.pt
./models/species/species_classes
./rules/severity_rules.csv  (опционально)
./data/                     (сюда будут складываться артефакты)


Запуск CPU:

docker compose -f docker-compose.cpu.yml up -d --build


Запуск GPU (нужны драйверы/NVIDIA runtime):

docker compose -f docker-compose.gpu.yml up -d --build


Сервис поднимется на http://localhost:8000.

API

GET /health → {"status":"ok"}

POST /infer — multipart (file), ответ: JSON со списками plants[], defects[], путями артефактов.

Статика артефактов: смонтирована на /artifacts, физически это ${OUT_DIR} (по умолчанию /data/out).

Пример запроса
curl -sS -X POST "http://localhost:8000/infer" \
  -F "file=@/path/to/image.jpg;type=image/jpeg" \
  -o out.json


Фрагмент ответа:

{
  "request_id": "b6811401-8218-4b08-be4f-4f8a414c0fb5",
  "status": "OK",
  "overlay_path": "/data/out/b6811401-8218-4b08-be4f-4f8a414c0fb5/overlay.png",
  "report_json_path": "/data/out/b6811401-8218-4b08-be4f-4f8a414c0fb5/report.json",
  "plants": [...],
  "defects": [...],
  "extras": {...}
}


Скачать артефакты через HTTP:

curl -sS -o overlay.png \
  "http://localhost:8000/artifacts/b6811401-8218-4b08-be4f-4f8a414c0fb5/overlay.png"

curl -sS -o report.json \
  "http://localhost:8000/artifacts/b6811401-8218-4b08-be4f-4f8a414c0fb5/report.json"

Docker Compose (схема монтирования)

CPU (пример):

services:
  treehealth:
    build:
      context: .
      dockerfile: Dockerfile.cpu
    env_file: .env
    volumes:
      - ./data:/data
      - ./weights:/srv/app/weights
      - ./models:/srv/app/models
      - ./rules:/srv/app/rules
    ports:
      - "8000:8000"
    restart: unless-stopped


GPU аналогично, но с runtime: nvidia и NVIDIA_VISIBLE_DEVICES=all.

Типичные проблемы

Permission denied: '/data/out/...'
В контейнере выдать права:

docker compose exec treehealth sh -lc \
  'chown -R appuser:appuser /data && find /data -type d -exec chmod 755 {} \; && find /data -type f -exec chmod 644 {} \;'


Плохая скорость первого запроса
На старте прогревается MiDaS; это нормально. Дальше будет быстрее.

Лицензия
