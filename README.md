# Plant-Health Infer Service

Серверный ML-пайплайн для анализа фотографий зелёных насаждений: сегментация растений и дефектов (YOLO11-L Seg), эвристики (наклон, доля сухих ветвей), классификация вида (EfficientNet-B0), генерация оверлея и машиночитаемого отчёта через REST API (FastAPI + Uvicorn).

## Содержание

* [Структура репозитория](#структура-репозитория)
* [Требования](#требования)
* [Быстрый старт (Docker)](#быстрый-старт-docker)
* [Локальный запуск без Docker](#локальный-запуск-без-docker)
* [Конфигурация (.env)](#конфигурация-env)
* [API](#api)
* [Доступ к артефактам](#доступ-к-артефактам)
* [Типичные проблемы](#типичные-проблемы)
* [Авторы](#авторы)

## Структура репозитория

```
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
models/
  species/
    model_ts.pt
    species_classes.json
    species_ru_map.json
tools/
    analyze_reports.py # Универсальный анализатор отчётов из data/out/**/report.json
weights/
  defect/
    defect_seg.pt
  plant/
    plant_seg.pt
.env.example
.gitattributes
Dockerfile.cpu
Dockerfile.gpu
docker-compose.cpu.yml
docker-compose.gpu.yml
requirements.txt
```

> Веса и модель видов лежат под Git LFS. После клонирования выполните `git lfs pull`.

## Требования

* Docker + Docker Compose (v2)
* Для GPU-варианта: установленный NVIDIA Container Toolkit и драйвер
* Git LFS (если клонируете репозиторий с весами)

## Быстрый старт (Docker)

1. Клонирование и LFS:

```bash
git clone https://github.com/Lin-Lini/plant-health.git
cd plant-health
git lfs pull
```

2. Скопируйте шаблон окружения и при необходимости отредактируйте значения:

```bash
cp .env.example .env
```

3. Запуск CPU:

```bash
docker compose -f docker-compose.cpu.yml up -d --build
```

Запуск GPU:

```bash
docker compose -f docker-compose.gpu.yml up -d --build
```

4. Проверка доступности сервиса:

```bash
curl -s http://localhost:8000/health
```

## Локальный запуск без Docker

```bash
python -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
# переменные окружения возьмутся из .env (рекомендуется установить python-dotenv) 
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Конфигурация (.env)

Ключевые переменные:

```
# общие
PORT=8000
OUT_DIR=/data/out
DEVICE=cpu            # или cuda:0

# веса
PLANT_SEG_WEIGHTS=/srv/app/weights/plant/plant_seg.pt
DEFECT_SEG_WEIGHTS=/srv/app/weights/defect/defect_seg.pt

# классификация видов
SPECIES_TS=/srv/app/models/species/model_ts.pt
SPECIES_CLASSES=/srv/app/models/species/species_classes.json
SPECIES_RU_MAP=/srv/app/models/species/species_ru_map.json

# пороги
THRESH_PLANT=0.25
THRESH_DEFECT=0.01
```

В `docker-compose.*.yml` директории хоста монтируются так, чтобы указанные пути внутри контейнера существовали:

* `./weights  -> /srv/app/weights`
* `./models   -> /srv/app/models`
* `./data     -> /data`  (артефакты инференса)
* при необходимости `./rules -> /srv/app/rules`

## API

* `GET /health` — проверка живости
* `POST /infer` — `multipart/form-data` с полем `file` (изображение). Возвращает JSON:

  * `request_id: str`
  * `plants[]: { id, cls, conf, area, tilt_deg, dry_ratio, species, health_score, health_grade }`
  * `defects[]: { id, cls, conf, area, plant_id, area_ratio, severity }`
  * `overlay_path: str`
  * `report_json_path: str`

### Пример запроса

```bash
curl -sS -X POST "http://localhost:8000/infer" \
  -F "file=@/path/to/image.jpg;type=image/jpeg" \
  -o out.json
```

Фрагмент ответа:

```json
{
  "request_id": "b6811401-8218-4b08-be4f-4f8a414c0fb5",
  "status": "OK",
  "overlay_path": "/data/out/b6811401-8218-4b08-be4f-4f8a414c0fb5/overlay.png",
  "report_json_path": "/data/out/b6811401-8218-4b08-be4f-4f8a414c0fb5/report.json",
  "plants": [...],
  "defects": [...]
}
```

## Доступ к артефактам

Каталог `${OUT_DIR}` публикуется статикой на префиксе `/artifacts`. Скачивание:

```bash
curl -sS -o overlay.png \
  "http://localhost:8000/artifacts/<request_id>/overlay.png"

curl -sS -o report.json \
  "http://localhost:8000/artifacts/<request_id>/report.json"
```

## Типичные проблемы

* **Permission denied при записи в `/data/out`**
  Выдайте права изнутри контейнера:

  ```bash
  SVC=$(docker compose ps --services | head -n 1)
  docker compose exec "$SVC" sh -lc \
    'chown -R appuser:appuser /data && find /data -type d -exec chmod 755 {} \; && find /data -type f -exec chmod 644 {} \;'
  ```

  Windows CMD: откройте интерактивную оболочку и выполните команды вручную:

  ```bat
  docker compose exec <service> sh
  chown -R appuser:appuser /data
  find /data -type d -exec chmod 755 {} \;
  find /data -type f -exec chmod 644 {} \;
  exit
  ```

* **Первый запрос долгий**
  На старте прогреваются модели (в т. ч. depth/фичи). Дальше быстрее.

* **GPU не виден**
  Проверьте `nvidia-smi` на хосте и наличие `--gpus all`/NVIDIA runtime в системе.

## Авторы

* ML: **Полина Чудинова**
* Разработчики: **Полина Чудинова, Михаил Вознюк, Илья Цветков**
* Разметка данных: **Дарья Логвинова, Фёдор Нестеров**
