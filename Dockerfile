FROM python:3.11-slim

WORKDIR /code

# libgomp1 - системна залежність LightGBM (OpenMP) на Debian; без неї
# `import lightgbm` падає з OSError на завантаженні lib_lightgbm.so.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
# data/ потрібна ML-пайплайну (app/ml/) - тренує/будує feature-датасет
# з цих CSV при першому запиті аналізу прогнозу.
COPY data ./data

# Render/Railway підставляють свій $PORT - тому беремо його з середовища
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
