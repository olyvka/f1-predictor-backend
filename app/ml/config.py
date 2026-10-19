"""Шляхи й константи ML-пайплайна. Значення TRAIN_END/VAL_END/TEST_START/
MIN_YEAR/MAX_REL ідентичні CELL 1 f1_pipeline_v2.py - міняти їх означає
міняти сам спліт і перенавчати моделі."""
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(_ROOT, "data")

# Кешовані артефакти (engineered dataset + треновані моделі) - важкі
# (~20с) і не залежать від конкретного запиту, тому переживають рестарт
# процесу. У Docker/на Render без persistent disk каталог ефемерний -
# просто перебудується на наступному холодному старті, це ок.
MODELS_DIR = os.getenv("F1_MODELS_DIR", os.path.join(_ROOT, "models"))

# Результат аналізу однієї гонки - JSON, окремий файл на (year, round).
# Рахується один раз "на тригер" і далі віддається з кешу (див. app/analysis.py).
ANALYSIS_CACHE_DIR = os.getenv("F1_ANALYSIS_CACHE_DIR", os.path.join(_ROOT, "analysis_cache"))

MIN_YEAR = 2000
TRAIN_END = 2021
VAL_END = 2023
TEST_START = 2024
MAX_REL = 20  # верхня межа relevance для LambdaRank (21 - position, clip 0..20)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_CACHE_DIR, exist_ok=True)
