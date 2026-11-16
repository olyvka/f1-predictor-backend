"""
Оновити data/*.csv з Kaggle вручну (те саме, що робить сервер у фоні,
див. app/data_update.py). Зручно для локальної розробки або cron:

    python scripts/update_data.py
"""
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import data_update  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Модель тут не перенавчаємо: процес одразу завершиться, а сервер і так
    # перебудує її при наступному запиті прогнозу (змінився fingerprint CSV).
    result = data_update.update(warm_models=False)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    sys.exit(0 if result["status"] in ("updated", "up_to_date") else 1)
