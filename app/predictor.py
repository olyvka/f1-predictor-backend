"""
Сюди перенесеш свою логіку передбачення.
Поки що — заглушка, яка повертає результати гонки як "передбачення",
щоб фронт вже міг малювати відповідний UI.

Коли будеш підключати реальну модель:
1. Замінити тіло predict_race() на виклик своєї моделі.
2. Вхідні дані брати через fastf1_client (лапи, погода, стартова решітка тощо)
   замість завантажених CSV/датасетів.
"""
from app.fastf1_client import get_race_results


def predict_race(year: int, round_number: int) -> list[dict]:
    # TODO: замінити на реальну модель передбачення
    results = get_race_results(year, round_number)
    predicted = [
        {
            "predicted_position": r["Position"],
            "driver": r["FullName"],
            "team": r["TeamName"],
            "confidence": None,  # сюди - впевненість моделі, коли буде готова
        }
        for r in results
    ]
    return predicted
