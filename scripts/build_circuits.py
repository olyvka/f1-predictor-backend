"""
Генерація SVG-контурів трас для фронтенду.

Джерело - публічний API MultiViewer (api.multiviewer.app), той самий, з
якого FastF1 бере Session.get_circuit_info(): контур траси (X/Y у тих
самих одиницях, що й позиційні дані F1, 1/10 м), кут повороту схеми (щоб
траса стояла так, як її показують у трансляції) і номери поворотів.
Телеметрія FastF1 тут не потрібна - і з жовтня 2026 вона недоступна:
livetiming.formula1.com/static відповідає 401 на запити без авторизації.

MultiViewer зберігає кілька версій кожної траси за роками (напр. Барселона
з шиканою до 2022 і без неї з 2023). Для етапу беремо найновішу версію, не
пізнішу за рік етапу - а не ту, яку API віддає на /circuits/<key>/<year>
сам: для Сінгапуру 2022 він повертає вже перебудовану трасу 2023 року.

Результат - по одному JSON на сезон: <out>/<year>.json
    {"year": 2024, "rounds": {"8": {"circuit": "Monte Carlo", "layout_year": 2022,
      "viewBox": "0 0 1000 640", "path": "M...Z",
      "start": {"x":..,"y":..,"angle":..},
      "corners": [{"n":"1","x":..,"y":..,"lx":..,"ly":..}],
      "pit_loss": {"normal": 19.5, "sc": 11.2, "vsc": 13.9}}}}
Етапи, для яких траси немає в MultiViewer, у файл не потрапляють -
скрипт перелічує їх наприкінці.

Запуск з кореня проєкту (venv має бути активований):
    python scripts/build_circuits.py --out ../f1-predictor-frontend/public/circuits 2018 2026
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import requests

# Дозволяє запускати скрипт напряму (python scripts/build_circuits.py),
# не встановлюючи проєкт як пакет.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import fastf1_client  # noqa: E402 (календар сезону + timeout-патч на requests)

API = "https://api.multiviewer.app/api/v1"
# Представляємось, щоб запити з цього скрипта було видно в логах API.
HEADERS = {"User-Agent": "f1-predictor/1.0 (build_circuits.py)"}

VIEW_WIDTH = 1000  # ширина viewBox; висота - за пропорціями траси
PADDING = 40  # запас по краях, щоб не обрізались підписи поворотів
TRACK_POINTS = 300  # точок у контурі після рівномірного перевибору
LABEL_OFFSET = 550  # винос підпису повороту від траси, в одиницях траси (1/10 м)

# Location з календаря FastF1 -> circuitKey MultiViewer. Ключі сталі між
# роками; FastF1 сам визначає їх із SessionInfo живого таймінгу F1, який
# тепер закритий, тож тримаємо відповідність тут.
LOCATION_TO_CIRCUIT = {
    "Austin": 9,
    "Baku": 144,
    "Barcelona": 15,
    "Budapest": 4,
    "Hockenheim": 34,
    "Imola": 6,
    "Istanbul": 59,
    "Jeddah": 149,
    "Las Vegas": 152,
    "Le Castellet": 28,
    "Lusail": 150,
    "Marina Bay": 61,
    "Singapore": 61,
    "Melbourne": 10,
    "Mexico City": 65,
    "Miami": 151,
    "Miami Gardens": 151,
    "Monaco": 22,
    "Monte Carlo": 22,
    "Montréal": 23,
    "Monza": 39,
    "Mugello": 146,
    "Nürburgring": 72,
    "Portimão": 147,
    "Sakhir": 63,
    "Shanghai": 49,
    "Silverstone": 2,
    "Sochi": 79,
    "Spa-Francorchamps": 7,
    "Spielberg": 19,
    "Suzuka": 46,
    "São Paulo": 14,
    "Yas Island": 70,
    "Yas Marina": 70,
    "Zandvoort": 55,
}
# Sakhir Grand Prix 2020 їхав зовнішнім кільцем - той самий Location
# "Sakhir", але окрема траса.
EVENT_TO_CIRCUIT = {"Sakhir Grand Prix": 148}


def get_json(path: str):
    res = requests.get(f"{API}/{path}", headers=HEADERS, timeout=(10, 30))
    res.raise_for_status()
    return res.json()


def rotate(xy: np.ndarray, angle: float) -> np.ndarray:
    """Поворот точок на angle (радіани) - як у прикладі FastF1 до get_circuit_info()."""
    rot = np.array([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]])
    return np.matmul(xy, rot)


def resample(xy: np.ndarray, n: int) -> np.ndarray:
    """Рівномірні за відстанню n точок замість нерівномірних сирих семплів."""
    seg = np.sqrt((np.diff(xy, axis=0) ** 2).sum(axis=1))
    dist = np.concatenate([[0], np.cumsum(seg)])
    targets = np.linspace(0, dist[-1], n, endpoint=False)
    return np.column_stack([np.interp(targets, dist, xy[:, 0]), np.interp(targets, dist, xy[:, 1])])


def build_outline(circuit: dict) -> dict:
    track = np.column_stack([circuit["x"], circuit["y"]]).astype(float)
    track = track[~np.isnan(track).any(axis=1)]
    angle = (circuit.get("rotation") or 0) / 180 * math.pi

    track = rotate(resample(track, TRACK_POINTS), angle)
    track[:, 1] *= -1  # SVG: вісь Y вниз

    corners = []
    for c in circuit.get("corners") or []:
        at = np.array([c["trackPosition"]["x"], c["trackPosition"]["y"]], dtype=float)
        off = rotate(np.array([LABEL_OFFSET, 0.0]), c["angle"] / 180 * math.pi)
        p, lbl = rotate(at, angle), rotate(at + off, angle)
        p[1] *= -1
        lbl[1] *= -1
        corners.append({"n": f"{c['number']}{c.get('letter') or ''}", "p": p, "l": lbl})

    # Масштаб у viewBox з урахуванням підписів поворотів.
    allpts = np.vstack([track] + [c["l"] for c in corners])
    lo, hi = allpts.min(axis=0), allpts.max(axis=0)
    scale = (VIEW_WIDTH - 2 * PADDING) / (hi[0] - lo[0])
    height = round((hi[1] - lo[1]) * scale + 2 * PADDING)

    def to_view(pt):
        return [round(float((pt[0] - lo[0]) * scale + PADDING), 1), round(float((pt[1] - lo[1]) * scale + PADDING), 1)]

    view = [to_view(pt) for pt in track]
    path = "M" + " L".join(f"{x} {y}" for x, y in view) + " Z"
    # Контур MultiViewer - коло від лінії старт/фініш, тож перша точка і є нею.
    dx, dy = view[3][0] - view[0][0], view[3][1] - view[0][1]

    return {
        "viewBox": f"0 0 {VIEW_WIDTH} {height}",
        "path": path,
        "start": {"x": view[0][0], "y": view[0][1], "angle": round(math.degrees(math.atan2(dy, dx)), 1)},
        "corners": [
            {"n": c["n"], **dict(zip(("x", "y"), to_view(c["p"]))), **dict(zip(("lx", "ly"), to_view(c["l"])))}
            for c in corners
        ],
    }


def pit_loss(circuit: dict) -> dict | None:
    """Середня втрата часу на піт-стоп, с: звичайна, під SC і під VSC
    (MultiViewer віддає рядки, напр. {"normal": "25.43", "sc": "16.11", ...})."""
    raw = circuit.get("pitLoss") or {}
    out = {}
    for k in ("normal", "sc", "vsc"):
        try:
            out[k] = round(float(raw[k]), 2)
        except (KeyError, TypeError, ValueError):
            out[k] = None
    return out if any(v is not None for v in out.values()) else None


def layout_year(versions: list[int], year: int) -> int:
    """Найновіша версія траси, не пізніша за рік етапу; для років до першої
    версії - найраніша наявна."""
    earlier = [v for v in versions if v <= year]
    return max(earlier) if earlier else min(versions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("years", nargs="+", type=int, help="рік або діапазон: 2024 / 2018 2026")
    parser.add_argument("--out", required=True, type=Path, help="папка для <year>.json (public/circuits фронтенду)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    catalog = get_json("circuits")  # {"22": {"name": "Monte Carlo", "years": [2022, 2021, 2019]}, ...}
    outlines = {}  # (circuitKey, layout_year) -> outline, щоб не качати ту саму версію двічі
    missing = []

    for year in range(args.years[0], args.years[-1] + 1):
        rounds = {}
        for event in fastf1_client.get_season_schedule(year):
            rnd, name, location = event["RoundNumber"], event["EventName"], event["Location"]
            key = EVENT_TO_CIRCUIT.get(name) or LOCATION_TO_CIRCUIT.get(location)
            meta = catalog.get(str(key)) if key else None
            if not meta:
                missing.append(f"{year} R{rnd} {name} ({location})")
                continue
            ly = layout_year(meta["years"], year)
            if (key, ly) not in outlines:
                info = get_json(f"circuits/{key}/{ly}")
                outlines[(key, ly)] = {**build_outline(info), "pit_loss": pit_loss(info)}
            rounds[str(rnd)] = {"circuit": meta["name"], "layout_year": ly, **outlines[(key, ly)]}

        (args.out / f"{year}.json").write_text(
            json.dumps({"year": year, "rounds": rounds}, ensure_ascii=False, separators=(",", ":"))
        )
        print(f"{year}: {len(rounds)} етапів")

    print(f"\nВерсій трас завантажено: {len(outlines)}")
    if missing:
        print("Немає траси в MultiViewer:")
        for m in missing:
            print("  -", m)


if __name__ == "__main__":
    main()
