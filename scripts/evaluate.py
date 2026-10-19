"""
Walk-forward оцінка моделі проти baseline "фініш = кваліфікація" з
bootstrap довірчими інтервалами (app/ml/evaluate.py).

    python scripts/evaluate.py                    # усі сезони 2006+
    python scripts/evaluate.py --years 2018-2025  # діапазон сезонів
    python scripts/evaluate.py --years 2020-2026 --ablate pr_,wx_
        # + A/B: та сама модель без фіч з цими префіксами (що дають саме вони)

Тренує модель заново на кожен сезон (~10-15с на сезон), кеш моделей
бекенду не чіпає.
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from app.ml.dataset import get_dataset  # noqa: E402
from app.ml.evaluate import compare, walk_forward, without_features  # noqa: E402


def _parse_years(value: str) -> list[int]:
    start, _, end = value.partition("-")
    return list(range(int(start), int(end or start) + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=_parse_years, default=None, help="напр. 2018-2025")
    parser.add_argument("--boot", type=int, default=2000, help="кількість bootstrap-вибірок")
    parser.add_argument("--ablate", default=None,
                        help="префікси фіч через кому - додатково порівняти з моделлю без них")
    args = parser.parse_args()

    dataset = get_dataset()
    print("Walk-forward (train <= Y-2, val = Y-1, test = Y):")
    result = walk_forward(dataset, args.years)

    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print("\nПо сезонах (середнє по гонках):")
    print(result.by_year().to_string())
    print(f"\nУсього {len(result.races)} гонок. Різниця model - baseline, 95% bootstrap CI:")
    print(result.summary(n_boot=args.boot).to_string(index=False))

    if args.ablate:
        prefixes = tuple(p.strip() for p in args.ablate.split(",") if p.strip())
        print(f"\nWalk-forward без фіч {prefixes}:")
        reduced = walk_forward(without_features(dataset, prefixes), args.years)
        print(f"\nA/B: з фічами {prefixes} проти без них (ті самі гонки), 95% bootstrap CI:")
        print(compare(result, reduced, n_boot=args.boot).to_string(index=False))


if __name__ == "__main__":
    main()
