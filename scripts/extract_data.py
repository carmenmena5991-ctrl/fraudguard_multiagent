"""Extrae los zip de Kaggle (IEEE-CIS Fraud Detection) en data/raw.

Uso:
    python scripts/extract_data.py /ruta/train_transaction.csv.zip /ruta/train_identity.csv.zip
"""
import sys
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def main(paths: list[str]) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for p in paths:
        with zipfile.ZipFile(p) as z:
            for name in z.namelist():
                # Evita path traversal: solo se aceptan nombres planos .csv
                if "/" in name or "\\" in name or not name.endswith(".csv"):
                    print(f"omitido: {name}")
                    continue
                z.extract(name, RAW)
                print(f"extraído: {RAW / name}")


if __name__ == "__main__":
    main(sys.argv[1:])
