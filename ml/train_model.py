#!/usr/bin/env python3
"""
train_model.py
Allena un modello LightGBM (regressione) sul dataset costruito da
build_training_data.py per predire il rendimento atteso a 10 giorni di ogni
ETF, usando tutti i 150 ETF in pool (non un modello per ticker: troppo pochi
campioni per i settori piu' giovani, es. AI_TECH nato nel 2024).

Validazione: split cronologico walk-forward (mai casuale, per non introdurre
look-ahead bias) — training sul passato, test sul periodo successivo mai
visto dal modello.

Output:
  - ml/model.joblib       modello allenato
  - ml/model_meta.json    feature usate, metriche di validazione, data training
"""

import json
import sys
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

ROOT = Path(__file__).parent
DATA_FILE = ROOT / "training_data.csv"
MODEL_FILE = ROOT / "model.joblib"
META_FILE = ROOT / "model_meta.json"

FEATURES_NUM = [
    "er", "volume_ratio", "baff", "kama_gap_pct", "ao", "ao_improving",
    "rsi14", "sar_bullish", "bars_since_flip",
    "buy3", "buy2", "best_buy", "super_best_buy", "super_best_buy_2", "perf_oggi",
]
FEATURE_CAT = "zona"
TARGET = "fwd_return_10d"

TEST_FRACTION = 0.2  # ultima porzione cronologica usata come test (mai vista in training)
MIN_ROWS = 5000


def main():
    if not DATA_FILE.exists():
        print("training_data.csv non trovato — esegui prima build_training_data.py", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    df = df.dropna(subset=[TARGET])
    if len(df) < MIN_ROWS:
        print(f"Solo {len(df)} righe con label valida (< {MIN_ROWS}) — dataset ancora troppo piccolo, salto il training", file=sys.stderr)
        sys.exit(1)

    df[FEATURE_CAT] = df[FEATURE_CAT].astype("category")
    df = df.sort_values("date").reset_index(drop=True)

    # ── split cronologico walk-forward: train sul passato, test sul futuro ──
    cutoff_idx = int(len(df) * (1 - TEST_FRACTION))
    cutoff_date = df["date"].iloc[cutoff_idx]
    train = df[df["date"] < cutoff_date]
    test = df[df["date"] >= cutoff_date]

    X_train = train[FEATURES_NUM + [FEATURE_CAT]]
    y_train = train[TARGET]
    X_test = test[FEATURES_NUM + [FEATURE_CAT]]
    y_test = test[TARGET]

    print(f"Train: {len(train)} righe (fino al {train['date'].max().date()})")
    print(f"Test:  {len(test)} righe (dal {test['date'].min().date()} al {test['date'].max().date()})")

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_train, y_train, categorical_feature=[FEATURE_CAT])

    pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred_test)
    r2 = r2_score(y_test, pred_test)
    ic = float(np.corrcoef(pred_test, y_test)[0, 1]) if len(y_test) > 1 else None

    # baseline naive: predire sempre la media del train -> per capire se il modello aggiunge valore
    baseline_pred = np.full(len(y_test), y_train.mean())
    baseline_mae = mean_absolute_error(y_test, baseline_pred)

    print(f"MAE modello: {mae:.3f}  |  MAE baseline (media train): {baseline_mae:.3f}")
    print(f"R2 test: {r2:.4f}  |  IC (corr pred/realizzato): {ic}")

    # ── retrain finale su tutto il dataset (train+test) per l'uso in produzione ──
    X_all = df[FEATURES_NUM + [FEATURE_CAT]]
    y_all = df[TARGET]
    model_final = lgb.LGBMRegressor(**model.get_params())
    model_final.fit(X_all, y_all, categorical_feature=[FEATURE_CAT])

    joblib.dump({
        "model": model_final,
        "features_num": FEATURES_NUM,
        "feature_cat": FEATURE_CAT,
        "zona_categories": list(df[FEATURE_CAT].cat.categories),
    }, MODEL_FILE)

    meta = {
        "trained_at": datetime.datetime.now().isoformat(),
        "horizon_days": 10,
        "rows_train": len(train),
        "rows_test": len(test),
        "test_period": [str(test["date"].min().date()), str(test["date"].max().date())],
        "mae_test": round(float(mae), 4),
        "mae_baseline_test": round(float(baseline_mae), 4),
        "r2_test": round(float(r2), 4),
        "ic_test": round(ic, 4) if ic is not None else None,
        "features": FEATURES_NUM + [FEATURE_CAT],
    }
    META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Modello salvato: {MODEL_FILE}")
    print(f"Metadati salvati: {META_FILE}")

    if mae >= baseline_mae:
        print("ATTENZIONE: il modello non batte la baseline naive (media storica) sul test — "
              "score ML da trattare con cautela finche' non migliora.", file=sys.stderr)


if __name__ == "__main__":
    main()
