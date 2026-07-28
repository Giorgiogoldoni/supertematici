#!/usr/bin/env python3
"""
compute_signal_kpi.py
Calcola il KPI storico dei segnali (Best Buy, Super Best Buy, Super Best Buy 2)
a partire dal dataset ricostruito da build_training_data.py.

Metodologia: orizzonte FISSO di HORIZON_DAYS giorni di mercato dal giorno in cui
il segnale e' attivo (non "fino a chiusura del flag" — il flag e' troppo
transitorio, resta vero per 1 sola barra nell'85% dei casi, quindi "fino a
chiusura" produrrebbe un rendimento ~0% per costruzione, non un KPI utile).

Obiettivo di successo configurabile (default: rendimento >= +3% entro
HORIZON_DAYS giorni di mercato).

Output: signal_kpi.json nella root del repo, letto da index.html per mostrare
il win-rate storico sotto il titolo di ogni sezione card.
"""

import json
import sys
import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
DATA_FILE = Path(__file__).parent / "training_data.csv"
OUT_FILE = ROOT / "signal_kpi.json"

HORIZON_DAYS = 5     # orizzonte fisso di valutazione (giorni di mercato)
SUCCESS_TARGET = 3.0  # rendimento %, soglia di "successo"

SIGNALS = {
    "pre_signal": "Pre-Signal",
    "best_buy": "Best Buy",
    "super_best_buy": "Super Best Buy",
    "super_best_buy_2": "Super Best Buy 2",
}


def main():
    if not DATA_FILE.exists():
        print("training_data.csv non trovato — esegui prima build_training_data.py", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(DATA_FILE, parse_dates=["date"]).sort_values(["ticker", "date"])
    df["fwd_ret"] = df.groupby("ticker")["prezzo"].transform(
        lambda s: (s.shift(-HORIZON_DAYS) / s - 1) * 100
    )

    base = df.dropna(subset=["fwd_ret"])
    baseline = {
        "n": int(len(base)),
        "win_rate_pct": round(float((base["fwd_ret"] >= SUCCESS_TARGET).mean() * 100), 1),
        "mean_return_pct": round(float(base["fwd_ret"].mean()), 3),
    }

    kpi = {
        "generato": datetime.datetime.now().isoformat(),
        "horizon_days": HORIZON_DAYS,
        "success_target_pct": SUCCESS_TARGET,
        "baseline": baseline,
        "segnali": {},
    }

    for col, label in SIGNALS.items():
        sub = df[df[col] == 1].dropna(subset=["fwd_ret"])
        n = len(sub)
        if n == 0:
            kpi["segnali"][col] = {"label": label, "n": 0}
            continue
        kpi["segnali"][col] = {
            "label": label,
            "n": int(n),
            "win_rate_pct": round(float((sub["fwd_ret"] >= SUCCESS_TARGET).mean() * 100), 1),
            "mean_return_pct": round(float(sub["fwd_ret"].mean()), 3),
            "median_return_pct": round(float(sub["fwd_ret"].median()), 3),
            "vs_baseline_win_rate_pp": round(
                float((sub["fwd_ret"] >= SUCCESS_TARGET).mean() * 100) - baseline["win_rate_pct"], 1),
            "vs_baseline_mean_return_pp": round(float(sub["fwd_ret"].mean()) - baseline["mean_return_pct"], 3),
        }
        print(f"{label}: n={n}  win-rate={kpi['segnali'][col]['win_rate_pct']}%  "
              f"media={kpi['segnali'][col]['mean_return_pct']:+.2f}%  "
              f"(baseline: win-rate={baseline['win_rate_pct']}%, media={baseline['mean_return_pct']:+.2f}%)")

    OUT_FILE.write_text(json.dumps(kpi, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Salvato: {OUT_FILE}")


if __name__ == "__main__":
    main()
