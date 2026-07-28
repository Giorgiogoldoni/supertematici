#!/usr/bin/env python3
"""
build_training_data.py
Ricostruisce, giorno per giorno, feature tecniche + label (rendimento a
HORIZON_DAYS giorni) per ogni ETF a partire dallo storico salvato in
data/charts/TICKER.json (OHLCV + kama_fast/kama_slow/sar/sar_trend/ao/rsi14).

Le feature ricalcano esattamente la logica di fetch_supertematici.py
(compute_indicators) ma vettorizzata su tutta la serie storica, non solo
sull'ultima barra, cosi' da poter costruire un dataset di training.

Filtri applicati:
  - ticker scartato se ha meno di MIN_BARS barre di storico
  - barre scartate se il rapporto giorno-su-giorno del prezzo supera SANITY
    (probabile rebase/split ETP non allineato da Yahoo)
  - label scartata se la finestra di valutazione a HORIZON_DAYS attraversa
    una di queste discontinuita'

Output: ml/training_data.csv con una riga per (ticker, data) e colonne
feature + label 'fwd_return_10d'.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
CHARTS_DIR = ROOT / "data" / "charts"
OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON_DAYS = 10          # orizzonte del rendimento target
MIN_BARS = 250             # ~1 anno di storico minimo per includere il ticker
JUMP_RATIO_HI = 2.5        # sopra questo rapporto giorno/giorno -> discontinuita' sospetta
JUMP_RATIO_LO = 0.4        # sotto questo rapporto -> discontinuita' sospetta

# ── stessi parametri/soglie di fetch_supertematici.py ──────────────────────
ER_N = 10
VOL_AVG_N = 20
SAR_FLIP_WINDOW = 3
SBB_VOL_MIN = 1.2
SBB_ER_MIN = 0.20
SANITY_PERFOGGI = 4.0


def efficiency_ratio(close: pd.Series, n: int) -> pd.Series:
    change = (close - close.shift(n)).abs()
    volatility = close.diff().abs().rolling(n).sum()
    er = change / volatility.replace(0, np.nan)
    return er.fillna(0)


def rsi(close: pd.Series, n: int) -> pd.Series:
    """Stessa formula di fetch_supertematici.py — usata per ricalcolare RSI5
    quando i chart json storici non lo contengono ancora (retrocompatibilita')."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def vol_ratio_series(vol: pd.Series) -> pd.Series:
    vol_avg = vol.rolling(VOL_AVG_N).mean()
    return (vol / vol_avg.replace(0, np.nan)).fillna(0)


def baff_series(price_above_kf: pd.Series) -> pd.Series:
    """Barre consecutive nello stesso stato (sopra/sotto KAMA), per ogni indice."""
    groups = (price_above_kf != price_above_kf.shift()).cumsum()
    return price_above_kf.groupby(groups).cumcount() + 1


def bars_since_flip_series(sar_flip: pd.Series) -> pd.Series:
    """Barre trascorse dall'ultimo flip SAR, per ogni indice (grande se nessun flip prima)."""
    idx = pd.Series(np.where(sar_flip.values, np.arange(len(sar_flip)), np.nan), index=sar_flip.index)
    last_true = idx.ffill()
    out = pd.Series(np.arange(len(sar_flip)), index=sar_flip.index) - last_true
    return out.fillna(len(sar_flip)).astype(int)


def ao_improving_series(ao: pd.Series) -> pd.Series:
    """True se le 3 barre precedenti (i-3,i-2,i-1) sono in aumento — stessa
    definizione di compute_indicators (non include la barra corrente)."""
    a1 = ao.shift(1)
    a2 = ao.shift(2)
    a3 = ao.shift(3)
    return (a1 > a2) & (a2 > a3)


def zona_series(price: pd.Series, kf: pd.Series, ks: pd.Series) -> pd.Series:
    out = pd.Series("NEUTRA", index=price.index, dtype=object)
    out[(price > kf) & (kf > ks)] = "LONG_CONF"
    out[(price > kf) & (price <= ks) & ~((price > kf) & (kf > ks))] = "LONG_EARLY"
    stop_mask = price < ks * 0.98
    out[stop_mask] = "STOP"
    uscita_mask = (price < ks) & ~stop_mask
    out[uscita_mask & ~(price > kf)] = "USCITA"
    return out


def build_ticker_frame(ticker: str, chart: dict) -> pd.DataFrame | None:
    n = len(chart.get("date", []))
    if n < MIN_BARS:
        return None

    df = pd.DataFrame({
        "date": pd.to_datetime(chart["date"]),
        "open": chart["open"], "high": chart["high"], "low": chart["low"],
        "close": chart["close"], "volume": chart["volume"],
        "kama_fast": chart["kama_fast"], "kama_slow": chart["kama_slow"],
        "sar": chart["sar"], "sar_trend": chart["sar_trend"],
        "ao": chart["ao"], "rsi14": chart["rsi14"],
    }).set_index("date")

    close = df["close"]
    df["rsi5"] = pd.Series(chart["rsi5"], index=df.index) if "rsi5" in chart else rsi(close, 5)

    # ── sanita' dati: rapporto giorno/giorno anomalo -> possibile rebase/split ─
    ratio = close / close.shift(1)
    anomaly = (ratio > JUMP_RATIO_HI) | (ratio < JUMP_RATIO_LO)
    df["_anomaly"] = anomaly.fillna(False)

    er = efficiency_ratio(close, ER_N)
    volr = vol_ratio_series(df["volume"])
    price_above_kf = close > df["kama_fast"]
    baff = baff_series(price_above_kf)
    sar_trend = df["sar_trend"]
    sar_flip = sar_trend != sar_trend.shift(1)
    sar_flip.iloc[0] = False
    bars_since_flip = bars_since_flip_series(sar_flip)
    sar_bullish = close > df["sar"]
    ao_improving = ao_improving_series(df["ao"])
    perf_oggi = close.pct_change() * 100

    # ── cross RSI5/RSI14 (stessa definizione del segnale rsi_cross in produzione) ──
    r5, r14 = df["rsi5"], df["rsi14"]
    rsi_cross_bull = (r5.shift(1) <= r14.shift(1)) & (r5 > r14)
    rsi_cross_bear = (r5.shift(1) >= r14.shift(1)) & (r5 < r14)

    zona = zona_series(close, df["kama_fast"], df["kama_slow"])
    ks = df["kama_slow"]
    gap_pct = ((df["kama_fast"] - ks) / ks.replace(0, np.nan) * 100).fillna(0)

    buy3 = zona.eq("LONG_CONF") & (volr >= 1.3) & (er >= 0.20) & sar_bullish
    buy2 = zona.eq("LONG_EARLY") & (volr >= 1.3) & (er >= 0.20) & sar_bullish
    best_buy = buy3 | buy2

    super_best_buy = (sar_bullish & (bars_since_flip <= SAR_FLIP_WINDOW) & (volr >= SBB_VOL_MIN)
                       & (er >= SBB_ER_MIN) & (perf_oggi.abs() <= SANITY_PERFOGGI))
    super_best_buy_2 = super_best_buy & zona.isin(["LONG_CONF", "LONG_EARLY"])

    out = pd.DataFrame({
        "ticker": ticker,
        "prezzo": close,
        "er": er, "volume_ratio": volr, "baff": baff,
        "kama_gap_pct": gap_pct, "ao": df["ao"], "ao_improving": ao_improving.astype(float),
        "rsi14": df["rsi14"], "rsi5": df["rsi5"],
        "rsi_cross_bull": rsi_cross_bull.astype(float), "rsi_cross_bear": rsi_cross_bear.astype(float),
        "pre_signal": rsi_cross_bull.astype(float),
        "sar_bullish": sar_bullish.astype(float),
        "bars_since_flip": bars_since_flip, "zona": zona,
        "buy3": buy3.astype(float), "buy2": buy2.astype(float), "best_buy": best_buy.astype(float),
        "super_best_buy": super_best_buy.astype(float), "super_best_buy_2": super_best_buy_2.astype(float),
        "perf_oggi": perf_oggi,
    })

    # ── label: rendimento a HORIZON_DAYS, scartata se attraversa un'anomalia ──
    fwd_close = close.shift(-HORIZON_DAYS)
    fwd_return = (fwd_close / close - 1) * 100
    anomaly_ahead = df["_anomaly"].shift(-1).rolling(HORIZON_DAYS, min_periods=1).max().shift(-(HORIZON_DAYS - 1)).fillna(0)
    # rolling forward sum di anomalie nei prossimi HORIZON_DAYS giorni
    anomaly_window = df["_anomaly"][::-1].rolling(HORIZON_DAYS, min_periods=1).max()[::-1]
    out["fwd_return_10d"] = fwd_return.where(~anomaly_window.astype(bool) & ~df["_anomaly"])

    out = out.iloc[MIN_BARS - 60:]  # scarta il warm-up iniziale degli indicatori (KAMA/ER/ADX)
    out = out.dropna(subset=["kama_gap_pct"])
    return out.reset_index()


def main():
    index_file = CHARTS_DIR / "index.json"
    if not index_file.exists():
        print("data/charts/index.json non trovato — esegui prima fetch_supertematici.py", file=sys.stderr)
        sys.exit(1)
    chart_index = json.loads(index_file.read_text(encoding="utf-8"))

    frames = []
    skipped = []
    for ticker, chart_file in chart_index.items():
        path = CHARTS_DIR / chart_file
        if not path.exists():
            skipped.append(ticker)
            continue
        chart = json.loads(path.read_text(encoding="utf-8"))
        frame = build_ticker_frame(ticker, chart)
        if frame is None:
            skipped.append(ticker)
            continue
        frames.append(frame)

    if not frames:
        print("Nessun ticker con storico sufficiente — impossibile costruire il dataset", file=sys.stderr)
        sys.exit(1)

    dataset = pd.concat(frames, ignore_index=True)
    dataset_labeled = dataset.dropna(subset=["fwd_return_10d"])

    print(f"Ticker inclusi: {len(frames)} / {len(chart_index)} (scartati per storico insufficiente/dati mancanti: {len(skipped)})")
    print(f"Righe totali: {len(dataset)}, righe con label valida: {len(dataset_labeled)}")

    out_path = OUT_DIR / "training_data.csv"
    dataset.to_csv(out_path, index=False)
    print(f"Salvato: {out_path}")


if __name__ == "__main__":
    main()
