# SUPERTEMATICI

Dashboard di analisi tecnica per un universo di 150 ETF tematici (18 settori:
AI_TECH, ROBOTICA, CYBERSECURITY, DIGITALE_ECOMMERCE_FINTECH, DIFESA,
ENERGIA_RINNOVABILE, NUCLEARE_URANIO, ENERGIA_FOSSILE, SALUTE_BIOTECH,
AGRIFOOD, CRYPTO_BLOCKCHAIN, MOBILITA_EV, INFRASTRUTTURE, MATERIE_PRIME,
MEDIA_GAMING, ACQUA_AMBIENTE, SEMICONDUTTORI, SETTORE_AMPIO_VALUTARE).

Architettura stateless (ricalcata su [azionario](https://github.com/Giorgiogoldoni/azionario)):
backend Python + GitHub Action che genera JSON statici, HTML solo renderer.
Nessun motore di portfolio/posizioni: solo segnali.

## File

- `fetch_supertematici.py` — scarica storico Yahoo Finance (max disponibile)
  e calcola indicatori (ER, KAMA fast/slow, baff, SAR, AO, RVI, RSI14/RSI5,
  ADX, momentum 1M/3M/6M)
- `tickers_supertematici.json` — universo dei 150 ETF (nome, ticker, settore, paese)
- `supertematici.json` — riepilogo generato per la tabella
- `data/charts/TICKER.json` — serie storica per il grafico
- `regole/TICKER_Regole.html` — scheda regole operative per ETF
- `index.html` — dashboard (filtro Cerca/Settore/Segnale, no tab fissi)
- `.github/workflows/update.yml` — aggiornamento automatico lun-ven 08:30 CET/CEST

## Segnali

- **Best Buy** (conf./early) — segnale `BUY3`/`BUY2`: zona KAMA allineata +
  AO>0 + volume + baff + ER + gap KAMA + SAR rialzista
- **Super Best Buy** — flip SAR rialzista fresco (≤2 barre) + AO>0 e in
  miglioramento (3 barre) + volume ≥1.5x + |perf oggi| ≤4%
- **Super Best Buy 2** (sperimentale) — stessa base senza richiedere AO>0,
  solo AO in miglioramento
