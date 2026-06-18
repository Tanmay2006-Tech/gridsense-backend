# Gridlock Hackathon 2.0 — BEST PIPELINE v3
## AI Ops Co-pilot for Traffic Police · Bangalore

---

## Quick Start

### Kaggle Notebook
```python
import kagglehub
path = kagglehub.dataset_download("tanmaytripathi7525/gridlock-round2-theme2")
from pipeline import run
lgbm, cat, xgbc = run(kaggle=True)
```

### Local
```bash
pip install -r requirements.txt
python pipeline.py
python api.py
```

---

## Results

| Metric | Score |
|--------|-------|
| **CV F1 (5-fold macro)** | **0.640** |
| **Hold-out F1 (macro)** | **0.630** |
| **Accuracy** | **0.638** |
| High class recall | 77% |
| Features | 63 |
| Training rows | 3,192 |

### Improvement over baseline
| Version | CV F1 | Hold-out F1 |
|---------|-------|-------------|
| v1 (basic) | 0.560 | 0.593 |
| **v3 (final)** | **0.640** | **0.630** |
| Gain | +0.080 | +0.037 |

---

## Key Feature Groups (63 total)

1. **Time** (13): hour, dow, month + sin/cos encoding + peak flags
2. **Domain** (9): severity score, road closure, is_planned, priority, is_major_corridor
3. **NEW untapped** (11): is_authenticated, has_kgid, has_cargo, has_end_addr, has_junction, has_endpoint, desc_len_log, has_veh_no, gba_enc, gba_high_rate, is_client2
4. **Interactions** (8): severity × road_cl, severity × peak, corridor × severity, etc.
5. **Statistical maps** (9): log-scaled cause/corridor/zone median durations + load counts
6. **Class rates** (6): P(class|cause), P(High|zone/PS/corridor)
7. **Categorical encoded** (8): event_cause, corridor, zone, police_station, etc.

---

## Ensemble Weights
- LightGBM: 30% (Optuna-tuned: n=317, lr=0.0178, depth=6, leaves=83)
- CatBoost:  40% (depth=8, iterations=700, lr=0.04)
- XGBoost:   30% (n=800, depth=6, lr=0.04)

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Status |
| GET | `/summary` | Dashboard header cards |
| GET | `/analytics` | All chart data |
| POST | `/predict` | Predict + recommend |
| GET | `/similar/<id>` | Similar past events |
| GET | `/events` | Filtered event list |
| GET | `/hotspots` | Map data |
| POST | `/feedback` | Officer feedback |
| GET | `/feedback` | Feedback + accuracy |
| GET | `/meta/corridors` | Corridors list |
| GET | `/meta/zones` | Zones list |
| GET | `/meta/causes` | Causes list |
| GET | `/meta/corporations` | GBA corporations |
