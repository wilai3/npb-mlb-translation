"""
Winsorize extreme per-player ratios, recompute translation factors with
confidence intervals, and compare v1 vs v2.
"""

import os
import sqlite3
import numpy as np
import pandas as pd

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
DB_PATH  = os.path.join(DATA_DIR, "npb_mlb.db")
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)

conn = sqlite3.connect(DB_PATH)

# ──────────────────────────────────────────────
# Load existing player ratio tables and v1 factors
# ──────────────────────────────────────────────
hit = pd.read_sql("SELECT * FROM player_ratios_hitting",       conn)
pit = pd.read_sql("SELECT * FROM player_ratios_pitching",      conn)
fac_hit_v1 = pd.read_sql("SELECT * FROM translation_factors_hitting",  conn)
fac_pit_v1 = pd.read_sql("SELECT * FROM translation_factors_pitching", conn)

# ──────────────────────────────────────────────
# STEP 1 — Winsorize extreme ratios
# ──────────────────────────────────────────────

WINSOR_PIT = ["HR9_ratio", "BB9_ratio", "ERA_ratio"]
WINSOR_HIT = ["HR_rate_ratio"]
P_LO, P_HI = 10, 90

def winsorize(series, lo_pct, hi_pct):
    lo = np.percentile(series.dropna(), lo_pct)
    hi = np.percentile(series.dropna(), hi_pct)
    return series.clip(lower=lo, upper=hi), lo, hi

hit_v2 = hit.copy()
pit_v2 = pit.copy()

print("=" * 70)
print("WINSORIZATION BOUNDS  (10th / 90th percentile)")
print("=" * 70)

print("\nHITTING")
for col in WINSOR_HIT:
    clipped, lo, hi = winsorize(hit_v2[col], P_LO, P_HI)
    n_clipped = ((hit_v2[col] < lo) | (hit_v2[col] > hi)).sum()
    hit_v2[col] = clipped
    print(f"  {col:<20}  p10={lo:.4f}  p90={hi:.4f}  rows clipped={n_clipped}")

print("\nPITCHING")
for col in WINSOR_PIT:
    clipped, lo, hi = winsorize(pit_v2[col], P_LO, P_HI)
    n_clipped = ((pit_v2[col] < lo) | (pit_v2[col] > hi)).sum()
    pit_v2[col] = clipped
    print(f"  {col:<20}  p10={lo:.4f}  p90={hi:.4f}  rows clipped={n_clipped}")

# ──────────────────────────────────────────────
# STEP 2 — Recompute translation factors
# ──────────────────────────────────────────────

HIT_RATIO_COLS = ["BA_ratio", "OBP_ratio", "SLG_ratio",
                  "HR_rate_ratio", "BB_rate_ratio", "K_rate_ratio"]
PIT_RATIO_COLS = ["ERA_ratio", "WHIP_ratio", "K9_ratio", "BB9_ratio", "HR9_ratio"]

STAT_NAMES = {
    "BA_ratio": "BA",    "OBP_ratio": "OBP",    "SLG_ratio": "SLG",
    "HR_rate_ratio": "HR_rate", "BB_rate_ratio": "BB_rate", "K_rate_ratio": "K_rate",
    "ERA_ratio": "ERA",  "WHIP_ratio": "WHIP",
    "K9_ratio": "K_per9", "BB9_ratio": "BB_per9", "HR9_ratio": "HR_per9",
}

def compute_factors(df, ratio_cols, weight_col):
    rows = []
    for col in ratio_cols:
        valid = df[["name", col, weight_col]].dropna()
        if valid.empty:
            continue
        w = valid[weight_col].values.astype(float)
        r = valid[col].values.astype(float)
        w_sum   = w.sum()
        factor  = (w * r).sum() / w_sum
        wstd    = np.sqrt((w * (r - factor) ** 2).sum() / w_sum)
        n       = len(valid)
        ci_half = 1.96 * wstd / np.sqrt(n)
        rows.append({
            "stat":         STAT_NAMES[col],
            "factor":       round(factor, 4),
            "n_players":    n,
            "weighted_std": round(wstd, 4),
            "min_ratio":    round(r.min(), 4),
            "max_ratio":    round(r.max(), 4),
            "ci_lower":     round(factor - ci_half, 4),
            "ci_upper":     round(factor + ci_half, 4),
        })
    return pd.DataFrame(rows)

fac_hit_v2 = compute_factors(hit_v2, HIT_RATIO_COLS, "harm_PA")
fac_pit_v2 = compute_factors(pit_v2, PIT_RATIO_COLS, "harm_IP")

# ──────────────────────────────────────────────
# STEP 3 — Add CIs to v1 factors as well (for comparison)
# ──────────────────────────────────────────────

def add_ci(df):
    df = df.copy()
    df["ci_lower"] = (df["factor"] - 1.96 * df["weighted_std"] / np.sqrt(df["n_players"])).round(4)
    df["ci_upper"] = (df["factor"] + 1.96 * df["weighted_std"] / np.sqrt(df["n_players"])).round(4)
    return df

fac_hit_v1 = add_ci(fac_hit_v1)
fac_pit_v1 = add_ci(fac_pit_v1)

# ──────────────────────────────────────────────
# STEP 4 — Side-by-side comparison
# ──────────────────────────────────────────────

def compare(v1, v2, label):
    merged = v1[["stat", "factor", "weighted_std", "ci_lower", "ci_upper"]].merge(
        v2[["stat", "factor", "weighted_std", "ci_lower", "ci_upper"]],
        on="stat", suffixes=("_v1", "_v2"),
    )
    merged["factor_delta"] = (merged["factor_v2"] - merged["factor_v1"]).round(4)
    merged["wstd_delta"]   = (merged["weighted_std_v2"] - merged["weighted_std_v1"]).round(4)
    print(f"\n{'=' * 80}")
    print(f"V1 vs V2 COMPARISON — {label}")
    print(f"{'=' * 80}")
    print(merged.to_string(index=False))

compare(fac_hit_v1, fac_hit_v2, "HITTING")
compare(fac_pit_v1, fac_pit_v2, "PITCHING")

# ──────────────────────────────────────────────
# Final summary of v2 factors with CIs
# ──────────────────────────────────────────────

print("\n")
print("=" * 70)
print("V2 HITTING TRANSLATION FACTORS  (with 95% CI)")
print("=" * 70)
print(fac_hit_v2[["stat", "factor", "n_players", "weighted_std",
                   "min_ratio", "max_ratio", "ci_lower", "ci_upper"]].to_string(index=False))

print()
print("=" * 70)
print("V2 PITCHING TRANSLATION FACTORS  (with 95% CI)")
print("=" * 70)
print(fac_pit_v2[["stat", "factor", "n_players", "weighted_std",
                   "min_ratio", "max_ratio", "ci_lower", "ci_upper"]].to_string(index=False))

# ──────────────────────────────────────────────
# STEP 5 — Save CSVs and load into DB
# ──────────────────────────────────────────────

fac_hit_v2.to_csv(os.path.join(DATA_DIR, "translation_factors_hitting_v2.csv"), index=False)
fac_pit_v2.to_csv(os.path.join(DATA_DIR, "translation_factors_pitching_v2.csv"), index=False)

print()
for df, table in [
    (hit_v2,       "player_ratios_hitting_v2"),
    (pit_v2,       "player_ratios_pitching_v2"),
    (fac_hit_v2,   "translation_factors_hitting_v2"),
    (fac_pit_v2,   "translation_factors_pitching_v2"),
]:
    df.to_sql(table, conn, if_exists="replace", index=False)
    print(f"  → saved {table} ({len(df)} rows) to DB")

conn.close()
print("\nDone.")
