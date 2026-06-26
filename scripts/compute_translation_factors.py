"""
Compute NPB→MLB translation factors from paired player windows.

Steps:
  1. Build 2-season NPB and MLB transition windows per player (dataset='core').
  2. Pool counting stats across the window; compute rate stats.
  3. Compute per-player ratio MLB/NPB for each rate.
  4. Weighted-average across players (weight = harmonic mean of window size).
  5. Save CSVs and load into npb_mlb.db.
"""

import sqlite3
import numpy as np
import pandas as pd

DB_PATH = "npb_mlb.db"

# ──────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)

npb_hit = pd.read_sql("SELECT * FROM npb_hitting  WHERE dataset='core'", conn)
mlb_hit = pd.read_sql("SELECT * FROM mlb_hitting  WHERE dataset='core'", conn)
npb_pit = pd.read_sql("SELECT * FROM npb_pitching WHERE dataset='core'", conn)
mlb_pit = pd.read_sql("SELECT * FROM mlb_pitching WHERE dataset='core'", conn)

# ──────────────────────────────────────────────
# STEP 1 — Window helpers
# ──────────────────────────────────────────────

def npb_window(df, player, n=2):
    rows = df[df["name"] == player].sort_values("year_ID", ascending=False)
    return rows.head(n)

def mlb_window(df, player, n=2):
    rows = df[df["name"] == player].sort_values("Year", ascending=True)
    return rows.head(n)

def safe_div(a, b):
    return float(a) / float(b) if b and b != 0 else np.nan

def harm_mean(a, b):
    return 2 * a * b / (a + b) if (a + b) > 0 else np.nan

# ──────────────────────────────────────────────
# STEP 2 & 3 — Per-player hitter ratios
# ──────────────────────────────────────────────

MIN_PA = 50
hit_rows = []

for player in sorted(npb_hit["name"].unique()):
    nw = npb_window(npb_hit, player)
    mw = mlb_window(mlb_hit, player)

    if mw.empty:
        continue

    npb_PA = nw["PA"].sum()
    mlb_PA = mw["PA"].sum()

    if npb_PA < MIN_PA or mlb_PA < MIN_PA:
        continue

    # ── NPB pooled counting stats ──
    n_AB  = nw["AB"].sum();  n_H   = nw["H"].sum()
    n_2B  = nw["2B"].sum();  n_3B  = nw["3B"].sum()
    n_HR  = nw["HR"].sum();  n_BB  = nw["BB"].sum();  n_SO = nw["SO"].sum()

    # ── MLB pooled counting stats ──
    m_AB  = mw["AB"].sum();  m_H   = mw["H"].sum()
    m_2B  = mw["2B"].sum();  m_3B  = mw["3B"].sum()
    m_HR  = mw["HR"].sum();  m_BB  = mw["BB"].sum();  m_SO = mw["SO"].sum()

    # ── NPB rates ──
    # SLG total-bases formula: H + 2B + 2·3B + 3·HR  (= TB where 1B are not double-counted)
    # Derivation: TB = 1B + 2·2B + 3·3B + 4·HR = (H-2B-3B-HR) + 2·2B + 3·3B + 4·HR
    #             = H + 2B + 2·3B + 3·HR
    n_BA  = safe_div(n_H,          n_AB)
    n_OBP = safe_div(n_H + n_BB,   n_AB + n_BB)
    n_SLG = safe_div(n_H + n_2B + 2*n_3B + 3*n_HR, n_AB)
    n_HR_rate = safe_div(n_HR, npb_PA)
    n_BB_rate = safe_div(n_BB, npb_PA)
    n_K_rate  = safe_div(n_SO, npb_PA)

    # ── MLB rates ──
    m_BA  = safe_div(m_H,          m_AB)
    m_OBP = safe_div(m_H + m_BB,   m_AB + m_BB)
    m_SLG = safe_div(m_H + m_2B + 2*m_3B + 3*m_HR, m_AB)
    m_HR_rate = safe_div(m_HR, mlb_PA)
    m_BB_rate = safe_div(m_BB, mlb_PA)
    m_K_rate  = safe_div(m_SO, mlb_PA)

    npb_years = sorted(nw["year_ID"].tolist())
    mlb_years = sorted(mw["Year"].tolist())

    hit_rows.append({
        "name":        player,
        "npb_years":   str(npb_years),
        "mlb_years":   str(mlb_years),
        "npb_PA":      int(npb_PA),
        "mlb_PA":      int(mlb_PA),
        "harm_PA":     harm_mean(npb_PA, mlb_PA),
        "npb_BA":      round(n_BA,  4),
        "mlb_BA":      round(m_BA,  4),
        "BA_ratio":    safe_div(m_BA,  n_BA),
        "npb_OBP":     round(n_OBP, 4),
        "mlb_OBP":     round(m_OBP, 4),
        "OBP_ratio":   safe_div(m_OBP, n_OBP),
        "npb_SLG":     round(n_SLG, 4),
        "mlb_SLG":     round(m_SLG, 4),
        "SLG_ratio":   safe_div(m_SLG, n_SLG),
        "npb_HR_rate": round(n_HR_rate, 4),
        "mlb_HR_rate": round(m_HR_rate, 4),
        "HR_rate_ratio": safe_div(m_HR_rate, n_HR_rate),
        "npb_BB_rate": round(n_BB_rate, 4),
        "mlb_BB_rate": round(m_BB_rate, 4),
        "BB_rate_ratio": safe_div(m_BB_rate, n_BB_rate),
        "npb_K_rate":  round(n_K_rate, 4),
        "mlb_K_rate":  round(m_K_rate, 4),
        "K_rate_ratio": safe_div(m_K_rate, n_K_rate),
    })

player_hit_df = pd.DataFrame(hit_rows)

# ──────────────────────────────────────────────
# STEP 2 & 3 — Per-player pitcher ratios
# ──────────────────────────────────────────────

MIN_IP = 20
pit_rows = []

for player in sorted(npb_pit["name"].unique()):
    nw = npb_window(npb_pit, player)
    mw = mlb_window(mlb_pit, player)

    if mw.empty:
        continue

    npb_IP = nw["IP"].sum()
    mlb_IP = mw["IP"].sum()

    if npb_IP < MIN_IP or mlb_IP < MIN_IP:
        continue

    # ── NPB pooled (IP-weighted for ERA/WHIP; pooled counts for rate stats) ──
    n_SO = nw["SO"].sum(); n_BB = nw["BB"].sum(); n_HR = nw["HR"].sum()
    n_ERA  = (nw["earned_run_avg"] * nw["IP"]).sum() / npb_IP
    n_WHIP = (nw["whip"]          * nw["IP"]).sum() / npb_IP
    n_K9   = (n_SO / npb_IP) * 9
    n_BB9  = (n_BB / npb_IP) * 9
    n_HR9  = (n_HR / npb_IP) * 9

    # ── MLB pooled ──
    m_SO = mw["SO"].sum(); m_BB = mw["BB"].sum(); m_HR = mw["HR"].sum()
    m_ERA  = (mw["ERA"]  * mw["IP"]).sum() / mlb_IP
    m_WHIP = (mw["WHIP"] * mw["IP"]).sum() / mlb_IP
    m_K9   = (m_SO / mlb_IP) * 9
    m_BB9  = (m_BB / mlb_IP) * 9
    m_HR9  = (m_HR / mlb_IP) * 9

    npb_years = sorted(nw["year_ID"].tolist())
    mlb_years = sorted(mw["Year"].tolist())

    pit_rows.append({
        "name":       player,
        "npb_years":  str(npb_years),
        "mlb_years":  str(mlb_years),
        "npb_IP":     round(npb_IP, 1),
        "mlb_IP":     round(mlb_IP, 1),
        "harm_IP":    harm_mean(npb_IP, mlb_IP),
        "npb_ERA":    round(n_ERA,  3),
        "mlb_ERA":    round(m_ERA,  3),
        "ERA_ratio":  safe_div(m_ERA, n_ERA),
        "npb_WHIP":   round(n_WHIP, 3),
        "mlb_WHIP":   round(m_WHIP, 3),
        "WHIP_ratio": safe_div(m_WHIP, n_WHIP),
        "npb_K9":     round(n_K9,  3),
        "mlb_K9":     round(m_K9,  3),
        "K9_ratio":   safe_div(m_K9, n_K9),
        "npb_BB9":    round(n_BB9, 3),
        "mlb_BB9":    round(m_BB9, 3),
        "BB9_ratio":  safe_div(m_BB9, n_BB9),
        "npb_HR9":    round(n_HR9, 3),
        "mlb_HR9":    round(m_HR9, 3),
        "HR9_ratio":  safe_div(m_HR9, n_HR9),
    })

player_pit_df = pd.DataFrame(pit_rows)

# ──────────────────────────────────────────────
# STEP 4 — League-level weighted translation factors
# ──────────────────────────────────────────────

def compute_factors(df, ratio_cols, weight_col):
    rows = []
    for stat in ratio_cols:
        valid = df[["name", stat, weight_col]].dropna()
        if valid.empty:
            continue
        w = valid[weight_col].values.astype(float)
        r = valid[stat].values.astype(float)
        w_sum  = w.sum()
        factor = (w * r).sum() / w_sum
        wstd   = np.sqrt((w * (r - factor) ** 2).sum() / w_sum)
        rows.append({
            "stat":         stat,
            "factor":       round(factor, 4),
            "n_players":    len(valid),
            "weighted_std": round(wstd, 4),
            "min_ratio":    round(r.min(), 4),
            "max_ratio":    round(r.max(), 4),
        })
    return pd.DataFrame(rows)

HIT_RATIO_COLS = ["BA_ratio", "OBP_ratio", "SLG_ratio",
                  "HR_rate_ratio", "BB_rate_ratio", "K_rate_ratio"]
PIT_RATIO_COLS = ["ERA_ratio", "WHIP_ratio", "K9_ratio", "BB9_ratio", "HR9_ratio"]

hit_factors = compute_factors(player_hit_df, HIT_RATIO_COLS, "harm_PA")
pit_factors = compute_factors(player_pit_df, PIT_RATIO_COLS, "harm_IP")

# Clean up stat names for the summary table
STAT_NAMES = {
    "BA_ratio": "BA",    "OBP_ratio": "OBP",    "SLG_ratio": "SLG",
    "HR_rate_ratio": "HR_rate", "BB_rate_ratio": "BB_rate", "K_rate_ratio": "K_rate",
    "ERA_ratio": "ERA",  "WHIP_ratio": "WHIP",
    "K9_ratio": "K_per9", "BB9_ratio": "BB_per9", "HR9_ratio": "HR_per9",
}
hit_factors["stat"] = hit_factors["stat"].map(STAT_NAMES)
pit_factors["stat"] = pit_factors["stat"].map(STAT_NAMES)

# ──────────────────────────────────────────────
# Print per-player tables
# ──────────────────────────────────────────────

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)
pd.set_option("display.float_format", "{:.4f}".format)

print("=" * 80)
print("PER-PLAYER HITTER RATIOS  (MLB / NPB rate, last-2 NPB → first-2 MLB)")
print("=" * 80)
hit_display = player_hit_df[[
    "name", "npb_years", "mlb_years", "npb_PA", "mlb_PA",
    "npb_BA", "mlb_BA", "BA_ratio",
    "npb_OBP", "mlb_OBP", "OBP_ratio",
    "npb_SLG", "mlb_SLG", "SLG_ratio",
    "npb_HR_rate", "mlb_HR_rate", "HR_rate_ratio",
    "npb_BB_rate", "mlb_BB_rate", "BB_rate_ratio",
    "npb_K_rate", "mlb_K_rate", "K_rate_ratio",
]].copy()
print(hit_display.to_string(index=False))

print()
print("=" * 80)
print("PER-PLAYER PITCHER RATIOS  (MLB / NPB rate, last-2 NPB → first-2 MLB)")
print("=" * 80)
pit_display = player_pit_df[[
    "name", "npb_years", "mlb_years", "npb_IP", "mlb_IP",
    "npb_ERA", "mlb_ERA", "ERA_ratio",
    "npb_WHIP", "mlb_WHIP", "WHIP_ratio",
    "npb_K9", "mlb_K9", "K9_ratio",
    "npb_BB9", "mlb_BB9", "BB9_ratio",
    "npb_HR9", "mlb_HR9", "HR9_ratio",
]].copy()
print(pit_display.to_string(index=False))

print()
print("=" * 80)
print("LEAGUE-LEVEL HITTING TRANSLATION FACTORS")
print("=" * 80)
print(hit_factors.to_string(index=False))

print()
print("=" * 80)
print("LEAGUE-LEVEL PITCHING TRANSLATION FACTORS")
print("=" * 80)
print(pit_factors.to_string(index=False))

# ──────────────────────────────────────────────
# STEP 5 — Save CSVs and load into DB
# ──────────────────────────────────────────────

hit_factors.to_csv("translation_factors_hitting.csv",   index=False)
pit_factors.to_csv("translation_factors_pitching.csv",  index=False)
player_hit_df.to_csv("player_ratios_hitting.csv",       index=False)
player_pit_df.to_csv("player_ratios_pitching.csv",      index=False)

for df, table in [
    (hit_factors,    "translation_factors_hitting"),
    (pit_factors,    "translation_factors_pitching"),
    (player_hit_df,  "player_ratios_hitting"),
    (player_pit_df,  "player_ratios_pitching"),
]:
    df.to_sql(table, conn, if_exists="replace", index=False)
    print(f"  → saved {table} ({len(df)} rows) to CSV and DB")

conn.close()
print("\nDone.")
