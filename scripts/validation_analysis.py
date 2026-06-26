"""
Phase 4 — Validation: project NPB→MLB performance for recent crossover players
using translation factors (naive) and OLS regression, then compare to actuals.
"""

import sqlite3
import numpy as np
import pandas as pd

DB_PATH = "npb_mlb.db"
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

conn = sqlite3.connect(DB_PATH)

# ── Players ──────────────────────────────────────────────────────────────────
HITTERS  = ["Munetaka Murakami", "Seiya Suzuki", "Shohei Ohtani"]
PITCHERS = ["Kodai Senga", "Yoshinobu Yamamoto", "Tatsuya Imai", "Shohei Ohtani"]
CORE_SET = {"Seiya Suzuki", "Shohei Ohtani"}   # in training data — not truly OOS

# ── Load factor and regression tables ────────────────────────────────────────
fac_hit = pd.read_sql("SELECT * FROM translation_factors_hitting_v2",  conn).set_index("stat")
fac_pit = pd.read_sql("SELECT * FROM translation_factors_pitching_v2", conn).set_index("stat")
reg_hit = pd.read_sql("SELECT * FROM regression_results_hitting",  conn)
reg_pit = pd.read_sql("SELECT * FROM regression_results_pitching", conn)

# Index by target stat (strip "mlb_") for easy lookup
def reg_by_stat(df):
    d = df.copy()
    d["stat"] = d["target_stat"].str.replace("mlb_", "", regex=False)
    return d.set_index("stat")

rh = reg_by_stat(reg_hit)
rp = reg_by_stat(reg_pit)

# ── Raw source tables ─────────────────────────────────────────────────────────
npb_hit_raw = pd.read_sql("SELECT * FROM npb_hitting",  conn)
npb_pit_raw = pd.read_sql("SELECT * FROM npb_pitching", conn)
mlb_hit_raw = pd.read_sql("SELECT * FROM mlb_hitting",  conn)
mlb_pit_raw = pd.read_sql("SELECT * FROM mlb_pitching", conn)

conn.close()

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def safe_div(a, b):
    return float(a) / float(b) if b and b != 0 else np.nan

def npb_hit_rates(df, player):
    """Pool last-2 NPB seasons and return hitting rate dict + metadata."""
    rows = df[df["name"] == player].sort_values("year_ID", ascending=False).head(2)
    last_year  = rows["year_ID"].max()
    final_age  = rows.loc[rows["year_ID"] == last_year, "age"].values[0]
    npb_years  = sorted(rows["year_ID"].tolist())

    PA = rows["PA"].sum(); AB = rows["AB"].sum(); H = rows["H"].sum()
    d2 = rows["2B"].sum(); d3 = rows["3B"].sum()
    HR = rows["HR"].sum(); BB = rows["BB"].sum(); SO = rows["SO"].sum()

    return {
        "npb_years":     npb_years,
        "npb_PA":        int(PA),
        "npb_final_age": int(final_age),
        "BA":      safe_div(H, AB),
        "OBP":     safe_div(H + BB, AB + BB),
        "SLG":     safe_div(H + d2 + 2*d3 + 3*HR, AB),
        "HR_rate": safe_div(HR, PA),
        "BB_rate": safe_div(BB, PA),
        "K_rate":  safe_div(SO, PA),
    }

def npb_pit_rates(df, player):
    """Pool last-2 NPB seasons and return pitching rate dict + metadata."""
    rows = df[df["name"] == player].sort_values("year_ID", ascending=False).head(2)
    last_year  = rows["year_ID"].max()
    final_age  = rows.loc[rows["year_ID"] == last_year, "age"].values[0]
    npb_years  = sorted(rows["year_ID"].tolist())

    IP = rows["IP"].sum()   # raw float — consistent with Phase 2
    SO = rows["SO"].sum(); BB = rows["BB"].sum(); HR = rows["HR"].sum()
    ERA  = (rows["earned_run_avg"] * rows["IP"]).sum() / IP
    WHIP = (rows["whip"]           * rows["IP"]).sum() / IP

    return {
        "npb_years":     npb_years,
        "npb_IP":        round(IP, 1),
        "npb_final_age": int(final_age),
        "ERA":    round(ERA,  3),
        "WHIP":   round(WHIP, 3),
        "K9":     round(safe_div(SO, IP) * 9, 3),
        "BB9":    round(safe_div(BB, IP) * 9, 3),
        "HR9":    round(safe_div(HR, IP) * 9, 3),
    }

def mlb_hit_rates_pooled(df, player, max_seasons=None):
    """Pool MLB hitting seasons (all, or first N by Year) and return rate dict."""
    rows = df[df["name"] == player].sort_values("Year")
    if max_seasons:
        rows = rows.head(max_seasons)
    if rows.empty:
        return None, []
    years = sorted(rows["Year"].tolist())
    PA = rows["PA"].sum(); AB = rows["AB"].sum(); H = rows["H"].sum()
    d2 = rows["2B"].sum(); d3 = rows["3B"].sum()
    HR = rows["HR"].sum(); BB = rows["BB"].sum(); SO = rows["SO"].sum()
    rates = {
        "BA":      safe_div(H, AB),
        "OBP":     safe_div(H + BB, AB + BB),
        "SLG":     safe_div(H + d2 + 2*d3 + 3*HR, AB),
        "HR_rate": safe_div(HR, PA),
        "BB_rate": safe_div(BB, PA),
        "K_rate":  safe_div(SO, PA),
    }
    return rates, years

def mlb_pit_rates_pooled(df, player, max_seasons=None):
    """Pool MLB pitching seasons and return rate dict."""
    rows = df[df["name"] == player].sort_values("Year")
    if max_seasons:
        rows = rows.head(max_seasons)
    if rows.empty:
        return None, []
    years = sorted(rows["Year"].tolist())
    IP = rows["IP"].sum()
    SO = rows["SO"].sum(); BB = rows["BB"].sum(); HR = rows["HR"].sum()
    ERA  = (rows["ERA"]  * rows["IP"]).sum() / IP
    WHIP = (rows["WHIP"] * rows["IP"]).sum() / IP
    rates = {
        "ERA":  round(ERA,  3),
        "WHIP": round(WHIP, 3),
        "K9":   round(safe_div(SO, IP) * 9, 3),
        "BB9":  round(safe_div(BB, IP) * 9, 3),
        "HR9":  round(safe_div(HR, IP) * 9, 3),
    }
    return rates, years

def naive_proj(npb_rate, stat, fac_table):
    """Multiply NPB rate by translation factor."""
    if stat in fac_table.index:
        return npb_rate * fac_table.loc[stat, "factor"]
    return np.nan

def regression_proj(npb_rate, age, stat, reg_table):
    """
    Apply OLS projection if p(npb_coef) < 0.05.
    Returns (projected_value, method_used).
    """
    if stat not in reg_table.index:
        return None, "no_model"
    row = reg_table.loc[stat]
    if row["significant"]:
        proj = row["intercept"] + row["npb_coef"] * npb_rate + row["age_coef"] * age
        return proj, "regression"
    return None, "naive_fallback"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1–5: per-player tables
# ─────────────────────────────────────────────────────────────────────────────

all_hit_rows = []
all_pit_rows = []

# ══════════════════════════════════════════════════════════════════════════════
#  HITTERS
# ══════════════════════════════════════════════════════════════════════════════

HIT_STATS = ["BA", "OBP", "SLG", "HR_rate", "BB_rate", "K_rate"]

print("=" * 78)
print("PHASE 4 VALIDATION — HITTERS")
print("=" * 78)

for player in HITTERS:
    npb = npb_hit_rates(npb_hit_raw, player)
    age = npb["npb_final_age"]
    in_sample_flag = " [IN-SAMPLE — core training set]" if player in CORE_SET else " [OUT-OF-SAMPLE]"

    actual_career, car_yrs = mlb_hit_rates_pooled(mlb_hit_raw, player)
    actual_f2,     f2_yrs  = mlb_hit_rates_pooled(mlb_hit_raw, player, max_seasons=2)

    print(f"\n{'─'*78}")
    print(f"  {player.upper()}{in_sample_flag}")
    print(f"  NPB window: {npb['npb_years']}  |  pooled PA: {npb['npb_PA']}  |  npb_final_age: {age}")
    if actual_career:
        print(f"  MLB career: {car_yrs}   |  First-2 seasons: {f2_yrs}")
    else:
        print("  MLB career: NO DATA")
    print(f"{'─'*78}")
    print(f"  {'stat':<12} {'npb_rate':>9} {'naive_proj':>11} {'reg_proj':>10} {'method':>13}"
          f" {'actual_f2':>10} {'actual_car':>11} {'naive_err_f2':>13} {'reg_err_f2':>11}")
    print(f"  {'-'*12} {'-'*9} {'-'*11} {'-'*10} {'-'*13} {'-'*10} {'-'*11} {'-'*13} {'-'*11}")

    for stat in HIT_STATS:
        npb_rate = npb[stat]
        n_proj   = naive_proj(npb_rate, stat, fac_hit)
        r_proj_val, method = regression_proj(npb_rate, age, stat, rh)
        final_reg = r_proj_val if method == "regression" else n_proj

        act_f2  = actual_f2[stat]  if actual_f2  else np.nan
        act_car = actual_career[stat] if actual_career else np.nan

        n_err = round(n_proj     - act_f2,  4) if not np.isnan(act_f2) else np.nan
        r_err = round(final_reg  - act_f2,  4) if not np.isnan(act_f2) else np.nan

        method_label = method if method == "regression" else "naive*"

        print(f"  {stat:<12} {npb_rate:>9.4f} {n_proj:>11.4f} {final_reg:>10.4f} {method_label:>13}"
              f" {act_f2:>10.4f} {act_car:>11.4f} {n_err:>13.4f} {r_err:>11.4f}")

        all_hit_rows.append({
            "player":        player,
            "in_sample":     player in CORE_SET,
            "stat":          stat,
            "npb_rate":      round(npb_rate, 4),
            "naive_proj":    round(n_proj, 4),
            "reg_proj":      round(final_reg, 4),
            "method":        method_label,
            "actual_f2":     round(act_f2,  4) if not np.isnan(act_f2)  else None,
            "actual_career": round(act_car, 4) if not np.isnan(act_car) else None,
            "naive_err_f2":  round(n_err,  4) if not np.isnan(n_err)   else None,
            "reg_err_f2":    round(r_err,  4) if not np.isnan(r_err)   else None,
            "npb_window":    str(npb["npb_years"]),
            "mlb_career_yrs": str(car_yrs),
            "mlb_f2_yrs":   str(f2_yrs),
            "npb_final_age": age,
        })

# ══════════════════════════════════════════════════════════════════════════════
#  PITCHERS
# ══════════════════════════════════════════════════════════════════════════════

PIT_STATS = ["ERA", "WHIP", "K9", "BB9", "HR9"]
PIT_FAC_KEY = {"ERA": "ERA", "WHIP": "WHIP", "K9": "K_per9", "BB9": "BB_per9", "HR9": "HR_per9"}

print()
print("=" * 78)
print("PHASE 4 VALIDATION — PITCHERS")
print("=" * 78)

for player in PITCHERS:
    npb = npb_pit_rates(npb_pit_raw, player)
    age = npb["npb_final_age"]
    in_sample_flag = " [IN-SAMPLE — core training set]" if player in CORE_SET else " [OUT-OF-SAMPLE]"

    actual_career, car_yrs = mlb_pit_rates_pooled(mlb_pit_raw, player)
    actual_f2,     f2_yrs  = mlb_pit_rates_pooled(mlb_pit_raw, player, max_seasons=2)

    print(f"\n{'─'*78}")
    print(f"  {player.upper()}{in_sample_flag}")
    print(f"  NPB window: {npb['npb_years']}  |  pooled IP: {npb['npb_IP']}  |  npb_final_age: {age}")
    if actual_career:
        print(f"  MLB career: {car_yrs}   |  First-2 seasons: {f2_yrs}")
    else:
        print("  MLB career: NO DATA")
    print(f"{'─'*78}")
    print(f"  {'stat':<7} {'npb_rate':>9} {'naive_proj':>11} {'reg_proj':>10} {'method':>13}"
          f" {'actual_f2':>10} {'actual_car':>11} {'naive_err_f2':>13} {'reg_err_f2':>11}")
    print(f"  {'-'*7} {'-'*9} {'-'*11} {'-'*10} {'-'*13} {'-'*10} {'-'*11} {'-'*13} {'-'*11}")

    for stat in PIT_STATS:
        npb_rate  = npb[stat]
        fac_key   = PIT_FAC_KEY[stat]
        n_proj    = naive_proj(npb_rate, fac_key, fac_pit)
        r_proj_val, method = regression_proj(npb_rate, age, stat, rp)
        final_reg = r_proj_val if method == "regression" else n_proj

        act_f2  = actual_f2[stat]     if actual_f2     else np.nan
        act_car = actual_career[stat] if actual_career  else np.nan

        n_err = round(n_proj    - act_f2, 4) if not np.isnan(act_f2) else np.nan
        r_err = round(final_reg - act_f2, 4) if not np.isnan(act_f2) else np.nan

        method_label = method if method == "regression" else "naive*"

        print(f"  {stat:<7} {npb_rate:>9.4f} {n_proj:>11.4f} {final_reg:>10.4f} {method_label:>13}"
              f" {act_f2:>10.4f} {act_car:>11.4f} {n_err:>13.4f} {r_err:>11.4f}")

        all_pit_rows.append({
            "player":        player,
            "in_sample":     player in CORE_SET,
            "stat":          stat,
            "npb_rate":      round(npb_rate, 4),
            "naive_proj":    round(n_proj, 4),
            "reg_proj":      round(final_reg, 4),
            "method":        method_label,
            "actual_f2":     round(act_f2,  4) if not np.isnan(act_f2)  else None,
            "actual_career": round(act_car, 4) if not np.isnan(act_car) else None,
            "naive_err_f2":  round(n_err,  4) if not np.isnan(n_err)   else None,
            "reg_err_f2":    round(r_err,  4) if not np.isnan(r_err)   else None,
            "npb_window":    str(npb["npb_years"]),
            "mlb_career_yrs": str(car_yrs),
            "mlb_f2_yrs":   str(f2_yrs),
            "npb_final_age": age,
        })

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Scouting summaries
# ─────────────────────────────────────────────────────────────────────────────

hit_val_df = pd.DataFrame(all_hit_rows)
pit_val_df = pd.DataFrame(all_pit_rows)

def scouting_summary(val_df, player, position):
    rows = val_df[val_df["player"] == player].copy()
    has_actual = rows["actual_f2"].notna().any()
    if not has_actual:
        print(f"  No MLB actuals available yet — projection only.")
        return

    rows["abs_naive_err"] = rows["naive_err_f2"].abs()
    rows["abs_reg_err"]   = rows["reg_err_f2"].abs()

    mae_naive = rows["abs_naive_err"].mean()
    mae_reg   = rows["abs_reg_err"].mean()
    better    = "regression" if mae_reg < mae_naive else "naive (factor-based)"

    best_stat  = rows.loc[rows["abs_naive_err"].idxmin(),  "stat"]
    worst_stat = rows.loc[rows["abs_naive_err"].idxmax(), "stat"]

    reg_rows = rows[rows["method"] == "regression"]
    if not reg_rows.empty:
        best_reg_stat = reg_rows.loc[reg_rows["abs_reg_err"].idxmin(), "stat"]
    else:
        best_reg_stat = "N/A"

    in_sample_note = " (note: in-sample — model was trained on this player)" \
                     if player in CORE_SET else ""

    print(f"\n  SCOUTING SUMMARY — {player}{in_sample_note}")
    print(f"  • Better overall method: {better}  (MAE naive={mae_naive:.4f}, reg={mae_reg:.4f})")
    print(f"  • Best leading indicator: {best_stat} (smallest naive error)")
    print(f"  • Worst leading indicator: {worst_stat} (largest naive error)")
    if reg_rows.empty:
        print(f"  • Regression applied to: none (no significant stats for this group)")
    else:
        print(f"  • Best regression stat: {best_reg_stat}")

    # One-sentence scouting verdict
    pa_or_ip = rows.iloc[0].get("npb_final_age")
    naive_dir = "would have" if better == "naive (factor-based)" else "would have underperformed vs"
    print(f"  • Verdict: The model {naive_dir} the naive factor projection; "
          f"{best_stat} was the most portable NPB skill — "
          f"scouts could have relied on it as the strongest predictor of MLB output.")

print()
print("=" * 78)
print("STEP 6 — SCOUTING SUMMARIES")
print("=" * 78)

print("\n─── HITTERS ───")
for player in HITTERS:
    scouting_summary(hit_val_df, player, "hitter")

print("\n─── PITCHERS ───")
for player in PITCHERS:
    scouting_summary(pit_val_df, player, "pitcher")

# ─────────────────────────────────────────────────────────────────────────────
# Overall MAE summary
# ─────────────────────────────────────────────────────────────────────────────

def overall_mae(df, label):
    valid = df.dropna(subset=["naive_err_f2", "reg_err_f2"])
    if valid.empty:
        print(f"\n  {label}: no actuals available.")
        return
    mae_n = valid["naive_err_f2"].abs().mean()
    mae_r = valid["reg_err_f2"].abs().mean()
    oos   = valid[~valid["in_sample"]]
    print(f"\n  {label} — all players:         naive MAE={mae_n:.4f}  reg MAE={mae_r:.4f}")
    if not oos.empty:
        print(f"  {label} — OOS players only:     "
              f"naive MAE={oos['naive_err_f2'].abs().mean():.4f}  "
              f"reg MAE={oos['reg_err_f2'].abs().mean():.4f}")
    else:
        print(f"  {label} — OOS players only:     no out-of-sample actuals")

print()
print("=" * 78)
print("OVERALL ACCURACY SUMMARY  (error = projected − actual first-2 seasons)")
print("  * naive* = regression model not significant, fell back to factor")
print("=" * 78)
overall_mae(hit_val_df, "HITTING ")
overall_mae(pit_val_df, "PITCHING")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Save
# ─────────────────────────────────────────────────────────────────────────────

hit_val_df.to_csv("validation_results_hitting.csv",  index=False)
pit_val_df.to_csv("validation_results_pitching.csv", index=False)

conn = sqlite3.connect(DB_PATH)
print()
for df, table in [
    (hit_val_df, "validation_results_hitting"),
    (pit_val_df, "validation_results_pitching"),
]:
    df.to_sql(table, conn, if_exists="replace", index=False)
    print(f"  → saved {table} ({len(df)} rows) to CSV and DB")
conn.close()
print("\nDone.")
