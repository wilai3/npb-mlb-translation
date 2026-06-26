"""
OLS regression: NPB rates + debut age → MLB rates.
LOO-CV MAE for generalization assessment.
"""

import ast
import sqlite3
import numpy as np
import pandas as pd
import statsmodels.api as sm

DB_PATH = "npb_mlb.db"
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

conn = sqlite3.connect(DB_PATH)

# ──────────────────────────────────────────────────────────────
# STEP 1 — Build regression datasets
# ──────────────────────────────────────────────────────────────

def last_npb_year(npb_years_str):
    """Parse '[2005, 2006]' → 2006."""
    return max(ast.literal_eval(npb_years_str))

hit_ratios = pd.read_sql("SELECT * FROM player_ratios_hitting_v2",  conn)
pit_ratios = pd.read_sql("SELECT * FROM player_ratios_pitching_v2", conn)

hit_ratios["last_npb_year"] = hit_ratios["npb_years"].apply(last_npb_year)
pit_ratios["last_npb_year"] = pit_ratios["npb_years"].apply(last_npb_year)

# Look up age in the player's final NPB season
npb_hit_ages = pd.read_sql(
    "SELECT name, year_ID, age FROM npb_hitting WHERE dataset='core'", conn
)
npb_pit_ages = pd.read_sql(
    "SELECT name, year_ID, age FROM npb_pitching WHERE dataset='core'", conn
)

hit_data = hit_ratios.merge(
    npb_hit_ages.rename(columns={"year_ID": "last_npb_year", "age": "npb_final_age"}),
    on=["name", "last_npb_year"], how="left",
)
pit_data = pit_ratios.merge(
    npb_pit_ages.rename(columns={"year_ID": "last_npb_year", "age": "npb_final_age"}),
    on=["name", "last_npb_year"], how="left",
)

# Report any missing age lookups
for label, df in [("hitters", hit_data), ("pitchers", pit_data)]:
    missing = df[df["npb_final_age"].isna()]["name"].tolist()
    if missing:
        print(f"WARNING — missing npb_final_age for {label}: {missing}")

print(f"Regression dataset: {len(hit_data)} hitters, {len(pit_data)} pitchers\n")

# ──────────────────────────────────────────────────────────────
# STEP 2 & 3 — OLS regressions + result extraction
# ──────────────────────────────────────────────────────────────

HIT_PAIRS = [
    ("mlb_BA",      "npb_BA"),
    ("mlb_OBP",     "npb_OBP"),
    ("mlb_SLG",     "npb_SLG"),
    ("mlb_HR_rate", "npb_HR_rate"),
    ("mlb_BB_rate", "npb_BB_rate"),
    ("mlb_K_rate",  "npb_K_rate"),
]
PIT_PAIRS = [
    ("mlb_ERA",  "npb_ERA"),
    ("mlb_WHIP", "npb_WHIP"),
    ("mlb_K9",   "npb_K9"),
    ("mlb_BB9",  "npb_BB9"),
    ("mlb_HR9",  "npb_HR9"),
]

def run_regressions(df, pairs):
    """Run OLS for each (target, predictor) pair and return results list + fitted models."""
    results = []
    models  = {}
    for target, npb_stat in pairs:
        sub = df[[target, npb_stat, "npb_final_age"]].dropna()
        y   = sub[target].values
        X   = sm.add_constant(sub[[npb_stat, "npb_final_age"]].values)
        res = sm.OLS(y, X).fit()

        stat_label = target.replace("mlb_", "")
        results.append({
            "target_stat": target,
            "npb_stat":    npb_stat,
            "r_squared":   round(res.rsquared, 4),
            "npb_coef":    round(res.params[1], 4),
            "npb_pvalue":  round(res.pvalues[1], 4),
            "age_coef":    round(res.params[2], 4),
            "age_pvalue":  round(res.pvalues[2], 4),
            "intercept":   round(res.params[0], 4),
            "n_players":   int(res.nobs),
            "significant": bool(res.pvalues[1] < 0.05),
        })
        models[stat_label] = (res, sub, npb_stat, target)
    return pd.DataFrame(results).sort_values("r_squared", ascending=False), models

hit_results, hit_models = run_regressions(hit_data, HIT_PAIRS)
pit_results, pit_models = run_regressions(pit_data, PIT_PAIRS)

print("=" * 70)
print("OLS REGRESSION SUMMARY — HITTERS  (sorted by R²)")
print("=" * 70)
print(hit_results.to_string(index=False))

print()
print("=" * 70)
print("OLS REGRESSION SUMMARY — PITCHERS  (sorted by R²)")
print("=" * 70)
print(pit_results.to_string(index=False))

# ── Full statsmodels output for top-3 models ──

def print_top3(models_dict, results_df, label):
    top3 = results_df.head(3)["target_stat"].str.replace("mlb_", "").tolist()
    print(f"\n{'=' * 70}")
    print(f"FULL STATSMODELS OUTPUT — TOP 3 {label} MODELS")
    print(f"{'=' * 70}")
    for stat in top3:
        res, sub, npb_stat, target = models_dict[stat]
        print(f"\n--- {target} ~ {npb_stat} + npb_final_age ---")
        print(res.summary())

print_top3(hit_models, hit_results, "HITTING")
print_top3(pit_models, pit_results, "PITCHING")

# ──────────────────────────────────────────────────────────────
# STEP 4 — Leave-One-Out Cross-Validation (MAE)
# ──────────────────────────────────────────────────────────────

def loo_mae(df, target, npb_stat):
    sub = df[[target, npb_stat, "npb_final_age"]].dropna().reset_index(drop=True)
    n   = len(sub)
    errors = []
    for i in range(n):
        train = sub.drop(i)
        test  = sub.iloc[[i]]
        y_tr  = train[target].values
        X_tr  = sm.add_constant(train[[npb_stat, "npb_final_age"]].values, has_constant="add")
        y_te  = test[target].values
        X_te  = sm.add_constant(test[[npb_stat, "npb_final_age"]].values, has_constant="add")
        res   = sm.OLS(y_tr, X_tr).fit()
        pred  = res.predict(X_te)[0]
        errors.append(abs(pred - y_te[0]))
    return round(np.mean(errors), 5)

print("\n")
print("=" * 70)
print("LOO-CV MEAN ABSOLUTE ERROR — HITTERS")
print("=" * 70)
loo_hit = []
for _, row in hit_results.iterrows():
    mae = loo_mae(hit_data, row["target_stat"], row["npb_stat"])
    loo_hit.append({"stat": row["target_stat"].replace("mlb_",""), "r_squared": row["r_squared"], "loo_mae": mae})
loo_hit_df = pd.DataFrame(loo_hit).sort_values("r_squared", ascending=False)
print(loo_hit_df.to_string(index=False))

print()
print("=" * 70)
print("LOO-CV MEAN ABSOLUTE ERROR — PITCHERS")
print("=" * 70)
loo_pit = []
for _, row in pit_results.iterrows():
    mae = loo_mae(pit_data, row["target_stat"], row["npb_stat"])
    loo_pit.append({"stat": row["target_stat"].replace("mlb_",""), "r_squared": row["r_squared"], "loo_mae": mae})
loo_pit_df = pd.DataFrame(loo_pit).sort_values("r_squared", ascending=False)
print(loo_pit_df.to_string(index=False))

# Merge LOO MAE into results tables
hit_results = hit_results.merge(
    loo_hit_df[["stat", "loo_mae"]].rename(columns={"stat": "_s"}),
    left_on=hit_results["target_stat"].str.replace("mlb_",""),
    right_on="_s", how="left"
).drop(columns=["_s", "key_0"], errors="ignore")

pit_results = pit_results.merge(
    loo_pit_df[["stat", "loo_mae"]].rename(columns={"stat": "_s"}),
    left_on=pit_results["target_stat"].str.replace("mlb_",""),
    right_on="_s", how="left"
).drop(columns=["_s", "key_0"], errors="ignore")

# ──────────────────────────────────────────────────────────────
# Final interpretation
# ──────────────────────────────────────────────────────────────

HIT_INTERP = {
    "BA":      ("BA",       "BA"),
    "OBP":     ("OBP",      "OBP"),
    "SLG":     ("SLG",      "SLG"),
    "HR_rate": ("HR_rate",  "HR rate"),
    "BB_rate": ("BB_rate",  "BB rate"),
    "K_rate":  ("K_rate",   "K rate"),
}
PIT_INTERP = {
    "ERA":  ("ERA",  "ERA"),
    "WHIP": ("WHIP", "WHIP"),
    "K9":   ("K9",   "K/9"),
    "BB9":  ("BB9",  "BB/9"),
    "HR9":  ("HR9",  "HR/9"),
}

def interpret_row(row, stat_label, scale_note=""):
    r2   = row["r_squared"]
    pval = row["npb_pvalue"]
    coef = row["npb_coef"]
    sig  = "statistically significant (p<0.05)" if row["significant"] else f"not significant (p={pval:.2f})"
    strength = (
        "strong" if r2 >= 0.5 else
        "moderate" if r2 >= 0.3 else
        "weak"
    )
    return (
        f"  {stat_label}: R²={r2:.2f}, {strength} relationship, NPB predictor {sig} — "
        f"{'NPB ' + stat_label + ' translates meaningfully' if r2 >= 0.3 else 'NPB ' + stat_label + ' is a poor standalone predictor'}; "
        f"each 1-unit increase in NPB {stat_label} predicts a {coef:+.3f} change in MLB {stat_label}."
    )

print("\n")
print("=" * 70)
print("PLAIN-ENGLISH INTERPRETATION")
print("=" * 70)
print("\nHITTERS:")
for _, row in hit_results.iterrows():
    stat = row["target_stat"].replace("mlb_", "")
    print(interpret_row(row, stat))
print("\nPITCHERS:")
for _, row in pit_results.iterrows():
    stat = row["target_stat"].replace("mlb_", "")
    print(interpret_row(row, stat))

# ──────────────────────────────────────────────────────────────
# STEP 5 — Save CSVs and load into DB
# ──────────────────────────────────────────────────────────────

hit_results.to_csv("regression_results_hitting.csv",  index=False)
pit_results.to_csv("regression_results_pitching.csv", index=False)

print()
for df, table in [
    (hit_results, "regression_results_hitting"),
    (pit_results, "regression_results_pitching"),
]:
    df.to_sql(table, conn, if_exists="replace", index=False)
    print(f"  → saved {table} ({len(df)} rows) to CSV and DB")

conn.close()
print("\nDone.")
