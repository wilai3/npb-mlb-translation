"""
NPB-to-MLB Statistical Translation Dashboard
"""

import os
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "npb_mlb.db")

HIT_STATS = ["BA", "OBP", "SLG", "HR_rate", "BB_rate", "K_rate"]
PIT_STATS = ["ERA", "WHIP", "K9", "BB9", "HR9"]
# Factor table uses different keys for pitcher per-9 stats
PIT_FAC_KEY = {"ERA": "ERA", "WHIP": "WHIP", "K9": "K_per9", "BB9": "BB_per9", "HR9": "HR_per9"}

STAT_DISPLAY = {
    "BA": "BA", "OBP": "OBP", "SLG": "SLG",
    "HR_rate": "HR/PA", "BB_rate": "BB/PA", "K_rate": "K/PA",
    "ERA": "ERA", "WHIP": "WHIP", "K9": "K/9", "BB9": "BB/9", "HR9": "HR/9",
}

# Reliability tier per stat (from regression R² results)
RELIABILITY = {
    "OBP": "green", "BB_rate": "green", "K_rate": "green", "K9": "green",
    "BA": "yellow", "WHIP": "yellow",
    "SLG": "red", "HR_rate": "red", "ERA": "red", "BB9": "red", "HR9": "red",
}
RELIABILITY_COLORS = {"green": "#d4edda", "yellow": "#fff3cd", "red": "#f8d7da"}

RELIABILITY_INFO = {
    "BA":      ("🟡", "R²=0.43 (sig.) — moderate; NPB contact partially transfers"),
    "OBP":     ("🟢", "R²=0.57 (sig.) — strong; plate discipline is portable"),
    "SLG":     ("🔴", "R²=0.27 (not sig.) — low; power is highly context-dependent"),
    "HR_rate": ("🔴", "R²=0.41 (not sig.) — caution; HR rate varies widely"),
    "BB_rate": ("🟢", "R²=0.64 (sig.) — strong; walk skills travel well"),
    "K_rate":  ("🟢", "R²=0.77 (sig.) — most reliable hitter stat"),
    "ERA":     ("🔴", "R²=0.10 (not sig.) — low; ERA is context-dependent"),
    "WHIP":    ("🟡", "R²=0.07 (not sig.) — moderate guide only"),
    "K9":      ("🟢", "R²=0.77 (sig.) — most reliable pitcher stat; strikeout stuff travels"),
    "BB9":     ("🔴", "R²=0.04 (not sig.) — low; walk rate poorly predicted"),
    "HR9":     ("🔴", "R²=0.09 (not sig.) — low; HR/9 varies widely on crossover"),
}

# Reference MLB averages for gauges and summaries
MLB_AVGS = {
    "BA": .250, "OBP": .320, "SLG": .410, "HR_rate": .030,
    "BB_rate": .082, "K_rate": .220,
    "ERA": 4.20, "WHIP": 1.28, "K9": 8.5, "BB9": 3.1, "HR9": 1.20,
}

# Gauge display ranges
GAUGE_CFG = {
    # stat: (min, max, lower_is_better)
    "K_rate":  (0.05, 0.35, True),
    "BB_rate": (0.02, 0.20, False),
    "OBP":     (0.25, 0.50, False),
    "K9":      (4.0,  14.0, False),
    "ERA":     (1.5,  6.5,  True),
    "WHIP":    (0.70, 1.80, True),
}

BAR_COLORS = {
    "Naive Proj":  "#1f77b4",
    "Reg Proj":    "#ff7f0e",
    "Actual MLB":  "#7f7f7f",
}

# ─────────────────────────────────────────────────────────────────────────────
# DB Loaders — all cached
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_players():
    with sqlite3.connect(DB_PATH) as c:
        return pd.read_sql("SELECT * FROM players ORDER BY position_type, name", c)

@st.cache_data
def load_factors():
    with sqlite3.connect(DB_PATH) as c:
        fh = pd.read_sql("SELECT * FROM translation_factors_hitting_v2", c).set_index("stat")
        fp = pd.read_sql("SELECT * FROM translation_factors_pitching_v2", c).set_index("stat")
        rh = pd.read_sql("SELECT * FROM regression_results_hitting", c)
        rp = pd.read_sql("SELECT * FROM regression_results_pitching", c)
    rh["_key"] = rh["target_stat"].str.replace("mlb_", "", regex=False)
    rp["_key"] = rp["target_stat"].str.replace("mlb_", "", regex=False)
    return fh, fp, rh.set_index("_key"), rp.set_index("_key")

@st.cache_data
def load_npb():
    with sqlite3.connect(DB_PATH) as c:
        nh = pd.read_sql("SELECT * FROM npb_hitting",  c)
        np_ = pd.read_sql("SELECT * FROM npb_pitching", c)
    return nh, np_

@st.cache_data
def load_mlb():
    with sqlite3.connect(DB_PATH) as c:
        mh = pd.read_sql("SELECT * FROM mlb_hitting",  c)
        mp = pd.read_sql("SELECT * FROM mlb_pitching", c)
    return mh, mp

# ─────────────────────────────────────────────────────────────────────────────
# Computation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_div(a, b):
    return float(a) / float(b) if b and not np.isnan(float(b)) and float(b) != 0 else np.nan

def _strip_suffix(name):
    """Remove (H)/(P) suffix for raw table lookups."""
    return name.replace(" (H)", "").replace(" (P)", "")


def get_npb_hit_profile(npb_df, player_name):
    """Returns (raw_table_df, rates_dict) for last 2 NPB seasons."""
    name = _strip_suffix(player_name)
    rows = npb_df[npb_df["name"] == name].sort_values("year_ID", ascending=False).head(2)
    if rows.empty:
        return None, None

    raw = rows[["year_ID", "age", "team_ID", "PA", "AB", "H", "2B", "3B",
                "HR", "BB", "SO", "batting_avg", "onbase_perc", "slugging_perc"]].copy()
    raw.columns = ["Year", "Age", "Team", "PA", "AB", "H", "2B", "3B",
                   "HR", "BB", "SO", "BA", "OBP", "SLG"]
    raw = raw.sort_values("Year").reset_index(drop=True)

    PA = rows["PA"].sum(); AB = rows["AB"].sum(); H = rows["H"].sum()
    d2 = rows["2B"].sum(); d3 = rows["3B"].sum()
    HR = rows["HR"].sum(); BB = rows["BB"].sum(); SO = rows["SO"].sum()
    final_age = int(rows.loc[rows["year_ID"].idxmax(), "age"])

    rates = {
        "npb_years":     sorted(rows["year_ID"].tolist()),
        "npb_final_age": final_age,
        "npb_PA":        int(PA),
        "BA":      _safe_div(H, AB),
        "OBP":     _safe_div(H + BB, AB + BB),
        "SLG":     _safe_div(H + d2 + 2*d3 + 3*HR, AB),
        "HR_rate": _safe_div(HR, PA),
        "BB_rate": _safe_div(BB, PA),
        "K_rate":  _safe_div(SO, PA),
    }
    return raw, rates


def get_npb_pit_profile(npb_df, player_name):
    """Returns (raw_table_df, rates_dict) for last 2 NPB seasons."""
    name = _strip_suffix(player_name)
    rows = npb_df[npb_df["name"] == name].sort_values("year_ID", ascending=False).head(2)
    if rows.empty:
        return None, None

    raw = rows[["year_ID", "age", "team_ID", "IP", "SO", "BB", "HR",
                "earned_run_avg", "whip"]].copy()
    raw.columns = ["Year", "Age", "Team", "IP", "SO", "BB", "HR", "ERA", "WHIP"]
    raw = raw.sort_values("Year").reset_index(drop=True)

    IP = rows["IP"].sum()
    SO = rows["SO"].sum(); BB = rows["BB"].sum(); HR = rows["HR"].sum()
    ERA  = (rows["earned_run_avg"] * rows["IP"]).sum() / IP
    WHIP = (rows["whip"]           * rows["IP"]).sum() / IP
    final_age = int(rows.loc[rows["year_ID"].idxmax(), "age"])

    rates = {
        "npb_years":     sorted(rows["year_ID"].tolist()),
        "npb_final_age": final_age,
        "npb_IP":        round(IP, 1),
        "ERA":  round(ERA,  3),
        "WHIP": round(WHIP, 3),
        "K9":   round(_safe_div(SO, IP) * 9, 3),
        "BB9":  round(_safe_div(BB, IP) * 9, 3),
        "HR9":  round(_safe_div(HR, IP) * 9, 3),
    }
    return raw, rates


def get_mlb_hit_actuals(mlb_df, player_name, max_seasons=2):
    name = _strip_suffix(player_name)
    rows = mlb_df[mlb_df["name"] == name].sort_values("Year").head(max_seasons)
    if rows.empty:
        return None, []
    years = sorted(rows["Year"].tolist())
    PA = rows["PA"].sum(); AB = rows["AB"].sum(); H = rows["H"].sum()
    d2 = rows["2B"].sum(); d3 = rows["3B"].sum()
    HR = rows["HR"].sum(); BB = rows["BB"].sum(); SO = rows["SO"].sum()
    return {
        "BA":      _safe_div(H, AB),
        "OBP":     _safe_div(H + BB, AB + BB),
        "SLG":     _safe_div(H + d2 + 2*d3 + 3*HR, AB),
        "HR_rate": _safe_div(HR, PA),
        "BB_rate": _safe_div(BB, PA),
        "K_rate":  _safe_div(SO, PA),
    }, years


def get_mlb_pit_actuals(mlb_df, player_name, max_seasons=2):
    name = _strip_suffix(player_name)
    rows = mlb_df[mlb_df["name"] == name].sort_values("Year").head(max_seasons)
    if rows.empty:
        return None, []
    years = sorted(rows["Year"].tolist())
    IP = rows["IP"].sum()
    SO = rows["SO"].sum(); BB = rows["BB"].sum(); HR = rows["HR"].sum()
    ERA  = (rows["ERA"]  * rows["IP"]).sum() / IP
    WHIP = (rows["WHIP"] * rows["IP"]).sum() / IP
    return {
        "ERA":  round(ERA,  3),
        "WHIP": round(WHIP, 3),
        "K9":   round(_safe_div(SO, IP) * 9, 3),
        "BB9":  round(_safe_div(BB, IP) * 9, 3),
        "HR9":  round(_safe_div(HR, IP) * 9, 3),
    }, years


def compute_projections(npb_rates, age, stats, fac_df, reg_df, pit_fac_key=None):
    """Returns list of projection dicts for each stat."""
    out = []
    for stat in stats:
        npb_val = npb_rates.get(stat, np.nan)
        fac_key = (pit_fac_key or {}).get(stat, stat)

        if fac_key not in fac_df.index or (isinstance(npb_val, float) and np.isnan(npb_val)):
            continue

        frow     = fac_df.loc[fac_key]
        naive    = npb_val * frow["factor"]
        ci_lower = npb_val * frow["ci_lower"]
        ci_upper = npb_val * frow["ci_upper"]

        if stat in reg_df.index:
            rrow = reg_df.loc[stat]
            if bool(rrow["significant"]):
                reg  = rrow["intercept"] + rrow["npb_coef"] * npb_val + rrow["age_coef"] * age
                meth = "regression"
            else:
                reg, meth = naive, "naive*"
        else:
            reg, meth = naive, "naive*"

        out.append({
            "stat":     stat,
            "npb":      round(float(npb_val), 4),
            "naive":    round(float(naive),   4),
            "reg":      round(float(reg),     4),
            "method":   meth,
            "ci_lower": round(float(ci_lower), 4),
            "ci_upper": round(float(ci_upper), 4),
        })
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────────────────────────────────────

def build_bar_chart(proj_list, actuals):
    rows = []
    for p in proj_list:
        lbl = STAT_DISPLAY.get(p["stat"], p["stat"])
        rows.append({"Stat": lbl, "Value": p["naive"], "Method": "Naive Proj"})
        rows.append({"Stat": lbl, "Value": p["reg"], "Method": "Reg Proj"})
        if actuals:
            v = actuals.get(p["stat"])
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                rows.append({"Stat": lbl, "Value": v, "Method": "Actual MLB"})

    fig = px.bar(
        pd.DataFrame(rows), x="Stat", y="Value",
        color="Method", barmode="group",
        color_discrete_map=BAR_COLORS,
        template="plotly_white",
    )
    fig.update_layout(
        legend_title_text="", xaxis_title="", yaxis_title="Rate / Value",
        font=dict(size=13), height=380, margin=dict(t=20, b=20, l=0, r=0),
    )
    return fig


def build_gauge(label, value, mlb_avg, min_val, max_val, lower_is_better=False):
    if lower_is_better:
        steps = [
            {"range": [min_val,          mlb_avg * 0.85], "color": "#c8e6c9"},
            {"range": [mlb_avg * 0.85,   mlb_avg * 1.15], "color": "#fff9c4"},
            {"range": [mlb_avg * 1.15,   max_val],        "color": "#ffcdd2"},
        ]
    else:
        steps = [
            {"range": [min_val,          mlb_avg * 0.85], "color": "#ffcdd2"},
            {"range": [mlb_avg * 0.85,   mlb_avg * 1.15], "color": "#fff9c4"},
            {"range": [mlb_avg * 1.15,   max_val],        "color": "#c8e6c9"},
        ]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": f"<b>{label}</b>", "font": {"size": 14}},
        number={"font": {"size": 22}, "valueformat": ".3f"},
        gauge={
            "axis":       {"range": [min_val, max_val], "tickwidth": 1},
            "bar":        {"color": "#1f77b4", "thickness": 0.35},
            "bgcolor":    "white",
            "borderwidth": 1,
            "bordercolor": "#ccc",
            "steps": steps,
            "threshold": {
                "line": {"color": "#d32f2f", "width": 3},
                "thickness": 0.75,
                "value": mlb_avg,
            },
        },
    ))
    fig.update_layout(
        height=220, margin=dict(t=55, b=10, l=15, r=15),
        paper_bgcolor="white",
        annotations=[dict(
            text=f"<span style='color:#d32f2f'>— MLB avg: {mlb_avg}</span>",
            xref="paper", yref="paper", x=0.5, y=-0.05,
            showarrow=False, font=dict(size=11),
        )],
    )
    return fig


def render_proj_table(proj_list, actuals, stat_keys_order):
    """Render a colour-coded projection table via pandas Styler."""
    rows = []
    stat_keys = []  # parallel list — avoids a hidden _key column that breaks Styler shape
    for p in proj_list:
        stat    = p["stat"]
        actual  = (actuals or {}).get(stat)
        actual_str = f"{actual:.4f}" if actual is not None and not np.isnan(actual) else "—"
        reg_str    = f"{p['reg']:.4f}" + ("*" if p["method"] != "regression" else "")
        rows.append({
            "Stat":       STAT_DISPLAY.get(stat, stat),
            "NPB Rate":   f"{p['npb']:.4f}",
            "Naive Proj": f"{p['naive']:.4f}",
            "Reg Proj":   reg_str,
            "Actual MLB": actual_str,
            "95% CI":     f"{p['ci_lower']:.4f} – {p['ci_upper']:.4f}",
        })
        stat_keys.append(stat)

    df = pd.DataFrame(rows)

    def color_row(row):
        # row.name is the integer row index; row.index matches the df columns exactly
        key   = stat_keys[row.name]
        color = RELIABILITY_COLORS.get(RELIABILITY.get(key, "yellow"), "")
        return [f"background-color: {color}" if col == "Stat" else "" for col in row.index]

    return df.style.apply(color_row, axis=1).hide(axis="index")

# ─────────────────────────────────────────────────────────────────────────────
# Plain-English summary
# ─────────────────────────────────────────────────────────────────────────────

def _level(value, avg, lower_is_better=False):
    ratio = avg / value if lower_is_better and value else value / avg if avg else 1
    if ratio >= 1.20: return "well above-average"
    if ratio >= 1.08: return "above-average"
    if ratio >= 0.92: return "near MLB average"
    if ratio >= 0.80: return "below-average"
    return "well below-average"


def generate_summary(proj_dict, position):
    if position == "hitter":
        ba  = proj_dict.get("BA",      MLB_AVGS["BA"])
        obp = proj_dict.get("OBP",     MLB_AVGS["OBP"])
        slg = proj_dict.get("SLG",     MLB_AVGS["SLG"])
        hr  = proj_dict.get("HR_rate", MLB_AVGS["HR_rate"])
        bb  = proj_dict.get("BB_rate", MLB_AVGS["BB_rate"])
        k   = proj_dict.get("K_rate",  MLB_AVGS["K_rate"])

        contact_lvl = _level(ba,  MLB_AVGS["BA"])
        obp_lvl     = _level(obp, MLB_AVGS["OBP"])
        bb_lvl      = _level(bb,  MLB_AVGS["BB_rate"])
        k_lvl       = _level(k,   MLB_AVGS["K_rate"], lower_is_better=True)

        power_desc = (
            "above-average HR potential" if hr >= MLB_AVGS["HR_rate"] * 1.20
            else "below-average HR potential" if hr <= MLB_AVGS["HR_rate"] * 0.70
            else "near-average HR potential"
        )
        return (
            f"This player projects as a **{contact_lvl} contact hitter** with "
            f"**{obp_lvl} on-base ability** (OBP: .{round(obp*1000):03d}). "
            f"Walk rate is **{bb_lvl}** (BB/PA: {bb:.3f}); "
            f"strikeout rate is **{k_lvl}** (K/PA: {k:.3f}). "
            f"Power shows **{power_desc}** (HR/PA: {hr:.3f}, SLG: {slg:.3f}) — "
            f"*SLG and HR rate carry low model reliability; treat as a rough guide.*"
        )
    else:
        era  = proj_dict.get("ERA",  MLB_AVGS["ERA"])
        whip = proj_dict.get("WHIP", MLB_AVGS["WHIP"])
        k9   = proj_dict.get("K9",   MLB_AVGS["K9"])
        bb9  = proj_dict.get("BB9",  MLB_AVGS["BB9"])
        hr9  = proj_dict.get("HR9",  MLB_AVGS["HR9"])

        k9_lvl   = _level(k9,  MLB_AVGS["K9"])
        era_lvl  = _level(era, MLB_AVGS["ERA"],  lower_is_better=True)
        bb9_lvl  = _level(bb9, MLB_AVGS["BB9"],  lower_is_better=True)
        whip_lvl = _level(whip,MLB_AVGS["WHIP"], lower_is_better=True)
        hr_desc  = (
            "elevated" if hr9 >= MLB_AVGS["HR9"] * 1.20
            else "suppressed" if hr9 <= MLB_AVGS["HR9"] * 0.80
            else "near-average"
        )
        return (
            f"This pitcher projects with **{k9_lvl} strikeout stuff** (K/9: {k9:.1f}) — "
            f"the most reliable signal in the model (R²=0.77). "
            f"ERA projects **{era_lvl}** ({era:.2f}) with **{whip_lvl} WHIP** ({whip:.2f}). "
            f"Walk rate is **{bb9_lvl}** (BB/9: {bb9:.1f}); "
            f"HR/9 is **{hr_desc}** ({hr9:.2f}). "
            f"*ERA, WHIP, BB/9, and HR/9 carry low model reliability — use K/9 as primary signal.*"
        )

# ─────────────────────────────────────────────────────────────────────────────
# Page layout
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NPB→MLB Translation",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚾ NPB → MLB Statistical Translation")

# Load all data once
players_df           = load_players()
fac_hit, fac_pit, reg_hit, reg_pit = load_factors()
npb_hit_df, npb_pit_df = load_npb()
mlb_hit_df, mlb_pit_df = load_mlb()

# ─── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Player Lookup")
    pos_lookup = st.radio("Position", ["Hitter", "Pitcher"], horizontal=True, key="pos_lookup")

    pos_type = "hitter" if pos_lookup == "Hitter" else "pitcher"
    player_pool = players_df[players_df["position_type"] == pos_type]

    def player_label(row):
        return f"{row['name']} | {row['npb_team']} → {row['mlb_team']}"

    player_labels = player_pool.apply(player_label, axis=1).tolist()
    player_names  = player_pool["name"].tolist()

    selected_label = st.selectbox("Player", player_labels, key="player_sel")
    selected_name  = player_names[player_labels.index(selected_label)]

    st.divider()

    st.header("Custom Projection")
    pos_custom = st.radio("Position", ["Hitter", "Pitcher"], horizontal=True, key="pos_custom")

    debut_age = st.slider("Age at MLB debut", min_value=20, max_value=35, value=27)

    st.subheader("NPB Input Stats")
    if pos_custom == "Hitter":
        c_ba  = st.slider("BA",      0.200, 0.400, 0.280, 0.001, format="%.3f")
        c_obp = st.slider("OBP",     0.250, 0.480, 0.340, 0.001, format="%.3f")
        c_slg = st.slider("SLG",     0.300, 0.650, 0.430, 0.001, format="%.3f")
        c_hr  = st.slider("HR/PA",   0.000, 0.100, 0.030, 0.001, format="%.3f")
        c_bb  = st.slider("BB/PA",   0.030, 0.200, 0.080, 0.001, format="%.3f")
        c_k   = st.slider("K/PA",    0.050, 0.350, 0.160, 0.001, format="%.3f")
        custom_npb = {"BA": c_ba, "OBP": c_obp, "SLG": c_slg,
                      "HR_rate": c_hr, "BB_rate": c_bb, "K_rate": c_k}
    else:
        c_era  = st.slider("ERA",  1.50, 6.00, 2.80, 0.01, format="%.2f")
        c_whip = st.slider("WHIP", 0.80, 1.60, 1.10, 0.01, format="%.2f")
        c_k9   = st.slider("K/9",  4.0,  12.0, 7.5,  0.1,  format="%.1f")
        c_bb9  = st.slider("BB/9", 1.0,   5.0, 2.5,  0.1,  format="%.1f")
        c_hr9  = st.slider("HR/9", 0.1,   1.5, 0.5,  0.1,  format="%.1f")
        custom_npb = {"ERA": c_era, "WHIP": c_whip, "K9": c_k9,
                      "BB9": c_bb9, "HR9": c_hr9}

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["🔍 Player Lookup", "⚙️ Custom Projection"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Player Lookup
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    is_hitter = (pos_lookup == "Hitter")

    # Fetch NPB profile
    if is_hitter:
        raw_npb, npb_rates = get_npb_hit_profile(npb_hit_df, selected_name)
    else:
        raw_npb, npb_rates = get_npb_pit_profile(npb_pit_df, selected_name)

    if npb_rates is None:
        st.warning(f"No NPB data found for **{selected_name}**.")
        st.stop()

    age       = npb_rates["npb_final_age"]
    npb_years = npb_rates["npb_years"]
    size_label = f"PA: {npb_rates['npb_PA']}" if is_hitter else f"IP: {npb_rates['npb_IP']}"

    # Fetch MLB actuals
    if is_hitter:
        actuals, actual_years = get_mlb_hit_actuals(mlb_hit_df, selected_name)
    else:
        actuals, actual_years = get_mlb_pit_actuals(mlb_pit_df, selected_name)

    # Compute projections
    if is_hitter:
        proj_list = compute_projections(npb_rates, age, HIT_STATS, fac_hit, reg_hit)
    else:
        proj_list = compute_projections(npb_rates, age, PIT_STATS, fac_pit, reg_pit, PIT_FAC_KEY)

    # ── SECTION A: NPB Profile ──────────────────────────────────────────────
    st.subheader(f"A — NPB Profile: {selected_name}")

    player_info = players_df[players_df["name"] == selected_name].iloc[0]
    col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
    col_meta1.metric("NPB Team",       player_info["npb_team"])
    col_meta2.metric("MLB Team",       player_info["mlb_team"])
    col_meta3.metric("Dataset",        player_info["dataset"].capitalize())
    col_meta4.metric("Age at Crossover", f"{age}")

    st.caption(
        f"NPB window: {npb_years}  |  {size_label}  |  "
        + (f"MLB actuals: {actual_years}" if actual_years else "No MLB data available yet")
    )

    st.dataframe(
        raw_npb.style.format({
            col: "{:.3f}" for col in raw_npb.select_dtypes("float").columns
        }),
        width='stretch', hide_index=True,
    )

    st.divider()

    # ── Pooled NPB rate summary ──────────────────────────────────────────────
    stats_list  = HIT_STATS if is_hitter else PIT_STATS
    npb_summary = {STAT_DISPLAY.get(s, s): f"{npb_rates[s]:.4f}" for s in stats_list if s in npb_rates}
    st.caption("Pooled NPB rates (last 2 seasons):")
    st.dataframe(
        pd.DataFrame([npb_summary]),
        width='stretch', hide_index=True,
    )

    st.divider()

    # ── SECTION B: Projection vs Actual ─────────────────────────────────────
    st.subheader("B — Projection vs Actual")

    if player_info["dataset"] == "core":
        st.info(
            "ℹ️ This player is in the **core training set** — projections are in-sample "
            "and will naturally be closer to actuals than for held-out players.",
            icon=None,
        )

    st.plotly_chart(build_bar_chart(proj_list, actuals), width='stretch')

    st.caption("\\* = regression model not significant; naive (factor) projection used instead")

    styled_tbl = render_proj_table(proj_list, actuals, stats_list)
    st.dataframe(styled_tbl, width='stretch', hide_index=True)

    st.divider()

    # ── SECTION C: Reliability Guide ─────────────────────────────────────────
    with st.expander("C — Reliability Guide", expanded=False):
        st.markdown("**Color coding in the table above:**")
        cols_leg = st.columns(3)
        cols_leg[0].markdown(
            "<span style='background:#d4edda;padding:2px 8px;border-radius:3px'>"
            "🟢 Green</span> = High reliability (R²≥0.57, significant)",
            unsafe_allow_html=True,
        )
        cols_leg[1].markdown(
            "<span style='background:#fff3cd;padding:2px 8px;border-radius:3px'>"
            "🟡 Yellow</span> = Moderate reliability",
            unsafe_allow_html=True,
        )
        cols_leg[2].markdown(
            "<span style='background:#f8d7da;padding:2px 8px;border-radius:3px'>"
            "🔴 Red</span> = Low reliability (not significant)",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        show_stats = (HIT_STATS if is_hitter else PIT_STATS)
        for stat in show_stats:
            icon, desc = RELIABILITY_INFO[stat]
            st.markdown(f"**{icon} {STAT_DISPLAY.get(stat, stat)}** — {desc}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Custom Projection
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    is_hit_custom = (pos_custom == "Hitter")
    stats_custom  = HIT_STATS if is_hit_custom else PIT_STATS
    fac_custom    = fac_hit   if is_hit_custom else fac_pit
    reg_custom    = reg_hit   if is_hit_custom else reg_pit
    fac_key_map   = None      if is_hit_custom else PIT_FAC_KEY

    proj_custom = compute_projections(
        custom_npb, debut_age, stats_custom,
        fac_custom, reg_custom, fac_key_map,
    )

    # Projection dict {stat: reg_proj} for summary
    proj_vals = {p["stat"]: p["reg"] for p in proj_custom}

    st.subheader("Projection Table")
    st.caption(
        f"Position: **{pos_custom}**  |  Age at debut: **{debut_age}**  |  "
        "95% CI based on translation-factor uncertainty across historical player sample."
    )
    st.caption("\\* = regression not significant; naive (factor) projection used")
    st.dataframe(
        render_proj_table(proj_custom, None, stats_custom),
        width='stretch', hide_index=True,
    )

    st.divider()

    # ── Gauge indicators for 3 most reliable stats ──────────────────────────
    if is_hit_custom:
        gauge_stats = ["K_rate", "BB_rate", "OBP"]
        gauge_labels = {"K_rate": "K/PA", "BB_rate": "BB/PA", "OBP": "OBP"}
    else:
        gauge_stats = ["K9", "ERA", "WHIP"]
        gauge_labels = {"K9": "K/9", "ERA": "ERA", "WHIP": "WHIP"}

    st.subheader("Key Metrics vs MLB Average")
    st.caption(
        "Red line = MLB league average. "
        "🟢 = better than avg, 🟡 = near avg, 🔴 = worse than avg"
    )
    g_cols = st.columns(3)
    for i, stat in enumerate(gauge_stats):
        if stat not in proj_vals:
            continue
        min_v, max_v, lower_better = GAUGE_CFG[stat]
        fig = build_gauge(
            label          = gauge_labels[stat],
            value          = proj_vals[stat],
            mlb_avg        = MLB_AVGS[stat],
            min_val        = min_v,
            max_val        = max_v,
            lower_is_better= lower_better,
        )
        g_cols[i].plotly_chart(fig, width='stretch')

    st.divider()

    # ── Plain-English summary ────────────────────────────────────────────────
    st.subheader("Scouting Summary")
    st.markdown(generate_summary(proj_vals, "hitter" if is_hit_custom else "pitcher"))

    st.caption(
        "Projections use translation factors (v2, winsorized) for non-significant stats "
        "and OLS regression for significant ones (K/PA, BB/PA, OBP, BA for hitters; "
        "K/9 for pitchers). Based on data from 11 core hitters and 13 core pitchers."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Built with data from Baseball Reference | NPB-MLB Translation Project"
)
