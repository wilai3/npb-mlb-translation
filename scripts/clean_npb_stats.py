import pandas as pd
import sqlite3

# ── Load raw data ──────────────────────────────────────────────────────────
players = pd.read_csv('npb_mlb_player_list.csv')
hitting_raw = pd.read_csv('npb_hitting_stats_raw.csv')
pitching_raw = pd.read_csv('npb_pitching_stats_raw.csv')

print(f"Loaded: {len(hitting_raw)} hitting rows, {len(pitching_raw)} pitching rows, {len(players)} players")

# ── Compute MLB debut year ─────────────────────────────────────────────────
# npb_seasons like "1990-1994" → last NPB year 1994 → MLB debut 1995
players['mlb_debut_year'] = players['npb_seasons'].str.split('-').str[1].astype(int) + 1

# Strip "(P)" / "(H)" from Ohtani's two entries before building the name map
players['name_norm'] = players['name'].str.replace(r'\s*\([PH]\)$', '', regex=True).str.strip()

# name → debut year (Ohtani appears twice with identical debut year; drop_duplicates is safe)
name_debut = players.drop_duplicates('name_norm').set_index('name_norm')['mlb_debut_year']

# ── HITTING ───────────────────────────────────────────────────────────────
h = hitting_raw.copy()
h['mlb_debut_year'] = h['name'].map(name_debut)

hitting_raw_n = len(h)

# Step 1: keep only pre-MLB seasons
after_debut_filter_h = h[h['year_ID'] < h['mlb_debut_year']]
dropped_post_mlb_h = hitting_raw_n - len(after_debut_filter_h)

# Step 2: drop PA < 10
h = after_debut_filter_h[after_debut_filter_h['PA'] >= 10].copy()
dropped_pa_h = len(after_debut_filter_h) - len(h)

# Step 3: add years_before_mlb
h['years_before_mlb'] = h['mlb_debut_year'] - h['year_ID']

hitting_clean_n = len(h)
h.to_csv('npb_hitting_stats_clean.csv', index=False)

# ── PITCHING ──────────────────────────────────────────────────────────────
p = pitching_raw.copy()
p['mlb_debut_year'] = p['name'].map(name_debut)

pitching_raw_n = len(p)

# Convert IP to float for numeric comparison (innings pitched stored as "235.0", "0.0", etc.)
p['IP'] = pd.to_numeric(p['IP'], errors='coerce')

# Step 1: keep only pre-MLB seasons
after_debut_filter_p = p[p['year_ID'] < p['mlb_debut_year']]
dropped_post_mlb_p = pitching_raw_n - len(after_debut_filter_p)

# Step 2: drop IP = 0 (noise / zero-out appearances)
after_ip_filter = after_debut_filter_p[after_debut_filter_p['IP'] > 0]
dropped_ip_p = len(after_debut_filter_p) - len(after_ip_filter)

# Step 3: fill GS nulls with 0 (relief pitchers have no starts recorded)
after_ip_filter = after_ip_filter.copy()
after_ip_filter['GS'] = after_ip_filter['GS'].fillna(0).astype(int)

# Step 4: drop the Matsuzaka row where earned_run_avg is null
matsuzaka_null = (after_ip_filter['name'] == 'Daisuke Matsuzaka') & after_ip_filter['earned_run_avg'].isna()
dropped_matsuzaka = matsuzaka_null.sum()
p = after_ip_filter[~matsuzaka_null].copy()

# Step 5: add years_before_mlb
p['years_before_mlb'] = p['mlb_debut_year'] - p['year_ID']

pitching_clean_n = len(p)
p.to_csv('npb_pitching_stats_clean.csv', index=False)

# ── SQLite ────────────────────────────────────────────────────────────────
conn = sqlite3.connect('npb_mlb.db')
h.to_sql('npb_hitting',  conn, if_exists='replace', index=False)
p.to_sql('npb_pitching', conn, if_exists='replace', index=False)
players.to_sql('players', conn, if_exists='replace', index=False)
conn.close()

# ── Summary ───────────────────────────────────────────────────────────────
SEP = '=' * 62

print(f'\n{SEP}')
print('ROWS BEFORE / AFTER CLEANING')
print(SEP)
print(f'  Hitting:  {hitting_raw_n:>4} raw  →  {hitting_clean_n:>3} clean'
      f'  (post-MLB: -{dropped_post_mlb_h}, PA<10: -{dropped_pa_h})')
print(f'  Pitching: {pitching_raw_n:>4} raw  →  {pitching_clean_n:>3} clean'
      f'  (post-MLB: -{dropped_post_mlb_p}, IP=0: -{dropped_ip_p},'
      f' Matsuzaka null ERA: -{dropped_matsuzaka})')

print(f'\n{SEP}')
print('HITTING — seasons kept per player')
print(SEP)
h_summary = (
    h.groupby('name')['year_ID']
     .agg(seasons='count', first_year='min', last_year='max')
     .sort_values('seasons', ascending=False)
)
print(h_summary.to_string())

print(f'\n{SEP}')
print('PITCHING — seasons kept per player')
print(SEP)
p_summary = (
    p.groupby('name')['year_ID']
     .agg(seasons='count', first_year='min', last_year='max')
     .sort_values('seasons', ascending=False)
)
print(p_summary.to_string())

print(f'\n{SEP}')
print('SQLite tables in npb_mlb.db')
print(SEP)
conn = sqlite3.connect('npb_mlb.db')
for tbl, df in [('npb_hitting', h), ('npb_pitching', p), ('players', players)]:
    n = conn.execute(f'SELECT COUNT(*) FROM {tbl}').fetchone()[0]
    print(f'  {tbl:<15} {n:>4} rows')
conn.close()

print(f'\nOutputs: npb_hitting_stats_clean.csv, npb_pitching_stats_clean.csv, npb_mlb.db')
