"""
scrape_npb_stats.py
-------------------
Scrapes NPB seasonal stats for each player from Baseball Reference's
international register (register/player.fcgi?id=<register_id>).

Saves two CSVs:
  - npb_hitting_stats_raw.csv
  - npb_pitching_stats_raw.csv

Usage:
    python scrape_npb_stats.py

Setup:
    Fill in REGISTER_ID_MAP below with each player's register ID.
    Find IDs by searching: https://www.baseball-reference.com/register/
    The URL format is: register/player.fcgi?id=<register_id>
    Example: Nomo  -> nomo--001hid
             Ohtani -> otani-000sho

Notes:
    - 4-second delay between requests to avoid rate limiting (~2 min runtime)
    - Run from the same directory as npb_mlb_player_list.csv
    - Ohtani is handled automatically — both hitting and pitching tables scraped
    - If a player returns no data, run with DEBUG=True to inspect table IDs
"""

import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
import time

PLAYER_LIST = "npb_mlb_player_list.csv"
HEADERS     = {"User-Agent": "Mozilla/5.0"}
DELAY       = 4    # seconds between requests
DEBUG       = True  # set True to print table IDs found on each page

# ============================================================
# REGISTER ID MAP
# Fill in each player's Baseball Reference register ID.
# Find at: https://www.baseball-reference.com/register/
# Format:  bbref_id -> register_id
# ============================================================
REGISTER_ID_MAP = {
    # --- PITCHERS ---
    "nomohi01":   "nomo--001hid",   # Hideo Nomo        ✓ confirmed
    "irabuhi01":  "irabu-001hid",               # Hideki Irabu
    "yoshima01":  "yoshii001mas",               # Masato Yoshii
    "hasegsh01":  "hasega001shi",               # Shigetoshi Hasegawa
    "sasakka01":  "sasaki001kaz",               # Kazuhiro Sasaki
    "otsukak01": "otsuka001aki",               # Akinori Otsuka
    "matsuda01":  "matsuz001dai",               # Daisuke Matsuzaka
    "kurodhi01":  "kuroda001hir",               # Hiroki Kuroda
    "igarary01":  "igaras001ryo",               # Ryota Igarashi
    "iwakuhi01":  "iwakum001his",               # Hisashi Iwakuma
    "darviyu01":  "darvis001yu-",               # Yu Darvish
    "tanakma01":  "tanaka003mas",               # Masahiro Tanaka
    "sengako01":  "senga-000kod",               # Kodai Senga        (validation)
    "yamamyo01":  "yamamo004yos",               # Yoshinobu Yamamoto (validation)
    "imaita01":   "imai--000tat",               # Tatsuya Imai       (validation)
    # --- HITTERS ---
    "suzukic01":  "suzuki001ich",               # Ichiro Suzuki
    "matsuhi01":  "matsui001hid",               # Hideki Matsui
    "shinjts01":  "shinjo001tsu",               # Tsuyoshi Shinjo
    "tagucso01":  "taguch001so-",               # So Taguchi
    "matsuka01":  "matsui001kaz",               # Kazuo Matsui
    "iwamuak01":  "iwamur001aki",               # Akinori Iwamura
    "fukudko01":  "fukudo001kos",               # Kosuke Fukudome
    "aokino01":   "aoki--001nor",               # Nori Aoki
    "tsutsyo01":  "tsutsu000yos",               # Yoshi Tsutsugo
    "suzukse01":  "suzuki001sei",               # Seiya Suzuki
    "murakmu01":  "muraka000mun",               # Munetaka Murakami  (validation)
    # --- TWO-WAY ---
    "ohtansh01":  "otani-000sho",   # Shohei Ohtani      ✓ confirmed
}

# Key NPB hitting stat columns to keep
# These are standard Baseball Reference register column names
NPB_HIT_COLS = [
    "year_ID", "age", "team_ID", "lg_ID", "G", "PA", "AB", "R", "H",
    "2B", "3B", "HR", "RBI", "BB", "SO", "batting_avg", "onbase_perc", "slugging_perc", "onbase_plus_slugging"
]
 
# Key NPB pitching stat columns to keep
NPB_PIT_COLS = [
    "year_ID", "age", "team_ID", "lg_ID", "W", "L", "earned_run_avg", "G", "GS",
    "IP", "H", "HR", "BB", "SO", "whip"
]
 
 
def get_register_page(register_id: str) -> BeautifulSoup | None:
    """Fetch and parse a Baseball Reference register player page."""
    url = f"https://www.baseball-reference.com/register/player.fcgi?id={register_id}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return BeautifulSoup(r.content, "html.parser")
    except Exception as e:
        print(f"  ERROR fetching {register_id}: {e}")
        return None
 
 
def debug_tables(soup: BeautifulSoup, name: str):
    """Print all table IDs found on the page, including inside HTML comments."""
    print(f"  [{name}] Tables found on page:")
    for t in soup.find_all("table"):
        print(f"    - {t.get('id', '(no id)')} (visible)")
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment_soup = BeautifulSoup(comment, "html.parser")
        for t in comment_soup.find_all("table"):
            tid = t.get("id")
            if tid:
                print(f"    - {tid} (in comment)")
 
 
def extract_stats(soup: BeautifulSoup, register_id: str, name: str,
                  table_id: str, col_list: list) -> pd.DataFrame:
    """
    Generic extractor — finds a table by ID and pulls all NPB rows.
    Works for both hitting and pitching tables.
    """
    rows = []
 
    # Baseball Reference wraps secondary tables in HTML comments.
    # Try direct find first, then search inside comments.
    table = soup.find("table", {"id": table_id})
    if not table:
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for comment in comments:
            comment_soup = BeautifulSoup(comment, "html.parser")
            table = comment_soup.find("table", {"id": table_id})
            if table:
                break
 
    if not table:
        print(f"  No table '{table_id}' found for {name}")
        return pd.DataFrame()
 
    tbody = table.find("tbody")
    if not tbody:
        print(f"  No tbody in table '{table_id}' for {name}")
        return pd.DataFrame()
 
    # Collect all league labels seen (for debugging)
    leagues_seen = set()
 
    for tr in tbody.find_all("tr"):
        # Skip section header rows
        if tr.get("class") and any(c in tr.get("class") for c in ["thead", "spacer"]):
            continue
 
        # Find league cell — register pages use "lg_ID" data-stat
        lg_td = tr.find("td", {"data-stat": "lg_ID"})
        if not lg_td:
            continue
 
        lg = lg_td.text.strip()
        leagues_seen.add(lg)
 
        # Keep only NPB rows
        # Register pages may use "NPB" or "Jpn" depending on era
        if lg not in ("JPPL", "JPCL"):
            continue
 
        row = {"register_id": register_id, "name": name}
        for stat in col_list:
            td = tr.find("td", {"data-stat": stat}) or tr.find("th", {"data-stat": stat})
            row[stat] = td.text.strip() if td else None
 
        rows.append(row)
 
    if DEBUG:
        print(f"  Leagues seen: {leagues_seen}")
 
    return pd.DataFrame(rows)
 
 
def main():
    players = pd.read_csv(PLAYER_LIST)
 
    # Deduplicate on bbref_id — Ohtani appears twice (P and H),
    # we fetch his page once and extract both tables
    unique_players = players.drop_duplicates(subset="bbref_id")[
        ["bbref_id", "name", "position_type"]
    ]
 
    hitting_frames  = []
    pitching_frames = []
    skipped         = []
 
    for _, row in unique_players.iterrows():
        bbref_id      = row["bbref_id"]
        name          = row["name"].replace(" (P)", "").replace(" (H)", "")
        position_type = row["position_type"]
 
        register_id = REGISTER_ID_MAP.get(bbref_id, "")
        if not register_id:
            print(f"SKIP {name} — register ID not filled in yet")
            skipped.append(name)
            continue
 
        print(f"Fetching: {name} ({register_id})")
        soup = get_register_page(register_id)
 
        if soup is None:
            time.sleep(DELAY)
            continue
 
        if DEBUG:
            debug_tables(soup, name)
 
        # Fetch hitting stats
        if position_type == "hitter" or bbref_id == "ohtansh01":
            df = extract_stats(soup, register_id, name,
                               table_id="standard_batting",
                               col_list=NPB_HIT_COLS)
            if not df.empty:
                print(f"  Found {len(df)} NPB hitting seasons")
                hitting_frames.append(df)
 
        # Fetch pitching stats
        if position_type == "pitcher":
            df = extract_stats(soup, register_id, name,
                               table_id="standard_pitching",
                               col_list=NPB_PIT_COLS)
            if not df.empty:
                print(f"  Found {len(df)} NPB pitching seasons")
                pitching_frames.append(df)
 
        # Ohtani pitching: separate register entry
        if bbref_id == "ohtansh01":
            ohtani_pitch_id = REGISTER_ID_MAP.get("ohtansh01_pitch", "")
            if ohtani_pitch_id:
                print(f"  Fetching Ohtani pitching register ({ohtani_pitch_id})")
                soup_p = get_register_page(ohtani_pitch_id)
                if soup_p:
                    df = extract_stats(soup_p, ohtani_pitch_id, name,
                                       table_id="standard_pitching",
                                       col_list=NPB_PIT_COLS)
                    if not df.empty:
                        print(f"  Found {len(df)} NPB pitching seasons (Ohtani)")
                        pitching_frames.append(df)
                time.sleep(DELAY)
            else:
                print("  SKIP Ohtani pitching — register ID not filled in yet")
 
        time.sleep(DELAY)
 
    # Save outputs
    if hitting_frames:
        hits_df = pd.concat(hitting_frames, ignore_index=True)
        hits_df.to_csv("npb_hitting_stats_raw.csv", index=False)
        print(f"\nSaved npb_hitting_stats_raw.csv — {len(hits_df)} rows")
    else:
        print("\nNo hitting data collected.")
 
    if pitching_frames:
        pit_df = pd.concat(pitching_frames, ignore_index=True)
        pit_df.to_csv("npb_pitching_stats_raw.csv", index=False)
        print(f"Saved npb_pitching_stats_raw.csv — {len(pit_df)} rows")
    else:
        print("No pitching data collected.")
 
    if skipped:
        print(f"\nSkipped {len(skipped)} players (register ID missing):")
        for s in skipped:
            print(f"  - {s}")
 
 
if __name__ == "__main__":
    main()
