"""
scrape_mlb_stats.py
-------------------
Scrapes MLB career stats from Baseball Reference main player pages.

URL pattern:
    https://www.baseball-reference.com/players/{first_letter}/{bbref_id}.shtml

Tables targeted:
    - batting_standard  (hitters + Ohtani)
    - pitching_standard (pitchers + Ohtani)

Only keeps rows where:
    - lg_ID is "AL" or "NL"
    - Year >= MLB debut year (npb_seasons end + 1)

Saves:
    - mlb_hitting_stats_raw.csv
    - mlb_pitching_stats_raw.csv

Notes:
    - 4-second delay between requests
    - BBref wraps most tables in HTML comments; handled via BeautifulSoup Comment
    - Ohtani: single page fetch, both batting and pitching tables extracted
    - Validation players (2026 rookies) will return 0 rows gracefully
"""

import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
import time

PLAYER_LIST = "npb_mlb_player_list.csv"
HEADERS     = {"User-Agent": "Mozilla/5.0"}
DELAY       = 4
DEBUG       = False

MLB_LEAGUES = {"AL", "NL"}

# Main player pages use different data-stat names from register pages:
#   league  → comp_name_abbr   (not lg_ID)
#   year    → year_id          (lowercase, not year_ID)
#   team    → team_name_abbr   (not team_ID)
#   hitting stats have b_ prefix, pitching stats have p_ prefix
STAT_LEAGUE = "comp_name_abbr"
STAT_YEAR   = "year_id"
STAT_TEAM   = "team_name_abbr"

MLB_HIT_STATS = [
    "year_id", "age", "team_name_abbr", "comp_name_abbr",
    "b_games", "b_pa", "b_ab", "b_r", "b_h",
    "b_doubles", "b_triples", "b_hr", "b_rbi", "b_bb", "b_so",
    "b_batting_avg", "b_onbase_perc", "b_slugging_perc", "b_onbase_plus_slugging",
]

MLB_PIT_STATS = [
    "year_id", "age", "team_name_abbr", "comp_name_abbr",
    "p_w", "p_l", "p_earned_run_avg", "p_g", "p_gs",
    "p_ip", "p_h", "p_hr", "p_bb", "p_so", "p_whip",
]

# Rename to user-facing column headers
HIT_RENAME = {
    "year_id": "Year", "age": "Age", "team_name_abbr": "Tm", "comp_name_abbr": "Lg",
    "b_games": "G", "b_pa": "PA", "b_ab": "AB", "b_r": "R", "b_h": "H",
    "b_doubles": "2B", "b_triples": "3B", "b_hr": "HR", "b_rbi": "RBI",
    "b_bb": "BB", "b_so": "SO",
    "b_batting_avg": "BA", "b_onbase_perc": "OBP",
    "b_slugging_perc": "SLG", "b_onbase_plus_slugging": "OPS",
}

PIT_RENAME = {
    "year_id": "Year", "age": "Age", "team_name_abbr": "Tm", "comp_name_abbr": "Lg",
    "p_w": "W", "p_l": "L", "p_earned_run_avg": "ERA",
    "p_g": "G", "p_gs": "GS", "p_ip": "IP",
    "p_h": "H", "p_hr": "HR", "p_bb": "BB", "p_so": "SO", "p_whip": "WHIP",
}


def get_player_page(bbref_id: str) -> BeautifulSoup | None:
    first = bbref_id[0]
    url = f"https://www.baseball-reference.com/players/{first}/{bbref_id}.shtml"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return BeautifulSoup(r.content, "html.parser")
    except Exception as e:
        print(f"  ERROR fetching {bbref_id}: {e}")
        return None


def debug_tables(soup: BeautifulSoup, name: str):
    """Print all table IDs on the page, including those inside HTML comments."""
    print(f"  [{name}] Tables found:")
    for t in soup.find_all("table"):
        print(f"    - {t.get('id', '(no id)')} (visible)")
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment_soup = BeautifulSoup(comment, "html.parser")
        for t in comment_soup.find_all("table"):
            tid = t.get("id")
            if tid:
                print(f"    - {tid} (in comment)")


def extract_mlb_stats(
    soup: BeautifulSoup,
    bbref_id: str,
    name: str,
    table_id: str,
    stat_cols: list,
    mlb_debut_year: int,
) -> pd.DataFrame:
    """
    Pull rows from a BBref table. Keeps only AL/NL rows at or after
    mlb_debut_year. Searches inside HTML comments when the table isn't
    rendered directly (BBref standard behaviour for secondary tables).
    """
    # Direct find first; fall back to comment search
    table = soup.find("table", {"id": table_id})
    if not table:
        for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
            comment_soup = BeautifulSoup(comment, "html.parser")
            table = comment_soup.find("table", {"id": table_id})
            if table:
                break

    if not table:
        print(f"  No table '{table_id}' found for {name}")
        return pd.DataFrame()

    tbody = table.find("tbody")
    if not tbody:
        return pd.DataFrame()

    rows = []
    leagues_seen = set()

    for tr in tbody.find_all("tr"):
        tr_classes = tr.get("class") or []
        if any(c in tr_classes for c in ("thead", "spacer", "partial_table")):
            continue

        # Main player pages use comp_name_abbr for league, not lg_ID
        lg_td = tr.find("td", {"data-stat": STAT_LEAGUE})
        if not lg_td:
            continue
        lg = lg_td.text.strip()
        leagues_seen.add(lg)

        if lg not in MLB_LEAGUES:
            continue

        # year_id (lowercase) can be a <th> on some rows
        year_cell = (tr.find("td", {"data-stat": STAT_YEAR})
                     or tr.find("th", {"data-stat": STAT_YEAR}))
        if not year_cell:
            continue
        year_text = year_cell.text.strip().replace("*", "").replace("+", "")
        try:
            year = int(year_text)
        except ValueError:
            continue  # "Career", "162-Game Avg.", etc.

        if year < mlb_debut_year:
            continue

        row = {"bbref_id": bbref_id, "name": name}
        for stat in stat_cols:
            cell = tr.find("td", {"data-stat": stat}) or tr.find("th", {"data-stat": stat})
            row[stat] = cell.text.strip() if cell else None

        rows.append(row)

    if DEBUG:
        print(f"  Leagues seen in '{table_id}': {leagues_seen}")

    return pd.DataFrame(rows)


def main():
    players = pd.read_csv(PLAYER_LIST)

    # Derive MLB debut year: end year of npb_seasons range + 1
    players["mlb_debut_year"] = (
        players["npb_seasons"].str.split("-").str[1].astype(int) + 1
    )

    # Deduplicate on bbref_id so Ohtani's page is only fetched once.
    # Both batting and pitching tables are extracted from his single page.
    unique_players = players.drop_duplicates(subset="bbref_id")

    hitting_frames  = []
    pitching_frames = []

    for _, row in unique_players.iterrows():
        bbref_id       = row["bbref_id"]
        name           = row["name"].replace(" (P)", "").replace(" (H)", "")
        position_type  = row["position_type"]
        mlb_debut_year = int(row["mlb_debut_year"])

        print(f"Fetching: {name} ({bbref_id}), MLB debut {mlb_debut_year}")
        soup = get_player_page(bbref_id)

        if soup is None:
            time.sleep(DELAY)
            continue

        if DEBUG:
            debug_tables(soup, name)

        # Hitting: hitters and Ohtani
        if position_type == "hitter" or bbref_id == "ohtansh01":
            df = extract_mlb_stats(
                soup, bbref_id, name,
                table_id="players_standard_batting",
                stat_cols=MLB_HIT_STATS,
                mlb_debut_year=mlb_debut_year,
            )
            if not df.empty:
                print(f"  {len(df)} MLB hitting seasons")
                hitting_frames.append(df)
            else:
                print(f"  No MLB hitting data")

        # Pitching: pitchers and Ohtani
        if position_type == "pitcher" or bbref_id == "ohtansh01":
            df = extract_mlb_stats(
                soup, bbref_id, name,
                table_id="players_standard_pitching",
                stat_cols=MLB_PIT_STATS,
                mlb_debut_year=mlb_debut_year,
            )
            if not df.empty:
                print(f"  {len(df)} MLB pitching seasons")
                pitching_frames.append(df)
            else:
                print(f"  No MLB pitching data")

        time.sleep(DELAY)

    if hitting_frames:
        hits_df = pd.concat(hitting_frames, ignore_index=True)
        hits_df.rename(columns=HIT_RENAME, inplace=True)
        hits_df.to_csv("mlb_hitting_stats_raw.csv", index=False)
        print(f"\nSaved mlb_hitting_stats_raw.csv — {len(hits_df)} rows")
    else:
        print("\nNo hitting data collected.")

    if pitching_frames:
        pit_df = pd.concat(pitching_frames, ignore_index=True)
        pit_df.rename(columns=PIT_RENAME, inplace=True)
        pit_df.to_csv("mlb_pitching_stats_raw.csv", index=False)
        print(f"Saved mlb_pitching_stats_raw.csv — {len(pit_df)} rows")
    else:
        print("No pitching data collected.")


if __name__ == "__main__":
    main()
