"""
fetch_mlb_stats.py
------------------
Fetches MLB seasonal stats for each player in npb_mlb_player_list.csv
using pybaseball (FanGraphs). Saves two CSVs:
  - mlb_hitting_stats_raw.csv
  - mlb_pitching_stats_raw.csv

Usage:
    pip install pybaseball
    python fetch_mlb_stats.py

Notes:
    - pybaseball pulls from FanGraphs. Rate limiting is handled internally.
    - We filter to only include seasons AFTER each player's MLB debut
      (i.e., their post-NPB MLB career), not any NPB seasons.
    - Ohtani is handled as two entries: hitting and pitching.
    - FanGraphs player IDs differ from bbref IDs. The ID mapping below
      was manually verified. If a player returns no data, double-check
      their FanGraphs ID at fangraphs.com/players.
"""

import requests

# FanGraphs blocks Python's default User-Agent with a 403.
# Monkey-patch requests.get before pybaseball loads so every outgoing
# call carries a browser-like header.
_orig_get = requests.get
def _get_with_ua(url, **kwargs):
    headers = kwargs.pop('headers', None) or {}
    headers.setdefault(
        'User-Agent',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    )
    return _orig_get(url, headers=headers, **kwargs)
requests.get = _get_with_ua

import pandas as pd
from pybaseball import batting_stats, pitching_stats, cache
import time

cache.enable()  # cache FanGraphs responses to disk; avoids duplicate fetches

PLAYER_LIST = "npb_mlb_player_list.csv"

# Manual FanGraphs ID mapping (bbref_id -> fangraphs_id)
# Verify at: https://www.fangraphs.com/players/<name>/<id>
FANGRAPHS_ID_MAP = {
    # Pitchers
    "nomohi01":   "666",   # Hideo Nomo
    "irabuhi01":  "1255",   # Hideki Irabu
    "yoshima01":  "807",   # Masato Yoshii
    "hasegsh01":  "1082",   # Shigetoshi Hasegawa
    "sasakka01":  "1098",   # Kazuhiro Sasaki
    "otsukak01": "1895",   # Akinori Otsuka
    "matsuda01":  "7775",   # Daisuke Matsuzaka
    "kurodhi01":  "3283",   # Hiroki Kuroda
    "igarary01":  "10232",   # Ryota Igarashi
    "iwakuhi01":  "13048",   # Hisashi Iwakuma
    "darviyu01":  "13074",   # Yu Darvish
    "tanakma01":  "15764",   # Masahiro Tanaka
    "sengako01":  "31838",   # Kodai Senga
    "yamamyo01":  "33825",   # Yoshinobu Yamamoto
    "imaita01":   "37124",   # Tatsuya Imai
    # Hitters
    "suzukic01":  "1101",   # Ichiro Suzuki
    "matsuhi01":  "1659",   # Hideki Matsui
    "shinjts01":  "1132",   # Tsuyoshi Shinjo
    "tagucso01":  "1186",   # So Taguchi
    "matsuka01":  "1854",   # Kazuo Matsui
    "iwamuak01":  "7781",   # Akinori Iwamura
    "fukudko01":  "3263",   # Kosuke Fukudome
    "aokino01":   "13075",   # Nori Aoki
    "tsutsyo01":  "27459",   # Yoshi Tsutsugo
    "suzukse01":  "30116",   # Seiya Suzuki
    "murakmu01":  "37120",   # Munetaka Murakami
    # Ohtani (two-way)
    "ohtansh01":  "19755",   # Shohei Ohtani
}


def get_fangraphs_ids():
    """
    Alternative: use pybaseball's playerid_lookup to find FanGraphs IDs
    programmatically instead of the manual map above.
    """
    from pybaseball import playerid_lookup

    players = pd.read_csv(PLAYER_LIST)
    results = []

    for _, row in players.iterrows():
        name = row["name"].replace(" (P)", "").replace(" (H)", "")
        parts = name.split()
        last, first = parts[-1], parts[0]

        try:
            lookup = playerid_lookup(last, first)
            if not lookup.empty:
                fg_id = lookup.iloc[0].get("key_fangraphs")
                print(f"  {name}: fg_id = {fg_id}")
                results.append({"name": name, "bbref_id": row["bbref_id"], "fg_id": fg_id})
            else:
                print(f"  {name}: NOT FOUND")
                results.append({"name": name, "bbref_id": row["bbref_id"], "fg_id": None})
        except Exception as e:
            print(f"  {name}: ERROR — {e}")
            results.append({"name": name, "bbref_id": row["bbref_id"], "fg_id": None})

        time.sleep(1)

    return pd.DataFrame(results)


def _filter_player(leaderboard: pd.DataFrame, fg_id: str, name: str, mlb_start: int) -> pd.DataFrame:
    """Extract one player's rows from a pre-fetched leaderboard and tag with name."""
    rows = leaderboard[
        (leaderboard["IDfg"] == int(fg_id)) &
        (leaderboard["Season"] >= mlb_start)
    ].copy()
    if not rows.empty:
        rows.insert(0, "name", name)
    return rows


def main():
    players = pd.read_csv(PLAYER_LIST)

    # Determine the earliest MLB debut year across all mapped players so we
    # fetch the leaderboard just once per stat type instead of once per player.
    debut_years = {}
    for _, row in players.iterrows():
        bbref_id = row["bbref_id"]
        if bbref_id not in FANGRAPHS_ID_MAP:
            continue
        npb_end = int(str(row["npb_seasons"]).split("-")[-1])
        debut_years[bbref_id] = npb_end + 1

    if not debut_years:
        print("No players with FanGraphs IDs found — nothing to fetch.")
        return

    earliest = min(debut_years.values())  # e.g. 1995 for Nomo
    end_year = 2026
    print(f"Fetching full FanGraphs leaderboards {earliest}–{end_year} (qual=0) …")

    try:
        all_batting  = batting_stats(earliest, end_year, qual=0)
        print(f"  Batting leaderboard: {len(all_batting)} rows")
    except Exception as e:
        print(f"  ERROR fetching batting leaderboard: {e}")
        all_batting = pd.DataFrame()

    try:
        all_pitching = pitching_stats(earliest, end_year, qual=0)
        print(f"  Pitching leaderboard: {len(all_pitching)} rows")
    except Exception as e:
        print(f"  ERROR fetching pitching leaderboard: {e}")
        all_pitching = pd.DataFrame()

    hitting_frames  = []
    pitching_frames = []

    for _, row in players.iterrows():
        bbref_id      = row["bbref_id"]
        name          = row["name"].replace(" (P)", "").replace(" (H)", "")
        position_type = row["position_type"]

        fg_id = FANGRAPHS_ID_MAP.get(bbref_id)
        if not fg_id:
            print(f"  SKIP {name} — no FanGraphs ID mapped")
            continue

        mlb_start = debut_years[bbref_id]

        if position_type == "hitter" and not all_batting.empty:
            df = _filter_player(all_batting, fg_id, name, mlb_start)
            print(f"  {name}: {len(df)} MLB hitting seasons (debut {mlb_start})")
            if not df.empty:
                hitting_frames.append(df)

        elif position_type == "pitcher" and not all_pitching.empty:
            df = _filter_player(all_pitching, fg_id, name, mlb_start)
            print(f"  {name}: {len(df)} MLB pitching seasons (debut {mlb_start})")
            if not df.empty:
                pitching_frames.append(df)

        # Ohtani appears twice in the player list (P + H); handle both sides
        if bbref_id == "ohtansh01":
            if not all_batting.empty:
                hit_df = _filter_player(all_batting, fg_id, name, mlb_start)
                if not hit_df.empty:
                    hitting_frames.append(hit_df)
            if not all_pitching.empty:
                pit_df = _filter_player(all_pitching, fg_id, name, mlb_start)
                if not pit_df.empty:
                    pitching_frames.append(pit_df)

    # Save outputs
    if hitting_frames:
        hits_df = pd.concat(hitting_frames, ignore_index=True)
        hits_df.to_csv("mlb_hitting_stats_raw.csv", index=False)
        print(f"\nSaved mlb_hitting_stats_raw.csv — {len(hits_df)} rows")
    else:
        print("\nNo hitting data collected.")

    if pitching_frames:
        pit_df = pd.concat(pitching_frames, ignore_index=True)
        pit_df.to_csv("mlb_pitching_stats_raw.csv", index=False)
        print(f"Saved mlb_pitching_stats_raw.csv — {len(pit_df)} rows")
    else:
        print("No pitching data collected.")


if __name__ == "__main__":
    main()
