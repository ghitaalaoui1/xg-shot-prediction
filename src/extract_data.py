from mplsoccer import Sbopen
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOTS_OUTPUT_PATH = os.path.join(BASE_DIR, "data", "raw", "shots.csv")
FREEZE_OUTPUT_PATH = os.path.join(BASE_DIR, "data", "raw", "freeze.csv")

# (competition_id, season_id) : les 12 competitions avec donnees 360 (cf. inventory_360.py)
# L'Euro 2024 sert de holdout (cf. split.py)
COMPETITIONS = [
    (43, 106),    # FIFA World Cup 2022
    (72, 107),    # Women's World Cup 2023
    (55, 282),    # UEFA Euro 2024
    (55, 43),     # UEFA Euro 2020
    (11, 90),     # La Liga 2020/2021
    (9, 281),     # 1. Bundesliga 2023/2024
    (7, 235),     # Ligue 1 2022/2023
    (53, 315),    # UEFA Women's Euro 2025
    (53, 106),    # UEFA Women's Euro 2022
    (7, 108),     # Ligue 1 2021/2022
    (44, 107),    # Major League Soccer 2023
    (1267, 107),  # African Cup of Nations 2023
]

parser = Sbopen()

def extract_all_shots():
    all_shots = []
    all_freeze = []

    for competition_id, season_id in COMPETITIONS:
        matches = parser.match(competition_id=competition_id, season_id=season_id)

        for _, match in matches.iterrows():
            events, related, freeze, tactics = parser.event(match['match_id'])
            shots = events[events['type_name'] == 'Shot'].copy()

            # Exclure les penaltys (position fixe, pas de defenseur, cas a part)
            shots = shots[shots['sub_type_name'] != 'Penalty']
            shots['is_goal'] = (shots['outcome_name'] == 'Goal').astype(int)
            # Ex. "UEFA Euro 2024" : doit correspondre a HOLDOUT_COMPETITION dans split.py
            shots['competition'] = f"{match['competition_name']} {match['season_name']}"

            all_shots.append(shots)
            all_freeze.append(freeze)

    full_shots = pd.concat(all_shots, ignore_index=True)
    full_freeze = pd.concat(all_freeze, ignore_index=True)

    full_shots.to_csv(SHOTS_OUTPUT_PATH, index=False)
    full_freeze.to_csv(FREEZE_OUTPUT_PATH, index=False)

    print(f"Total de tirs extraits (hors penaltys) : {len(full_shots)}")
    print(f"Total de lignes freeze frame : {len(full_freeze)}")
    print(f"Tirs sauvegardes dans : {SHOTS_OUTPUT_PATH}")
    print(f"Freeze frames sauvegardes dans : {FREEZE_OUTPUT_PATH}")

if __name__ == "__main__":
    extract_all_shots()

import pandas as pd

shots = pd.read_csv('data/raw/shots.csv')
freeze = pd.read_csv('data/raw/freeze.csv')

print(f"Nombre total de tirs (hors penaltys) : {len(shots)}")
print(f"Nombre de lignes freeze frame : {len(freeze)}")
print(f"\nDistribution de is_goal :")
print(shots['is_goal'].value_counts())
print(f"\nTaux de but : {shots['is_goal'].mean() * 100:.1f}%")