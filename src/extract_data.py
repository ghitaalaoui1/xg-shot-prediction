from mplsoccer import Sbopen
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOTS_OUTPUT_PATH = os.path.join(BASE_DIR, "data", "raw", "shots_worldcup2022.csv")
FREEZE_OUTPUT_PATH = os.path.join(BASE_DIR, "data", "raw", "freeze_worldcup2022.csv")

parser = Sbopen()

def extract_all_shots():
    matches = parser.match(competition_id=43, season_id=106)
    all_shots = []
    all_freeze = []

    for match_id in matches['match_id']:
        events, related, freeze, tactics = parser.event(match_id)
        shots = events[events['type_name'] == 'Shot'].copy()

        # Exclure les penaltys (position fixe, pas de defenseur, cas a part)
        shots = shots[shots['sub_type_name'] != 'Penalty']
        shots['is_goal'] = (shots['outcome_name'] == 'Goal').astype(int)

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

shots = pd.read_csv('data/raw/shots_worldcup2022.csv')
freeze = pd.read_csv('data/raw/freeze_worldcup2022.csv')

print(f"Nombre total de tirs (hors penaltys) : {len(shots)}")
print(f"Nombre de lignes freeze frame : {len(freeze)}")
print(f"\nDistribution de is_goal :")
print(shots['is_goal'].value_counts())
print(f"\nTaux de but : {shots['is_goal'].mean() * 100:.1f}%")