import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOTS_PATH = os.path.join(BASE_DIR, "data", "raw", "shots_worldcup2022.csv")
FREEZE_PATH = os.path.join(BASE_DIR, "data", "raw", "freeze_worldcup2022.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "shots_features.csv")

GOAL_X, GOAL_Y = 120, 40
POST_Y1, POST_Y2 = 36, 44


def compute_distance_angle(shots):
    shots = shots.copy()
    shots['distance_to_goal'] = np.sqrt((GOAL_X - shots['x'])**2 + (GOAL_Y - shots['y'])**2)

    angle_1 = np.arctan2(POST_Y1 - shots['y'], GOAL_X - shots['x'])
    angle_2 = np.arctan2(POST_Y2 - shots['y'], GOAL_X - shots['x'])
    shots['shot_angle'] = np.abs(angle_1 - angle_2)
    return shots


def _sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


def _point_in_triangle(pt, v1, v2, v3):
    d1 = _sign(pt, v1, v2)
    d2 = _sign(pt, v2, v3)
    d3 = _sign(pt, v3, v1)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def _keeper_lateral_offset(shot_point, keeper_point):
    # Distance perpendiculaire entre le gardien et la bissectrice de l'angle de tir
    # (ligne tireur -> milieu "angulaire" des deux poteaux), position theorique ideale du gardien.
    # 0 = gardien parfaitement centre dans l'angle, valeur elevee = gardien decale lateralement.
    sx, sy = shot_point
    u1 = np.array([GOAL_X - sx, POST_Y1 - sy])
    u2 = np.array([GOAL_X - sx, POST_Y2 - sy])
    bisector = u1 / np.linalg.norm(u1) + u2 / np.linalg.norm(u2)
    bisector /= np.linalg.norm(bisector)
    kx, ky = keeper_point[0] - sx, keeper_point[1] - sy
    return abs(bisector[0] * ky - bisector[1] * kx)


def compute_defenders_and_keeper(shots, freeze):
    n_defenders = []
    dist_to_keeper = []
    keeper_offset = []

    for _, shot in shots.iterrows():
        shot_id = shot['id']
        shot_point = (shot['x'], shot['y'])
        v2 = (GOAL_X, POST_Y1)
        v3 = (GOAL_X, POST_Y2)

        frame = freeze[freeze['id'] == shot_id]
        opponents = frame[frame['teammate'] == False]

        non_keeper_opponents = opponents[opponents['position_name'] != 'Goalkeeper']
        count = sum(
            _point_in_triangle((row['x'], row['y']), shot_point, v2, v3)
            for _, row in non_keeper_opponents.iterrows()
        )
        n_defenders.append(count)

        keeper = opponents[opponents['position_name'] == 'Goalkeeper']
        if len(keeper) > 0:
            kx, ky = keeper.iloc[0]['x'], keeper.iloc[0]['y']
            dist = np.sqrt((kx - shot['x'])**2 + (ky - shot['y'])**2)
            offset = _keeper_lateral_offset(shot_point, (kx, ky))
        else:
            dist = np.nan
            offset = np.nan
        dist_to_keeper.append(dist)
        keeper_offset.append(offset)

    shots = shots.copy()
    shots['defenders_in_cone'] = n_defenders
    shots['distance_to_keeper'] = dist_to_keeper
    shots['keeper_lateral_offset'] = keeper_offset
    return shots


def build_features():
    shots = pd.read_csv(SHOTS_PATH)
    freeze = pd.read_csv(FREEZE_PATH)

    print(f"Tirs charges : {len(shots)}")

    shots = compute_distance_angle(shots)
    shots = compute_defenders_and_keeper(shots, freeze)

    keep_cols = [
        'is_goal', 'distance_to_goal', 'shot_angle', 'defenders_in_cone',
        'distance_to_keeper', 'keeper_lateral_offset', 'under_pressure', 'shot_first_time',
        'body_part_name', 'technique_name', 'play_pattern_name'
    ]
    final_df = shots[keep_cols].copy()

    final_df['under_pressure'] = final_df['under_pressure'].fillna(False).astype(int)
    final_df['shot_first_time'] = final_df['shot_first_time'].fillna(False).astype(int)

    print(f"\nValeurs manquantes avant encodage :\n{final_df.isnull().sum()}")

    final_df = pd.get_dummies(final_df, columns=['body_part_name', 'technique_name', 'play_pattern_name'], drop_first=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nFeatures sauvegardees dans : {OUTPUT_PATH}")
    print(f"Colonnes finales ({final_df.shape[1]}) : {final_df.columns.tolist()}")

    return final_df

if __name__ == "__main__":
    build_features()