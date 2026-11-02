"""Carte de chaleur xG : ce que le modele a appris, pas ce qui s'est produit.

Genere reports/xg_surface.png, deux panneaux a echelle commune :
sans defenseur dans le cone, puis avec trois.
"""

import os

import joblib
import numpy as np
import pandas as pd
from mplsoccer import VerticalPitch

from features import FEATURES, compute_distance_angle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "logistic.pkl")
OUTPUT = os.path.join(BASE_DIR, "reports", "xg_surface.png")

# Situation representee, a citer dans la legende
SCENARIO = {
    "body_part_name": "Right Foot",
    "technique_name": "Normal",
    "play_pattern_name": "Regular Play",
}
KEEPER_OFF_LINE = 0.5        # gardien quasiment sur sa ligne
KEEPER_LATERAL_OFFSET = 0.3  # gardien bien place dans l'angle
BINS = (30, 24)


def build_grid(defenders):
    """Grille de tirs fictifs, construite avec les fonctions du projet."""
    xs = np.linspace(80.5, 119.5, 120)
    ys = np.linspace(0.5, 79.5, 160)
    grid = pd.DataFrame(
        [(x, y) for x in xs for y in ys], columns=["x", "y"]
    )

    grid = compute_distance_angle(grid)

    for col in FEATURES:
        if col not in grid.columns:
            grid[col] = 0

    grid["defenders_in_cone"] = defenders
    grid["keeper_off_line"] = KEEPER_OFF_LINE
    grid["keeper_lateral_offset"] = KEEPER_LATERAL_OFFSET

    for prefix, value in SCENARIO.items():
        col = f"{prefix}_{value}"
        if col not in FEATURES:
            candidates = [f for f in FEATURES if f.startswith(prefix)]
            raise SystemExit(f"Colonne introuvable : {col}\nDisponibles : {candidates}")
        grid[col] = 1

    return grid


def main():
    artifact = joblib.load(MODEL_PATH)
    model, scaler = artifact["model"], artifact["scaler"]

    grids = {}
    for defenders in (0, 3):
        grid = build_grid(defenders)
        X = grid[FEATURES]
        if scaler is not None:
            X = scaler.transform(X)
        grid["xg"] = model.predict_proba(X)[:, 1]
        grids[defenders] = grid

    vmax = max(g["xg"].max() for g in grids.values())

    pitch = VerticalPitch(pitch_type="statsbomb", half=True, line_zorder=3,
                          line_color="#222222", pitch_color="white")
    fig, axs = pitch.draw(nrows=1, ncols=2, figsize=(12, 8))

    for ax, defenders in zip(axs, (0, 3)):
        grid = grids[defenders]
        stats = pitch.bin_statistic(grid["x"], grid["y"], values=grid["xg"],
                                    statistic="mean", bins=BINS)
        pcm = pitch.heatmap(stats, ax=ax, cmap="YlOrRd", vmin=0, vmax=vmax,
                            edgecolors="none", zorder=1)
        plural = "s" if defenders != 1 else ""
        ax.set_title(f"{defenders} defender{plural} in the shooting cone", fontsize=13)

    fig.colorbar(pcm, ax=axs, shrink=0.6, label="Goal probability (xG)")
    fig.suptitle(
        "Model xG surface\n"
        "Right foot, normal technique, open play, keeper on his line",
        fontsize=14,
    )

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    print(f"Figure sauvegardee dans : {OUTPUT}")


if __name__ == "__main__":
    main()