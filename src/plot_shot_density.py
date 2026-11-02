"""KDE des positions de tir : ou l'on tire, et d'ou l'on marque."""

import os

import pandas as pd
from mplsoccer import VerticalPitch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOTS = os.path.join(BASE_DIR, "data", "raw", "shots.csv")
OUTPUT = os.path.join(BASE_DIR, "reports", "shot_density.png")


def main():
    shots = pd.read_csv(SHOTS).dropna(subset=["x", "y"])
    goals = shots[shots["is_goal"] == 1]

    panels = [
        (shots, f"All shots (n = {len(shots)})"),
        (goals, f"Goals only (n = {len(goals)})"),
    ]

    pitch = VerticalPitch(pitch_type="statsbomb", half=True, line_zorder=3,
                          line_color="#dddddd", pitch_color="#1a1a1a")
    fig, axs = pitch.draw(nrows=1, ncols=2, figsize=(12, 7))
    fig.set_facecolor("#1a1a1a")

    for ax, (data, titre) in zip(axs, panels):
        pitch.kdeplot(data["x"], data["y"], ax=ax, fill=True, levels=60,
                      cmap="magma", thresh=0.01, zorder=1)
        ax.set_title(titre, color="white", fontsize=12, pad=8)

    fig.suptitle("Shot and goal density", color="white", fontsize=14, y=0.93)
    fig.text(0.5, 0.06, "Each panel has its own colour scale: "
             "compare the shape of the areas, not the intensity.",
             ha="center", color="#aaaaaa", fontsize=9)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Figure sauvegardee dans : {OUTPUT}")


if __name__ == "__main__":
    main()