"""xG explorer: goal probability by shot location and situation.

Run with: streamlit run app.py
"""

import os
import sys

import joblib
import matplotlib.pyplot as plt
import streamlit as st
from mplsoccer import VerticalPitch

# Les modules de src/ s'importent sans prefixe (from features import ...)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import pandas as pd  # noqa: E402

from surface import build_frame, grid_positions, options_for, predict  # noqa: E402

MODELS = {
    "Logistic regression": "logistic.pkl",
    "XGBoost": "xgboost.pkl",
    "Calibrated random forest": "rf_calibrated.pkl",
    "KNN": "knn.pkl",
}

# Categorie supprimee par drop_first a l'encodage : toutes les indicatrices a zero
REFERENCE = {
    "body_part_name": "Head",
    "technique_name": "Backheel",
    "play_pattern_name": "From Corner",
}

LABELS = {
    "body_part_name": "Body part",
    "technique_name": "Technique",
    "play_pattern_name": "Play pattern",
}

DEFAULTS = {
    "body_part_name": "Right Foot",
    "technique_name": "Normal",
    "play_pattern_name": "Regular Play",
}


@st.cache_resource
def load_artifact(filename):
    return joblib.load(os.path.join(BASE_DIR, "models", filename))


@st.cache_data
def xg_grid(model_file, scenario_items):
    """Surface xG pour un scenario. Recalculee seulement si le scenario change."""
    scenario = dict(scenario_items)
    positions = grid_positions()
    frame = build_frame(positions, scenario)
    frame["xg"] = predict(load_artifact(model_file), frame)
    return frame[["x", "y", "xg"]]


def categorical_scenario(prefix, choice):
    """Une modalite choisie -> les indicatrices correspondantes."""
    if choice == REFERENCE[prefix]:
        return {}
    return {f"{prefix}_{choice}": 1}


st.set_page_config(page_title="xG Explorer", layout="wide")
st.title("xG Explorer")

with st.sidebar:
    st.header("Shot situation")
    model_name = st.selectbox("Model", list(MODELS))

    scenario = {}
    for prefix, label in LABELS.items():
        options = [REFERENCE[prefix]] + options_for(prefix)
        choice = st.selectbox(label, options, index=options.index(DEFAULTS[prefix]))
        scenario.update(categorical_scenario(prefix, choice))

    scenario["under_pressure"] = int(st.checkbox("Under pressure"))
    scenario["shot_first_time"] = int(st.checkbox("First-time shot"))
    scenario["defenders_in_cone"] = st.slider("Defenders in the shooting cone", 0, 6, 0)
    scenario["keeper_off_line"] = st.slider(
        "Keeper off his line (yd)", -2.0, 10.0, 0.5, 0.5,
        help="How far the keeper has come out towards the shooter.")
    scenario["keeper_lateral_offset"] = st.slider(
        "Keeper lateral offset (yd)", 0.0, 5.0, 0.3, 0.1,
        help="0 = keeper centred in the shooting angle.")

    st.header("Shot location")
    shot_x = st.slider("Distance from the goal line (yd)", 0.5, 59.5, 12.0, 0.5)
    shot_y = st.slider("Lateral position (yd, 40 = centre)", 0.5, 79.5, 40.0, 0.5)

model_file = MODELS[model_name]
grid = xg_grid(model_file, tuple(sorted(scenario.items())))

# Le curseur ne declenche qu'une prediction ponctuelle, pas la grille
x = 120 - shot_x
point = build_frame(pd.DataFrame({"x": [x], "y": [shot_y]}), scenario)
xg_point = predict(load_artifact(model_file), point)[0]

col_pitch, col_info = st.columns([3, 1])

with col_info:
    st.metric("Shot xG", f"{xg_point:.3f}")
    st.caption(
        f"Distance to goal: {point['distance_to_goal'].iloc[0]:.1f} yd  \n"
        f"Shot angle: {point['shot_angle'].iloc[0] * 180 / 3.14159:.1f}°"
    )
    st.caption(f"Max xG on the pitch: {grid['xg'].max():.3f}")

with col_pitch:
    pitch = VerticalPitch(pitch_type="statsbomb", half=True, line_zorder=3,
                          line_color="#222222", pitch_color="white")
    fig, ax = pitch.draw(figsize=(8, 6))
    stats = pitch.bin_statistic(grid["x"], grid["y"], values=grid["xg"],
                                statistic="mean", bins=(40, 34))
    pcm = pitch.heatmap(stats, ax=ax, cmap="YlOrRd", vmin=0,
                        vmax=max(grid["xg"].max(), 1e-6),
                        edgecolors="none", zorder=1)
    pitch.scatter(x, shot_y, ax=ax, s=150, color="#1f77b4",
                  edgecolors="black", zorder=4)
    fig.colorbar(pcm, ax=ax, shrink=0.7, label="Goal probability (xG)")
    st.pyplot(fig)
    plt.close(fig)
