"""Construction des situations de tir, partagee par l'app et les figures."""

import numpy as np
import pandas as pd

from features import FEATURES, compute_distance_angle


def options_for(prefix):
    """Les modalites disponibles pour une variable categorielle."""
    return [f[len(prefix) + 1:] for f in FEATURES if f.startswith(prefix + "_")]


def build_frame(positions, scenario):
    """positions : DataFrame avec x et y. scenario : dict colonne -> valeur."""
    frame = compute_distance_angle(positions.copy())

    for col in FEATURES:
        if col not in frame.columns:
            frame[col] = 0

    for col, value in scenario.items():
        if col not in FEATURES:
            raise ValueError(f"Variable inconnue : {col}")
        frame[col] = value

    return frame


def predict(artifact, frame):
    X = frame[artifact["features"]]
    if artifact["scaler"] is not None:
        X = artifact["scaler"].transform(X)
    return artifact["model"].predict_proba(X)[:, 1]


def grid_positions(nx=150, ny=170):
    xs = np.linspace(60.5, 119.4, nx)
    ys = np.linspace(0.5, 79.5, ny)
    return pd.DataFrame([(x, y) for x in xs for y in ys], columns=["x", "y"])