"""Decoupage des donnees, partage par tous les scripts d'entrainement."""

import numpy as np

HOLDOUT_COMPETITION = "UEFA Euro 2024"
SEED = 42


def split_by_match(df, test_size=0.2, seed=SEED):
    """Decoupe en gardant tous les tirs d'un match du meme cote."""
    matches = df["match_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(matches)
    cut = int(len(matches) * (1 - test_size))
    train_ids, test_ids = matches[:cut], matches[cut:]
    return (
        df[df["match_id"].isin(train_ids)].copy(),
        df[df["match_id"].isin(test_ids)].copy(),
    )


def load_splits(df, seed=SEED):
    """Retourne (train, test, holdout). Le seul point d'entree autorise."""
    holdout = df[df["competition"] == HOLDOUT_COMPETITION].copy()
    pool = df[df["competition"] != HOLDOUT_COMPETITION].copy()
    train, test = split_by_match(pool, seed=seed)
    return train, test, holdout