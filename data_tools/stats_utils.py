from itertools import combinations
from math import acos, sqrt, pi
import pandas as pd
import numpy as np

def compute_vad_coverage(df: pd.DataFrame,
                         centers_cols: list[str],
                         radius_col: str,
                         n_samples: int = 200_000,
                         seed: int = 0) -> float:
    """Monte‑Carlo estimate of PAD‑cube coverage by the union of emotion spheres.

    :param df: DataFrame with columns
    :type df: pd.DataFrame
    :param centers_cols: Names of the columns containing the VAD means.
    :type centers_cols: list[str]
    :param radius_col: Which *_std column to use as sphere radii.
    :type radius_col: str
    :param n_samples: Number of random points to draw inside the cube (↑ for lower error).
    :type n_samples: int
    :param seed: RNG seed for reproducible results.
    :type seed: int
    :return: Fraction of the cube covered (0 … 1).
    :rtype: float
    """
    rng      = np.random.default_rng(seed)
    samples  = rng.uniform(-1, 1, size=(n_samples, 3))      # cube points
    centres  = df[centers_cols].values
    radii_sq = df[radius_col].values ** 2                   # radius² for speed

    covered = np.zeros(n_samples, dtype=bool)
    for c, r2 in zip(centres, radii_sq):
        d2 = np.sum((samples - c) ** 2, axis=1)             # distance²
        covered |= d2 <= r2                                 # union of spheres

    return covered.mean()     

def _intersection_volume(r1: float, r2: float, d: float) -> float:
    """Exact volume of the intersection of two spheres.

    :param r1: Radius of the first sphere (non‑negative).
    :param r2: Radius of the second sphere (non‑negative).
    :param d: Centre‑to‑centre distance between the two spheres (non‑negative).
    :return: Volume of the overlap region (0 if they do not intersect).
    :rtype: float
    """
    # No intersection
    if d >= r1 + r2:
        return 0.0
    # One sphere completely inside the other
    if d <= abs(r1 - r2):
        return 4.0 / 3.0 * pi * min(r1, r2) ** 3

    # Partial overlap – use the classic formula
    term1 = r1**2 * acos((d**2 + r1**2 - r2**2) / (2 * d * r1))
    term2 = r2**2 * acos((d**2 + r2**2 - r1**2) / (2 * d * r2))
    term3 = 0.5 * sqrt(
        (-d + r1 + r2) *
        ( d + r1 - r2) *
        ( d - r1 + r2) *
        ( d + r1 + r2)
    )
    return term1 + term2 - term3


def find_intersecting_pairs(df: pd.DataFrame,
                            radius_col: str,
                            with_volume: bool = False) -> pd.DataFrame:
    """Identify all intersecting sphere pairs (and optionally their volumes).

    :param df: DataFrame with columns
    :type df: pd.DataFrame
    :param radius_col: Which *_std column to use as the radius for each emotion.
    :type radius_col: str
    :param with_volume: Whether to compute & return the intersection volume for each pair.
    :type with_volume: bool
    :return: One row per intersecting pair with columns:
             ['idx1', 'idx2', 'distance', 'r1', 'r2'] plus
             'intersection_volume' if requested.
    :rtype: pd.DataFrame
    """
    centres = df[['pleasure_mean', 'arousal_mean', 'dominance_mean']].values
    radii   = df[radius_col].values

    records = []
    for i, j in combinations(range(len(df)), 2):
        d = np.linalg.norm(centres[i] - centres[j])
        if d <= radii[i] + radii[j]:             # spheres intersect
            rec = dict(idx1=i, idx2=j,
                       distance=d, r1=radii[i], r2=radii[j])
            if with_volume:
                rec['intersection_volume'] = _intersection_volume(
                    radii[i], radii[j], d
                )
            records.append(rec)

    return pd.DataFrame.from_records(records)

def compute_pad_coverage_voxel(df: pd.DataFrame,
                               radius_col: str,
                               resolution: int = 120,
                               chunk: int = 40) -> float:
    """Deterministic voxel (grid) coverage of the PAD cube.

    :param df: DataFrame with columns
            'pleasure_mean', 'arousal_mean', 'dominance_mean', and `radius_col`
    :type df: pd.DataFrame
    :param radius_col: One of the *_std columns to use as sphere radii.
    :type radius_col: str
    :param resolution: Number of voxels per axis (higher → finer & slower; memory ∝ resolution³).
    :type resolution: int
    :param chunk: Number of z‑slices processed at once to keep RAM reasonable.
    :type chunk: int
    :return: Fraction of the 8‑unit PAD cube covered by the union of spheres.
    :rtype: float
    """
    step = 2.0 / resolution                         # edge length of one voxel
    xy_centres = np.linspace(-1 + step/2, 1 - step/2, resolution)
    xv, yv = np.meshgrid(xy_centres, xy_centres, indexing='ij')
    xy_points = np.stack((xv.ravel(), yv.ravel()), axis=1)  # (res², 2)

    centres = df[['pleasure_mean', 'arousal_mean', 'dominance_mean']].values
    radii_sq = df[radius_col].values ** 2

    covered_voxels = 0
    for z_start in range(0, resolution, chunk):
        # Work on chunk‑sized blocks along the z‑axis to save memory
        z_centres = np.linspace(-1 + step/2, 1 - step/2, resolution)[
            z_start : z_start + chunk
        ]
        points = np.column_stack((
            np.tile(xy_points, (len(z_centres), 1)),
            np.repeat(z_centres, resolution * resolution)
        ))

        inside = np.zeros(points.shape[0], dtype=bool)
        for c, r2 in zip(centres, radii_sq):
            d2 = np.sum((points - c) ** 2, axis=1)
            inside |= d2 <= r2
            if inside.all():                      # short‑circuit if fully covered
                break
        covered_voxels += inside.sum()

    voxel_volume = step ** 3
    total_volume = covered_voxels * voxel_volume
    return total_volume / 8.0