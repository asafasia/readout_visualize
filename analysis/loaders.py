import os
import json
import numpy as np
from typing import Tuple, Dict, Any

from .utils import two_state_discriminator
from scipy.stats import skew, kurtosis
from matplotlib import pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import kurtosis
from scipy.stats import chi2
# import pingouin as pg   # <-- pip install pingouin
# import gzip


def load_json(path: str) -> Dict[str, Any]:
    """Load and parse JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def extract_data(path: str, file_type: str = "json") -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Extract sweep parameters and acquired data from a JSON file.

    Parameters
    ----------
    path : str
        Full file path to the JSON data file.

    Returns
    -------
    lengths : np.ndarray
    amplitudes : np.ndarray
    acquired_data : list[dict]
    """
    if file_type == "npz":
        loaded = np.load(path, allow_pickle=True)
        data = loaded["data"].item()
    elif file_type == "json":
        data = load_json(path)
    elif file_type == "gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported file_type: {file_type}")

    sweep = data["sweep_params"]
    lengths = np.asarray(sweep["lengths"])
    amplitudes = np.asarray(sweep["amplitudes"])
    acquired_data = data["acquired_data"]

    return lengths, amplitudes, acquired_data


def find_fidelity(IQ_data):
    I0 = np.array(IQ_data['ground']['I'])
    Q0 = np.array(IQ_data['ground']['Q'])
    I1 = np.array(IQ_data['excited']['I'])
    Q1 = np.array(IQ_data['excited']['Q'])

    _, _, fidelity, _, _, _, _ = two_state_discriminator(
        I0, Q0, I1, Q1, b_print=False, b_plot=False)

    return fidelity


def find_separation(IQ_data):
    I0 = np.array(IQ_data['ground']['I'])
    Q0 = np.array(IQ_data['ground']['Q'])
    I1 = np.array(IQ_data['excited']['I'])
    Q1 = np.array(IQ_data['excited']['Q'])

    ground = I0 + 1j*Q0
    excited = I1 + 1j*Q0

    distance = np.mean(ground) - np.mean(excited)
    varience = np.var(ground)/2+np.var(excited)/2

    return np.abs(distance)/np.sqrt(varience)


def find_deveiation(IQ_data, plot=False, print=False, ax=None):

    return check_gaussian_2d(IQ_data, plot=plot, print=print, ax=ax)


def gaussian_deviation_score(x):
    """Compute a score indicating deviation from Gaussian distribution."""
    skewness = skew(x)
    kurt = kurtosis(x, fisher=True)  # Fisher's definition (subtract 3)

    # A simple combined score
    JB = (skewness**2 / 6 + kurt**2 / 24)
    return JB


def plot_confidence_ellipse(mean, cov, ax, n_std=2.0, **kwargs):
    # Eigenvalues and vectors
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)

    ellipse = Ellipse(xy=(mean[0], mean[1]), width=width, height=height,
                      angle=theta, linestyle='--', color='black', fill=False, linewidth=2)
    ax.add_patch(ellipse)


def check_gaussian_2d(iq_data, plot=False, print=False, ax=None):

    ground_data = np.array([iq_data['ground']['I'], iq_data['ground']['Q']]).T
    excited_data = np.array(
        [iq_data['excited']['I'], iq_data['excited']['Q']]).T

    double_data = [ground_data, excited_data]

    max_score = 0

    for i, data in enumerate(double_data):
        X = data[:, 0]
        Y = data[:, 1]

        # -----------------------
        # 1. Scatter + Gaussian ellipse
        # -----------------------
        mean = data.mean(axis=0)
        cov = np.cov(data, rowvar=False)

        if plot:
            if ax is None:
                plt.figure(figsize=(6, 6))
                plt.scatter(X, Y, s=8, alpha=0.4)

            # plt.scatter(X, Y, s=8, alpha=0.4)
            plot_confidence_ellipse(mean, cov, ax, n_std=2,
                                    edgecolor=f'C{1-i}', lw=2, fill=False)
            # plt.title("2D Scatter + Gaussian Ellipse")
            # plt.xlabel("X")
            # plt.ylabel("Y")
            # plt.axis('equal')

        # -----------------------
        # 2. Mardia's Skewness & Kurtosis
        # -----------------------

        mardia = pg.multivariate_normality(data, alpha=0.05)
        if print:
            print("=== Multivariate Normality Tests ===")
            print(mardia)

            if mardia.normal:
                print("\n✅ Final Verdict: The distribution looks multivariate Gaussian.")
            else:
                print(
                    "\n❌ Final Verdict: The distribution is NOT multivariate Gaussian.")

        if mardia.hz > max_score:
            max_score = mardia.hz
    return max_score
