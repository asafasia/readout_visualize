from matplotlib import pyplot as plt
import numpy as np
from analysis.loaders import check_gaussian_2d
from .utils import find_closest_keys


def plot_single_iq_data(ax1, ax2, IQ_data, **kargs) -> None:
    """
    Plot the IQ data for a single length and amplitude setting.

    :param str l_key: The length key to plot.
    :param str a_key: The amplitude key to plot.
    :param ReadoutDataset dataset: The dataset containing the data.
    """

    Ig = IQ_data["ground"]["I"]
    Qg = IQ_data["ground"]["Q"]
    Ie = IQ_data["excited"]["I"]
    Qe = IQ_data["excited"]["Q"]

    threshold = np.mean(Ig)/2+np.mean(Ie)/2

    ax1.plot(Ig, Qg, 'b.', alpha=0.3, label="Ground State")
    ax1.plot(Ie, Qe, 'r.', alpha=0.3, label="Excited State")
    ax1.set_xlabel("I Quadrature")
    ax1.set_ylabel("Q Quadrature")
    ax1.legend()
    ax1.set_xlim(kargs.get("xlim"))
    ax1.set_ylim(kargs.get("ylim"))
    ax1.axvline(threshold, color='gray', lw=2.5, ls='--')

    ax1.grid(True)
    ax2.hist(Ig, bins=50,color='b', density=True, alpha=0.5, label="Ground State")
    ax2.hist(Ie, bins=50, color='r', density=True, alpha=0.5, label="Excited State")
    ax2.grid(True)
    ax2.set_xlabel("I Quadrature")
    ax2.set_ylabel("Probability Density")
    ax2.set_xlim(kargs.get("xlim"))
    ax2.set_ylim(0, 6)
    ax2.axvline(threshold, color='gray', lw=2.5, ls='--')

    ax2.legend()


def plot_map(ax, x_values: np.ndarray, y_values: np.ndarray, data_map: np.ndarray, **kargs: dict):
    c = ax.pcolormesh(
        x_values * 1e9,
        y_values,
        data_map,
        shading="auto",
        **kargs
    )

    return c


if __name__ == "__main__":
    # Example usage
    # dataset = ReadoutDataset(qubit="q16", with_twpa=True)
    # plot_single_iq_data(length=100.0, amplitude=0.5, dataset=dataset)
    pass
