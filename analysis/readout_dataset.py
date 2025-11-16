from pathlib import Path
import numpy as np
from config import DATA_DIR
from analysis.loaders import extract_data, find_separation, find_deveiation, check_gaussian_2d
from analysis.plotting import plot_single_iq_data, plot_map
from matplotlib import pyplot as plt


class ReadoutDataset:
    """
    Load and organize readout-optimization datasets for a specific qubit.
    Automatically extracts:
        - lengths (ns)
        - amplitudes (V)
        - acquired_data (nested dictionary)
        - fidelity map
        - separation map
    """

    def __init__(self, qubit: str, with_twpa: bool, file_type: str = "json") -> None:
        self.qubit = qubit
        self.with_twpa = with_twpa
        self.data_root = DATA_DIR / qubit
        self.file_type = file_type
        self.file_name = self._select_file(file_type=file_type)

        (
            self.lengths,
            self.amplitudes,
            self.acquired_data,
        ) = extract_data(self.file_name, file_type=self.file_type)

        self.fidelity_map = self._compute_fidelity_map()
        self.separation_map = self._compute_separation_map()
        # self.gaussian_score_map = self._comptute_guassian_score_map()
        self.max_fidelity = np.max(self.fidelity_map)
        self.max_separation = np.max(self.separation_map)
    # -------------------------------------------------------------------------
    # File Selection
    # -------------------------------------------------------------------------

    def _select_file(self, file_type: str) -> Path:
        """Return the single matching dataset file, otherwise raise."""
        # ext = "*.npz" if file_type == "npz" else "*.json"
        if file_type == "npz":
            ext = "*.npz"
        elif file_type == "json":
            ext = "*.json"
        elif file_type == "gz":
            ext = "*.gz"
        else:
            raise ValueError(f"Unsupported file_type: {file_type}")

        files = list(self.data_root.glob(ext))
        files = [f for f in files if ("with_twpa" in f.name) == self.with_twpa]

        if len(files) == 0:
            raise FileNotFoundError(
                f"No dataset found for qubit='{self.qubit}', with_twpa={self.with_twpa}"
            )
        if len(files) > 1:
            raise ValueError(
                f"Multiple dataset files found for qubit='{self.qubit}', "
                f"with_twpa={self.with_twpa}. Please clean the directory.\n"
                f"Files: {[f.name for f in files]}"
            )
        return files[0]

    # -------------------------------------------------------------------------
    # Internal Data Processing
    # -------------------------------------------------------------------------
    def _compute_fidelity_map(self) -> np.ndarray:
        """Return a matrix fidelity[length_index][amp_index]."""
        return np.array([
            [entry["fidelity"] for _, entry in sorted(
                amp_dict.items(), key=lambda x: float(x[0]))]
            for _, amp_dict in sorted(self.acquired_data.items(), key=lambda x: float(x[0]))
        ])

    def _compute_separation_map(self) -> np.ndarray:
        """Return a matrix of separations computed from IQ data."""

        return np.array([
            [
                find_separation(entry["IQ_data"])
                for _, entry in sorted(amp_dict.items(), key=lambda x: float(x[0]))
            ]
            for _, amp_dict in sorted(self.acquired_data.items(), key=lambda x: float(x[0]))
        ])

    def _comptute_guassian_score_map(self) -> np.ndarray:
        data = self.acquired_data
        m = []
        for l_key in data.keys():
            v = []
            for a_key in data[l_key].keys():
                IQ_data = data[l_key][a_key]["IQ_data"]
                score = check_gaussian_2d(IQ_data)
                v.append(score)
            m.append(v)
        return np.array(m)

    def _compute_general_map(self, func, *args) -> np.ndarray:
        """Return a matrix computed from IQ data using the provided function."""

        # return np.array([
        #     [
        #         func(entry["IQ_data"], *args)
        #         for _, entry in sorted(amp_dict.items(), key=lambda x: float(x[0]))
        #     ]
        #     for _, amp_dict in sorted(self.acquired_data.items(), key=lambda x: float(x[0]))
        # ])

    def find_max_fidelity_settings(self) -> tuple[float, float]:
        """Return the (length, amplitude) pair that gives the max fidelity."""
        index = np.unravel_index(
            np.argmax(self.fidelity_map), self.fidelity_map.shape)
        length = float(sorted(self.acquired_data.keys())[index[0]])
        amplitude = float(sorted(
            self.acquired_data[str(length)].keys())[index[1]])
        return length, amplitude

    def find_max_separation_settings(self) -> tuple[float, float]:
        """Return the (length, amplitude) pair that gives the max separation."""
        index = np.unravel_index(
            np.argmax(self.separation_map), self.separation_map.shape)
        length = float(sorted(self.acquired_data.keys())[index[0]])
        amplitude = float(sorted(
            self.acquired_data[str(length)].keys())[index[1]])
        return length, amplitude

    def get_fidelity_at(self, length: float, amplitude: float) -> float:
        """Return the fidelity at the closest (length, amplitude) setting."""
        l_key = min(self.acquired_data.keys(),
                    key=lambda x: abs(float(x) - length))
        a_key = min(
            self.acquired_data[l_key].keys(),
            key=lambda x: abs(float(x) - amplitude),
        )
        return self.acquired_data[str(l_key)][str(a_key)]["fidelity"]

    def get_iq_data_at(self, length: float, amplitude: float) -> dict:
        """Return the IQ data at the closest (length, amplitude) setting."""
        l_key = min(self.acquired_data.keys(),
                    key=lambda x: abs(float(x) - length))
        a_key = min(
            self.acquired_data[l_key].keys(),
            key=lambda x: abs(float(x) - amplitude),
        )
        return self.acquired_data[str(l_key)][str(a_key)]["IQ_data"]
    # -------------------------------------------------------------------------
    # Convenience
    # -------------------------------------------------------------------------

    def plot_single_iq_data(self, length: float, amplitude: float, axs=None, **kargs) -> None:
        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        data = _give(self.acquired_data, float(length), float(amplitude))
        IQ_data = data["IQ_data"]

        plot_single_iq_data(axs[0], axs[1], IQ_data, **kargs, ax=axs[0])

        fidelity = data["fidelity"]
        seperation = find_separation(IQ_data)
        deveiation = find_deveiation(IQ_data, plot=True, ax=axs[0])

        print(f"Fidelity: {fidelity:.2f} %")
        print(f"Separation: {seperation:.2f} a.u.")
        print(f"Gaussian Deviation: {deveiation:.2f} a.u.")

    def plot_fidelity_map(self, ax, point: tuple[float, float] = None, **kargs) -> None:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        c = plot_map(
            ax,
            self.lengths,
            self.amplitudes,
            self.fidelity_map.T,
            **kargs,
        )

        if not point is None:
            ax.plot(point[0]/1e-9, point[1], 'ro', mfc='none', mec='r', markeredgewidth=3,
                    markersize=13, label=f'Chosen Point fidelity={self.get_fidelity_at(point[0], point[1]):.2f}%')
            ax.legend(loc='upper right')

        ax.set_xlabel("Readout Length (ns)")
        ax.set_ylabel("Readout Amplitude (V)")

        return c

    def __repr__(self) -> str:
        return (
            f"ReadoutDataset(qubit='{self.qubit}', with_twpa={self.with_twpa}, "
            f"file='{self.file_name.name}')"
        )


def _give(acquired_data, length: float, amplitude: float):
    l_key = min(acquired_data.keys(),
                key=lambda x: abs(float(x) - length))
    a_key = min(
        acquired_data[l_key].keys(),
        key=lambda x: abs(float(x) - amplitude),
    )
    return acquired_data[str(l_key)][str(a_key)]


if __name__ == "__main__":

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    dataset = ReadoutDataset("q16", with_twpa=False, file_type="gz")

    dataset.plot_single_iq_data(length=300.0e-9, amplitude=0.1, axs=axs[:2])
    dataset.plot_fidelity_map(ax=axs[2], point=(300.0e-9, 0.1))
    plt.show()
