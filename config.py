from pathlib import Path


PARENT_DIR = Path(__file__).parent

DATA_DIR = PARENT_DIR / "data"  # relative to config.py location

QUBITS = [f'q{i}' for i in range(1, 21)]


# print(f"Data directory is set to: {DATA_DIR.resolve()}")

if __name__ == "__main__":
    print(PARENT_DIR.resolve())
    print(DATA_DIR.resolve())
