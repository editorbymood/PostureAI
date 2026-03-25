"""
Dataset Handling — Loading, Synthetic Generation & Augmentation
================================================================
Handles both real collected data and synthetic data generation.
Synthetic data is used for initial model testing and demonstration.
"""

import os
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.feature_eng import NUM_FEATURES, NUM_CLASSES, POSTURE_LABELS


class PostureDataset:
    """Load posture data from .npy or .csv files in a directory."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load(self) -> tuple:
        """Load all data files, returns (features, labels)."""
        all_features = []
        all_labels = []

        if not self.data_dir.exists():
            return None, None

        # Load .npy files
        for f in self.data_dir.glob("*.npy"):
            try:
                data = np.load(f, allow_pickle=True)
                for sample in data:
                    if isinstance(sample, dict):
                        features = sample.get('features', [])
                        label = sample.get('label', 0)
                        if len(features) == NUM_FEATURES:
                            all_features.append(features)
                            all_labels.append(label)
            except Exception as e:
                print(f"⚠️ Error loading {f}: {e}")

        # Load .csv files
        for f in self.data_dir.glob("*.csv"):
            try:
                data = np.loadtxt(f, delimiter=',', skiprows=1)
                # Expect: 12 features + 1 label column
                if data.shape[1] == NUM_FEATURES + 1:
                    all_features.extend(data[:, :NUM_FEATURES].tolist())
                    all_labels.extend(data[:, -1].astype(int).tolist())
            except Exception as e:
                print(f"⚠️ Error loading {f}: {e}")

        if len(all_features) == 0:
            return None, None

        return np.array(all_features, dtype=np.float32), np.array(all_labels, dtype=np.int64)


def load_all_data(data_dir: str, temporal: bool = False,
                  window_size: int = 30) -> tuple:
    """
    Load data and optionally create temporal windows.

    Args:
        data_dir: Path to data directory
        temporal: If True, create sliding windows for LSTM
        window_size: Number of frames per window
    """
    dataset = PostureDataset(data_dir)
    X, y = dataset.load()

    if X is None:
        return None, None

    if temporal:
        return create_windows(X, y, window_size)

    return X, y


def create_windows(X: np.ndarray, y: np.ndarray,
                   window_size: int = 30) -> tuple:
    """Create sliding windows from sequential data."""
    if len(X) < window_size:
        # Pad with copies
        pad = np.tile(X[0], (window_size - len(X), 1))
        X = np.vstack([pad, X])
        y = np.concatenate([np.full(window_size - len(y), y[0]), y])

    windows = []
    labels = []
    for i in range(len(X) - window_size + 1):
        windows.append(X[i:i + window_size])
        labels.append(y[i + window_size - 1])  # Label = last frame

    return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.int64)


def generate_synthetic_data(n_samples: int = 2000, temporal: bool = False,
                            window_size: int = 30) -> tuple:
    """
    Generate synthetic posture data for testing.

    Simulates realistic feature distributions for each posture class
    so we can train and evaluate models without real data.
    """
    np.random.seed(42)
    samples_per_class = n_samples // NUM_CLASSES

    all_X = []
    all_y = []

    # Define feature distributions per class:
    #   (mean, std) for each of the 12 features
    class_distributions = {
        0: {  # Good Posture
            'shoulder_angle': (3, 3),     'neck_inclination': (8, 5),
            'torso_inclination': (5, 4),  'hip_angle': (2, 2),
            'head_lateral_offset': (0, 0.08),
            'left_torso_ratio': (0.45, 0.05),  'right_torso_ratio': (0.45, 0.05),
            'symmetry_score': (0.92, 0.04),
            'head_drop_ratio': (0.02, 0.03),  'ear_alignment': (8, 5),
            'shoulder_hip_width': (1.3, 0.15), 'avg_confidence': (0.85, 0.08),
        },
        1: {  # Forward Lean
            'shoulder_angle': (5, 4),     'neck_inclination': (30, 8),
            'torso_inclination': (25, 7), 'hip_angle': (3, 3),
            'head_lateral_offset': (0, 0.1),
            'left_torso_ratio': (0.48, 0.06),  'right_torso_ratio': (0.48, 0.06),
            'symmetry_score': (0.85, 0.06),
            'head_drop_ratio': (0.18, 0.06),  'ear_alignment': (28, 8),
            'shoulder_hip_width': (1.15, 0.15), 'avg_confidence': (0.8, 0.1),
        },
        2: {  # Left Lean
            'shoulder_angle': (18, 5),    'neck_inclination': (12, 6),
            'torso_inclination': (15, 6), 'hip_angle': (10, 5),
            'head_lateral_offset': (-0.35, 0.1),
            'left_torso_ratio': (0.42, 0.06),  'right_torso_ratio': (0.5, 0.06),
            'symmetry_score': (0.65, 0.08),
            'head_drop_ratio': (0.05, 0.04),  'ear_alignment': (15, 6),
            'shoulder_hip_width': (1.25, 0.15), 'avg_confidence': (0.82, 0.09),
        },
        3: {  # Right Lean
            'shoulder_angle': (18, 5),    'neck_inclination': (12, 6),
            'torso_inclination': (15, 6), 'hip_angle': (10, 5),
            'head_lateral_offset': (0.35, 0.1),
            'left_torso_ratio': (0.5, 0.06),   'right_torso_ratio': (0.42, 0.06),
            'symmetry_score': (0.65, 0.08),
            'head_drop_ratio': (0.05, 0.04),  'ear_alignment': (15, 6),
            'shoulder_hip_width': (1.25, 0.15), 'avg_confidence': (0.82, 0.09),
        },
        4: {  # Neck Strain
            'shoulder_angle': (6, 4),     'neck_inclination': (22, 7),
            'torso_inclination': (10, 5), 'hip_angle': (3, 3),
            'head_lateral_offset': (0.05, 0.12),
            'left_torso_ratio': (0.46, 0.05),  'right_torso_ratio': (0.46, 0.05),
            'symmetry_score': (0.88, 0.05),
            'head_drop_ratio': (0.12, 0.05),  'ear_alignment': (35, 8),
            'shoulder_hip_width': (1.2, 0.15), 'avg_confidence': (0.83, 0.09),
        },
    }

    for class_idx in range(NUM_CLASSES):
        dist = class_distributions[class_idx]
        features = np.zeros((samples_per_class, NUM_FEATURES), dtype=np.float32)

        for i, key in enumerate(dist):
            mean, std = dist[key]
            features[:, i] = np.random.normal(mean, std, samples_per_class)

        # Clip to valid ranges
        features[:, 0] = np.clip(features[:, 0], 0, 45)   # shoulder_angle
        features[:, 1] = np.clip(features[:, 1], 0, 60)   # neck
        features[:, 2] = np.clip(features[:, 2], 0, 60)   # torso
        features[:, 3] = np.clip(features[:, 3], 0, 45)   # hip
        features[:, 4] = np.clip(features[:, 4], -1, 1)   # head offset
        features[:, 7] = np.clip(features[:, 7], 0, 1)    # symmetry
        features[:, 11] = np.clip(features[:, 11], 0, 1)  # confidence

        all_X.append(features)
        all_y.append(np.full(samples_per_class, class_idx, dtype=np.int64))

    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    # Shuffle
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    if temporal:
        return create_windows_synthetic(X, y, window_size)

    return X, y


def create_windows_synthetic(X, y, window_size):
    """Create temporal windows from synthetic data."""
    windows = []
    labels = []

    for class_idx in range(NUM_CLASSES):
        mask = y == class_idx
        class_X = X[mask]
        n_windows = len(class_X) // window_size

        for i in range(n_windows):
            window = class_X[i * window_size:(i + 1) * window_size]
            # Add temporal noise
            noise = np.random.normal(0, 0.5, window.shape).astype(np.float32)
            window = window + noise
            windows.append(window)
            labels.append(class_idx)

    return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.int64)


if __name__ == "__main__":
    # Test synthetic data generation
    X, y = generate_synthetic_data(1000, temporal=False)
    print(f"Single-frame: X={X.shape}, y={y.shape}")
    print(f"  Class counts: {np.bincount(y)}")

    X_t, y_t = generate_synthetic_data(1000, temporal=True, window_size=30)
    print(f"Temporal: X={X_t.shape}, y={y_t.shape}")
    print(f"  Class counts: {np.bincount(y_t)}")
