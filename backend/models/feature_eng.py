"""
Feature Engineering — Server-Side Validation & Constants
========================================================
Ensures consistency between frontend and backend feature definitions.
"""

# Feature names must match frontend/js/featureEngine.js exactly
FEATURE_NAMES = [
    'shoulder_angle',        # 0
    'neck_inclination',      # 1
    'torso_inclination',     # 2
    'hip_angle',             # 3
    'head_lateral_offset',   # 4
    'left_torso_ratio',      # 5
    'right_torso_ratio',     # 6
    'symmetry_score',        # 7
    'head_drop_ratio',       # 8
    'ear_alignment',         # 9
    'shoulder_hip_width',    # 10
    'avg_confidence',        # 11
]

NUM_FEATURES = len(FEATURE_NAMES)

POSTURE_LABELS = [
    'Good Posture',
    'Forward Lean',
    'Left Lean',
    'Right Lean',
    'Neck Strain',
]

NUM_CLASSES = len(POSTURE_LABELS)

# Valid ranges for each feature (for anomaly detection)
FEATURE_RANGES = {
    'shoulder_angle':      (0, 45),
    'neck_inclination':    (0, 60),
    'torso_inclination':   (0, 60),
    'hip_angle':           (0, 45),
    'head_lateral_offset': (-1.0, 1.0),
    'left_torso_ratio':    (0, 2.0),
    'right_torso_ratio':   (0, 2.0),
    'symmetry_score':      (0, 1.0),
    'head_drop_ratio':     (-0.5, 1.0),
    'ear_alignment':       (0, 60),
    'shoulder_hip_width':  (0.5, 3.0),
    'avg_confidence':      (0, 1.0),
}

# Thresholds for rule-based classification
THRESHOLDS = {
    'shoulder_angle_max':    12,
    'neck_inclination_max':  20,
    'torso_inclination_max': 18,
    'head_offset_max':       0.3,
    'symmetry_min':          0.7,
    'ear_alignment_max':     25,
}


class FeatureValidator:
    """Validate feature vectors and detect anomalies."""

    def validate(self, features) -> list:
        """
        Check feature values are in expected ranges.
        Returns list of issues/warnings.
        """
        issues = []
        for i, (name, (lo, hi)) in enumerate(FEATURE_RANGES.items()):
            if i < len(features):
                val = features[i]
                if val < lo or val > hi:
                    issues.append(f"{name} out of range: {val:.3f} (expected {lo}–{hi})")
        return issues

    def check_thresholds(self, features) -> dict:
        """Apply threshold-based posture checks."""
        results = {}
        results['shoulder_ok'] = features[0] <= THRESHOLDS['shoulder_angle_max']
        results['neck_ok'] = features[1] <= THRESHOLDS['neck_inclination_max']
        results['torso_ok'] = features[2] <= THRESHOLDS['torso_inclination_max']
        results['head_ok'] = abs(features[4]) <= THRESHOLDS['head_offset_max']
        results['symmetry_ok'] = features[7] >= THRESHOLDS['symmetry_min']
        results['ear_ok'] = features[9] <= THRESHOLDS['ear_alignment_max']
        results['all_ok'] = all(results.values())
        return results
