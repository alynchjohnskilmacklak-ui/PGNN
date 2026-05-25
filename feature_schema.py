"""Single source of truth for feature/label column definitions and indices.

All modules that construct or consume the 14-feature training input or the
16-feature raw dataset must import column names and indices from here.
"""

FEATURE_COLS = [
    "x", "y", "z", "v0_actual", "rho", "slant_range",
    "wind_x", "wind_y", "wind_z", "cant_angle", "T_powder_C",
    "in_low_branch", "in_high_branch", "T0_C", "P0_Pa", "alt_gun",
]
LABEL_COLS = ["alpha_deg", "theta_deg"]

_ALL_COLS = FEATURE_COLS + LABEL_COLS
COLUMN_INDEX: dict[str, int] = {name: i for i, name in enumerate(_ALL_COLS)}

BRANCH_COLS = ("in_low_branch", "in_high_branch")

TRAINING_FEATURE_COLS = [name for name in FEATURE_COLS if name not in BRANCH_COLS]
N_TRAINING_FEATURES = len(TRAINING_FEATURE_COLS)


def training_feature_idx(name: str) -> int:
    """Return the column index of *name* in the 14-feature training array
    (after removing in_low_branch and in_high_branch)."""
    raw_idx = COLUMN_INDEX[name]
    dropped = sum(1 for bc in BRANCH_COLS if COLUMN_INDEX[bc] < raw_idx)
    return raw_idx - dropped


def build_inference_input(
    x_target: float, y_target: float, z_target: float,
    v0_actual: float, rho: float,
    wind_x: float, wind_y: float, wind_z: float,
    cant_angle: float, T_powder_C: float,
    T0_C: float, P0_Pa: float, alt_gun: float,
) -> "np.ndarray":
    """Build a (1, 14) float32 array suitable for model inference."""
    import numpy as np
    slant_range = float(np.sqrt(x_target ** 2 + y_target ** 2 + z_target ** 2))
    return np.array(
        [[x_target, y_target, z_target, v0_actual, rho, slant_range,
          wind_x, wind_y, wind_z, cant_angle, T_powder_C,
          T0_C, P0_Pa, alt_gun]],
        dtype=np.float32,
    )
