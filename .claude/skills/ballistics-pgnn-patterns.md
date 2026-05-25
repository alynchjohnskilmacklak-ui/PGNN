---
name: ballistics-pgnn-patterns
description: Coding patterns extracted from ballistic-pgnn — physics-guided neural networks for projectile trajectory prediction with differentiable simulation.
version: 1.0.0
source: local-git-analysis
analyzed_commits: 7
generated: 2026-05-13
---

# Ballistics PGNN Patterns

## Project Summary

Physics-Guided Neural Network (PGNN) for predicting firing angles (theta, alpha) to hit a 3D target under environmental conditions (wind, altitude, temperature, pressure, cant angle). Uses differentiable RK4 trajectory simulation inside the training loss to enforce physical consistency.

## Commit Conventions

This project uses **Chinese-language checkpoint commits** with descriptive titles:

```
存档点 #N: <description>
```

- `#1`: Initial version
- `#2`: Bug fixes (P0/P1)
- `#3`: Dev tooling (VS Code debug config)
- `#4`: New module (Benchmark)
- `#5`: Feature implementation from spec
- `#6`: Major upgrade (PGNN differentiable sim)
- `#7`: Performance fix (training speed)

7 commits total, all on 2026-05-08. Short-lived project with rapid iteration.

## Code Architecture

```
project_root/
├── ballistics.py        # Physics engine (atmosphere, drag, RK4, trajectory)
├── generate_dataset.py  # Parallel trajectory dataset generation
├── train_model.py       # NN architecture + training loop + physics loss
├── predict.py           # Inference: NN prediction + Newton-Raphson refinement
├── main.py              # Pipeline orchestration + benchmark experiments
├── Benchmark.py         # Benchmark comparison framework
├── test_mini_rk4.py     # Numerical validation (NumPy ref vs PyTorch impl)
├── artifacts_127/       # Output directory for trained models
└── artifacts_<caliber>/ # Per-ammo output directories
```

Flat structure — no subpackages. Each file is a self-contained module with clear responsibility:
- **Physics** (`ballistics.py`): Pure NumPy, no ML dependencies
- **Data** (`generate_dataset.py`): Multiprocessing trajectory generation
- **Training** (`train_model.py`): PyTorch model, dataset, physics loss, training loop
- **Inference** (`predict.py`): Model loading, prediction, Newton-Raphson refinement
- **Orchestration** (`main.py`): Pipeline, experiments, CLI

## Key Patterns

### Dataclass Configuration

Use `@dataclass` for typed configuration objects:

```python
@dataclass
class ProjectileParams:
    mass: float = 0.04828
    caliber: float = 0.01295
    v0_base: float = 840.0
    # ...

@dataclass
class TrainConfig:
    out_dir: str = "artifacts"
    seed: int = 42
    batch_size: int = 1536
    # ...
```

### Min-Max Normalization

Features are normalized to [0, 1] via min-max scaling. Scaling parameters are saved as JSON:

```python
def minmax_fit(X: np.ndarray) -> dict:
    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    span = np.where((xmax - xmin) < 1e-12, 1.0, (xmax - xmin))
    return {"xmin": xmin.tolist(), "xmax": xmax.tolist(), "span": span.tolist()}

def minmax_transform(X: np.ndarray, scaler: dict) -> np.ndarray:
    xmin = np.array(scaler["xmin"], dtype=np.float32)
    span = np.array(scaler["span"], dtype=np.float32)
    return (X - xmin) / span
```

Scaler saved as JSON alongside model weights.

### Dual-Branch Architecture (Low/High Angle)

Two separate `SingleBranchDNN` models are trained — one for low angles (0-55°) and one for high angles (45-85°). The overlap zone (45-55°) is handled by both models. At inference time, both branches predict and the best solution is chosen:

```python
# Training
train(trajectory_mode=0)  # low branch
train(trajectory_mode=1)  # high branch

# Inference
th_low, al_low = low_model.predict(X)
th_high, al_high = high_model.predict(X)
# Both refined via Newton-Raphson, best chosen by min(err_3d, t_hit)
```

### Physics-Guided Loss (PGNN)

The differentiable physics loss runs mini-RK4 integration inside PyTorch:

```python
class SmoothPhysicsLoss(nn.Module):
    def forward(self, pred_angles, y_true, X_raw):
        # 1. Build initial state from predicted angles
        state = self._make_initial_state_batch(pred_angles, v0, cant_angle)
        # 2. Integrate via mini_rk4_step over N steps
        for _ in range(max_steps):
            state = mini_rk4_step(state, dt, env)
            # 3. Detect x-target crossing, interpolate y_hit, z_hit
        # 4. Loss = smooth_l1(y_hit - y_target) + smooth_l1(z_hit - z_target)
        # 5. Missed targets get x-shortfall penalty
```

Physics loss uses **scheduled warmup** (epochs 1-8: disabled, 9-18: progressive, 19+: full) to avoid early instability from large gradient discrepancies.

### Chunked Forward for Physics Loss

Large batch physics integration is handled in chunks of 256 to manage memory:

```python
if B <= chunk_size:
    return self._forward_chunk(...)
for start in range(0, B, chunk_size):
    chunk_loss = self._forward_chunk(...)
return torch.cat(all_losses, dim=0)
```

### Model Persistence

```
artifacts_<caliber>/
├── dataset.npy              # Training data (NumPy)
├── dataset.csv              # Training data (CSV)
├── dataset_meta.json        # Mass, caliber, v0, sample count
├── dual_model_low.pt        # PyTorch state_dict
├── dual_scaler_low.json     # Min-max scaler parameters
├── train_history_low.json   # Per-epoch metrics
├── train_history_low.csv    # Per-epoch metrics (CSV)
├── loss_curve_low.png       # Training curve plot
├── dual_model_high.pt
├── dual_scaler_high.json
├── train_history_high.json
├── train_history_high.csv
├── loss_curve_high.png
└── pipeline_result_refined.json  # Final evaluation summary
```

Models are saved as `state_dict` via `torch.save(model.state_dict(), path)`, loaded with `model.load_state_dict(torch.load(path))`.

### Inference: NN → Newton-Raphson → Choose Best

The full solve pipeline:

1. **NN Predict**: Low and high branches each predict (theta, alpha)
2. **Newton-Raphson Refine**: Central-difference Jacobian, line-search step acceptance
3. **Grid Fallback**: If Newton fails to converge to tolerance, coarse-to-fine grid search
4. **Choose Best**: Min of (err_3d, t_hit) across converged solutions
5. **Save Artifacts**: 3D trajectory plot, error comparison bar chart, CSV comparison table

```python
result = solve_target_unified(
    x_target, y_target, z_target,
    v0_actual, rho, wind_x, wind_y, wind_z,
    cant_angle, T_powder_C, alt_gun, T0_C, P0_Pa,
    dir_path="artifacts", save_plot=True
)
```

### Trajectory Group Split

Data is split by trajectory group (unique combination of theta, alpha, v0, rho, wind, cant, T0, P0, alt) to prevent data leakage — all samples from the same trajectory go to the same split:

```python
group_ids, _ = build_trajectory_groups(arr)
split = trajectory_group_split(group_ids, seed=42, ratios=(0.70, 0.15, 0.15))
```

### Naming Conventions

- **Functions/Variables**: `snake_case`
- **Classes**: `PascalCase` (e.g., `ProjectileParams`, `AngleDataset`, `ResidualBlock`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `LOW_THETA_MAX`, `ALPHA_ABS_MAX`)
- **Private helpers**: Leading underscore (e.g., `_make_initial_state`, `_forward_chunk`, `_per_elem_smooth_l1`)
- **Torch equivalents**: Suffixed with `_torch` (e.g., `smooth_cd_g7_torch`, `isa_density_torch`)

### Parallel Dataset Generation

Uses `concurrent.futures.ProcessPoolExecutor` with `os.cpu_count() // 2` workers:

```python
with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, os.cpu_count() // 2)) as executor:
    future_to_th = {executor.submit(_process_single_angle_worker, arg): arg[0] for arg in worker_args}
    for future in concurrent.futures.as_completed(future_to_th):
        local_trajs, _ = future.result()
        trajectories.extend(local_trajs)
```

Requires `multiprocessing.freeze_support()` on Windows (`if __name__ == "__main__": freeze_support()`).

### Training Infrastructure

- **Optimizer**: AdamW with fused kernel when available (`fused=True` fallback)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=4)
- **Gradient clipping**: `clip_grad_norm_(max_norm=1.0)`
- **AMP**: `torch.amp.autocast("cuda")` for fp16 forward, physics loss forced to fp32
- **EMA**: `ModelEMA` class with decay=0.999, applied during evaluation
- **Early stopping**: Based on validation loss, patience configurable
- **NaN/Inf protection**: Parameter health check per epoch, per-batch loss check
- **Input noise**: Training-only Gaussian noise (`xb + torch.randn_like(xb) * 0.001`)

### Visualization Conventions

- High DPI: `dpi=400` for loss curves, `dpi=220` for prediction plots
- `bbox_inches="tight"`, `plt.tight_layout()`
- Separate figure per plot (no subplots in a single figure)
- Consistent color palette for error bars: `["#4C78A8", "#72B7B2", "#F58518", "#E45756", "#54A24B"]`

## Workflows

### Adding a New Ammo Type

1. Add entry to `AMMO_CONFIGS` in `generate_dataset.py` with mass, caliber, v0_base, out_dir
2. Set `CURRENT_AMMO` to the new key
3. Run `main.py --mode full` to generate dataset + train both branches

### Running a Full Pipeline

```bash
python main.py --mode full
```

Generates dataset → trains low model → trains high model → verifies with 500 test samples → saves all results.

### Running Benchmark Experiments (B2)

```bash
python main.py --mode b2 --run_e3
```

Runs controlled experiments comparing pure-data vs PGNN at different data sizes.

### Making a Prediction

```python
from predict import solve_target_unified
result = solve_target_unified(x_target=800, y_target=5, z_target=0, ...)
```

## Testing Patterns

- Single validation script (`test_mini_rk4.py`) — numerical comparison only
- Not pytest-based; runs as standalone script with `exit(1)` on failure
- Tests the differentiable physics implementation against the NumPy reference
- Test cases cover: angle ranges, cant_angle, wind, altitude combinations
- Thresholds: position error < 2m, velocity error < 2m/s at 1.0s simulation

## Feature Vector Layout

18 input features (after removing branch indicator columns):

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | x | Target x (m) |
| 1 | y | Target y (m) |
| 2 | z | Target z (m) |
| 3 | v0_actual | Actual muzzle velocity (m/s) |
| 4 | rho | Air density at gun (kg/m³) |
| 5 | slant_range | 3D distance to target |
| 6 | wind_x | Wind x-component (m/s) |
| 7 | wind_y | Wind y-component (m/s) |
| 8 | wind_z | Wind z-component (m/s) |
| 9 | cant_angle | Gun cant angle (deg) |
| 10 | T_powder_C | Powder temperature (°C) |
| 11 | theta_low_vac | Vacuum low angle (deg) |
| 12 | theta_high_vac | Vacuum high angle (deg) |
| 13 | alpha_geom | Geometric azimuth (deg) |
| 14 | t_flight_est | Vacuum flight time (s) |
| 15 | T0_C | Air temperature at gun (°C) |
| 16 | P0_Pa | Air pressure at gun (Pa) |
| 17 | alt_gun | Gun altitude (m) |

2 output labels: `[alpha_deg, theta_deg]` (order: alpha first, then theta — verified via `LABEL_COLS = ["alpha_deg", "theta_deg"]`)
