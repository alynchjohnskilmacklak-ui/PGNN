import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
import multiprocessing

from ballistics import (
    ProjectileParams,
    simulate_trajectory,
    interpolate_yz_on_xgrid,
    get_atmosphere,
)
from train_model import LOW_THETA_MAX, HIGH_THETA_MIN

# =========================================================
# 药温耦合工具
# =========================================================


def isa_pressure(alt_m: float, P_sea: float = 101325.0) -> float:
    return P_sea * (1.0 - 2.25577e-5 * alt_m) ** 5.25588


# =========================================================
# 🎯 武器库配置区
# =========================================================
AMMO_CONFIGS = {
    "12.7mm_B32": {
        "mass": 0.04828,
        "caliber": 0.01295,
        "v0_base": 840.0,
        "out_dir": "artifacts_127"
    },
    "14.5mm_B32": {
        "mass": 0.06344,
        "caliber": 0.01450,
        "v0_base": 980.0,
        "out_dir": "artifacts_145"
    }
}

CURRENT_AMMO = "12.7mm_B32"

FEATURE_COLS = [
    "x", "y", "z", "v0_actual", "rho", "slant_range",
    "wind_x", "wind_y", "wind_z", "cant_angle", "T_powder_C",
    "in_low_branch", "in_high_branch", "T0_C", "P0_Pa", "alt_gun",
]
LABEL_COLS = ["alpha_deg", "theta_deg"]
COLUMN_INDEX = {name: i for i, name in enumerate(FEATURE_COLS + LABEL_COLS)}


def _stratified_alt_samples(rng: np.random.Generator, alt_range, n_samples: int, alt_bins: int):
    alt_min, alt_max = float(alt_range[0]), float(alt_range[1])
    if n_samples <= 0: return [], 0
    alt_bins = int(min(max(1, alt_bins), n_samples))
    edges = np.linspace(alt_min, alt_max, alt_bins + 1)
    base, rem = n_samples // alt_bins, n_samples % alt_bins
    alts = []
    for b in range(alt_bins):
        k = base + (1 if b < rem else 0)
        alts.extend(rng.uniform(edges[b], edges[b + 1], size=k).tolist())
    rng.shuffle(alts)
    return alts, alt_bins


def _build_theta_list(theta_min: float, theta_max: float, theta_step: float):
    if theta_step <= 0: raise ValueError("theta_step must be positive.")
    thetas = np.arange(theta_min, theta_max + 1e-12, theta_step, dtype=np.float64)
    if len(thetas) < 2: raise ValueError("Need at least two angle samples.")
    return thetas.tolist()


def _process_single_angle_worker(args):
    # ✅ 新增：alpha_range (方向角采样范围)
    (th, seed, alt_range, samples_per_angle, alt_bins, v0_base,
     wind_x_range, wind_y_range, wind_z_range, cant_angle_range,
     alpha_range,                      # ← 新增
     T0_C_range, P0_Pa_range, dt, t_max, base_params_dict) = args

    rng = np.random.default_rng(seed)
    local_trajectories = []
    alts, bins_used = _stratified_alt_samples(rng, alt_range, samples_per_angle, alt_bins)

    for alt_m in alts:
        wind_x    = float(rng.uniform(wind_x_range[0],    wind_x_range[1]))
        wind_y    = float(rng.uniform(wind_y_range[0],    wind_y_range[1]))
        wind_z    = float(rng.uniform(wind_z_range[0],    wind_z_range[1]))
        cant_angle= float(rng.uniform(cant_angle_range[0],cant_angle_range[1]))
        # ✅ 采样方向角 alpha (方位角, 偏航方向偏差)
        alpha_deg = float(rng.uniform(alpha_range[0], alpha_range[1]))

        T0_C  = float(rng.uniform(T0_C_range[0],  T0_C_range[1]))
        P0_Pa = float(isa_pressure(alt_m) * rng.uniform(0.95, 1.05))

        T_powder = T0_C + float(rng.normal(0, 5.0))

        T0_K = T0_C + 273.15
        rho, _ = get_atmosphere(alt_m, alt_m, T0_K, P0_Pa)

        p = ProjectileParams(**base_params_dict)
        p.v0_base       = v0_base
        p.T_powder_C    = T_powder
        p.wind_x        = wind_x
        p.wind_y        = wind_y
        p.wind_z        = wind_z
        p.cant_angle_deg= cant_angle
        p.alt_gun       = float(alt_m)
        p.T0_C          = T0_C
        p.P0_Pa         = float(P0_Pa)

        # ✅ 将采样到的方向角 alpha_deg 传入弹道仿真
        traj = simulate_trajectory(
            theta_deg=th,
            alpha_deg=alpha_deg,   # ← 真实方向角参与物理仿真
            params=p,
            dt=dt,
            t_max=t_max
        )

        traj["T_powder_C"]    = T_powder
        traj["rho"]           = float(rho)
        traj["alt_gun"]       = float(alt_m)
        traj["wind_x"]        = wind_x
        traj["wind_y"]        = wind_y
        traj["wind_z"]        = wind_z
        traj["cant_angle_deg"]= cant_angle
        traj["alpha_deg"]     = alpha_deg   # ✅ 记录真实方向角标签
        traj["T0_C"]          = T0_C
        traj["P0_Pa"]         = P0_Pa

        local_trajectories.append(traj)

    return local_trajectories, bins_used


def generate_dataset(
        out_dir: str | None = None,
        theta_min: int = 0,
        theta_max: int = 75,
        theta_step: float = 1.0,
        dt: float = 0.05,
        x_step: float = 10.0,
        t_max: float = 200.0,
        params: ProjectileParams | None = None,
        plot_examples: bool = True,
        samples_per_angle: int = 30,
        alt_range=(0.0, 2000.0),
        alt_bins: int = 30,
        keep_prob: float = 0.8,
        wind_x_range=(-20.0, 20.0),
        wind_y_range=(-5.0, 5.0),
        wind_z_range=(-15.0, 15.0),
        cant_angle_range=(-15.0, 15.0),
        alpha_range=(-10.0, 10.0),        # ✅ 新增：方向角范围 (度)
        T0_C_range=(-30.0, 45.0),
        P0_Pa_range=(60000.0, 105000.0),
        seed: int = 42,
) -> dict:
    cfg = AMMO_CONFIGS[CURRENT_AMMO]
    if out_dir is None: out_dir = cfg["out_dir"]
    v0_base = cfg["v0_base"]

    os.makedirs(out_dir, exist_ok=True)
    if params is None: params = ProjectileParams(mass=cfg["mass"], caliber=cfg["caliber"])

    thetas = _build_theta_list(theta_min, theta_max, theta_step)
    trajectories, ranges = [], []

    print(f"🚀 准备生成 3D 实战级 [{CURRENT_AMMO}] 弹道数据: 共 {len(thetas)} 个角度...")

    worker_args = []
    for i, th in enumerate(thetas):
        args = (
            th, seed + i, alt_range, samples_per_angle, alt_bins, v0_base,
            wind_x_range, wind_y_range, wind_z_range, cant_angle_range,
            alpha_range,                  # ✅ 传入方向角范围
            T0_C_range, P0_Pa_range, dt, t_max, params.__dict__
        )
        worker_args.append(args)

    completed_count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, os.cpu_count() // 2)) as executor:
        future_to_th = {executor.submit(_process_single_angle_worker, arg): arg[0] for arg in worker_args}
        for future in concurrent.futures.as_completed(future_to_th):
            th_val = future_to_th[future]
            try:
                local_trajs, _ = future.result()
                trajectories.extend(local_trajs)
                ranges.extend([t["range"] for t in local_trajs])
                completed_count += 1
                if completed_count % 5 == 0 or completed_count == len(thetas):
                    print(f"✅ 进度: {completed_count}/{len(thetas)}")
            except Exception as exc:
                print(f"❌ 角度 {th_val:.1f}° 产生异常: {exc}")

    max_range = float(np.max(ranges)) if ranges else 0.0
    if max_range <= 0: raise RuntimeError("Max range <= 0. Check parameters.")

    x_grid = np.arange(0.0, max_range + 1e-9, x_step, dtype=np.float64)
    rng = np.random.default_rng(seed)
    rows = []

    for traj in trajectories:
        th       = float(traj["theta_deg"])
        alpha    = float(traj["alpha_deg"])   # ✅ 读取真实方向角
        y_grid, z_grid = interpolate_yz_on_xgrid(traj, x_grid)

        valid   = np.isfinite(y_grid) & (y_grid >= 0.0)
        x_valid = x_grid[valid]
        y_valid = y_grid[valid]
        z_valid = z_grid[valid]

        if 0.0 < keep_prob < 1.0 and len(x_valid) > 0:
            mask    = rng.random(len(x_valid)) < keep_prob
            x_valid = x_valid[mask]
            y_valid = y_valid[mask]
            z_valid = z_valid[mask]

        in_low_branch = 1.0 if th <= LOW_THETA_MAX else 0.0
        in_high_branch = 1.0 if th >= HIGH_THETA_MIN else 0.0

        for xv, yv, zv in zip(x_valid, y_valid, z_valid):
            slant_range = float(np.sqrt(xv**2 + yv**2 + zv**2))
            rows.append((
                float(xv), float(yv), float(zv),
                float(traj["v0_actual"]),
                float(traj["rho"]),
                slant_range,
                float(traj["wind_x"]),
                float(traj["wind_y"]),
                float(traj["wind_z"]),
                float(traj["cant_angle_deg"]),
                float(traj["T_powder_C"]),
                in_low_branch,
                in_high_branch,
                float(traj["T0_C"]),
                float(traj["P0_Pa"]),
                float(traj["alt_gun"]),
                alpha,
                float(th),
            ))

    cols = FEATURE_COLS + LABEL_COLS
    df = pd.DataFrame(rows, columns=cols)
    df = df.sort_values(by=["theta_deg", "alpha_deg", "v0_actual", "x"],
                        kind="mergesort").reset_index(drop=True)

    print(f"📊 提取完毕，总计生成了 {len(df)} 行 3D 实战级训练数据！正在保存...")

    csv_path, npy_path, meta_path = [
        os.path.join(out_dir, f) for f in ("dataset.csv", "dataset.npy", "dataset_meta.json")
    ]
    df.to_csv(csv_path, index=False)
    np.save(npy_path, df[cols].to_numpy(dtype=np.float32))

    meta = {
        "ammo_type":        CURRENT_AMMO,
        "mass_kg":          cfg["mass"],
        "caliber_m":        cfg["caliber"],
        "v0_base":          float(v0_base),
        "max_range":        float(max_range),
        "num_samples":      int(len(df)),
        "samples_per_angle":int(samples_per_angle),
        "alpha_range_deg":  list(alpha_range),
        "note": "3D Unified dataset with wind_z, cant_angle, T_powder, and alpha_deg label."
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "out_dir":     out_dir,
        "csv_path":    csv_path,
        "npy_path":    npy_path,
        "num_samples": int(len(df))
    }


if __name__ == "__main__":
    multiprocessing.freeze_support()
    info = generate_dataset()
    print("Dataset generated:", info)
