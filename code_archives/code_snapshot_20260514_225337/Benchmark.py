"""
benchmark_methods.py
=====================

独立对照实验：在干净的测试集上对比 4 种弹道求解策略，回答
"NN pipeline 是否真的比简单方法好" 这个问题。

四种方法
--------
  A: 真空解析解 -> 牛顿法
  B: 固定初值 (30°/65°, alpha=0) -> 牛顿法
  C: NN 初值 -> 现有 solve_target_unified（grid + 牛顿）
  D: 纯 NN，不做 refine（衡量 NN 单独有多准）
  E: NN 初值 -> 牛顿法（跳过 grid search） <-- 你的设计原意候选

核心对照
--------
  C vs E：测 grid search 到底有没有在帮忙 / 是不是历史包袱
  A vs E：测 NN 初值是不是真的比真空解析初值更好
  D     ：诊断 NN 输出离牛顿吸引域有多远（< ~5° 才能保证 E 稳定）

每个方法报告
------------
  err_3d:    最终 3D 命中误差（米）
  n_sim:     simulate_trajectory 的调用次数（计算成本的真实代理）
  wall_ms:   单次查询墙钟时间（毫秒）
  converged: 是否落在 (y_tol, z_tol) 容差内
  iters:     牛顿迭代次数（仅 A/B 适用，C 含 grid 不计）

设计要点
--------
1) 测试集独立生成（与 dataset.csv 不相关），避免数据泄漏。
2) 测试集 P0_Pa 跟随 alt_gun 走 ISA 标准大气，物理一致。
3) 测试样本由"先采 (θ, α) 再正向仿真"得到，保证目标可达。
4) 所有方法共享同一份 simulate_trajectory（带计数包装）。
5) 模型缺失时自动跳过 C/D 并继续运行 A/B。

运行
----
    python benchmark_methods.py --n 500 --out_dir artifacts_127
    python benchmark_methods.py --n 100 --skip_nn   # 模型还没训练时
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import ballistics
from ballistics import (
    ProjectileParams,
    get_atmosphere,
    isa_pressure,
    time_and_y_at_x,
)
from ballistics import simulate_trajectory as _orig_simulate_trajectory


# ---------------------------------------------------------------------------
# 全局计数器：统计 simulate_trajectory 的调用次数
# 必须在 import predict 之前完成 monkey-patch，
# 否则 predict.py 里 `from ballistics import simulate_trajectory` 会绑定到原始函数
# ---------------------------------------------------------------------------
class _SimCounter:
    def __init__(self):
        self.n = 0

    def reset(self):
        self.n = 0

    def inc(self):
        self.n += 1


COUNTER = _SimCounter()


def _counted_simulate(*args, **kwargs):
    COUNTER.inc()
    return _orig_simulate_trajectory(*args, **kwargs)


# 替换模块级引用：这一步必须在 import predict 之前
ballistics.simulate_trajectory = _counted_simulate


# 现在可以安全 import predict 了
import predict  # noqa: E402

# 双保险：如果 predict.py 里已经 from ballistics import simulate_trajectory，
# 它在自己模块名空间里有了 simulate_trajectory 绑定，也替换它
predict.simulate_trajectory = _counted_simulate

from predict import (  # noqa: E402
    infer_model_dims_from_state_dict,
    load_branch_model,
    solve_target_unified,
)
from model_architecture import (  # noqa: E402
    HIGH_THETA_MAX,
    HIGH_THETA_MIN,
    LOW_THETA_MAX,
    LOW_THETA_MIN,
)
from train_model import (  # noqa: E402
    minmax_transform,
)


# ---------------------------------------------------------------------------
# 共用：评分一组角度（一次 simulate）
# ---------------------------------------------------------------------------
def score_angle(
    theta_deg: float,
    alpha_deg: float,
    x_target: float,
    y_target: float,
    z_target: float,
    params: ProjectileParams,
    dt: float = 0.01,
    t_max: float = 120.0,
):
    """正向仿真一条弹道并计算到目标的 3D 命中误差。返回 None 表示不可达。"""
    traj = ballistics.simulate_trajectory(
        theta_deg=theta_deg, alpha_deg=alpha_deg, params=params, dt=dt, t_max=t_max
    )
    if traj["range"] < x_target:
        return None
    t_hit, y_hit, z_hit = time_and_y_at_x(traj, x_target)
    if t_hit is None or not np.isfinite(y_hit) or not np.isfinite(z_hit):
        return None
    err_3d = math.sqrt((y_hit - y_target) ** 2 + (z_hit - z_target) ** 2)
    return {
        "theta": float(theta_deg),
        "alpha": float(alpha_deg),
        "y_hit": float(y_hit),
        "z_hit": float(z_hit),
        "y_err": float(abs(y_hit - y_target)),
        "z_err": float(abs(z_hit - z_target)),
        "err_3d": float(err_3d),
        "t_hit": float(t_hit),
    }


# ---------------------------------------------------------------------------
# 共用：朴素牛顿法 refine（central difference Jacobian + 线搜索）
# 故意不带 grid search，让 A 和 B 公平对比"初值好坏"
# ---------------------------------------------------------------------------
def newton_refine(
    th_guess: float,
    al_guess: float,
    x_target: float,
    y_target: float,
    z_target: float,
    params: ProjectileParams,
    th_min: float,
    th_max: float,
    al_min: float = -15.0,
    al_max: float = 15.0,
    y_tol: float = 2.0,
    z_tol: float = 2.0,
    max_iter: int = 15,
    dth: float = 0.05,
    dal: float = 0.05,
    dt: float = 0.01,
    t_max: float = 120.0,
):
    """返回 (best_solution_dict_or_None, n_newton_iters)"""
    th = float(np.clip(th_guess, th_min, th_max))
    al = float(np.clip(al_guess, al_min, al_max))
    best = None
    iters = 0

    for _ in range(max_iter):
        iters += 1
        base = score_angle(th, al, x_target, y_target, z_target, params, dt, t_max)
        if base is None:
            break
        if best is None or base["err_3d"] < best["err_3d"]:
            best = dict(base)

        E_y = base["y_hit"] - y_target
        E_z = base["z_hit"] - z_target
        if abs(E_y) < y_tol and abs(E_z) < z_tol:
            break

        # 中心差分 Jacobian
        thp = score_angle(th + dth, al, x_target, y_target, z_target, params, dt, t_max)
        thm = score_angle(th - dth, al, x_target, y_target, z_target, params, dt, t_max)
        alp = score_angle(th, al + dal, x_target, y_target, z_target, params, dt, t_max)
        alm = score_angle(th, al - dal, x_target, y_target, z_target, params, dt, t_max)
        if None in (thp, thm, alp, alm):
            break

        J = np.array(
            [
                [(thp["y_hit"] - thm["y_hit"]) / (2 * dth), (alp["y_hit"] - alm["y_hit"]) / (2 * dal)],
                [(thp["z_hit"] - thm["z_hit"]) / (2 * dth), (alp["z_hit"] - alm["z_hit"]) / (2 * dal)],
            ],
            dtype=np.float64,
        )
        rhs = np.array([E_y, E_z], dtype=np.float64)
        try:
            delta = np.linalg.solve(J, rhs)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(J, rhs, rcond=None)[0]

        d_th = float(np.clip(delta[0], -2.0, 2.0))
        d_al = float(np.clip(delta[1], -2.0, 2.0))

        # 简单线搜索：1.0 -> 0.5 -> 0.25 -> 0.1
        accepted = False
        for scale in (1.0, 0.5, 0.25, 0.1):
            cand_th = float(np.clip(th - scale * d_th, th_min, th_max))
            cand_al = float(np.clip(al - scale * d_al, al_min, al_max))
            cand = score_angle(cand_th, cand_al, x_target, y_target, z_target, params, dt, t_max)
            if cand is None:
                continue
            if cand["err_3d"] <= best["err_3d"] + 1e-6:
                th, al = cand_th, cand_al
                if cand["err_3d"] < best["err_3d"]:
                    best = dict(cand)
                accepted = True
                break
        if not accepted:
            break

    return best, iters


# ---------------------------------------------------------------------------
# 真空解析解（方法 A 的初值）
# ---------------------------------------------------------------------------
def vacuum_initial_guess(
    x_target: float, y_target: float, z_target: float, v0: float, g: float = 9.80665
):
    """
    无阻力无风的解析弹道倒推。
    
    水平射程 = sqrt(x² + z²)，方位角 alpha = atan2(z, x)
    俯仰角 θ 的两组解（低/高弹道）由二次方程给出：
        tan θ = [v² ± sqrt(v⁴ - g(g·x_eff² + 2·y·v²))] / (g·x_eff)
    """
    horizontal = math.sqrt(x_target ** 2 + z_target ** 2)
    if horizontal < 1.0:
        return None

    alpha = math.degrees(math.atan2(z_target, x_target))
    x_eff = horizontal

    disc = v0 ** 4 - g * (g * x_eff ** 2 + 2.0 * y_target * v0 ** 2)
    if disc < 0:
        return None  # 真空也不可达 -> 真实大气下更不可能

    sqrt_disc = math.sqrt(disc)
    tan_low = (v0 ** 2 - sqrt_disc) / (g * x_eff)
    tan_high = (v0 ** 2 + sqrt_disc) / (g * x_eff)
    return (
        math.degrees(math.atan(tan_low)),
        math.degrees(math.atan(tan_high)),
        alpha,
    )


# ---------------------------------------------------------------------------
# 测试样本生成（独立于 dataset.csv）
# ---------------------------------------------------------------------------
def build_params(sample: dict, base_params: ProjectileParams) -> ProjectileParams:
    """从 sample dict 构造一个完整的 ProjectileParams（保持 mass/caliber/temp_coeff 不变）。"""
    p = ProjectileParams(**base_params.__dict__)
    p.T_powder_C = float(sample["T_powder_C"])
    p.v0_base = float(sample["v0_actual"]) - p.temp_coeff * (p.T_powder_C - 15.0)
    p.wind_x = float(sample["wind_x"])
    p.wind_y = float(sample["wind_y"])
    p.wind_z = float(sample["wind_z"])
    p.cant_angle_deg = float(sample["cant_angle"])
    p.alt_gun = float(sample["alt_gun"])
    p.T0_C = float(sample["T0_C"])
    p.P0_Pa = float(sample["P0_Pa"])
    return p


def generate_test_samples(
    n_samples: int,
    base_params: ProjectileParams,
    seed: int = 99999,
    dt_gen: float = 0.01,
) -> list[dict]:
    """
    生成 n_samples 个独立测试样本。
    
    流程：(随机 env, θ, α) -> 正向仿真 -> 沿弹道挑一个 x 处的 (x, y, z) 作为目标。
    保证每个目标都是物理可达的，且每个样本都有对应的"真值角度"。
    
    这些 simulate 调用不计入实验成本（COUNTER 在每个样本开始时重置）。
    """
    rng = np.random.default_rng(seed)
    samples = []
    attempts = 0
    max_attempts = n_samples * 5

    print(f"[TEST SET] 生成 {n_samples} 个独立测试样本（seed={seed}）...")
    while len(samples) < n_samples and attempts < max_attempts:
        attempts += 1

        # —— 物理自洽的环境采样 ——
        alt_gun = float(rng.uniform(0.0, 1500.0))
        T0_C = float(rng.uniform(-20.0, 40.0))
        # P0 跟随 ISA，再 ±5% 抖动模拟天气
        P0_Pa = float(isa_pressure(alt_gun) * rng.uniform(0.95, 1.05))
        wind_x = float(rng.uniform(-15.0, 15.0))
        wind_y = float(rng.uniform(-3.0, 3.0))
        wind_z = float(rng.uniform(-12.0, 12.0))
        cant_angle = float(rng.uniform(-12.0, 12.0))
        T_powder = T0_C + float(rng.normal(0.0, 5.0))  # 单峰、温和耦合
        v0_actual = base_params.v0_base + base_params.temp_coeff * (T_powder - 15.0)

        T0_K = T0_C + 273.15
        rho, _ = get_atmosphere(alt_gun, alt_gun, T0_K, P0_Pa)

        # —— 真值角度：按低/高弹道近似均匀采样 ——
        # 低弹道占多，约 70%；高弹道 30%
        if rng.random() < 0.7:
            theta = float(rng.uniform(3.0, 45.0))
        else:
            theta = float(rng.uniform(50.0, 70.0))
        alpha = float(rng.uniform(-6.0, 6.0))

        # —— 正向仿真，挑目标点 ——
        sample_dict = {
            "T_powder_C": T_powder,
            "v0_actual": v0_actual,
            "wind_x": wind_x,
            "wind_y": wind_y,
            "wind_z": wind_z,
            "cant_angle": cant_angle,
            "alt_gun": alt_gun,
            "T0_C": T0_C,
            "P0_Pa": P0_Pa,
        }
        p = build_params(sample_dict, base_params)
        traj = _orig_simulate_trajectory(  # 直接用原函数，不计数
            theta_deg=theta, alpha_deg=alpha, params=p, dt=dt_gen, t_max=200.0
        )
        max_x = traj["range"]
        if max_x < 200.0:
            continue
        # 沿弹道在 30%–90% 处挑目标
        x_frac = float(rng.uniform(0.30, 0.90))
        x_target = max_x * x_frac
        t_hit, y_target, z_target = time_and_y_at_x(traj, x_target)
        if t_hit is None or y_target is None or y_target < 0.5:
            continue

        sample_dict.update({
            "rho": float(rho),
            "x_target": float(x_target),
            "y_target": float(y_target),
            "z_target": float(z_target),
            "true_theta": float(theta),
            "true_alpha": float(alpha),
        })
        samples.append(sample_dict)

    print(f"[TEST SET] 完成 {len(samples)}/{n_samples}（尝试 {attempts} 次）\n")
    return samples


# ---------------------------------------------------------------------------
# 选最优解：从多个候选解里挑落在容差内、err_3d 最小的
# ---------------------------------------------------------------------------
def pick_best(solutions: list, y_tol: float, z_tol: float):
    valid = [s for s in solutions if s is not None and s["y_err"] <= y_tol and s["z_err"] <= z_tol]
    if not valid:
        # 没有满足容差的也返回 err_3d 最小的（用于错误统计）
        all_solutions = [s for s in solutions if s is not None]
        if not all_solutions:
            return None, False
        return min(all_solutions, key=lambda s: s["err_3d"]), False
    return min(valid, key=lambda s: s["err_3d"]), True


# ---------------------------------------------------------------------------
# 方法 A：真空解析 -> 牛顿
# ---------------------------------------------------------------------------
def method_A(sample: dict, base_params: ProjectileParams, y_tol: float, z_tol: float, dt: float, t_max: float):
    p = build_params(sample, base_params)
    COUNTER.reset()
    t0 = time.perf_counter()

    guess = vacuum_initial_guess(
        sample["x_target"], sample["y_target"], sample["z_target"], sample["v0_actual"]
    )
    if guess is None:
        wall_ms = (time.perf_counter() - t0) * 1000
        return _failed_result("A", wall_ms)

    th_low, th_high, alpha = guess
    sol_low, it_low = newton_refine(
        th_low, alpha, sample["x_target"], sample["y_target"], sample["z_target"],
        p, LOW_THETA_MIN, LOW_THETA_MAX, y_tol=y_tol, z_tol=z_tol, dt=dt, t_max=t_max,
    )
    sol_high, it_high = newton_refine(
        th_high, alpha, sample["x_target"], sample["y_target"], sample["z_target"],
        p, HIGH_THETA_MIN, HIGH_THETA_MAX, y_tol=y_tol, z_tol=z_tol, dt=dt, t_max=t_max,
    )
    chosen, converged = pick_best([sol_low, sol_high], y_tol, z_tol)
    wall_ms = (time.perf_counter() - t0) * 1000
    return {
        "method": "A",
        "err_3d": chosen["err_3d"] if chosen else np.nan,
        "n_sim": COUNTER.n,
        "wall_ms": wall_ms,
        "iters": it_low + it_high,
        "converged": bool(converged),
        "chosen_theta": chosen["theta"] if chosen else np.nan,
        "chosen_alpha": chosen["alpha"] if chosen else np.nan,
    }


# ---------------------------------------------------------------------------
# 方法 B：固定初值 (30°/65°, alpha=0) -> 牛顿
# ---------------------------------------------------------------------------
def method_B(sample: dict, base_params: ProjectileParams, y_tol: float, z_tol: float, dt: float, t_max: float):
    p = build_params(sample, base_params)
    COUNTER.reset()
    t0 = time.perf_counter()

    sol_low, it_low = newton_refine(
        30.0, 0.0, sample["x_target"], sample["y_target"], sample["z_target"],
        p, LOW_THETA_MIN, LOW_THETA_MAX, y_tol=y_tol, z_tol=z_tol, dt=dt, t_max=t_max,
    )
    sol_high, it_high = newton_refine(
        65.0, 0.0, sample["x_target"], sample["y_target"], sample["z_target"],
        p, HIGH_THETA_MIN, HIGH_THETA_MAX, y_tol=y_tol, z_tol=z_tol, dt=dt, t_max=t_max,
    )
    chosen, converged = pick_best([sol_low, sol_high], y_tol, z_tol)
    wall_ms = (time.perf_counter() - t0) * 1000
    return {
        "method": "B",
        "err_3d": chosen["err_3d"] if chosen else np.nan,
        "n_sim": COUNTER.n,
        "wall_ms": wall_ms,
        "iters": it_low + it_high,
        "converged": bool(converged),
        "chosen_theta": chosen["theta"] if chosen else np.nan,
        "chosen_alpha": chosen["alpha"] if chosen else np.nan,
    }


# ---------------------------------------------------------------------------
# 共用：NN 推理两个分支（D 和 E 共用）
# ---------------------------------------------------------------------------
def nn_predict_both(
    sample: dict,
    loaded_low_model,
    loaded_low_scaler,
    loaded_high_model,
    loaded_high_scaler,
):
    """
    跑两次 NN 推理，分别拿低弹道和高弹道的初值。
    返回 ((th_low, al_low), (th_high, al_high))
    """
    import torch

    slant = math.sqrt(
        sample["x_target"] ** 2 + sample["y_target"] ** 2 + sample["z_target"] ** 2
    )
    X_in = np.array(
        [[
            sample["x_target"], sample["y_target"], sample["z_target"],
            sample["v0_actual"], sample["rho"], slant,
            sample["wind_x"], sample["wind_y"], sample["wind_z"],
            sample["cant_angle"], sample["T_powder_C"],
            sample["T0_C"], sample["P0_Pa"], sample["alt_gun"],
        ]],
        dtype=np.float32,
    )

    with torch.no_grad():
        x_low = torch.from_numpy(minmax_transform(X_in, loaded_low_scaler)).to(
            next(loaded_low_model.parameters()).device
        )
        pred_low = loaded_low_model(x_low)
        th_low, al_low = float(pred_low[0][0]), float(pred_low[0][1])

        x_high = torch.from_numpy(minmax_transform(X_in, loaded_high_scaler)).to(
            next(loaded_high_model.parameters()).device
        )
        pred_high = loaded_high_model(x_high)
        th_high, al_high = float(pred_high[0][0]), float(pred_high[0][1])

    return (th_low, al_low), (th_high, al_high)


# ---------------------------------------------------------------------------
# 方法 C：solve_target_unified（你现有的方案）
# ---------------------------------------------------------------------------
def method_C(
    sample: dict,
    base_params: ProjectileParams,
    out_dir: str,
    loaded_low_model,
    loaded_low_scaler,
    loaded_high_model,
    loaded_high_scaler,
    y_tol: float,
    z_tol: float,
    dt: float,
    t_max: float,
):
    COUNTER.reset()
    t0 = time.perf_counter()

    res = solve_target_unified(
        x_target=sample["x_target"],
        y_target=sample["y_target"],
        z_target=sample["z_target"],
        v0_actual=sample["v0_actual"],
        rho=sample["rho"],
        wind_x=sample["wind_x"],
        wind_y=sample["wind_y"],
        wind_z=sample["wind_z"],
        cant_angle=sample["cant_angle"],
        T_powder_C=sample["T_powder_C"],
        alt_gun=sample["alt_gun"],
        T0_C=sample["T0_C"],
        P0_Pa=sample["P0_Pa"],
        dir_path=out_dir,
        loaded_low_model=loaded_low_model,
        loaded_low_scaler=loaded_low_scaler,
        loaded_high_model=loaded_high_model,
        loaded_high_scaler=loaded_high_scaler,
        params=base_params,
        y_tol=y_tol,
        z_tol=z_tol,
        dt=dt,
        t_max=t_max,
        save_plot=False,
    )
    wall_ms = (time.perf_counter() - t0) * 1000
    chosen = res.get("chosen")
    return {
        "method": "C",
        "err_3d": chosen["err_3d"] if chosen else np.nan,
        "n_sim": COUNTER.n,
        "wall_ms": wall_ms,
        "iters": np.nan,  # C 用 grid + Newton，迭代数概念不直接可比
        "converged": chosen is not None,
        "chosen_theta": chosen["theta"] if chosen else np.nan,
        "chosen_alpha": chosen["alpha"] if chosen else np.nan,
    }


# ---------------------------------------------------------------------------
# 方法 D：纯 NN，不 refine（衡量 NN 单独能力）
# ---------------------------------------------------------------------------
def method_D(
    sample: dict,
    base_params: ProjectileParams,
    loaded_low_model,
    loaded_low_scaler,
    loaded_high_model,
    loaded_high_scaler,
    y_tol: float,
    z_tol: float,
    dt: float,
    t_max: float,
):
    p = build_params(sample, base_params)
    COUNTER.reset()
    t0 = time.perf_counter()

    (th_low, al_low), (th_high, al_high) = nn_predict_both(
        sample, loaded_low_model, loaded_low_scaler,
        loaded_high_model, loaded_high_scaler,
    )

    # 各正向仿真一次，挑误差小的（不做 refine）
    sol_low = score_angle(
        th_low, al_low, sample["x_target"], sample["y_target"], sample["z_target"],
        p, dt=dt, t_max=t_max,
    )
    sol_high = score_angle(
        th_high, al_high, sample["x_target"], sample["y_target"], sample["z_target"],
        p, dt=dt, t_max=t_max,
    )
    chosen, converged = pick_best([sol_low, sol_high], y_tol, z_tol)
    wall_ms = (time.perf_counter() - t0) * 1000
    return {
        "method": "D",
        "err_3d": chosen["err_3d"] if chosen else np.nan,
        "n_sim": COUNTER.n,
        "wall_ms": wall_ms,
        "iters": 0,
        "converged": bool(converged),
        "chosen_theta": chosen["theta"] if chosen else np.nan,
        "chosen_alpha": chosen["alpha"] if chosen else np.nan,
    }


# ---------------------------------------------------------------------------
# 方法 E：NN 初值 -> 牛顿（跳过 grid search）
# 这正是你"NN 给好初值 + 牛顿修正"的设计原意。
# 跟 A/B 共用同一份 newton_refine，跟 C 共用同一份 NN 推理 ——
# 是 apples-to-apples 测出 grid search 在不在帮忙的关键对照。
# ---------------------------------------------------------------------------
def method_E(
    sample: dict,
    base_params: ProjectileParams,
    loaded_low_model,
    loaded_low_scaler,
    loaded_high_model,
    loaded_high_scaler,
    y_tol: float,
    z_tol: float,
    dt: float,
    t_max: float,
):
    p = build_params(sample, base_params)
    COUNTER.reset()
    t0 = time.perf_counter()

    (th_low, al_low), (th_high, al_high) = nn_predict_both(
        sample, loaded_low_model, loaded_low_scaler,
        loaded_high_model, loaded_high_scaler,
    )

    sol_low, it_low = newton_refine(
        th_low, al_low, sample["x_target"], sample["y_target"], sample["z_target"],
        p, LOW_THETA_MIN, LOW_THETA_MAX, y_tol=y_tol, z_tol=z_tol, dt=dt, t_max=t_max,
    )
    sol_high, it_high = newton_refine(
        th_high, al_high, sample["x_target"], sample["y_target"], sample["z_target"],
        p, HIGH_THETA_MIN, HIGH_THETA_MAX, y_tol=y_tol, z_tol=z_tol, dt=dt, t_max=t_max,
    )
    chosen, converged = pick_best([sol_low, sol_high], y_tol, z_tol)
    wall_ms = (time.perf_counter() - t0) * 1000
    return {
        "method": "E",
        "err_3d": chosen["err_3d"] if chosen else np.nan,
        "n_sim": COUNTER.n,
        "wall_ms": wall_ms,
        "iters": it_low + it_high,
        "converged": bool(converged),
        "chosen_theta": chosen["theta"] if chosen else np.nan,
        "chosen_alpha": chosen["alpha"] if chosen else np.nan,
    }


def _failed_result(method: str, wall_ms: float):
    return {
        "method": method,
        "err_3d": np.nan,
        "n_sim": COUNTER.n,
        "wall_ms": wall_ms,
        "iters": 0,
        "converged": False,
        "chosen_theta": np.nan,
        "chosen_alpha": np.nan,
    }


# ---------------------------------------------------------------------------
# 主 benchmark
# ---------------------------------------------------------------------------
def run_benchmark(
    n_samples: int,
    out_dir: str,
    seed: int,
    skip_nn: bool,
    y_tol: float,
    z_tol: float,
    dt: float,
    t_max: float,
):
    # —— 加载 ammo 配置 ——
    from generate_dataset import AMMO_CONFIGS, CURRENT_AMMO
    cfg = AMMO_CONFIGS[CURRENT_AMMO]
    base_params = ProjectileParams(mass=cfg["mass"], caliber=cfg["caliber"])
    base_params.v0_base = cfg["v0_base"]
    print(f"[CONFIG] ammo={CURRENT_AMMO}, v0_base={cfg['v0_base']}, out_dir={out_dir}\n")

    # —— 测试集 ——
    samples = generate_test_samples(n_samples, base_params, seed=seed)
    if not samples:
        raise RuntimeError("No test samples generated.")

    # —— 加载 NN 模型（如果存在）——
    loaded = {"low_m": None, "low_s": None, "high_m": None, "high_s": None}
    has_nn = False
    if not skip_nn:
        low_pt = os.path.join(out_dir, "dual_model_low.pt")
        high_pt = os.path.join(out_dir, "dual_model_high.pt")
        low_js = os.path.join(out_dir, "dual_scaler_low.json")
        high_js = os.path.join(out_dir, "dual_scaler_high.json")
        if all(os.path.exists(p) for p in (low_pt, high_pt, low_js, high_js)):
            print("[NN] 加载模型...")
            try:
                ld = infer_model_dims_from_state_dict(low_pt)
                hd = infer_model_dims_from_state_dict(high_pt)
                loaded["low_m"], loaded["low_s"], _ = load_branch_model(
                    low_pt, low_js,
                    theta_min=LOW_THETA_MIN, theta_max=LOW_THETA_MAX,
                    in_dim=ld["in_dim"], hidden=ld["hidden"], dropout=ld["dropout"],
                )
                loaded["high_m"], loaded["high_s"], _ = load_branch_model(
                    high_pt, high_js,
                    theta_min=HIGH_THETA_MIN, theta_max=HIGH_THETA_MAX,
                    in_dim=hd["in_dim"], hidden=hd["hidden"], dropout=hd["dropout"],
                )
                has_nn = True
                print("[NN] OK\n")
            except Exception as e:
                print(f"[NN] 加载失败 ({e})，跳过 C/D\n")
        else:
            print(f"[NN] 模型文件缺失（{out_dir}/dual_model_*.pt），跳过 C/D\n")

    # —— 主循环 ——
    rows = []
    print(f"[BENCH] 开始评估，样本数 = {len(samples)}")
    print(f"        容差: y_tol={y_tol} m, z_tol={z_tol} m")
    print(f"        仿真: dt={dt}, t_max={t_max}\n")

    for i, s in enumerate(samples):
        if (i + 1) % 50 == 0 or i == len(samples) - 1:
            print(f"  进度 {i + 1}/{len(samples)}")

        common = dict(
            sample=s, base_params=base_params,
            y_tol=y_tol, z_tol=z_tol, dt=dt, t_max=t_max,
        )

        rA = method_A(**common)
        rB = method_B(**common)
        rA["sample_id"] = rB["sample_id"] = i
        rA["true_theta"] = rB["true_theta"] = s["true_theta"]
        rA["true_alpha"] = rB["true_alpha"] = s["true_alpha"]
        rows.append(rA)
        rows.append(rB)

        if has_nn:
            rC = method_C(
                s, base_params, out_dir,
                loaded["low_m"], loaded["low_s"], loaded["high_m"], loaded["high_s"],
                y_tol, z_tol, dt, t_max,
            )
            rD = method_D(
                s, base_params,
                loaded["low_m"], loaded["low_s"], loaded["high_m"], loaded["high_s"],
                y_tol, z_tol, dt, t_max,
            )
            rE = method_E(
                s, base_params,
                loaded["low_m"], loaded["low_s"], loaded["high_m"], loaded["high_s"],
                y_tol, z_tol, dt, t_max,
            )
            rC["sample_id"] = rD["sample_id"] = rE["sample_id"] = i
            rC["true_theta"] = rD["true_theta"] = rE["true_theta"] = s["true_theta"]
            rC["true_alpha"] = rD["true_alpha"] = rE["true_alpha"] = s["true_alpha"]
            rows.append(rC)
            rows.append(rD)
            rows.append(rE)

    df = pd.DataFrame(rows)
    return df, has_nn


# ---------------------------------------------------------------------------
# 汇总 + 输出
# ---------------------------------------------------------------------------
def summarize_and_save(df: pd.DataFrame, out_dir: str, has_nn: bool):
    methods = ["A", "B"] + (["C", "D", "E"] if has_nn else [])
    method_names = {
        "A": "A: Vacuum-init + Newton",
        "B": "B: Fixed-init + Newton",
        "C": "C: Prioritized NN-init + early-exit refine",
        "D": "D: Pure NN (no refine)",
        "E": "E: NN-init + Newton (no grid)",
    }

    # —— 文本汇总 ——
    print("\n" + "=" * 80)
    print("  Results Summary")
    print("=" * 80)
    rows_summary = []
    for m in methods:
        sub = df[df["method"] == m]
        succ = sub[sub["converged"]]
        # 关键统计
        row = {
            "method": method_names[m],
            "n": len(sub),
            "convergence_rate": float(succ.shape[0] / max(len(sub), 1)),
            "err_3d_median": float(succ["err_3d"].median()) if len(succ) else float("nan"),
            "err_3d_mean": float(succ["err_3d"].mean()) if len(succ) else float("nan"),
            "err_3d_p95": float(succ["err_3d"].quantile(0.95)) if len(succ) else float("nan"),
            "n_sim_mean": float(sub["n_sim"].mean()),
            "n_sim_median": float(sub["n_sim"].median()),
            "wall_ms_mean": float(sub["wall_ms"].mean()),
            "wall_ms_median": float(sub["wall_ms"].median()),
        }
        rows_summary.append(row)

    summary_df = pd.DataFrame(rows_summary)
    print(summary_df.to_string(index=False))

    # —— 写出 CSV ——
    detail_csv = os.path.join(out_dir, "benchmark_detail.csv")
    summary_csv = os.path.join(out_dir, "benchmark_summary.csv")
    df.to_csv(detail_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] {detail_csv}")
    print(f"[SAVE] {summary_csv}")

    # —— 画图 ——
    plot_path = os.path.join(out_dir, "benchmark_compare.png")
    _plot_compare(df, methods, method_names, plot_path)
    print(f"[SAVE] {plot_path}")

    # —— 关键结论 ——
    print("\n" + "=" * 80)
    print("  Key Verdict")
    print("=" * 80)
    if has_nn:
        def _stat(method_letter, col):
            row = summary_df.loc[summary_df["method"].str.startswith(f"{method_letter}:")]
            return float(row[col].iloc[0]) if len(row) else float("nan")

        c_med = _stat("C", "err_3d_median")
        e_med = _stat("E", "err_3d_median")
        a_med = _stat("A", "err_3d_median")
        d_med = _stat("D", "err_3d_median")
        c_t = _stat("C", "wall_ms_mean")
        e_t = _stat("E", "wall_ms_mean")
        a_t = _stat("A", "wall_ms_mean")
        c_sim = _stat("C", "n_sim_mean")
        e_sim = _stat("E", "n_sim_mean")
        c_conv = _stat("C", "convergence_rate")
        e_conv = _stat("E", "convergence_rate")

        speedup_e_vs_c = c_t / max(e_t, 1e-9)
        sim_save_e_vs_c = c_sim / max(e_sim, 1e-9)

        # ========= 主对照：C vs E（设计原意验证）=========
        print("  >>> Primary: exhaustive NN -> Newton (E) vs current prioritized solver (C)")
        print(f"      err_3d (median):   C={c_med:.4f} m   |   E={e_med:.4f} m")
        print(f"      wall_ms  (mean):   C={c_t:.1f}        |   E={e_t:.1f}     (E faster {speedup_e_vs_c:.1f}x)")
        print(f"      n_sim    (mean):   C={c_sim:.1f}      |   E={e_sim:.1f}    (E saves {sim_save_e_vs_c:.1f}x)")
        print(f"      convergence:       C={c_conv:.1%}      |   E={e_conv:.1%}")
        print()

        # ========= 辅助对照：A vs E（NN 有没有比真空初值更好）=========
        print("  >>> Secondary: NN 初值 (E) vs 真空解析初值 (A)")
        print(f"      err_3d (median):   A={a_med:.4f} m   |   E={e_med:.4f} m")
        print(f"      wall_ms  (mean):   A={a_t:.1f}        |   E={e_t:.1f}")
        print()

        # ========= 设计判定 =========
        print("  >>> Verdict:")
        # 主判定：E 是不是验证了你的设计
        e_acceptable_acc = (e_med <= max(c_med * 2.0, c_med + 0.05))  # 精度劣化 <2x 或 <5cm
        e_much_faster = (e_t < c_t * 0.5)
        e_robust = (e_conv >= 0.95)

        if e_acceptable_acc and e_much_faster and e_robust:
            print("      OK: E 精度持平、明显更快、收敛稳定 --")
            print("         你的设计 (NN initial -> Newton refine) 完全验证成功。")
            print("         建议: 把 solve_target_unified 里的 grid search 关掉，直接走 NN -> Newton。")
        elif e_much_faster and not e_robust:
            print("      WARN: E 快但收敛率掉了 -- grid search 在替你兜 NN 失误。")
            print("         建议: 检查 NN 不收敛的样本规律（看 detail.csv），针对性加训。")
            print("         过渡方案: 牛顿失败时再回落到 grid，热路径仍享受 E 的速度。")
        elif not e_acceptable_acc:
            print("      FAIL: E 精度比 C 差太多 -- grid search 不只是兜底，确实在精修。")
            print("         保留 C 的设计；如果嫌慢，缩小 grid 步数（比如 5x5 -> 3x3）。")
        else:
            print("      ? E 跟 C 差距不显著——可能是测试样本不够覆盖边界情况。")
            print("         建议: 把 --n 调到 1000+ 再跑一次。")

        # 辅助判定：NN 比真空好不好
        print()
        if e_med < a_med * 0.5 and a_med > 0.05:
            print("      OK: NN 初值确实比真空初值精度高 -> NN 训练有价值。")
        elif abs(e_med - a_med) / max(a_med, 1e-9) < 0.3:
            print("      WARN: NN 初值精度跟真空初值接近 -> 牛顿在做大部分工作，NN 是装饰。")
            print("         考虑: 让 NN 改学'残差' (NN 输出 = 真空初值 + Δ)，分工更明确。")

        # D 健康检查
        print()
        print(f"      [Health check] 纯 NN (D) median 误差 = {d_med:.2f} m")
        if d_med < 5.0:
            print("                     -> NN 输出已经在牛顿吸引域内，方法 E 应当稳定收敛。")
        else:
            print("                     -> NN 输出离真值偏远，牛顿可能发散，grid 兜底有意义。")
    else:
        print("  没有加载 NN（用 --skip_nn 或模型缺失），只比较了 A 和 B。")
    print("=" * 80 + "\n")


def _plot_compare(df: pd.DataFrame, methods: list, method_names: dict, out_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    # A=blue, B=orange, C=green, D=red, E=purple (E 用紫色突出 —— 它是设计原意候选)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#9467BD"]

    # (1) err_3d boxplot（仅成功样本，对数 y 轴）
    ax = axes[0, 0]
    data = []
    labels = []
    for m in methods:
        sub = df[(df["method"] == m) & (df["converged"]) & np.isfinite(df["err_3d"])]
        if len(sub):
            data.append(sub["err_3d"].values + 1e-6)  # 防止 log(0)
            labels.append(m)
    if data:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
    ax.set_yscale("log")
    ax.set_ylabel("err_3d (m), log scale")
    ax.set_title("3D Hit Error (converged samples)")
    ax.grid(True, alpha=0.3)

    # (2) n_sim boxplot
    ax = axes[0, 1]
    data = [df[df["method"] == m]["n_sim"].values for m in methods]
    bp = ax.boxplot(data, tick_labels=methods, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_ylabel("# simulate_trajectory calls")
    ax.set_title("Compute Cost (sim calls)")
    ax.grid(True, alpha=0.3)

    # (3) wall time boxplot
    ax = axes[1, 0]
    data = [df[df["method"] == m]["wall_ms"].values for m in methods]
    bp = ax.boxplot(data, tick_labels=methods, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_ylabel("wall time (ms)")
    ax.set_title("Wall Clock Time per Query")
    ax.grid(True, alpha=0.3)

    # (4) convergence rate bar
    ax = axes[1, 1]
    rates = []
    for m in methods:
        sub = df[df["method"] == m]
        rates.append(sub["converged"].mean() if len(sub) else 0)
    bars = ax.bar(methods, rates, color=colors[:len(methods)], alpha=0.85)
    for bar, r in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{r:.1%}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Convergence rate")
    ax.set_title("Within-tolerance Rate")
    ax.grid(True, axis="y", alpha=0.3)

    # 图例（在 figure 顶部）
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i], alpha=0.7) for i in range(len(methods))]
    fig.legend(handles, [method_names[m] for m in methods],
               loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=3, fontsize=10, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=100, help="测试样本数 (default: 100)")
    p.add_argument("--out_dir", type=str, default="artifacts_127", help="模型/输出目录")
    p.add_argument("--seed", type=int, default=99999, help="测试集随机种子（避开训练 seed 42）")
    p.add_argument("--skip_nn", action="store_true", help="跳过方法 C 和 D（模型未训练时用）")
    p.add_argument("--y_tol", type=float, default=2.0, help="y 方向命中容差 (m)")
    p.add_argument("--z_tol", type=float, default=2.0, help="z 方向命中容差 (m)")
    p.add_argument("--dt", type=float, default=0.05,
                   help="仿真步长 (default: 0.05, 跟 solve_target_unified 对齐；想更精确传 0.01)")
    p.add_argument("--t_max", type=float, default=120.0, help="仿真最大时长 (s)")
    args = p.parse_args()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    df, has_nn = run_benchmark(
        n_samples=args.n,
        out_dir=args.out_dir,
        seed=args.seed,
        skip_nn=args.skip_nn,
        y_tol=args.y_tol,
        z_tol=args.z_tol,
        dt=args.dt,
        t_max=args.t_max,
    )
    summarize_and_save(df, args.out_dir, has_nn)


if __name__ == "__main__":
    main()
