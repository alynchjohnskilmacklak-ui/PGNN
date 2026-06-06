# Ballistic Angle Prediction Pipeline

基于神经网络的弹道射角预测系统，用于根据目标位置和环境条件预测发射角度（theta/alpha），并通过数值求解器精修，实现高精度命中。

---

## 模块总览

```
ballistics.py          ← 物理引擎（大气、弹道仿真）
model_architecture.py  ← 神经网络结构定义
physics_loss.py        ← 可微物理损失函数
train_model.py         ← 数据预处理 + 训练循环
solver.py              ← 数值求解器（Broyden 快速路径 + grid 回退）
predict.py             ← 推理预测 + 结果可视化
generate_dataset.py    ← 合成数据集生成
main.py                ← 顶层流水线（数据集→训练→验证）
Benchmark.py           ← 独立对照实验（5种方法性能对比）
benchmark_newton_vs_broyden_solver.py ← Newton vs Broyden 求解器离线对比
```

---

## 各模块详细说明

### 1. `ballistics.py` — 物理引擎

外弹道仿真核心库，实现 3D 弹道数值积分。

| 函数/类 | 功能 |
|---------|------|
| `gravity_from_altitude(h_m)` | 计算海拔 h 处重力加速度 |
| `isa_pressure(alt_m, P_sea)` | ISA 标准大气压下海拔处气压 |
| `get_atmosphere(alt, alt_gun, T0, P0)` | 获取海拔处温度/密度（含对流层顶处理） |
| `calc_dynamic_cd_G7(ma)` | G7 弹道系数（亚/跨/超音速三段 sigmoid 平滑过渡） |
| `ProjectileParams` | 弹丸物理参数 dataclass（质量、口径、初速、风、倾角等） |
| `get_dynamic_wind(y)` | 对数风廓线（风速随高度变化） |
| `dynamics(t, state, params)` | ODE 右端项：3D 气动阻力 + 重力 + 风速耦合 |
| `rk4_step(...)` | 四阶 Runge-Kutta 积分步 |
| `simulate_trajectory(theta, alpha, params)` | 完整弹道仿真，返回各时刻位置/速度 |
| `interpolate_y_on_xgrid(traj, x_grid)` | 在 x 网格上插值 y |
| `interpolate_yz_on_xgrid(traj, x_grid)` | 在 x 网格上插值 (y, z) |
| `time_and_y_at_x(traj, x_target)` | 线性插值求弹道到达目标 x 处的时间、(y, z) |

---

### 2. `model_architecture.py` — 神经网络结构

纯模型定义，不含训练逻辑。

| 类/常量 | 功能 |
|---------|------|
| `LOW_THETA_MIN` (0°) | 低弹道 theta 下界 |
| `LOW_THETA_MAX` (55°) | 低弹道 theta 上界 |
| `HIGH_THETA_MIN` (45°) | 高弹道 theta 下界 |
| `HIGH_THETA_MAX` (85°) | 高弹道 theta 上界 |
| `ALPHA_ABS_MAX` (15°) | alpha 偏航角绝对值上界 |
| `ResidualBlock` | 残差 MLP 块：Linear → LayerNorm → SiLU → Dropout × 2，残差连接 |
| `AngleHead` | 输出头：sigmoid 映射 theta 到 [θ_min, θ_max]，tanh 映射 alpha 到 ±15° |
| `SingleBranchDNN` | 完整模型：stem（Linear+LN+SiLU）→ backbone（3×ResidualBlock）→ AngleHead |
| `ModelEMA` | 指数移动平均：训练时维护平滑权重，验证时使用 |

---

### 3. `physics_loss.py` — 物理损失函数

可微分的弹道物理引擎，嵌入训练循环作为正则化项，使模型预测的角度在物理上自洽。

| 函数/类 | 功能 |
|---------|------|
| `SmoothPhysicsLoss` | 核心物理引擎：用预测角度做 N 步 Euler 积分得到 (x,y,z)，与真实弹道位置比较 |
| `_per_elem_smooth_l1` | Smooth L1 损失，比 MSE 对异常值更鲁棒 |
| `_weighted_mean` | 加权均值（处理重要性采样） |
| `_sample_importance` | 重要性权重：对重叠区域 (42-58°) 和极端角度 (+60%/+20% 权重) 加强训练 |
| `_per_sample_angle_loss` | 逐样本角度损失：theta 权重 1.35，alpha 权重 1.0 |
| `_physics_residual_loss` | 物理残差：预测弹道 vs 真实弹道归一化位置偏差 |
| `_scheduled_lambda_phys` | 物理损失调度器：前 12 epoch 线性 warm-up |
| `_forward_loss_dict` | 组装函数：反归一化 → 计算 data loss + physics loss → 返回损失字典 |

---

### 4. `train_model.py` — 训练循环

数据预处理、训练逻辑、评估。

| 函数/类 | 功能 |
|---------|------|
| `AngleDataset` | PyTorch Dataset 封装（内存 → Tensor） |
| `minmax_fit(X)` | Min-Max 归一化拟合 |
| `minmax_transform(X, scaler)` | Min-Max 归一化变换 |
| `build_trajectory_groups(arr)` | 按弹道参数分组（theta/alpha/v0/风/倾角），防止同弹道泄漏到 train/val/test |
| `trajectory_group_split(group_ids)` | 按组划分 train/val/test (70/15/15) |
| `evaluate_model(model, loader, ...)` | DataLoader 评估（CPU 模式） |
| `evaluate_tensor_batches(model, X, y, ...)` | GPU 张量评估（缓存模式） |
| `TrainConfig` | 训练超参数 dataclass |
| `_save_history_csv(history, path)` | 保存训练历史 CSV |
| `_read_ammo_meta(out_dir)` | 读取弹药元数据（质量、口径） |
| `_make_adamw(params, lr, wd)` | AdamW 优化器（优先 fused 版本） |
| `_make_grad_scaler(use_cuda)` | AMP 混合精度 GradScaler |
| `train(...)` | **主训练函数**：数据加载 → 模型构建 → 训练循环（AMP + EMA + 早停 + 物理损失） |

---

### 5. `solver.py` — 数值求解器

从 NN 初值出发，用 Broyden 拟牛顿法快速精修角度；若精度不足则回退到 grid search 再精修。

| 函数 | 功能 |
|------|------|
| `_score_angle_at_target` | 给定 (theta, alpha)，正向仿真弹道，返回 3D 命中误差 |
| `_grid_search_refine` | 网格搜索：在初值周围搜索更优角度 |
| `_coarse_to_fine_refine` | 两级网格搜索：粗搜索 (±2°, 步长 0.5°) → 精搜索 (±0.5°, 步长 0.1°) |
| `_newton_only_refine` | 牛顿法精修（备用/对照）：中心差分 Jacobian，每轮 5 次仿真，最多 15 次迭代 |
| `_broyden_refine` | **Broyden 拟牛顿精修**：首轮中心差分 Jacobian，后续 good Broyden rank-1 更新，减少仿真次数 |
| `_refine_candidate` | **自适应策略**：NN 初值 → Broyden 快速精修 → 满足容差直接返回 → 不满足则 grid search → grid 结果再 Broyden → 选误差最小者 |

---

### 6. `predict.py` — 推理预测

加载模型 → NN 预测 → 数值精修 → 输出结果。

| 函数 | 功能 |
|------|------|
| `load_scaler(path)` | 加载 JSON 归一化器 |
| `load_branch_model(model_path, scaler_path, ...)` | 加载训练好的分支模型（low/high）和归一化器 |
| `infer_model_dims_from_state_dict(path)` | 从 .pt 权重文件反推模型结构参数（in_dim, hidden 等） |
| `_predict_branch_angles(model, scaler, device, X)` | 单次 NN 前向预测 |
| `_make_env_params(params, v0, wind, ...)` | 构造 ProjectileParams 环境对象 |
| `_save_solution_plot` | 绘制 3D 弹道对比图（low/high 精修后轨迹 vs 目标点） |
| `_save_nn_prediction_plot` | 绘制 3D 弹道对比图（low/high NN 原始预测） |
| `_save_prediction_compare_csv` | 保存预测对比表 |
| `_save_error_compare_plot` | 绘制误差对比图 |
| `solve_target_unified(...)` | **主求解入口**：双分支 NN 预测 → 精修 → 择优 → 出图 |

---

### 7. `generate_dataset.py` — 数据集生成

并行合成训练数据集。

| 函数/常量 | 功能 |
|----------|------|
| `AMMO_CONFIGS` | 弹药配置字典（12.7mm_B32, 14.5mm_B32 等） |
| `CURRENT_AMMO` | 当前选定弹药 |
| `_stratified_alt_samples(n, rng)` | 分层采样海拔 |
| `_process_single_angle_worker(args)` | 多进程 Worker：对单个 theta 角仿真多条随机弹道 |
| `generate_dataset(out_dir, seed, ...)` | **主函数**：ProcessPoolExecutor 并行仿真 → 网格化 → 特征/标签矩阵 → 保存 .npy/.csv |

---

### 8. `main.py` — 顶层流水线

连接所有模块，一键完成完整流程。

| 函数 | 功能 |
|------|------|
| `_remove_pipeline_outputs(out_dir)` | 清理旧产物 |
| `_generate_verify_samples(n, params, seed)` | 生成独立验证样本（不同 seed，防止数据泄漏） |
| `_verify_unified_model(out_dir, n, seed)` | 端到端验证：加载模型 → 500 样本反解 → 统计精度 |
| `full_pipeline_dual_task(seed, out_dir)` | **主入口**：生成数据集 → 训练 low 模型 → 训练 high 模型 → 验证 |

---

### 9. `Benchmark.py` — 对照实验

5 种求解方法的独立性能对比。

| 方法 | 策略 |
|------|------|
| **A** | 真空解析初值 → 牛顿法 |
| **B** | 固定初值 (30°/65°) → 牛顿法 |
| **C** | NN 初值 → grid + 牛顿法（当前流水线） |
| **D** | 纯 NN 预测（不精修，衡量 NN 独立精度） |
| **E** | NN 初值 → 牛顿法（跳过 grid search） |

| 函数 | 功能 |
|------|------|
| `_SimCounter` | 仿真调用计数器 |
| `score_angle(th, alpha, ...)` | 单次角度评分 |
| `newton_refine(...)` | 牛顿法精修 |
| `vacuum_initial_guess(x, y)` | 真空弹道解析解 |
| `build_params(sample, base_params)` | 从字典构造参数 |
| `generate_test_samples(n, params, seed)` | 独立测试集生成 |
| `method_A ... method_E` | 各方法实现 |
| `run_benchmark(out_dir, n)` | 运行全部方法 |
| `summarize_and_save(df, out_dir)` | 统计汇总 + 输出 |

---

### 10. `benchmark_newton_vs_broyden_solver.py` — Newton vs Broyden 离线对比

独立 benchmark 脚本，不依赖 NN 模型，对比两种迭代方法在相同初值下的性能。

| 方法 | 策略 |
|------|------|
| **Newton** | 每轮中心差分 Jacobian（5 次仿真/轮），复刻 `solver._newton_only_refine()` |
| **Broyden** | 首轮中心差分 Jacobian，后续 good Broyden rank-1 更新（不增加仿真） |

| 函数 | 功能 |
|------|------|
| `SimCounter` | 仿真调用计数器 |
| `counted_score_angle_at_target` | 带计数的角度评分（每调用一次 `simulate_trajectory` 递增） |
| `benchmark_newton_refine` | Newton 精修（带完整计数统计） |
| `benchmark_broyden_refine` | Broyden 精修（带完整计数统计） |
| `RefineResult` | 统一结果 dataclass（含 method/sample_id/converged/err_3d/wall_ms/simulate_calls 等） |
| `generate_benchmark_samples` | 正向仿真生成独立测试样本（不依赖训练数据） |
| `build_params_from_sample` | 从样本字典构造 `ProjectileParams` |
| `run_benchmark` | 主流程：生成样本 → Newton/Broyden 对比 → 汇总 DataFrame |
| `print_summary` | 按 method 分组输出收敛率、误差、仿真次数等汇总统计 |

输出文件：
- `benchmark_newton_vs_broyden_solver.csv` — 逐样本详细结果
- `benchmark_newton_vs_broyden_summary.csv` — 按方法汇总

命令行：
```bash
python benchmark_newton_vs_broyden_solver.py --n_samples 100 --seed 20260606
```

---

## 执行流程

```
main.full_pipeline_dual_task()
    │
    ├─ generate_dataset.generate_dataset()      ← 合成数据
    │       └─ ballistics.simulate_trajectory()
    │
    ├─ train_model.train(trajectory_mode=0)     ← 训练 Low 模型
    │       ├─ model_architecture.SingleBranchDNN
    │       └─ physics_loss.SmoothPhysicsLoss
    │
    ├─ train_model.train(trajectory_mode=1)     ← 训练 High 模型
    │
    └─ _verify_unified_model()                  ← 验证
            └─ predict.solve_target_unified()
                    ├─ model_architecture.SingleBranchDNN
                    └─ solver._refine_candidate()
```

## 环境依赖

- Python 3.11+
- PyTorch (CUDA 可选)
- NumPy, Pandas, Matplotlib
- 无需额外深度学习框架

---

## 项目文件结构与脚本说明

核心代码仍在根目录（`ballistics.py` / `solver.py` / `model_architecture.py` 等），非核心脚本已整理到 `scripts/` 子目录：

```
scripts/
├── benchmarks/     ← 求解器 benchmark、消融实验
├── experiments/    ← 参数扫描、噪声验证等训练实验
├── pipeline/       ← 流程脚本（暂空）
├── debug/          ← 调试脚本（暂空）
└── legacy/         ← 旧脚本（暂空）
```

根目录保留了 `benchmark_newton_vs_broyden_solver.py` 和 `ablation_refine_strategy.py` 作为兼容入口（wrapper），推荐使用 `scripts/benchmarks/` 下的实际脚本。

详细文档：
- 脚本索引：`docs/SCRIPT_INDEX.md`
- 代码地图：`docs/CODE_MAP.md`
- 运行手册：`docs/RUNBOOK.md`
