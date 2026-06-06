# 脚本索引

本文件记录项目中非核心脚本的用途、运行方式和状态。

---

## 1. Pipeline 流程脚本

| 文件 | 作用 | 输入 | 输出 | 推荐运行命令 | 状态 |
|---|---|---|---|---|---|
| (暂无) | — | — | — | — | — |

---

## 2. Benchmark 对照实验脚本

| 文件 | 作用 | 输入 | 输出 | 推荐运行命令 | 状态 |
|---|---|---|---|---|---|
| `scripts/benchmarks/benchmark_newton_vs_broyden_solver.py` | Newton vs Broyden 求解器离线对比。相同初值下对比两种迭代方法的耗时、仿真调用次数和最终误差 | 无（正向仿真生成测试样本） | `benchmark_newton_vs_broyden_solver.csv`、`benchmark_newton_vs_broyden_summary.csv` | `python scripts/benchmarks/benchmark_newton_vs_broyden_solver.py --n_samples 100` | 推荐 |
| `scripts/benchmarks/ablation_refine_strategy.py` | 消融实验：对比 6 种精修策略（nn_only / newton / broyden / grid_newton / grid_broyden / broyden_fast） | 无（正向仿真生成测试样本） | `ablation_refine_strategy_results.csv`、`ablation_refine_strategy_summary.csv` | `python scripts/benchmarks/ablation_refine_strategy.py --n 100` | 推荐 |
| `Benchmark.py`（根目录） | 5 种求解方法的独立性能对比（A/B/C/D/E），含 NN 模型加载 | 已训练模型（.pt）、测试样本 | `benchmark_detail.csv`、`benchmark_summary.csv`、`benchmark_compare.png` | `python Benchmark.py --n 100` | 推荐 |

---

## 3. Experiments 测试验证脚本

| 文件 | 作用 | 输入 | 输出 | 推荐运行命令 | 状态 |
|---|---|---|---|---|---|
| `scripts/experiments/run_hybrid_seed_sweep.py` | 混合模型多 seed 扫描：对比 MLP / KAN-MLP / KAN-MLP+PGNN 在不同 seed 下的训练稳定性 | `dataset.npy`（先运行 generate_dataset.py） | 各配置的训练历史 CSV、模型 .pt 文件 | `python scripts/experiments/run_hybrid_seed_sweep.py` | 可用 |
| `scripts/experiments/run_measurement_noise_validation.py` | 测量噪声鲁棒性验证：评估 NN+solver 在真实传感器噪声下的命中精度 | 已训练模型（.pt）、scaler JSON | 噪声条件下的命中误差统计 CSV | `python scripts/experiments/run_measurement_noise_validation.py --n 200` | 可用 |
| `scripts/experiments/run_model_type_sweep.py` | 模型类型扫描：对比 MLP / KAN / KAN-MLP 三种结构的训练效果 | `dataset.npy`（先运行 generate_dataset.py） | 各模型类型的训练历史 CSV | `python scripts/experiments/run_model_type_sweep.py` | 可用 |
| `scripts/experiments/run_pgnn_lambda_seed_sweep.py` | PGNN lambda+seed 联合扫描：确定最佳物理损失权重 | `dataset.npy`、依赖 `run_pgnn_noise_sweep.py` | 各 lambda/seed 组合的训练结果 CSV | `python scripts/experiments/run_pgnn_lambda_seed_sweep.py` | 可用 |
| `scripts/experiments/run_pgnn_noise_sweep.py` | PGNN 噪声配置模块：导出 INPUT_NOISE / LABEL_NOISE 供其他扫描脚本导入 | 无 | 仅导出噪声配置字典 | `python scripts/experiments/run_pgnn_noise_sweep.py` | 配置模块 |
| `scripts/experiments/run_pgnn_steps_sweep.py` | PGNN 物理步数扫描：确定最佳 physics_steps 参数 | `dataset.npy` | 各 steps 配置的训练结果 CSV | `python scripts/experiments/run_pgnn_steps_sweep.py` | 可用 |

---

## 4. Debug 调试脚本

| 文件 | 作用 | 输入 | 输出 | 推荐运行命令 | 状态 |
|---|---|---|---|---|---|
| (暂无) | — | — | — | — | — |

---

## 5. Legacy 旧脚本

| 文件 | 作用 | 输入 | 输出 | 推荐运行命令 | 状态 |
|---|---|---|---|---|---|
| (暂无) | — | — | — | — | — |

---

## 说明

- **根目录 wrapper**：`benchmark_newton_vs_broyden_solver.py` 和 `ablation_refine_strategy.py` 在根目录保留了兼容入口，实际代码在 `scripts/benchmarks/` 下。
- **状态字段**：推荐 / 可用 / 调试用 / 临时 / 旧版保留 / 待确认。
