# 运行手册

## 1. 推荐运行顺序

1. **生成数据** → `python main.py`（会自动执行 `generate_dataset.py`）
2. **训练模型** → `main.py` 自动训练 low 和 high 两个分支模型
3. **运行预测** → `main.py` 自动执行验证
4. **运行 benchmark** → `python Benchmark.py --n 100`
5. **查看输出文件** → 见下方输出文件说明

如只需单独生成数据：`python generate_dataset.py`

---

## 2. 常用命令

### 完整流程

```bash
python main.py
```

### 对照实验

```bash
# 5 种方法对比（需先训练模型）
python Benchmark.py --n 100 --out_dir artifacts_127

# 跳过 NN 方法（模型未训练时）
python Benchmark.py --n 100 --skip_nn
```

### 求解器 benchmark

```bash
# Newton vs Broyden 离线对比（不依赖 NN）
python scripts/benchmarks/benchmark_newton_vs_broyden_solver.py --n_samples 100 --seed 20260606
```

### 精修策略消融

```bash
python scripts/benchmarks/ablation_refine_strategy.py --n 100 --seed 20260606
```

### 扫描实验

```bash
# 模型类型对比
python scripts/experiments/run_model_type_sweep.py

# PGNN lambda 参数扫描
python scripts/experiments/run_pgnn_lambda_seed_sweep.py

# 物理步数扫描
python scripts/experiments/run_pgnn_steps_sweep.py

# 混合模型多 seed 稳定性
python scripts/experiments/run_hybrid_seed_sweep.py

# 测量噪声鲁棒性
python scripts/experiments/run_measurement_noise_validation.py --n 200
```

---

## 3. 输出文件说明

| 输出 | 位置 | 说明 |
|---|---|---|
| 训练数据集 | `artifacts_127/dataset.csv`、`dataset.npy` | 特征和标签数据 |
| 训练模型 | `artifacts_127/dual_model_low.pt`、`dual_model_high.pt` | 低/高弹道分支 PyTorch 模型 |
| 归一化器 | `artifacts_127/dual_scaler_low.json`、`dual_scaler_high.json` | Min-Max 归一化参数 |
| 训练历史 | `artifacts_127/training_history_low.csv`、`training_history_high.csv` | 各 epoch 的 loss 记录 |
| Benchmark 结果 | `artifacts_127/benchmark_detail.csv`、`benchmark_summary.csv` | 对照实验详细和汇总结果 |
| Benchmark 图表 | `artifacts_127/benchmark_compare.png` | 箱线图对比 |
| 求解器 benchmark | `benchmark_newton_vs_broyden_solver.csv`、`benchmark_newton_vs_broyden_summary.csv` | Newton vs Broyden 结果 |
| 消融结果 | `ablation_refine_strategy_results.csv`、`ablation_refine_strategy_summary.csv` | 策略消融结果 |
| 扫描结果 | `artifacts_127/` 下各 `*_sweep_*.csv` 文件 | 参数扫描实验输出 |

---

## 4. 常见问题

### ModuleNotFoundError

确认从 **项目根目录** 运行脚本，不要 cd 到子目录。

移动后的脚本（`scripts/` 下）已添加自动路径处理：
```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
```

### 模型文件缺失

先运行训练流程：
```bash
python main.py
```
或单独训练：
```python
from train_model import train
train(trajectory_mode=0)  # low branch
train(trajectory_mode=1)  # high branch
```

### benchmark 太慢

降低样本数或增大仿真步长：
```bash
python Benchmark.py --n 50
python scripts/benchmarks/benchmark_newton_vs_broyden_solver.py --n_samples 50
```

### 结果不稳定

检查并统一关键参数：
- `--seed`：随机种子（建议固定 20260606 或 42）
- `--dt`：仿真步长（0.05 较快，0.01 更精确）
- `--t_max`：最大仿真时间（120s 通常足够）
- `--y_tol` / `--z_tol`：命中容差（默认 2.0m）

### GPU 内存不足

减小 batch_size 或使用 CPU 模式：
```python
train(..., batch_size=4096)
```

### scripts/ 下脚本 import 失败

确认 `sys.path` 已设置。如果仍有问题，临时方案：
```bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
```
