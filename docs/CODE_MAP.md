# 项目代码地图

## 1. 核心模块

| 文件 | 作用 | 是否主流程依赖 |
|---|---|---|
| `ballistics.py` | 外弹道物理仿真：RK4 积分、ISA 大气模型、G7 弹道系数、3D 风场、命中截面插值 | 是 |
| `solver.py` | 数值精修求解器：grid search、Newton、Broyden、自适应快速回退策略（5 种 refine_mode） | 是 |
| `model_architecture.py` | 神经网络模型结构：MLP（残差块）、KAN、KAN-MLP hybrid | 是 |
| `physics_loss.py` | 训练阶段可微物理损失（PGNN）：Euler 积分 + 弹道位置约束 | 是 |
| `train_model.py` | 模型训练逻辑：数据加载、归一化、分组划分、AMP 混合精度、EMA、早停 | 是 |
| `predict.py` | 推理预测和求解入口：双分支 NN 预测、精修、择优、可视化 | 是 |
| `generate_dataset.py` | 合成训练数据生成：多进程并行仿真、网格化插值、CSV/NPY 输出 | 是 |
| `feature_schema.py` | 特征/标签列定义和推理输入构建 | 是 |
| `main.py` | 顶层完整流程入口：数据集生成 → 训练 → 验证 | 是 |

---

## 2. 主流程

```
数据生成：
    generate_dataset.py
        └─ ballistics.py（仿真）

训练：
    train_model.py
        ├─ model_architecture.py（网络结构）
        └─ physics_loss.py（物理损失）

预测：
    predict.py
        ├─ model_architecture.py（NN 推理）
        └─ solver.py（数值精修）

完整流程：
    main.py
        ├─ generate_dataset.py
        ├─ train_model.py
        └─ predict.py
```

---

## 3. 求解器流程

当前正式求解流程（`solver._refine_candidate`，默认 `refine_mode="broyden_fast"`）：

```
NN 初值
  └─ Broyden 快速精修
       ├─ 满足 y_tol/z_tol → 直接返回（跳过 grid search）
       └─ 不满足 → grid search 回退
            └─ grid 结果再次 Broyden 精修
                 └─ 选择 err_3d 最小的结果返回
```

支持的 refine_mode：
- `"broyden_fast"`：上述默认策略
- `"newton"`：直接 Newton 精修
- `"broyden"`：直接 Broyden 精修
- `"grid_newton"`：grid search → Newton
- `"grid_broyden"`：grid search → Broyden

---

## 4. 目录结构

```
项目根目录/
├── ballistics.py          ← 核心：物理引擎
├── solver.py              ← 核心：数值求解器
├── model_architecture.py  ← 核心：网络结构
├── physics_loss.py        ← 核心：物理损失
├── train_model.py         ← 核心：训练
├── predict.py             ← 核心：推理
├── generate_dataset.py    ← 核心：数据生成
├── feature_schema.py      ← 核心：特征定义
├── main.py                ← 核心：顶层流程
├── Benchmark.py           ← 入口：对照实验
├── benchmark_newton_vs_broyden_solver.py ← wrapper → scripts/benchmarks/
├── ablation_refine_strategy.py          ← wrapper → scripts/benchmarks/
├── scripts/
│   ├── benchmarks/
│   │   ├── benchmark_newton_vs_broyden_solver.py
│   │   └── ablation_refine_strategy.py
│   ├── experiments/
│   │   ├── run_hybrid_seed_sweep.py
│   │   ├── run_measurement_noise_validation.py
│   │   ├── run_model_type_sweep.py
│   │   ├── run_pgnn_lambda_seed_sweep.py
│   │   ├── run_pgnn_noise_sweep.py
│   │   └── run_pgnn_steps_sweep.py
│   ├── pipeline/
│   ├── debug/
│   └── legacy/
├── docs/
│   ├── SCRIPT_INDEX.md
│   ├── CODE_MAP.md
│   └── RUNBOOK.md
└── artifacts_*/           ← 输出目录（模型、数据、图表）
```

---

## 5. 常用运行入口

| 目的 | 命令 |
|---|---|
| 完整流程 | `python main.py` |
| 对照实验（A/B/C/D/E） | `python Benchmark.py --n 100` |
| Newton vs Broyden benchmark | `python scripts/benchmarks/benchmark_newton_vs_broyden_solver.py --n_samples 100` |
| 精修策略消融 | `python scripts/benchmarks/ablation_refine_strategy.py --n 100` |
| 模型类型扫描 | `python scripts/experiments/run_model_type_sweep.py` |
| PGNN lambda 扫描 | `python scripts/experiments/run_pgnn_lambda_seed_sweep.py` |
