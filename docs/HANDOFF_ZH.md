# TPMS Lab 交接说明（0.3.9.dev12，2026-11-05）

本文件写给下一位维护者（ChatGPT 或其他人）。内容包括：现在的代码做到了什么、证据放在哪里、怎么复现、还剩哪些工作，以及每项工作的验收标准。所有数字都来自 `evidence/` 下的 CSV 文件，可以直接复核。

## 1. 当前状态，一句话

TPMS Lab 从隐式场出发，直接生成共形的固相/流体两相四面体网格并交给 COMSOL。已有的 16 工况横向基准以 4 节点固相四面体计分。BCC 高质量及包络模式的中位低尾质量高于所测 fTetWild、Gmsh 和 TetGen 路线，耗时见第 5 节。但 dev8 之后新增的输出（BCC 格子、周期面、包络）都还没有在 COMSOL 里实算过。这是最优先的待办（见第 6 节 T1）。

## 2. 版本脉络（本轮）

| 版本 | 主要内容 | 默认输出是否变化 |
|---|---|---|
| dev2–dev5 | 审查修复、cut-layer 优化器、提速 | 是（各版 CHANGELOG 有说明） |
| dev6–dev7 | 体积/质量预算、缓存、相感知曲面移动（由他人完成） | 是 |
| dev8 | `lattice="bcc"` 背景格子 | 否（默认仍是 kuhn） |
| dev9 | `snap_alpha_boundary`、`--quadratic` 二次曲边导出、`periodic_faces`、壁厚分辨率报告、优化器批量建表 | 否 |
| dev10 | `optimize_mesh(effort="high")` | 否 |
| dev11 | `optimize_mesh(surface_envelope=mm)`；本文件 | 否 |
| dev12 | 移除 10 节点曲边导出，网格文件采用 4 节点四面体 | 否 |

dev8 起，所有新功能都是可选开关。在默认参数下，Kuhn 路径的输出与 dev7 逐字节一致（已用哈希核对）。优化器的两处提速也已核对：输出逐字节不变。

## 3. 推荐用法

```python
from tpmslab import Config, generate, optimize_mesh, save_model
cfg = Config(family="Gyroid", density_start=.25, density_end=.5, resolution=10,
             lattice="bcc", snap_alpha=0.3, snap_alpha_boundary=0.35,
             domain_mode="solid_fluid", edge_interpolation="implicit_root")
model = optimize_mesh(generate(cfg), effort="high")      # 再加 surface_envelope=0.0087 可进一步提高最差单元质量
save_model(model, "out_dir")                               # 导出 4 节点四面体
```

命令行对应的参数：

```
--lattice bcc --snap-alpha 0.3 --snap-alpha-boundary 0.35 --optimize --optimize-effort high --surface-envelope 0.0087 --periodic-faces
```

- **分辨率**：BCC 每个体素的单元数约为 Kuhn 的 2 倍，所以要得到相同单元数，分辨率取 Kuhn 的约 5/6。
- **最低分辨率**：每胞元采样数下限为 8（`Config.validate`）。
- **周期面**：`periodic_faces=True` 只在场确实周期的方向上生效，程序会数值自动判断；如果所有方向都不周期，会报错。

## 4. 代码地图（`src/tpmslab/`）

| 模块 | 作用 | 本轮改动 |
|---|---|---|
| `model.py` | `Config` 与校验；场与阈值 `field_and_threshold`、`raw_field` | 新增 `lattice`、`snap_alpha_boundary`、`periodic_faces` |
| `volume.py` | `sample_lattice`（格子与采样）、`_generate_volume`（逐背景四面体切割）、`generate_volume` 包装（周期强制、壁厚报告）、`bcc_background`、NASTRAN 输出 | BCC、周期、壁厚报告 |
| `snapping.py` | 切割前顶点吸附（isosurface stuffing 风格），含翻转与夹缩回退 | 盒面单独阈值、周期配对吸附 |
| `multidomain.py` | 固相加两个流体侧的三次切割与合并、隐式求根 | 未改（BCC 通过 `lattice_xyz` 兼容） |
| `optimize.py` | 优化器：字典式网格，翻转、边移除 DP、盒面翻转、帽翻转，内部与曲面光顺；带轮次预算和回退 | 批量建表、DP 去缓存开销、`effort`、`surface_envelope`、周期面冻结与配对 |
| `surface_smoothing.py` | 边界顶点沿隐式面和盒面滑动，带弦误差与相质量守卫 | `frozen`、`envelope`、`surface_distances` |
| `batch_smoothing.py` | 向量化的星形评分与独立批次 | 未改 |
| `periodic.py` | 周期轴检测、节点配对、坐标强制、校验、配对光顺 | 新增 |
| `reporting.py` | 密度剖面、`wall_resolution` | 新增壁厚报告 |
| `io.py` | `save_model`、清单 | 4 节点 NAS 与 NPZ 导出 |
| `comsol*.py` | COMSOL 启动器、导入、求解、重开核验 | 未改 |
| `cli.py`、`web.py`、`static/` | 命令行与网页界面 | 新增对应开关 |

**容易踩的坑**：
- `quality.tetra_metrics` 返回的是**体积绝对值**。判断单元是否翻转要用 `quality.signed_volumes`。
- 部分文件使用 CRLF 换行，修改时要保留：`__init__.py`、`web.py`、`comsol*.py`、`io.py`、`CHANGELOG.md`、`CITATION.cff`、`pyproject.toml`、`SOURCE_ORIGIN.json`。

## 5. 证据与复现

| 内容 | 脚本 | 结果位置 |
|---|---|---|
| BCC 与 Kuhn 对比 | `benchmarks/bcc_vs_kuhn.py` | `evidence/bcc_lattice/bcc_vs_kuhn.csv` |
| 四种 TPMS × 四档密度横向对比（Kuhn/BCC/Gmsh） | `benchmarks/bcc_crosscompare.py [--bcc-boundary-alpha 0.35 --bcc-effort high --bcc-envelope 0.00866 --only tpms_bcc]` | `evidence/dev9/crosscompare.csv`、`bcc_high_effort.csv`；包络结果在 `docs/COMPARISON_ZH.md` |
| TetGen、fTetWild、microgen | `benchmarks/other_tools.py`（microgen 要在单独的 venv 里运行，需要 numpy<2 和 cadquery-ocp 7.8.1.1.post1） | `evidence/bcc_lattice/other_tools.csv` |
| 历史二次单元体积误差 | dev9 历史试验 | `evidence/dev9/quadratic_volume.csv` |
| 图 | `benchmarks/bcc_figures.py SRC DST [level]` | `evidence/dev9/figures/` |
| 未采用的实验 | `benchmarks/itt_*.py`、`benchmarks/refine_experiment.py` | `evidence/itt_stage0/README.md`；CHANGELOG dev9 |
| 测试 | `python -m pytest -q`（326 个，约 2.4 分钟） | — |

**计时条件**：所有计时都在单线程 Linux 容器里测得（Python 3.13，numpy 2.4.6，Gmsh 4.15.2）。Windows 机器上的绝对时间会不同，各工具之间的相对关系可参考。

**主要结论**（16 个工况的中位数；详见 `docs/COMPARISON_ZH.md`）：

| 方案 | p01 | 最小质量 | 用时 |
|---|---|---|---|
| BCC 标准 | 0.51 | 0.38 | 2.7 s |
| BCC 高质量 | 0.57 | 0.40 | 4.1 s |
| BCC 高质量 + 包络 | 0.62 | 0.48 | 5.4 s |
| fTetWild | 0.48 | 0.35 | 10.3 s |

- **历史二次单元试验**：固相体积误差下降 5–64 倍，例如 Gyroid r13 从 0.335 pp 降到 0.006 pp。
- **包络模式的代价**：体积误差比 fTetWild 略大的工况占 12/16。

## 6. 待办（按优先级），每项附验收标准

**T1. COMSOL 实算验证（必须，由用户在本机运行）**

- **运行**：双击 `source\Run-Dev9Validation.cmd`。可以带用例名，只跑部分用例。结果写在 `evidence\dev9_validation\status.json`。
- **用例**：`bcc_flow`、`bcc_elasticity`、`periodic_import`、`bcc_convergence`（res 8/10/13/17）。
- **验收**：
  - 每个用例的 `status` 为 passed；
  - `relative_flux_imbalance` 小于 1e-6 量级；
  - `bcc_convergence` 的出口流量和应变能随分辨率单调收敛，可以做 Richardson 外推（`tpmslab.convergence.richardson`）。
- **周期面**：目前只验证了网格能导入。下一步要在 Java 模板里加周期条件，把 x0 对应 x1、y0 对应 y1；边界 id 规则是 2+2·轴+侧，流体相再加 10。

**T2. 包络模式保体积**

- **现象**：开启 `surface_envelope` 后，中位体积误差比精确曲面模式大 0.02–0.1 pp。
- **做法**（任选）：
  - 在 `surface_smoothing.surface_smooth` 的包络试探中加入"体积变化惩罚"，或按法向成对移动；
  - 或者只允许位于盒面交线上的界面节点使用包络。诊断显示最差单元集中在那里，见 CHANGELOG dev9。
- **验收**：16 工况的中位体积误差不大于精确曲面模式，同时保住大部分最小质量的提升（中位最小质量 ≥ 0.45）。

**T3. 提速（大规模多胞模型需要）**

- **热点**：`optimize.py` 中的 `_ring_triangulation`、`_metric`、`_try_replace`，以及 `volume._generate_volume` 中逐单元的 Python 切割循环。
- **做法**：numba 或 Cython；或者按空间分块并行。
- **约束**：在默认参数下输出必须逐字节不变。可以用对 points/tetra 做 sha256 的方式核对，参考 dev9 的做法。
- **验收**：12 工况计时集（4 设计 × res 10/13/17，BCC）的优化总时间降低 3 倍以上。

**T4. 周期面的质量与速度**

- **现状**：`periodic.pair_smooth` 是纯 Python 循环，耗时约为普通优化的 2 倍；周期面上也不做翻转。
- **做法**：
  - 把配对光顺向量化；
  - 加入"配对盒面翻转"：两面同时做相同的重连。
- **验收**：在开启周期面的情况下，p01 与非周期的差距小于 0.005。

**T5. 盒面边界层**

- **现象**：最差单元集中在曲面与盒面小角度相交的位置。把 IWP 的盒面放到其镜面对称面上（相位 0,0,0），最小质量从 0.26 升到 0.49。
- **可做**：提供"按对称面对齐相位"的建议工具，或在报告中输出交线处的夹角统计。
- **注意**：相位属于用户的设计参数，不能替用户自动修改。

**T6. 功能扩展**：圆柱等非盒形外形；更多 TPMS 族的周期性检查；大规模模型的内存估算（`estimate_lattice_bytes` 已经区分了 BCC）。

**T7. 发表级证据**：COMSOL 网格收敛研究，加上解析解或实验对照；对 fTetWild 和 Gmsh 先调参再对比；在标准硬件上重测计时。

## 7. 已知问题与注意事项

- `wall_resolution` 只支持 sheet 模式（壁厚取 2q/|∇f| 的一阶估计）。
- `snap_alpha_boundary=0.4` 在 Diamond r10 上会出现退化单元，所以推荐 0.35。
- 优化器的轮次预算可能在 IWP 上触发回退（报告里会写明原因），属正常现象。
- 本机没有 `.venv` 时，`Run-*.cmd` 会先调用 `Setup-Python.ps1` 安装环境。

## 8. 工作原则（建议沿用）

- 每项改进都**先约定通过标准，再做实验**；达不到就停下来，并如实记录。ITT 和自适应加密都是这样被否决的，相关记录保留在 `benchmarks/` 和 `evidence/` 中。
- 新功能默认关闭，保证已有用户的输出逐字节不变。
- 质量、体积和周期性都在结果数组上重新核对，不信任中间簿记。
