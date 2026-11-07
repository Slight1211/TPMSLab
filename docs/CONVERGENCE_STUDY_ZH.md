# 网格收敛与有效传输系数研究（0.3.9.dev0 更正）

本页保留 0.3.8.dev0 的原始 P1 求解数据，使用 0.3.9.dev0 更正不等比三网格的外推计算及解释，没有重新求解这些工况。Python → COMSOL 的当前入口见 [工作流说明](PYTHON_COMSOL_ZH.md)。

本页记录两项面向论文的求解验证：一项在包内用 P1 有限元完成（任何机器可复现），一项需要 Windows + 已授权的 COMSOL（在 6.2.0.290 上实算；脚本已备好，由作者在本机运行）。两项都只回答"网格改变后求解结果是否收敛到同一极限、收敛快慢如何"，不是对真实材料或实验的验证。

## A. 逐相稳态扩散（包内 P1 求解器）

`benchmarks/transport_convergence.py`

问题：对固体相和孔隙相分别求解 ∇²u = 0，z=0 面 u=0，z=Lz 面 u=1，其余边界（含固体/流体共享界面和外盒侧面）零通量。耗散 E = ∫|∇u|² 给出相对有效系数 k_eff/k_bulk = E/(Lx·Ly/Lz)。固体相的结果是相对有效导热率，孔隙相的结果是相对有效扩散率（迂曲度的倒数量级）。未与任一 z 面相连的孤立连通域被剔除（它们对通量贡献为零，否则刚度矩阵奇异）。

设计：Gyroid 0.25→0.50、Diamond 0.15→0.30、Primitive Schwartz 0.25→0.50，均为 5 mm 单胞、线性 z 梯度、片状结构。分辨率 8/12/16/24（Gyroid 另加 32）；变体：基线（`snap_alpha=0`）与吸附+求根（`snap_alpha=0.25`，`edge_interpolation="implicit_root"`）。固体相另用 Gmsh 4.15 重新参数化并按格距重划的网格、同一求解器求解，作为外部交叉检查（分辨率 ≤16，Gmsh 对更细的分片表面参数化极慢）。

Richardson 外推取最细三级分辨率，按 f(h)=L+C·h^p 解不等比三点方程，报告拟合阶 p 与拟合极限 L。分辨率 12/16/24 不满足等比条件，不能使用等比公式。三点拟合不能单独证明渐近区间或两种生成器的真实极限一致。

结果见 `transport_convergence_<date>/transport_convergence.csv` 与 `summary.json`（`richardson` 字段），表格在本文末尾由脚本生成。

## B. COMSOL 孔隙流动与线弹性（作者本机）

`benchmarks/comsol_convergence.py` + `Run-ComsolConvergence.cmd`

对同样的设计/分辨率/变体，调用包内已有的 COMSOL 桥接跑两个已定义的示例：双相网格上的固定壁蠕动流（入口 0.01 Pa，出口 0 Pa，μ=0.001 Pa·s）得到出口流量和表观渗透率 K = Q·μ·Lz/(Lx·Ly·Δp)；固体网格上的线弹性（底面固定，顶面 z 位移 −0.001·Lz）得到应变能与最大位移。每次运行保留独立目录（MPH、comsol.log、build.log、comsol_verification.json），结果逐行追加到 `comsol_convergence.csv`，中途失败不丢已有结果。

运行方法（在解压后的 `source` 目录）：

```
Run-ComsolConvergence.cmd
```

默认 3 个设计 × 分辨率 8/12/16/24 × 2 个变体 × 2 个问题 = 48 次 COMSOL 求解；分辨率 24 的双相网格约 30～35 万单元，蠕动流求解可能需要数分钟到十余分钟。可以先跑粗的三级：

```
powershell -ExecutionPolicy Bypass -File Run-ComsolConvergence.ps1 -Resolutions 8,12,16
```

或只跑流动 / 只跑弹性：`-SkipElasticity`、`-SkipFlow`。输出目录为 `source` 的上一级下的 `comsol_convergence_<时间戳>`，把整个目录（或至少 `comsol_convergence.csv`、`environment.json`、`errors.log` 和各运行目录的 `comsol_verification.json`、`comsol.log`）放回共享文件夹即可继续分析。

注意：弹性示例要求固体连通（`volume_components == 1`）；吸附变体的固体单独网格只能用吸附（根投影需要双相）。当前 CSV 将弹性变体标为 `snapped`，流动标为 `snapped_root`，并保存实际插值方式。COMSOL 自带的求解时间从 `comsol.log` 的 `Solution time` 行尽力解析，缺失则为空。

## 结果（A 项）

运行环境：Linux x86_64、Python 3.13.18、NumPy 2.5.3、SciPy 1.18.1；三个设计 × 分辨率 8/12/16/24 × 两变体，固体相与孔隙相各一次。Gmsh 交叉检查与 Gyroid 32 分辨率的补充运行因会话中断未完成，可按文中命令重跑。

**固体相相对有效导热率 k_eff/k_s**

| 设计 | 变体 | res 8 | res 12 | res 16 | res 24 | 表观阶 p | Richardson 极限 | res 24 单元数 | res 24 CG 迭代 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gyroid_central | baseline | 0.2829 | 0.2570 | 0.2471 | 0.2406 | 2.22 | 0.2361 | 88426 | 1045 |
| Gyroid_central | snapped_root | 0.2632 | 0.2497 | 0.2436 | 0.2387 | 1.66 | 0.2337 | 42413 | 274 |
| Diamond_low | baseline | 0.1904 | 0.1625 | 0.1531 | 0.1469 | 2.25 | 0.1428 | 92304 | 1031 |
| Diamond_low | snapped_root | 0.1679 | 0.1564 | 0.1501 | 0.1459 | 2.25 | 0.1430 | 32828 | 267 |
| Primitive_Schwartz_central | baseline | 0.2990 | 0.2835 | 0.2768 | 0.2710 | 1.49 | 0.2641 | 74068 | 914 |
| Primitive_Schwartz_central | snapped_root | 0.2969 | 0.2806 | 0.2751 | 0.2704 | 1.54 | 0.2651 | 38778 | 185 |

**孔隙相相对有效扩散率 D_eff/D_0**

| 设计 | 变体 | res 8 | res 12 | res 16 | res 24 | 表观阶 p | Richardson 极限 | res 24 单元数 | res 24 CG 迭代 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gyroid_central | baseline | 0.2990 | 0.3101 | 0.3138 | 0.3151 | 4.20 | 0.3154 | 105815 | 908 |
| Gyroid_central | snapped_root | 0.3244 | 0.3207 | 0.3191 | 0.3175 | 0.85 | 0.3134 | 61726 | 295 |
| Diamond_low | baseline | 0.3076 | 0.3179 | 0.3183 | 0.3168 | – | – | 137480 | 691 |
| Diamond_low | snapped_root | 0.3347 | 0.3255 | 0.3221 | 0.3187 | 1.01 | 0.3120 | 77632 | 275 |
| Primitive_Schwartz_central | baseline | 0.2824 | 0.2861 | 0.2854 | 0.2866 | – | – | 92110 | 1168 |
| Primitive_Schwartz_central | snapped_root | 0.3005 | 0.2937 | 0.2907 | 0.2882 | 1.51 | 0.2852 | 58550 | 232 |

解读：

- 固体相：当前有限序列均单调。更正后的两种拟合极限相对差约为 Gyroid 1.014%、Diamond 0.181%、Primitive 0.357%（以两估计均值为分母）。仍不能据三点拟合证明真实极限一致，也不能断言吸附变体在每一级都更接近极限。
- 孔隙相：res 24 两变体的系数绝对差约为 0.2373、0.1891、0.1604 个百分点，相对差约为 0.750%、0.595%、0.558%。Diamond 与 Primitive 基线非单调，不支持“夹住真实极限”的说法。表中“–”代表最细三级无正阶幂律拟合；可拟合的条目也仅是条件性估计，需要更细网格和渐近区间检验。
- 求解代价：同一分辨率下吸附+求根的单元数约为基线的一半，Jacobi-CG 迭代次数约为基线的 1/3～1/5；这是自由度与条件数共同作用的结果，不构成对商业求解器加速的证明。
- 以上是逐相标量椭圆问题的一致性与收敛证据，不是 Stokes、弹性、FSI 或 COMSOL 验证；后者见 B 项脚本。
