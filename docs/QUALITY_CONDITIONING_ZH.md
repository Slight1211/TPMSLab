# TPMS Lab 网格质量改进版 0.3.7.dev0

日期：2026-10-28（America/Toronto）。这是本地开发交付，没有上传或覆盖 Notion 原包，也没有发布到 PyPI/GitHub。

## 使用

在本交付目录双击 `Start-TpmsLab.cmd`。本机优先使用已经验证的环境；换机时需要 Python 3.11，首次启动会安装锁定依赖。
在“尺寸与采样”中勾选 **优化内部网格**，生成后展开“运行记录与模型说明”，查看优化前后低质量单元数量。下载完整包包含 NAS、NPZ、报告和哈希。

Python：

```python
from tpmslab import Config, generate, condition_mesh, quality_audit, save_model

base = generate(Config(resolution=12, domain_mode="solid_fluid"))
improved = condition_mesh(base)  # 默认每轮最多256个节点，共2轮；不修改base
print(improved["report"]["conditioning"]["before_by_phase"])
print(improved["report"]["conditioning"]["after_by_phase"])
save_model(improved, "conditioned_result")  # 必须是新目录

# 简写：generate(Config(...), condition=True)
# 显式门槛示例：不合格会抛出异常，调用者应先审查报告再决定是否导出。
# quality_audit(improved, minimum_quality=0.01, max_poor_fraction=0)
```

命令行：

```powershell
python -m tpmslab init-config params.json
python -m tpmslab generate --config params.json --out result_new --condition
```

可调工作量：`--condition-max-nodes 256 --condition-sweeps 2`。
可选导出门槛：`--min-quality 0.01 --max-poor-fraction 0`；阈值逐相检查，失败时不会创建输出目录。
浏览器“保存参数”的 `condition` 布尔字段也可由命令行读取。导出的 `config.json` 保持几何参数格式；是否优化及预算记录在 `report.json/conditioning` 中，复跑时须带对应参数。

## 改了什么

新增 `src/tpmslab/conditioning.py`。先由四面体重建外表面和域间共享面，锁定这些面上的全部节点。对低质量单元附近的内部节点，用解析导数和 SciPy SLSQP 搜索提高邻接单元最低 mean-ratio 质量的位置。

节点数、单元数、单元连接、材料/连通域标签和共享界面三角形均不变。每个候选都检查正体积、局部最低质量、低质量数量、局部分位数和邻接总体积，不把优化器的 success 标志当作通过证据。

最终再从坐标重算质量和体积。任一相的最低质量、第1百分位质量、Q<0.01数量不能变差；每相中位质量最多允许相对下降5%；每个原连通域的体积相对差不超过1e-10。失败时回退整套候选坐标并保留失败原因。5%是本实现预先设定的工程取舍，不是行业标准。

默认 `target_quality=0.15` 是选择待优化节点的关注范围，**不是保证所有输出单元Q≥0.15**。Q<0.01在这里用于描述质量尾部，也不是通用求解安全阈值。

## 实测结果与可检查数据

测试环境：Windows、Python 3.11.9、NumPy 2.4.6、SciPy 1.17.1、trimesh 5.2.0。核心没有新增依赖。基线使用原来的生成器数值模块；原始备份ZIP SHA-256为 `74d18bebc0788e7c10f6dbec8c5efe53fbf97d6d4e224fc43a34e5c5d799f970`。

探索性配对研究共12个网格：Gyroid、Primitive Schwartz、Diamond各三种线性密度范围（0.15→0.30、0.25→0.50、0.40→0.65，resolution=12），加Gyroid n=16、相位偏移的Primitive和Gyroid→Diamond混合结构。设置在初始Gyroid试验后选择，不是预注册或盲法独立验证。

- 11/12候选通过；Diamond高密度候选因固体中位质量相对下降约6.6%而回退。
- 包括回退样本，极差单元合计2710→1476，减少45.5%。总数只作描述，不能把单元当独立样本计算显著性。
- 通过候选的逐相Q第1百分位提高约14.9%～78.9%；逐相中位质量仍可能下降，最大约3.65%。
- 每个网格的节点数和单元数不变；全部边界/界面节点保持逐值相同；域标签和连通关系不变。
- 12/12独立几何审计通过，包括逐域体积、正体积和独立共享界面面归属检查。
- 12组基线和改进网格均通过完整长方体上的P1标量扩散校验：精确解u=z/Lz、侧面零通量、均匀导率，使用Jacobi预条件CG。该检查是有限元一致性和线性求解验证，不代表孔隙流动、固体弹性或FSI已经验证。
- CG迭代次数有升有降，12组次数比值的中位数约1.002；没有一般加速证据。计时期间存在其他验证任务，时间不用于跨软件性能比较。

数据在 `quality_validation_20261028/comparison.csv`、`summary.json` 和每个case的JSON中；逐节点候选记录、配置、基线/改进NAS/NPZ和哈希均保留。运行：

```powershell
python benchmarks/conditioning_study.py --out another_new_validation --save-meshes
python -m pytest -q
```

## 仍然存在的限制

被界面或外边界完全约束的劣质单元可能无法靠内部节点移动改善。低密度Diamond仍保留较多低质量单元；本次没有隐藏、删除小域，也没有改变几何来消掉这些单元。`--min-quality`等导出门槛可用于显式拒绝不满足要求的结果。

本次没有运行COMSOL求解或建立真实工况的收敛曲线，不能称为全面解决仿真稳定性。边界层、非线性材料、大变形接触、周期约束和光滑CAD均不属于本次改动。

初期Gmsh与TetGen候选出现质量尾部恶化或体积异常，未接入交付版。最终功能只使用已有NumPy/SciPy/trimesh依赖；不把外部后端尝试包装成通过的结果。

## 方法与工作流来源

- 数值优化器：[SciPy SLSQP文档](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)。采用既有局部优化思路，不主张算法首创。
- 本轮使用 scientific-critical-thinking 技能约束比较口径、保留失败与区分几何质量和求解结论；使用 computer-use/browser 能力检查界面生成与下载。
- 工作流来源：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. 当前查验记录为v2，https://doi.org/10.48550/arXiv.2609.00065 。该引用说明工作流来源，不构成网格算法验证。
