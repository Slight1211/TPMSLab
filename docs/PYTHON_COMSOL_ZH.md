# Python → COMSOL 工作流（0.3.9.dev7）

在 `source` 目录执行。网格坐标为 mm；材料参数 E、nu、rho 使用 Pa、无量纲、kg/m³。

## 1. 准备 Python

```powershell
powershell -ExecutionPolicy Bypass -File .\Setup-Python.ps1
```

可用 `-PythonExe 'C:\路径\python.exe'` 指定 Python 3.11+。依赖安装失败后可重跑；环境内存在 python.exe 不代表安装完成。此入口安装核心包，不需要 Flask。

## 2. Python 生成并保存网格

```python
from tpmslab import Config, generate, save_model, run_comsol

config = Config(
    family="Gyroid", resolution=12,
    domain_mode="solid_fluid",
    snap_alpha=0.25, edge_interpolation="implicit_root",
)
mesh = generate(config, progress=print)
folder = save_model(mesh, "../runs/gyroid_r12")  # 每次生成使用新目录
result = run_comsol(folder, solve=True, progress=print, build_timeout_s=600)
print(result["output_directory"])
```

`solid_fluid` 求解固定壁孔隙蠕动流；`solid` 求解静力弹性示例。弹性单相配置使用 `edge_interpolation="linear"`，可启用 `snap_alpha`。两者均为示例边界条件，尚不代表材料实验验证或网格收敛证明。

COMSOL 自动检测 C:/D: 盘安装目录；可显式设置：

```powershell
$env:COMSOL_BIN = 'C:\Program Files\COMSOL\COMSOL62\Multiphysics\bin\win64'
```

需本机可用许可证。不同 COMSOL 版本以实际导入、求解和重开记录为准。

本轮在 COMSOL 6.2.0.290 上完成了单相、双相 Java 编译兼容修正。用户授权本次文件访问后，解析半盒的弹性和流动示例均已实际求解、保存并重开核验；记录在 `evidence/authorized_smoke`。另有粗分辨率 Gyroid 配对工况，结果以 `evidence/gyroid_r8/comsol_convergence.csv` 为准。这些是工作流检查，不构成网格收敛或材料实验验证。

本次获准的偏好文件保存在 `evidence/authorized_preferences`。需要它时显式传入：

```python
result = run_comsol(folder, solve=True,
                    preferences_directory="../evidence/authorized_preferences",
                    progress=print)
```

程序把偏好文件复制到本次 attempt，并对进程设置独立 `-prefsdir`，不改变全局 COMSOL 设置。启用的是本次受信任脚本的文件访问权限，其他安全限制保留；未传该参数时沿用 COMSOL 默认设置。Windows 原生恢复文件名较长，程序自动使用短的临时 scratch/recovery 目录，MPH 和日志仍保存在 attempt 内。

兼容性依据：[6.2 MeshSequence API](https://doc.comsol.com/6.2/doc/com.comsol.help.comsol/api/com/comsol/model/MeshSequence.html) 使用 boolean 方法；[后续版本兼容说明](https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_release_text.06.015.html) 说明其改为 String 方法。桥接在运行时选择可用签名。

## 2b. 网格质量优化（0.3.9.dev7）

```python
mesh = generate(config, optimize=True)      # 或 tpmslab generate --optimize
print(mesh["report"]["optimization"])       # 翻转/边移除次数、边界顶点移动数、min/p01 前后
```

`optimize=True` 在生成后做局部处理，每轮只针对质量最差的 5% 单元：

- 切割层内部的 2-3 翻转与通用边移除（环长 3–7，动态规划选环多边形的最优三角化；3-2、4-4 是其特例）；
- 盒面边移除：盒面上共面的两个三角形换另一条对角线并重排相邻单元，区域与体积完全不变；
- 界面"帽"翻转（双相）：四个顶点都在界面上且两个面都是界面面的薄片单元，换到另一相并消去其共享边——等价于同一组曲面点换一种三角化，仅在质量提高且弦误差不增大时保留；
- 内部节点光顺：每个顶点试"邻点质心"和"使最差相邻四面体变为正四面体的位置"两个方向、四个步长，取最好的；
- 边界顶点在长方体面内、以及**沿精确隐式零水平集**滑动（每一步 Newton 重新投影，残差 < 1e-9 mm）。在能带来质量提升的位置中，优先选使相邻曲面三角形偏离水平集最少的位置，避免几何保真度变差。

两相仍共形，盒面和界面的三角剖分可能换对角线；界面帽翻转可能把单元移到另一相。边界数组在输出前重建。可与 `snap_alpha=0.25`、`edge_interpolation="implicit_root"` 一起使用。

dev6 默认启用 `chord_tolerance=0.0`，按每个批次当前邻域检查采样弦误差，修复旧版复用过期邻域的问题。每一轮结束时，逐相检查最小质量、p01 和低质量单元体积分数不恶化；中位质量相对输入最多下降 15%。每相累计体积变化默认不超过盒体积的 **0.3 个百分点**。超限整轮回退，并在 `optimization.rollback_reasons` 中说明原因。此阈值可调，不代表有限元求解误差上限。

dev7 将逐相最小质量检查提前到混相节点的每次移动，避免固体相改善却损害流体相。整轮质量、弦误差和体积预算保持原值。每轮的 `surface_phase_guard_rejected_trials` 记录因此被拒绝的试探。有效轮次可能增加，最终质量通常可进一步改善，但不能保证所有结构的优化总时间都缩短。

生成阶段复用顶点顺序完全一致的共享面 SVD，并提前筛掉明显不会获选的四面体剖分；临界判断和获选结果仍按原 NumPy 运算计算。优化阶段缓存固定坐标下的单元评分，移动坐标后清空；弦误差检查只展开实际关联面，并先排除质量不合格的试探，减少重复计算与临时内存。

```python
from tpmslab import generate, optimize_mesh

base = generate(config)
mesh = optimize_mesh(base, max_phase_volume_shift_pp=0.05)  # 更严格的体积预算
fixed = optimize_mesh(base, surface=False)                 # 固定边界顶点与各相体积
```

显式传 `chord_tolerance=None` 可关闭弦误差硬约束，传 `max_phase_volume_shift_pp=None` 可关闭体积预算。`smooth="slsqp"` 仍支持附加 SLSQP 光顺，并使用重连后的边界。优化后密度剖面、分相和分域体积全部从最终网格重算；密度分箱的 `slab_count` 是非等宽分箱求平均的权重。旧网格的 COMSOL 求解记录不会作为新网格的有效结果继续保留。

## 3. 命令行分步运行与重试

```powershell
.\.venv\Scripts\python.exe -m tpmslab init-config params.json
# 编辑 params.json 中的 domain_mode、snap_alpha、edge_interpolation 等
.\.venv\Scripts\python.exe -m tpmslab generate --config params.json --out ..\runs\case01
.\.venv\Scripts\python.exe -m tpmslab comsol --source ..\runs\case01 --import-only
.\.venv\Scripts\python.exe -m tpmslab comsol --source ..\runs\case01 --timeout 600
# 对本次已经授权的配置，可追加：
# --preferences-directory ..\evidence\authorized_preferences
```

弹性可加 `--material material.json`，文件示例：`{"E":1.5e9,"nu":0.3,"rho":950}`。重复 `comsol` 命令会创建新 attempt，无需重新生成网格。

`generate --mph/--solve`、`comsol`、`remesh-quality` 三个子命令现在接受相同的 COMSOL 启动选项：`--preferences-directory`、`--timeout`、`--max-tetrahedra`、`--np`，以及 `--element-order {1,2}`。`--element-order 2` 请求二次位移单元（弹性）或 P2+P1（蠕动流）；随附 evidence 全部使用默认值 1，阶次 2 尚未实算。每个 attempt 的 `run.json` 与 `comsol_verification.json` 记录实际使用的 `comsol_bin`、从批处理日志解析出的 `comsol_version`、`element_order` 和 `processes`。

双相流动示例只对同时接触 Z- 与 Z+ 的流体域施加 `spf` 物理场；闭孔或死端流体域仍作为网格域和材料保留，其 id 写入核验记录的 `excluded_fluid_domains`。只有不存在贯通域时才拒绝求解。

每次运行保存到 `comsol_runs/<attempt_id>/`：输入 NAS、副本参数、Java、设置、日志、MPH、`run.json` 及核验记录。模型根目录 `report.json` 的 `comsol_run` 表示最新状态；失败会清除当前成功标记，旧运行目录保留。`input_manifest.json` 固定生成时的输入哈希；运行前检查文件是否发生变化。

## 4. 收敛研究

先执行一个设计、一个粗分辨率，再增加工况：

```powershell
.\.venv\Scripts\python.exe benchmarks\comsol_convergence.py --out ..\study01 --designs Gyroid_central --resolutions 8 --timeout 120
# 中断后，使用完全相同选项并追加 --resume
.\.venv\Scripts\python.exe benchmarks\comsol_convergence.py --out ..\study01 --designs Gyroid_central --resolutions 8 --timeout 120 --resume
```

成功工况跳过，失败工况在新目录重试。CSV 包含实际插值方式、吸附参数、配置哈希及 COMSOL attempt。弹性吸附变体标为 `snapped`，流动吸附求根变体标为 `snapped_root`。
批处理也接受 `--preferences-directory`；恢复时检查其路径和初始文件哈希，避免静默换用不同配置。

P1 扩散脚本同样逐工况保存，支持 `--resume`。可只更正旧 CSV 的外推结果：

```powershell
.\.venv\Scripts\python.exe benchmarks\transport_convergence.py --out ..\corrected_summary --summarize-csv '旧结果\transport_convergence.csv'
```

外推使用不等比三网格幂律拟合；无正阶数、非单调等情况返回空估计及原因。拟合结果不证明真实极限一致。自定义场增加两处非对齐探针，发现漏界面则拒绝生成；有限探针不能保证任意高频公式的几何精度，仍需分辨率和 m_cal 加密检查。
