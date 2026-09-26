# Python 调用：梯度 TPMS 固体 / 流体双域

需要 Python 3.11 或以上。当前源码版本为 0.3.4，PyPI 已发布基线为 0.3.0；在本仓库根目录安装：

```sh
python -m pip install .
python examples/solid_fluid_api.py
```

示例默认创建新的 `results/时间戳` 文件夹，生成 5 mm 立方体的梯度 Gyroid（目标密度沿 z 从 0.25 增至 0.50），包括固体及两个互补孔隙域。默认只生成网格，不要求 COMSOL。

```python
from datetime import datetime
from pathlib import Path
from tpmslab import Config, generate, save_model

config = Config(
    family="Gyroid", domain_mode="solid_fluid",
    size=(5.0, 5.0, 5.0), cells=(1, 1, 1),
    density_start=0.25, density_end=0.50,
    gradient="linear", axis="z", resolution=12,
)
model = generate(config)
folder = save_model(model, Path("results") / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
print(folder / "mesh.nas")
```

坐标和尺寸单位为 mm。输出包括 mesh.nas（体网格及边界）、mesh.npz、固体/流体 STL 预览、参数、域编号和诊断报告。现有目录不会被覆盖。

本机安装有兼容的 COMSOL 及所需许可时，可接着调用：

```python
from tpmslab.comsol import build_mph
build_mph(folder, model["report"], solve=True)
```

或者使用完整示例：

```sh
python examples/solid_fluid_api.py --comsol create
python examples/solid_fluid_api.py --comsol solve
```

`create` 创建并重开验证未求解 MPH；`solve` 创建并求解固定壁面孔隙蠕动流，输出 `model_solved.mph`。此演示不是变形流固耦合。自动查找失败时，将 `COMSOL_BIN` 设为包含 comsolbatch.exe 和 comsolcompile.exe 的目录。

可选网页界面：

```sh
python -m pip install ".[web]"
tpmslab web --output tpmslab-output
```

可修改 family 为 Primitive_Schwartz、Diamond 等；可用 `tpmslab families` 查看内置名称。提高 resolution 会增加体单元和计算成本。COMSOL 验证范围和复现方法见 docs/SOLID_FLUID.md；算法说明见 docs/METHOD.md。


支持的分辨率范围：每个胞元边长方向 `resolution >= 8`。

分辨率及各轴胞元数不设固定上限；分辨率为至少 8 的整数，各轴胞元数为正整数。实际可处理规模取决于计算资源。

### 密度标定采样

`Config(m_cal=80)` 设置每轴密度标定中点采样数，总数为 80³。默认 56，要求整数且至少为 8，不设固定上限；与网格 `resolution` 独立。JSON 同样使用 `m_cal` 字段，旧配置省略时使用 56。导出配置记录该值。改变它可能改变标定阈值及生成几何；既有验证使用 56。
