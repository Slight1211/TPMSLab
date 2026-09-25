# Python 调用：梯度 TPMS 固体 / 流体双域

需要 Python 3.11 或以上。软件当前为 0.3.0rc1，尚未发布到 PyPI；在本仓库根目录安装：

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

可修改 family 为 Primitive_Schwartz、Diamond 等；可用 `tpmslab families` 查看内置名称。提高 resolution 会增加体单元和计算成本。论文示例及 COMSOL 实算范围见 paper/manuscript.pdf。
