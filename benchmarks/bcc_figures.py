"""Figures for bcc_crosscompare.py output (PNG, light surface, Noto Sans CJK)."""
import csv
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MESHERS = [  # fixed categorical order: slot 1..4
    ("tpms_bcc", "TPMS Lab · BCC 格子", "#2a78d6", "o"),
    ("tpms_kuhn", "TPMS Lab · Kuhn 格子 (dev7)", "#eb6834", "s"),
    ("gmsh_fixed", "Gmsh · 同一边界内部重剖", "#1baf7a", "^"),
    ("gmsh_surface", "Gmsh · 曲面+体全部重剖", "#eda100", "D"),
]
DESIGNS = [("Gyroid_central", "Gyroid 0.25→0.50"), ("Diamond_low", "Diamond 0.15→0.30"),
           ("Primitive_Schwartz_central", "Primitive 0.25→0.50"), ("IWP_sheet", "IWP 0.30→0.45")]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": ["Noto Sans CJK JP", "DejaVu Sans"], "font.size": 10,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
})


def load(path):
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8-sig")) if not r["error"]]
    for r in rows:
        for k in ("level", "tetrahedra"):
            r[k] = int(float(r[k]))
        for k in ("total_seconds", "q_min", "q_p01", "q_median", "volume_error_pp", "k_eff_over_bulk"):
            r[k] = float(r[k])
    return rows


def pick(rows, design, mesher):
    return sorted((r for r in rows if r["design"] == design and r["mesher"] == mesher), key=lambda r: r["level"])


def fig_tradeoff(rows, out):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.4), sharey="row")
    for col, (design, title) in enumerate(DESIGNS):
        for row, (key, ylabel) in enumerate((("q_p01", "固相 p01 质量（越高越好）"),
                                             ("total_seconds", "端到端用时 s（越低越好）"))):
            ax = axes[row, col]
            for mesher, label, color, marker in MESHERS:
                series = pick(rows, design, mesher)
                if not series:
                    continue
                x = [r["tetrahedra"] / 1000 for r in series]
                y = [r[key] for r in series]
                ax.plot(x, y, color=color, lw=2, marker=marker, ms=8, mec=SURFACE, mew=1.5, label=label, zorder=3)
                if mesher == "tpms_bcc":
                    ax.annotate("BCC", (x[-1], y[-1]), xytext=(5, 4 if row == 0 else -12), textcoords="offset points",
                                fontsize=9, color=INK, fontweight="bold")
            if row == 0:
                ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold")
            ax.set_xlabel("固相四面体数（千）")
            if col == 0:
                ax.set_ylabel(ylabel)
    axes[1, 0].set_ylim(bottom=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.945))
    fig.suptitle("BCC 格子：同等单元数下 p01 质量多数最高，用时普遍短于 Kuhn", x=0.02, y=0.995, ha="left", fontsize=15,
                 fontweight="bold", color=INK)
    fig.text(0.02, 0.008, "4 档网格密度：Kuhn r10/12/16/20 ↔ BCC r8/10/13/17（单元数相当）。TPMS Lab 两行均含 optimize_mesh；"
             "Gmsh 用时含提供边界的 BCC 生成。单线程 Linux 容器，Gmsh 4.15.2（Delaunay + Netgen 优化）。\n"
             "Gmsh 曲面+体全部重剖在 16 个工况中有 9 个报错（STL 曲面重参数化失败），图中缺点即失败。",
             fontsize=8.5, color=INK2, va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_bars(rows, out, level=2):
    metrics = [("total_seconds", "端到端用时 (s)，越低越好", "{:.1f}"),
               ("q_p01", "固相 p01 质量，越高越好", "{:.2f}"),
               ("q_min", "固相最小质量，越高越好", "{:.2f}"),
               ("q_median", "固相中位质量，越高越好", "{:.2f}")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6))
    width = 0.19
    x = np.arange(len(DESIGNS))
    for ax, (key, title, fmt) in zip(axes, metrics):
        for i, (mesher, label, color, _) in enumerate(MESHERS):
            values = []
            for design, _ in DESIGNS:
                r = [r for r in rows if r["design"] == design and r["mesher"] == mesher and r["level"] == level]
                values.append(r[0][key] if r else np.nan)
            pos = x + (i - 1.5) * (width + 0.01)
            ax.bar(pos, values, width, color=color, label=label, zorder=3, edgecolor=SURFACE, linewidth=0.8)
            for p, v in zip(pos, values):
                if not np.isfinite(v):
                    ax.annotate("×\n失败", (p, 0), xytext=(0, 2), textcoords="offset points", ha="center",
                                va="bottom", fontsize=7, color=INK2, linespacing=0.9)
            if mesher == "tpms_bcc":
                for p, v in zip(pos, values):
                    if np.isfinite(v):
                        ax.annotate(fmt.format(v), (p, v), xytext=(0, 3), textcoords="offset points",
                                    ha="center", fontsize=8, color=INK, fontweight="bold")
        ax.set_xticks(x, [t.split(" ")[0] for _, t in DESIGNS])
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
        ax.grid(axis="x", visible=False)
        if key != "total_seconds":
            ax.set_ylim(0, 1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.925))
    kuhn_res, bcc_res = [(10, 8), (12, 10), (16, 13), (20, 17)][level]
    fig.suptitle(f"同一密度档横向对比（Kuhn r{kuhn_res} ↔ BCC r{bcc_res}，单元数相当）", x=0.02, y=0.99,
                 ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_distribution(npz, out, title):
    data = np.load(npz)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    bins = np.linspace(0, 1, 51)
    for mesher, label, color, _ in MESHERS:
        if f"{mesher}_quality" not in data:
            continue
        q = data[f"{mesher}_quality"]
        hist, edges = np.histogram(q, bins=bins)
        ax.step(edges[1:], hist / len(q) * 100, where="pre", color=color, lw=2, label=f"{label}（{len(q):,} 个）")
        qs = np.sort(q)
        frac = np.arange(1, len(qs) + 1) / len(qs) * 100
        ax2.plot(qs, frac, color=color, lw=2)
    ax.axvline(0.952, color=INK2, lw=1, ls=":")
    ax.annotate("BCC 背景四面体 0.952", (0.952, ax.get_ylim()[1] * 0.92), xytext=(-6, 0),
                textcoords="offset points", ha="right", fontsize=8.5, color=INK2)
    ax.set_xlabel("单元质量（mean ratio，1 = 正四面体）")
    ax.set_ylabel("单元占比 (%)")
    ax.set_title("质量分布", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    low = min(float(data[f"{m}_quality"].min()) for m, _, _, _ in MESHERS if f"{m}_quality" in data)
    ax2.set_xlim(max(0.0, low - 0.03), 0.62)
    ax2.set_ylim(0, 10)
    ax2.set_xlabel("单元质量")
    ax2.set_ylabel("累计占比 (%)")
    ax2.set_title("最差 10% 的尾部（越靠右越好）", loc="left", fontweight="bold")
    fig.suptitle(title, x=0.02, y=0.99, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _slice(points, tetra, quality, axis, value):
    """Polygons where the plane x_axis = value cuts the tetrahedra (2-D, other axes)."""
    keep = [a for a in range(3) if a != axis]
    p = points[tetra]
    d = p[:, :, axis] - value
    hit = (d.min(axis=1) < 0) & (d.max(axis=1) > 0)
    polys, q = [], []
    for cell, dist, qq in zip(p[hit], d[hit], quality[hit]):
        pts = []
        for i in range(4):
            for j in range(i + 1, 4):
                if dist[i] * dist[j] < 0:
                    t = dist[i] / (dist[i] - dist[j])
                    pts.append(cell[i] + t * (cell[j] - cell[i]))
        pts = np.array(pts)[:, keep]
        c = pts.mean(axis=0)
        order = np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
        polys.append(pts[order])
        q.append(qq)
    return polys, np.array(q)


def fig_slice(npz, out, title, value=2.37):
    data = np.load(npz)
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    panels = [(m, label) for m, label, _, _ in MESHERS if f"{m}_points" in data][:3]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 5.0))
    for ax, (mesher, label) in zip(np.atleast_1d(axes), panels):
        polys, q = _slice(data[f"{mesher}_points"], data[f"{mesher}_tetra"], data[f"{mesher}_quality"], 2, value)
        collection = PolyCollection(polys, array=q, cmap=cmap, clim=(0, 1), edgecolors=SURFACE, linewidths=0.35)
        ax.add_collection(collection)
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_title(label, loc="left", fontsize=11, fontweight="bold")
        ax.text(0, -0.25, f"本截面最小 {q.min():.2f} · 中位 {np.median(q):.2f}", fontsize=9, color=INK2,
                transform=ax.transData)
    bar = fig.colorbar(collection, ax=np.atleast_1d(axes).tolist(), shrink=0.75, pad=0.02)
    bar.set_label("单元质量（深 = 好，1 = 正四面体）", color=INK2)
    bar.outline.set_visible(False)
    fig.suptitle(title, x=0.02, y=0.93, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


ALL_TOOLS = [  # (mesher, label); BCC highlighted, the rest share one neutral ink
    ("tpms_bcc_env", "BCC 高质量 + 包络 8.7 µm"),
    ("tpms_bcc_high", "TPMS Lab · BCC 高质量"),
    ("tpms_bcc", "TPMS Lab · BCC 标准"),
    ("tpms_kuhn", "TPMS Lab · Kuhn（dev7）"),
    ("ftetwild", "fTetWild"),
    ("gmsh_surface", "Gmsh 曲面+体重剖"),
    ("gmsh_fixed", "Gmsh 同边界内部"),
    ("tetgen", "TetGen 同边界"),
    ("microgen", "microgen（体素裁剪）"),
]
NEUTRAL = "#a3a29c"
OURS = {"tpms_bcc": "#2a78d6", "tpms_bcc_high": "#184f95", "tpms_bcc_env": "#0d366b"}


def load_errors(*paths):
    errs = set()
    for path in paths:
        if Path(path).is_file():
            for r in csv.DictReader(open(path, encoding="utf-8-sig")):
                if r["error"]:
                    errs.add((r["design"], int(float(r["level"])), r["mesher"]))
    return errs


def fig_all_tools(rows, out, level=2, errors=frozenset()):
    metrics = [("q_p01", "固相 p01 质量（越高越好）", "{:.2f}"), ("q_min", "固相最小质量（越高越好）", "{:.2f}"),
               ("total_seconds", "端到端用时 s（越低越好）", "{:.1f}")]
    fig, axes = plt.subplots(len(metrics), 4, figsize=(16, 10.5), sharey=True)
    y = np.arange(len(ALL_TOOLS))[::-1]
    for col, (design, title) in enumerate(DESIGNS):
        for row, (key, xlabel, fmt) in enumerate(metrics):
            ax = axes[row, col]
            for yi, (mesher, label) in zip(y, ALL_TOOLS):
                hit = [r for r in rows if r["design"] == design and r["mesher"] == mesher and r["level"] == level]
                color = OURS.get(mesher, NEUTRAL)
                if not hit:
                    ax.text(0.01, yi, "失败（报错）" if (design, level, mesher) in errors else "未运行", transform=ax.get_yaxis_transform(), va="center",
                            fontsize=8.5, color=INK2)
                    continue
                v = hit[0][key]
                ax.barh(yi, v, height=0.62, color=color, zorder=3)
                ax.annotate(fmt.format(v), (v, yi), xytext=(3, 0), textcoords="offset points", va="center",
                            fontsize=8.5, color=INK if mesher in OURS else INK2,
                            fontweight="bold" if mesher in OURS else "normal")
            ax.set_yticks(y, [label for _, label in ALL_TOOLS])
            ax.grid(axis="y", visible=False)
            if key != "total_seconds":
                ax.set_xlim(0, 0.72)
            else:
                ax.set_xlim(0, max(r["total_seconds"] for r in rows if r["design"] == design and r["level"] == level) * 1.25)
            if row == 0:
                ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold")
            ax.set_xlabel(xlabel)
    kuhn_res, bcc_res = LEVELS_FIG[level]
    fig.suptitle(f"与其他网格工具横向对比（Kuhn r{kuhn_res} ↔ BCC r{bcc_res} 密度档）", x=0.01, y=0.995,
                 ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.01, 0.005, "TetGen、fTetWild、Gmsh 同边界：输入为 TPMS Lab BCC 生成的同一固相边界，用时含该生成；目标边长 h 相同。"
             "fTetWild 单线程、包络 1e-3；microgen 1.2.1 为其原生体输出（结构化六面体网格裁剪后三角化），\n"
             "只给固相、无优化。只有 TPMS Lab 同时输出共形的固/流两相网格。单线程 Linux 容器。",
             fontsize=8.5, color=INK2, va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out, dpi=200)
    plt.close(fig)


LEVELS_FIG = [(10, 8), (12, 10), (16, 13), (20, 17)]


def fig_overview(rows, out, errors=frozenset()):
    """All 16 design x density cases per tool: p01 and end-to-end seconds (dots = cases, bar = median)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), sharey=True, gridspec_kw={"width_ratios": [1, 1, 1]})
    y = np.arange(len(ALL_TOOLS))[::-1]
    rng = np.random.default_rng(0)
    panels = [("q_p01", "固相 p01 质量（越高越好）", False), ("q_min", "固相最小质量（越高越好）", False),
              ("total_seconds", "端到端用时 s（对数轴，越低越好）", True)]
    for ax, (key, xlabel, logx) in zip(axes, panels):
        for yi, (mesher, label) in zip(y, ALL_TOOLS):
            values = np.array([r[key] for r in rows if r["mesher"] == mesher])
            failed = sum(1 for e in errors if e[2] == mesher)
            color = OURS.get(mesher, NEUTRAL)
            if len(values):
                jitter = rng.uniform(-0.18, 0.18, len(values))
                ax.scatter(values, yi + jitter, s=26, color=color, alpha=0.85, edgecolors=SURFACE, linewidths=0.6,
                           zorder=3)
                med = float(np.median(values))
                ax.plot([med, med], [yi - 0.32, yi + 0.32], color=INK if mesher in OURS else INK2, lw=2.2,
                        zorder=4)
                ax.annotate(("{:.2f}" if not logx else "{:.1f}").format(med), (med, yi + 0.32), xytext=(0, 2),
                            textcoords="offset points", ha="center", fontsize=8.5,
                            color=INK if mesher in OURS else INK2,
                            fontweight="bold" if mesher in OURS else "normal")
            if failed and key == "q_p01":
                ax.annotate(f"报错 {failed}/16", (0.0, yi), xytext=(4, -11), textcoords="offset points",
                            fontsize=8, color=INK2, xycoords=("axes fraction", "data"))
        ax.set_yticks(y, [label for _, label in ALL_TOOLS])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(xlabel)
        if logx:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        else:
            ax.set_xlim(0, 0.7)
    fig.suptitle("16 个工况（4 种 TPMS × 4 档密度）总览：点 = 单个工况，竖线 = 中位数", x=0.01, y=0.99, ha="left",
                 fontsize=15, fontweight="bold", color=INK)
    fig.text(0.01, 0.01, "TetGen、fTetWild、Gmsh 同边界的输入是 TPMS Lab BCC 生成的同一固相边界（用时已含）；fTetWild 单线程、包络 1e-3；"
             "microgen 1.2.1 为原生体素裁剪输出；“包络”= 界面节点可偏离精确曲面 ≤ 8.7 µm（与 fTetWild 默认包络相同）。只有 TPMS Lab 同时给出共形固/流两相网格。",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(out, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    rows = load(src / "crosscompare.csv")
    others = src / "other_tools.csv"
    if others.is_file():
        all_rows = rows + load(others)
        high = src / "bcc_high.csv"
        if high.is_file():
            extra = load(high)
            for r in extra:
                r["mesher"] = "tpms_bcc_high"
            all_rows += extra
        env = src / "bcc_envelope.csv"
        if env.is_file():
            extra = load(env)
            for r in extra:
                r["mesher"] = "tpms_bcc_env"
            all_rows += extra
        for level in range(4):
            fig_all_tools(all_rows, dst / f"5_all_tools_L{level}.png", level=level,
                          errors=load_errors(src / "crosscompare.csv", others))
        fig_overview(all_rows, dst / "0_overview_all_tools.png", errors=load_errors(src / "crosscompare.csv", others))
    fig_tradeoff(rows, dst / "1_quality_vs_time.png")
    fig_bars(rows, dst / "2_side_by_side.png", level=int(sys.argv[3]) if len(sys.argv) > 3 else 2)
    for npz in sorted(src.glob("cells_*.npz")):
        name = npz.stem.replace("cells_", "")
        pretty = dict(DESIGNS).get(name.split("_L")[0], name)
        fig_distribution(npz, dst / f"3_quality_distribution_{name}.png", f"单元质量分布：{pretty}（同一密度档，固相）")
        fig_slice(npz, dst / f"4_slice_{name}.png", f"z = 2.37 mm 截面，按单元质量着色：{pretty}")
