# SoftwareX LaTeX 初稿（含真实几何和 COMSOL 结果）

入口：manuscript.tex；编译结果：manuscript.pdf；参考文献：references.bib。

## 编译

使用包含 elsarticle 的 TeX Live / MiKTeX，或上传源码 ZIP 至 Overleaf，主文件选择 manuscript.tex，编译器选择 pdfLaTeX。

    pdflatex manuscript.tex
    bibtex manuscript
    pdflatex manuscript.tex
    pdflatex manuscript.tex

本机实际使用 Tectonic 0.17.0 成功编译；以上 pdfLaTeX / Overleaf 流程为标准使用方法，未在这些平台另行执行。ZIP 内也提供已生成的 manuscript.bbl。所有正文引用图片与主文件放在同一层。

## 模板依据

使用 Elsevier SoftwareX Original Software Publication LaTeX 模板的公开归档副本，保留 elsarticle 的 preprint、12pt、A4 设置、八项代码元数据和五个必需正文部分。原模板版权为 Elsevier 2026，使用 LPPL 1.2 或后续版本；未修改的模板存于本地 evidence 目录。

官方下载地址：https://legacyfileshare.elsevier.com/promis_misc/softwarex-osp-template.tex
归档来源：https://github.com/yongyin-leon/CubeScope/tree/main/docs/softwarex-template
作者指南：https://www.sciencedirect.com/journal/softwarex/publish/guide-for-authors

此次访问作者指南与官方下载返回 403，因此不声称已核验官网当前版的全部要求。归档 README 记载模板下载于 2026-04-24；归档模板本身要求五个部分、至多六幅图，正文 4000 词，另有正文页数说明。现稿采用四幅组合图，正文按较保守的 3000 词以内目标编写。PDF 是 12pt 预印版，总页数包含封面、元数据、图表及参考文献，不能直接当作正文页数。

## 四幅图

1. 软件架构与可选 COMSOL 流程（矢量 PDF）。
2. Gyroid 实际体网格的固液界面截面与局部放大。
3. Gyroid、Primitive、Diamond 的固体和互补流体几何；蓝、绿是两个不连通孔道。
4. 从已求解 MPH 提取的孔隙流速和独立固体算例的位移结果，并标明边界条件、单位。图不是虚构 COMSOL 界面截图；结果由导出的真实有限元场绘制。原生 COMSOL 图片另存于 evidence。

注意：文件名 fig2_geometry 与 fig3_interface 是生成时的名称，文章中的图号由 LaTeX 按引用顺序自动生成，依次为界面图 2、几何图 3。

## 证据与复现

evidence/provenance.json 记录原始网格和 MPH 的 SHA256；ExportPaper.java 从保存的解提取数值，不修改原 MPH、不重求解。COMSOL Eval 的流动可视化采样网格经过细化，因此采样网格单元数不等于正文的求解网格单元数。flow_solution.npz / elastic_solution.npz 保存带明确单位的结果数组。

复现绘图脚本保留在 evidence 中，按原工程目录运行，需要 NumPy、VTK、Matplotlib、Pillow。脚本仍引用原工程 work/paper-figures 和 outputs 路径；这部分是本地溯源材料，不是独立 Python 发布包。LaTeX 源码 ZIP 可独立编译，不需要 COMSOL 和 Python。

## 仍须作者填写

作者、单位、通讯邮箱、真实贡献、基金及利益冲突；公开 GitHub 仓库、明确版本/不可变归档与文档链接；人工审阅后的 AI 声明。现版本是 0.3.0rc1，尚未发布到 PyPI。归档模板要求公开 GitHub、README.md 与 Licence.txt，正式发布时须核对仓库许可证文件命名。

本文只确认 COMSOL 网格导入、独立固体弹性计算和固定壁面流动计算。未宣称光滑 STEP CAD、几何布尔运算、流固耦合或网格收敛已验证。
