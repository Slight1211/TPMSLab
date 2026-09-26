# Publishing TPMSLab 0.3.0 to PyPI

This workflow uploads the exact wheel and sdist already published on GitHub.
It does not rebuild or move the reviewed v0.3.0 tag. Both SHA-256 values and the
tag's source commit are checked before publication.

## One-time account setup

Sign in to https://pypi.org/ and complete its account/email/2FA requirements.
At https://pypi.org/manage/account/publishing/ add a pending GitHub publisher:

| Field | Value |
| --- | --- |
| PyPI project name | tpmslab |
| Owner | Slight1211 |
| Repository | TPMSLab |
| Workflow filename | publish-pypi.yml |
| Environment | pypi |

No password or API token needs to be sent to a collaborator or stored in this repository.
A pending publisher does not reserve a name; PyPI determines availability at first upload.

## Publish

Open GitHub Actions, choose **Publish v0.3.0 to PyPI**, then **Run workflow** on
**main**. Enable the **publish** checkbox after the pending publisher is configured.
A normal push only runs verification; it never publishes to PyPI.
The environment name must match the PyPI configuration exactly.

After success, verify https://pypi.org/project/tpmslab/0.3.0/ and install from PyPI
in a fresh environment:

```sh
python -m pip install tpmslab==0.3.0
tpmslab --help
```

For the optional browser interface use `python -m pip install "tpmslab[web]==0.3.0"`.
COMSOL is optional external software and requires a separate installation/licence.
Update the software availability statement only after PyPI confirms publication.
Do not overwrite the v0.3.0 tag or alter these verified distributions.

## 中文操作说明

在 PyPI 注册或登录后，进入账户 Publishing 页面，按上表新增 GitHub 待发布授权。
完成后运行上述 GitHub 工作流，并勾选 publish。普通推送只检查，不会上传。
发布成功后才可以使用 PyPI 的 pip 安装命令；配置授权本身不等于已经发布。
