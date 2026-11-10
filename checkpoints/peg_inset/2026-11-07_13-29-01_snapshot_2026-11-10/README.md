# Newton 4096 环境训练快照与跨机器续训

此目录冻结自 `logs/rsl_rl/peg_inset/2026-11-07_13-29-01`，打包日期为
2026-11-10（Asia/Shanghai）。源机器的训练保持运行；后续新检查点不会自动同步到此快照。

最新完整检查点为 **`model_12600.pt`**。此目录同时保留截至打包时的全部
64 个检查点（0、200、……、12600），便于选择较早模型继续实验。

## 另一台机器的操作

需要已安装 Git、uv，以及满足当前 CUDA/PyTorch 和 Isaac Lab 要求的 NVIDIA 驱动。
Python 3.12 及项目依赖由 `uv sync --locked` 按仓库中的 `uv.lock` 安装。
首次启动需要能下载 NVIDIA Factory USD 资产。

```bash
git clone --single-branch --branch codex/checkpoint-newton-20261110-12600 \
  https://github.com/jboyxs/PegInsert.git
cd PegInsert
uv sync --locked

# 只在 CPU 上验证模型、归一化、优化器和迭代编号可以恢复，不启动仿真。
uv run --locked python \
  checkpoints/peg_inset/2026-11-07_13-29-01_snapshot_2026-11-10/verify_resume.py

# 从 model_12600.pt 继续，追加 2400 次迭代，4096 个环境。
bash checkpoints/peg_inset/2026-11-07_13-29-01_snapshot_2026-11-10/resume.sh
```

这里的 RSL-RL 将 `--max_iterations` 解释为**加载检查点后追加的迭代数**。
默认命令从编号 12600 运行到编号 14999；加载后沿用保存的编号作为起点，
所以第一个续训编号仍为 12600。不是重新从随机策略训练，也不是再追加 15000 次。

例如，追加 5000 次迭代、改用 2048 个环境：

```bash
bash checkpoints/peg_inset/2026-11-07_13-29-01_snapshot_2026-11-10/resume.sh 5000 2048
```

改变环境数会改变 PPO 每次迭代的采样量和 minibatch 大小。
建议首先使用原来的 4096 环境、种子 42、Newton + MuJoCo Warp 配置。
本机已完成该规模的容量短测；目标机器仍需满足资源和依赖要求。

## 会恢复什么

检查点保存 Actor、Critic、二者的观测归一化统计、Adam 优化器动量与学习率、
训练迭代编号。续训命令显式保留优化器状态，并新建日志目录
`logs/rsl_rl/peg_inset/<时间>_resumed_from_12600/`。

**现有检查点未保存仿真环境状态、初态库、逐状态成功率历史、采样器状态、
课程状态、随机数生成器状态或累计日志计数。** 另一台机器会重新生成初态库，
重新统计成功率并从默认课程状态启动；这是模型和优化器续训，不能保证逐步重现源进程。
这些运行时数据没有对应的已保存目录，复制 TensorBoard 日志也不能恢复它们。
本次未向运行中的训练进程注入代码或尝试提取其内存状态。

## 文件与依赖

- `model_*.pt`：全部已完成检查点，最后一个为 `model_12600.pt`。
- `params/env.yaml`、`params/agent.yaml`：源训练保存的实际配置，供核对。
  其中包含源机器初始化后的绝对路径，续训时通过项目任务代码重新构造环境，
  不直接将该 YAML 当作新机器的环境配置加载。
- `manifest.json`：模型 SHA-256、源代码提交、原始命令和实际安装版本。
- `SHA256SUMS`：整个快照的文件校验清单，续训脚本会先校验。
- `events.out.tfevents.*`：打包时刻的 TensorBoard 快照，已裁去不完整尾记录。
- `diagnostics/`：训练曲线和已有诊断，供参考；它们不是恢复环境状态的输入。
- `git/PegInsert.diff`、`run.json`：源训练代码版本和运行信息。
- `verify_resume.py`、`resume.sh`：CPU 恢复验证及实际续训入口。

项目源码 `src/`、`pyproject.toml`、`uv.lock`、`.python-version` 位于仓库根目录，
与此快照一起通过该分支获取。源训练的任务代码与提交
`09ce3aedf9ec297d6f8d0da566833e159117f2dc` 一致。
无需复制 `.venv/`、CUDA 编译缓存或整份 `/tmp` 资产缓存；这些依赖和缓存会在目标机器创建。

原始配置保留稀疏奖励、自适应初态采样和重力课程。此快照提供可续训产物，
不表示模型已经在正常重力下通过独立评测。
