## PegInset（已实现）

已将本地 NIST 源码中的 `PegInsert4MM`（4 mm 方销插入）抽取到 `src/peginsert/`，Gym 任务名为 **`PegInset`**。环境类为 `PegInsetEnvCfg`，PPO 配置类为 `PegInsetPPORunnerCfg`。默认直接使用 4 mm 方销/方孔，无需再指定 `presets=peg_insert_4mm`。

```text
src/peginsert/
├── __init__.py                         # 注册任务，提供 peginsert 命令
├── tasks/pegact/
│   ├── assembly_keypoints.py           # NIST 板、4 mm 方销/孔、Franka 关键点
│   ├── assembly_profile{,_cfg}.py      # 装配路径采样和配置
│   ├── config/franka/
│   │   ├── __init__.py                 # Gym 注册 PegInset
│   │   ├── env_cfg.py                  # 动作、观测、奖励、终止、课程、物理配置
│   │   ├── scenes_cfg.py / assets_cfg.py
│   │   ├── task_cfg.py                 # 从 4 mm 变体提取的具体装配参数
│   │   └── agents/rsl_rl_ppo_cfg.py
│   ├── mdp/                           # 本地观测、奖励、终止、事件和课程实现
│   └── reset/env_cfg.py                # 三种重置策略、碰撞筛选、初态库
└── utils/                             # 几何、SDF、碰撞检查、重置组合器、采样器
```

运行（项目安装后通过 `isaaclab.tasks` 入口自动发现任务）：

项目使用 Python 3.12，已声明 `isaaclab[isaacsim]==3.0.0` 并在 `[tool.uv]` 中保存官方依赖覆盖配置；`uv sync --locked` 会一并安装 Isaac Sim 6.2.0.0 及扩展缓存。

```powershell
uv sync --locked
uv run isaaclab list_envs --keyword PegInset
uv run isaaclab train --task PegInset --rl_library rsl_rl
uv run isaaclab play --task PegInset --rl_library rsl_rl --checkpoint <checkpoint.pt>
```

也可用 `uv run peginsert train ...` 调用同一 CLI。默认物理后端仍为上游的 `isaacsim_physx`，需要安装并配置 Isaac Sim；支持以 `physics=physx` 或 `physics=newton_mjwarp` 选择其他后端，运行时也需要对应依赖。USD 资产仍来自 `ISAACLAB_NUCLEUS_DIR/Factory`。

训练 checkpoint 的独立成功率评估：

```powershell
uv run python scripts/evaluate.py --checkpoint <checkpoint.pt> --headless --episodes 1000 --num_envs 64 --seed 12345
# 小规模验证，保存到指定空目录
uv run python scripts/evaluate.py --checkpoint <checkpoint.pt> --headless --episodes 8 --num_envs 4 --state_table_size 16 --output logs/evaluation/smoke
```

脚本评估当前项目的 `PegInset` 环境，固定正常重力 `(0, 0, -9.81)`，关闭重力课程和难度调度，仅从碰撞筛选后的 `grasp_asset_in_air` 初态库均匀有放回采样。保留原空中抓取位姿扰动、IK 设置及成功阈值，不使用脚本专家收紧后的抓取分布。默认后端是 `newton_mjwarp`，应与训练一致；PhysX 训练结果可设置 `--physics isaacsim_physx` 或 `--physics physx`。省略 `--headless` 可显示窗口。

默认读取 checkpoint 同目录下的 `params/agent.yaml`，也可用 `--agent_cfg <agent.yaml>` 指定网络配置；找不到时使用项目默认网络配置并提示。严格加载 Actor 权重及其观测归一化统计，使用推理模式和均值动作，不添加训练时的探索噪声。默认评估 seed 为 12345，应与训练 seed 不同；每次启动独立生成初态库，`--state_table_size` 默认 512，可扩大初态覆盖或更换 seed 重复评估。初态库仍是有限集合，不能将重复采样次数当作独立初态数量。

输出目录默认 `logs/evaluation/<timestamp>`，必须为空。`episodes.jsonl` 保存每个完整回合的初态编号、回报、长度、终止原因及成功标记；`summary.json` 保存 `successful_episodes / completed_episodes`、失败数、纯超时数、平均回报/长度及运行条件，另保存 `metadata.json`、`env.yaml`、`agent.yaml`。成功必须触发原 `success` 条件且没有同时触发异常终止；同时成功和超时仍计成功。失败、超时作为互斥结果，`termination_counts` 则逐项记录，可重叠。为避免快速结束回合占比过高，预先给各并行环境分配固定回合配额，完成指定总数后停止。Ctrl+C、关闭仿真或 `--max_steps` 提前停止时，仅统计完整回合并将 `complete` 标为 `false`；未完成回合不进入成功率。

添加 `--success_gifs 3` 可将前 3 个成功回合导出为真实仿真画面的 GIF（仅支持 Newton 后端，`--num_envs` 不超过 4）。每帧包含机械臂全景和孔口近景，按 25 FPS 播放，成功终帧停留额外 0.8 秒。相机在场景构建前注册，终帧在自动重置前采集，避免下一回合画面混入。文件保存在输出目录的 `gifs/` 下；`summary.json` 的 `success_gifs` 记录对应回合和文件路径，`gifs_complete` 表示是否录满。评估仍运行指定的完整回合数，失败画面不导出；若成功次数不足，保留已录制的成功 GIF。

```powershell
uv run python scripts/evaluate.py --checkpoint <checkpoint.pt> --headless --episodes 24 --num_envs 4 --state_table_size 128 --seed 12345 --success_gifs 3
```

默认奖励包含成功奖励、动作幅度、动作变化、关节力矩和异常终止惩罚，以及稠密奖励 `assembly_progress`。新增项的实际单步贡献为 `2 × (γΦ(s下一步) − Φ(s当前))`，返回值已补偿奖励管理器的步长缩放。势函数为 `Φ = 0.30A + 0.25B + 0.45D`，其中 `A` 为接近孔口分数（距离尺度 50 mm），`B` 为横向与倾角对正分数（尺度 5 mm 和 0.1 rad），`D` 为对正分数乘以插入比例（孔口至装配位置 9 mm）。实现见 `mdp/rewards.py`，参数见 `PegInsetRewardsCfg.assembly_progress`。

成功或失败终止时，下一状态的势值取零以消除终点残余，因此终止步的塑形项可能为负。无限时域模式下的超时截断保留势值，用于价值自举。每次 reset 从新初态的实时几何初始化历史势值。PPO 与塑形默认共用 `task_cfg.DISCOUNT_FACTOR = 0.995`；运行时覆盖 PPO 的 `agent.algorithm.gamma` 时，应同步设置 `env.rewards.assembly_progress.params.gamma`。设 `env.rewards.assembly_progress.weight=0` 可关闭新增项。

仅发生超时时，环境在重置前将动作后的观测接入策略的五帧历史，PPO 以该历史计算 `V(sₜ₊₁)`，并在训练目标中加入一次 `γV(sₜ₊₁)`。若同一步也触发成功或失败终止，或配置为有限时域，训练目标只使用该步奖励。

动作、观测历史、成功阈值、物理参数、重置课程与 `play_mode()` 来自上游；环境实现不导入 `isaaclab_tasks.contrib.nist`，只复用 Isaac Lab 的通用接口和任务配置工具。移植代码保留原 BSD-3-Clause 版权声明，许可见 `src/peginsert/LICENSE-ISAACLAB`。

无需启动仿真的验证：

```powershell
uv run python -m unittest discover -s tests -v
```

以下保留原有源码阅读笔记。

### 从零实现现有的PegInsert任务

#### 1. 初步文件组织结构

```text
src/isaaclab_tutorial/tasks/pegact/
├── __init__.py                      # 任务族说明
├── config/franka/
│   ├── __init__.py                  # gym.register × 3（PegInsert/GearMesh/NutThread）
│   ├── env_cfg.py                   # ManagerBasedRLEnvCfg + 各 Cfg + play_mode
│   ├── scene_cfg.py / assets_cfg.py
│   └── agents/{__init__.py, rsl_rl_ppo_cfg.py}
├── mdp/{__init__.py, __init__.pyi, actions.py, observations.py, events.py,
│        rewards.py, terminations.py, geometry.py, task.py, control.py}
├── reset/{dataset.py, generator.py}  # 可选：预生成合法抓握初始状态
└── assets/                           # Factory USD（或明确用 ISAACLAB_NUCLEUS_DIR）
```

#### 2. 参照 nist 现成任务

[nist 源码目录](.venv/Lib/site-packages/isaaclab_tasks/contrib/nist/)

以下说明以当前虚拟环境中的源码为准。`nist` 是基于 Franka Panda 和 NIST 装配板的 Factory 装配任务族，使用 `ManagerBasedRLEnv`。它在 `config/__init__.py` 中只注册一个 Gym 环境 `IsaacContrib-Factory-Franka`，通过 `presets` 切换以下 **12 种任务变体**，默认是 `nut_thread_m16`。

| 任务 | preset 名称 | 目标 |
| --- | --- | --- |
| 螺母旋拧（1 种） | `nut_thread_m16` | 将 M16 螺母旋入固定螺栓 |
| 齿轮装配（3 种） | `gear_mesh_small`、`gear_mesh_medium`、`gear_mesh_large` | 将指定齿轮装到基座轴上，与场景中的其余齿轮啮合 |
| 圆杆插入（4 种） | `rod_insert_4mm`、`rod_insert_8mm`、`rod_insert_12mm`、`rod_insert_16mm` | 将对应尺寸的圆杆插入圆孔 |
| 矩形销插入（4 种） | `peg_insert_4mm`、`peg_insert_8mm`、`peg_insert_12mm`、`peg_insert_16mm` | 将对应尺寸的矩形销插入矩形孔，是本项目的主要参考 |

例如选择 `--task IsaacContrib-Factory-Franka presets=peg_insert_4mm`，即可使用 4 mm 矩形销插入配置；物理后端由 `physics` 选择，支持 `isaacsim_physx`（默认）、`physx` 和 `newton_mjwarp`。

文件结构与作用如下（同类文件合并展示）：

```text
nist/
├── __init__.py                       # 包说明
├── config/
│   ├── __init__.py                   # 注册 Gym 环境，绑定环境配置与训练配置
│   └── agents/
│       ├── __init__.py               # 训练配置包入口
│       └── rsl_rl_ppo_cfg.py         # RSL-RL PPO 网络、学习率、迭代次数等
├── factory_env_cfg.py                # 总配置：动作、观测、奖励、终止、事件、课程和物理后端
├── factory_scenes_cfg.py             # 公共场景及 12 种 fixed_asset / held_asset 组合
├── factory_assets_cfg.py             # 机器人、桌面、装配板、零件的 USD 与物理/接触参数
├── factory_presets.py                # 按任务选择装配路径、对齐点、抓取点、障碍物等
├── assembly_keypoints.py             # 零件和机器人上的几何参考点、位姿偏移和抓取尺寸
├── assembly_profile_cfg.py           # 装配路径分段、旋转及位姿噪声的配置
├── assembly_profile.py               # 路径插值与位姿采样实现，为重置和成功判定提供几何信息
├── reset_env_cfg.py                  # 组合重置策略，配置碰撞筛选与初始状态库
├── mdp/
│   ├── __init__.py / __init__.pyi     # 延迟导出 MDP 接口及类型声明，复用通用 MDP 项
│   ├── observations.py              # 不同参考系下的相对位姿与连杆速度
│   ├── events.py                    # 零件/装配板重置、IK 放置末端、抓取与装配初态生成
│   ├── rewards.py                   # 动作幅值/变化惩罚、装配成功奖励
│   ├── terminations.py              # 越界、装配进度上下文及成功终止
│   └── curriculums.py               # 根据成功率调整难度，插值修改课程参数
└── utils/
    ├── __init__.py / __init__.pyi     # 工具接口导出及类型声明
    ├── event_combinators.py          # 串联重置、按成功率选择策略、收集并采样有效初态
    ├── reset_state.py               # 读取/恢复机器人与物体状态
    ├── sampling/                    # sampler 及配置：均匀采样、基于成功率的 Beta 权重采样
    ├── collision_analyzer.py         # 基于表面点云和 SDF 检查碰撞/穿透
    ├── collision_analyzer_cfg.py     # 碰撞检查对象、采样点数和距离阈值
    ├── mesh_ops.py / sdf.py          # 网格转换、表面点采样与有符号距离计算
    ├── rigid_object_hasher.py        # 读取碰撞几何哈希与相对变换，用于几何复用
    ├── pose_offset.py               # 位姿偏移及坐标变换工具
    └── grid_downsample.py           # 网格分桶降采样与特征提取工具
```

任务运行时，策略根据相对位姿、末端速度、关节位置和上一动作等观测，输出机械臂关节位置增量与夹爪开合指令。奖励以装配成功为主，并惩罚动作、动作变化和关节力矩；成功、超时、越界或关节速度异常会结束回合。
训练重置会生成并筛选抓握/装配初态，存入状态库后按成功率采样，课程逐步将重力从零恢复至正常值；`play_mode()` 则使用正常重力和统一的空中抓取重置策略进行评估。

实现本项目时，可按 `config/__init__.py → factory_env_cfg.py → factory_scenes_cfg.py / factory_presets.py → mdp/` 的顺序阅读，再参考 `reset_env_cfg.py` 和 `utils/` 实现合法抓握初态。动作直接复用 Isaac Lab 的通用关节动作配置；USD 资产来自 `ISAACLAB_NUCLEUS_DIR/Factory`，不在 `nist` 目录内。
### 当前思路
top->down思路从factory_env_cfg.py 的FactoryEnvCfg出发，分模块去构建/抄袭

已实现基于真实销、孔位姿的脚本专家和 BC 示范采集，见 `src/peginsert/tasks/pegact/expert.py`、`demonstrations.py`、`expert_cfg.py` 和 `scripts/run_expert.py`。固定从 `grasp_asset_in_air` 重置策略生成的碰撞筛选初态开始，使用正常重力；保留销和孔的原随机位姿，不混入已装配初态。专家采集单独将夹爪朝向设为相对销的朝向，并将抓取平移扰动收紧至横向 ±0.5 mm、轴向 ±3 mm，倾角扰动收紧至 ±0.02 rad，保证起点接近真实抓持状态。另筛除抓取位置 IK 残差过大或夹爪开度大于 6 mm（单侧）的样本，避免闭爪前掉落；这使示范起点成为原空中初态的抓取就绪子分布。新增重置参数 `orientation_in_asset_frame` 默认关闭，不改变 PPO 的默认重置行为。Actor 学“这一步怎么动”，Critic 学“从这里按当前策略继续，预计还能拿多少奖励”。两者需要不同的训练标签。BC 网络训练和 Critic 预训练尚未实现。

```powershell
uv run python scripts/run_expert.py --num_envs 16 --episodes 100
```

采集默认使用 `newton_mjwarp`（Newton + MuJoCo Warp 求解器），窗口为 `newton_gl`，其中相机画面使用 `NewtonWarpRendererCfg`；通过 `launch_simulation` 启动并检查没有 Kit 依赖。批量采集可加 `--headless` 关闭窗口及图像渲染。专家只在重置时复用已有初态生成逻辑；轨迹中通过 `env.step(action)` 控制，不直接写关节位置、目标或物体位姿。

```powershell
# 无窗口采集：100 个完整尝试，只保存其中成功的回合
uv run python scripts/run_expert.py --headless --num_envs 16 --episodes 100 --output logs/expert/bc_run_01
# 小规模排查：保留失败轨迹，限制向量步数
uv run python scripts/run_expert.py --headless --num_envs 4 --episodes 8 --state_table_size 16 --max_steps 1200 --keep_failures
```

专家先闭爪稳定，再抬高、横移对齐、接近孔口、小步插入；方销朝向从四个等价方向中选择。DLS IK 使用末端世界坐标雅可比，并根据实际销与末端相对变换补偿偏心抓取。通过关节 PD 刚度将销的重力载荷前馈换算成关节增量，配合小幅速度阻尼和有界积分补偿，降低静差；这些补偿均包含在送给 Actor 同一动作接口的标签中。插入只在精细对齐后下压，停滞时回退并重新对齐，超过重试次数或明显滑脱则停止运动等待环境结束。成功由原任务终止条件判定，未放宽阈值。脚本能否在目标初态分布上提供足够成功示范，需要用采集统计评估。

动作标签与 PPO Actor 输出接口一致：`[panda_joint1…7 的原始增量指令, 夹爪指令]`，按运行时 action term 的实际关节顺序保存。前六轴的目标增量是 `action × 0.02 rad`，第七轴是 `action × 0.2 rad`，夹爪 `-1` 为闭合。专家将 IK 增量除以对应缩放后提交给同一 action manager；没有把增量换成绝对关节角或再次按 decimation 缩放。当前 Actor 是没有硬限幅的高斯输出；专家额外将标签限制在 `[-2, 2]`，同时将每个关节目标增量限制为 `0.05 rad`，不修改训练环境动作接口。

输出目录必须为空。`metadata.json` 记录观测布局、关节顺序、动作缩放、专家参数和运行配置，`env.yaml` 保存具体环境配置；`episodes.jsonl` 记录每次成功、失败或超时的统计；`summary.json` 汇总成功率。完整回合按固定 seed 的哈希分入 `train/` 与 `validation/`（默认验证比例 10%，小数据集可能没有验证回合），避免同一回合的相邻帧分散到两边。每条 `.pt` 包含：

- `policy_observation[T, D]`：动作执行前、未经归一化的原生五帧 `policy` 观测，直接作为 Actor 的输入；保持环境原有 history 排列。
- `action[T, 8]`：实际送入 `env.step` 的原始动作，作为 BC 标签。
- `reward`、`terminated`、`truncated`、`success`、`failure`：环境总奖励与终止标记；同时成功和异常终止不作为成功示范。
- `stage`、`geometry`、`expert_failure`：控制阶段、位置/朝向/滑脱误差与专家失败原因，供排查使用。

中断时丢弃尚未完成的回合；默认丢弃失败回合的张量但保留统计。若使用 `--keep_failures`，BC 时应筛选 `episode_success=True`。文件可用 `torch.load(path, weights_only=True)` 读取；用训练集估计 Actor 的观测归一化统计，回归动作均值。示范不包含重置后的观测充当上一回合的终态，也不提供 Critic 的回报或自举标签。初态库是有限集合，严格检验未见初态时还应另用不同 seed 和独立采集目录生成验证数据。

本机 RTX 3060 的小规模实测（seed=42、4 个环境、16 个初态、8 次完整尝试）成功 3 次，得到 850 对成功轨迹观测/动作；这是运行验证，不能作为泛化成功率。已逐步核对保存的五帧观测中最新 `prev_action` 与上一条动作标签完全相等，所有回合仅在末步出现终止标记。Newton GL + Warp 相机实测输出 640×480 RGB，未加载 `omni.kit.app`；20 项无仿真单元测试通过。当前版本已能采集成功示范，但还不是高成功率专家。

#### 1. 模仿学习：在仿真里生成示范

- **谁来示范：** 从已夹住方销的重置状态开始，用仿真中的销、孔位姿写一个脚本专家：先移到孔口上方，对齐中心和方销朝向，再沿孔轴小步插入；若下压后深度不再增加，就回退并重新对齐。用逆运动学把目标位姿转成关节增量，按环境的动作缩放换算后送入 `env.step`，夹爪保持闭合。现有初态库只提供起点，动作标签来自专家实际执行的轨迹。
- **怎么采、怎么用：** 随机改变初始偏移和姿态，每步保存动作前的五帧 `policy` 观测、送入环境的专家动作、环境总奖励，以及成功、失败或超时标记。按完整回合划分训练集和验证集。先用成功示范做行为克隆（BC），让 Actor 的动作均值拟合专家动作；再让 Actor 独立跑，看新初态下能否插入。若经常在孔边卡住，就在它自己走到的卡住状态补录专家纠偏动作，再训练一轮（[DAgger 思路](https://proceedings.mlr.press/v15/ross11a.html)）。

#### 2. Critic：先用实际回报预训练，再交给 PPO

- **最小做法：** 固定克隆好的 Actor，在仿真里用它采一批回合，成功和失败都保留。用环境实际返回的总奖励从后往前算标签：`G_t = r_t + 0.995 × G_{t+1}`；成功或失败终止时，末步之后的回报取零。第一版可跳过纯超时回合；若超时占多数，则在有初步价值估计后，用重置前最后五帧的价值自举，与现有 PPO 一致。用均方误差让 Critic 从五帧观测预测 `G_t`，再带着两个网络及其观测归一化统计启动在线 PPO。
- **如何用先验：** 现有总奖励已经通过势函数 `Φ` 编入了接近孔口、对正和插入深度，回报标签自然包含这些知识，不必再加一次。若想单独检验几何先验，可在预训练时加一个辅助输出，预测实际插入深度；价值输出仍拟合回报。这样即使方销很接近孔口却已经卡住，Critic 也能从后续失败中学到低回报。

**怎么判断值不值得做：** 在相同 PPO 环境步数下，对比「从零 PPO」「BC + PPO」「BC + Critic 预训练 + PPO」。用相同的未见初态和正常重力评估插入成功率，同时单列采集示范和预训练回合消耗的仿真步数，才能判断是否真的省数据。
