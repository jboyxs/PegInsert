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

保留上游的动作、观测历史、奖励、成功阈值、物理参数、重置课程与 `play_mode()` 行为；环境实现不导入 `isaaclab_tasks.contrib.nist`，只复用 Isaac Lab 的通用接口和任务配置工具。移植代码保留原 BSD-3-Clause 版权声明，许可见 `src/peginsert/LICENSE-ISAACLAB`。

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
