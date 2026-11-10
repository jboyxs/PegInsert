"""Register the standalone PegInset environment without loading the simulator."""

import gymnasium as gym

gym.register(
    id="PegInset",
    entry_point="peginsert.tasks.pegact.env:PegInsetEnv",
    disable_env_checker=True,
    kwargs={
        "default_agent": "rsl_rl",
        "env_cfg_entry_point": f"{__name__}.env_cfg:PegInsetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PegInsetPPORunnerCfg",
    },
)
