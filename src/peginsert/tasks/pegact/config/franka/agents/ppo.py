"""PPO with a pre-reset successor value for time-limit truncations."""

import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict


class PegInsetPPO(PPO):
    @torch.no_grad()
    def process_env_step(self, obs, rewards, dones, extras):
        if "time_outs" in extras:
            # Consume the timeout here so upstream PPO does not bootstrap again
            # from transition.values, which must keep V(s_t) for GAE.
            extras = extras.copy()
            del extras["time_outs"]
            final = extras["timeout_bootstrap"]
            if final is not None:
                ids, history = final
                terminal_obs = TensorDict({"policy": history.to(self.device)}, batch_size=[len(ids)])
                next_values = self.critic(terminal_obs).squeeze(-1)
                rewards = rewards.clone()
                rewards[ids.to(self.device)] += self.gamma * next_values
        super().process_env_step(obs, rewards, dones, extras)
