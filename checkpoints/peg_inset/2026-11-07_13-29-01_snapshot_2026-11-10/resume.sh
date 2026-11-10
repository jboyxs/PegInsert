#!/usr/bin/env bash
set -euo pipefail

snapshot_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_dir="$(cd -- "$snapshot_dir/../../.." && pwd)"
additional_iterations="${1:-2400}"
parallel_envs="${2:-4096}"
if [[ ! "$additional_iterations" =~ ^[1-9][0-9]*$ || ! "$parallel_envs" =~ ^[1-9][0-9]*$ ]]; then
  echo 'Usage: bash resume.sh [additional_iterations=2400] [num_envs=4096]' >&2
  exit 2
fi

cd -- "$snapshot_dir"
sha256sum --check SHA256SUMS
cd -- "$repository_dir"
exec uv run --locked isaaclab train \
  --task PegInset \
  --rl_library rsl_rl \
  --num_envs "$parallel_envs" \
  --max_iterations "$additional_iterations" \
  --checkpoint "$snapshot_dir/model_12600.pt" \
  --resume_optimizer_state \
  --run_name resumed_from_12600 \
  --seed 42 \
  --viz none \
  physics=newton_mjwarp
