"""Measure how often the trained policy survives each push strength (no viewer).
Run:  python eval.py            (uses g1_push.zip + vecnorm.pkl)
      python eval.py 2          (uses g1_push_stage2.zip + vecnorm_stage2.pkl)"""
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from g1_env import G1Push

suffix = f"_stage{sys.argv[1]}" if len(sys.argv) > 1 else ""
EPISODES = 20

env = VecNormalize.load(f"vecnorm{suffix}.pkl", DummyVecEnv([G1Push]))
env.training = False
env.norm_reward = False
model = PPO.load(f"g1_push{suffix}", device="cpu")
raw = env.envs[0]

print("push force   survived")
for force in (0, 60, 100, 150, 200, 250):
    raw.fixed_force = float(force)
    obs = env.reset()
    survived = done_count = 0
    while done_count < EPISODES:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = env.step(action)
        if dones[0]:
            done_count += 1
            survived += bool(infos[0].get("TimeLimit.truncated", False))
    print(f"{force:6d} N     {survived}/{EPISODES}")
