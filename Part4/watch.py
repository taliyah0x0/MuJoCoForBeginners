"""Watch the trained policy.   Run:  mjpython watch.py"""
import time

import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from g1_env import G1Push

env = VecNormalize.load("vecnorm.pkl", DummyVecEnv([lambda: G1Push(push_scale=1.0)]))
env.training = False
env.norm_reward = False
model = PPO.load("g1_push", device="cpu")

raw = env.envs[0]
obs = env.reset()
with mujoco.viewer.launch_passive(raw.m, raw.d) as v:
    while v.is_running():
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(action)       # resets automatically at episode end
        v.sync()
        time.sleep(0.02)
