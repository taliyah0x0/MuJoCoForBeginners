"""Sanity check, no training needed.   Run:  mjpython check.py
The robot should stand still, get pushed after 1-2 s, and fall. Then it resets."""
import time

import mujoco.viewer
import numpy as np

from g1_env import G1Push

env = G1Push(push_scale=1.0)
env.reset()
with mujoco.viewer.launch_passive(env.m, env.d) as v:
    while v.is_running():
        _, _, fell, timeout, _ = env.step(np.zeros(12, dtype=np.float32))
        if fell or timeout:
            env.reset()
        v.sync()
        time.sleep(0.02)
