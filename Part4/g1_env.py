"""Unitree G1 push-recovery environment (MuJoCo, CPU).

The robot stands, gets shoved in the back, and must stay on its feet.
Only the 12 leg joints are controlled; waist and arms are held at the default pose.
"""
import os

import gymnasium as gym
import mujoco
import numpy as np

XML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "mujoco_menagerie", "unitree_g1", "scene.xml")

N_LEG = 12            # actuators 0-11 are the two legs (6 each)
ACTION_SCALE = 0.25   # rad of joint-target offset per unit action
TARGET_HEIGHT = 0.78  # pelvis height when standing with slightly bent knees
EPISODE_STEPS = 250   # 5 s at 50 Hz


class G1Push(gym.Env):
    def __init__(self, push_scale=0.3):
        self.m = mujoco.MjModel.from_xml_path(XML)
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = 0.002
        self.decim = 10                                   # 500 Hz sim -> 50 Hz control
        self.pelvis = self.m.body("pelvis").id
        self.torso = self.m.body("torso_link").id
        self.key = self.m.key("stand").id
        self.push_scale = push_scale
        self.fixed_force = None

        # Default pose: the "stand" keyframe, with knees slightly bent.
        self.ctrl0 = self.m.key_ctrl[self.key].copy()
        for side in (0, 6):                               # left leg, right leg
            self.ctrl0[side + 0] = -0.1                   # hip pitch
            self.ctrl0[side + 3] = 0.3                    # knee
            self.ctrl0[side + 4] = -0.2                   # ankle pitch
        self.q0 = self.ctrl0[:N_LEG].copy()

        self.action_space = gym.spaces.Box(-1, 1, (N_LEG,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (6 + 3 * N_LEG,), np.float32)
        self.prev = np.zeros(N_LEG)
        self.t = 0

    def set_push_scale(self, scale):
        self.push_scale = scale

    def _obs(self):
        R = self.d.xmat[self.pelvis].reshape(3, 3)
        grav = R.T @ np.array([0.0, 0.0, -1.0])           # gravity direction in pelvis frame
        return np.concatenate([
            grav,
            self.d.qvel[3:6],                             # pelvis angular velocity
            self.d.qpos[7:7 + N_LEG] - self.q0,
            0.05 * self.d.qvel[6:6 + N_LEG],
            self.prev,
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.key)
        self.d.qpos[7:7 + N_LEG] = self.q0
        self.d.ctrl[:] = self.ctrl0
        self.d.xfrc_applied[:] = 0
        mujoco.mj_forward(self.m, self.d)
        self.t = 0
        self.prev = np.zeros(N_LEG)
        self.push_t = int(self.np_random.integers(50, 100))          # 1-2 s in
        # 100-200 N for 0.16 s is 16-32 N.s: enough to need a step, still recoverable.
        force = self.push_scale * self.np_random.uniform(100, 200)   # newtons
        if self.fixed_force is not None:                             # used by eval.py
            force = self.fixed_force
        yaw = self.np_random.uniform(-0.35, 0.35)                    # +/- 20 deg
        self.push_vec = force * np.array([np.cos(yaw), np.sin(yaw), 0.0])
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        self.d.ctrl[:] = self.ctrl0
        self.d.ctrl[:N_LEG] = self.q0 + ACTION_SCALE * a
        pushing = self.push_t <= self.t < self.push_t + 8            # 0.16 s shove
        self.d.xfrc_applied[self.torso, :3] = self.push_vec if pushing else 0.0
        for _ in range(self.decim):
            mujoco.mj_step(self.m, self.d)
        self.t += 1

        obs = self._obs()
        height = self.d.qpos[2]
        up = -obs[2]                                      # 1 when perfectly upright
        fell = bool(height < 0.5 or up < 0.5)
        reward = (
            1.0
            + up
            + np.exp(-20 * (height - TARGET_HEIGHT) ** 2)
            - 0.5 * np.sum(self.d.qvel[:2] ** 2)          # come to rest
            - 0.1 * np.sum((self.d.qpos[7:7 + N_LEG] - self.q0) ** 2)   # return to stance
            - 0.01 * np.sum((a - self.prev) ** 2)         # smooth actions
            - 50.0 * fell
        )
        self.prev = a.copy()
        return obs, float(reward), fell, self.t >= EPISODE_STEPS, {}
