"""Train PPO on the G1 push-recovery task.   Run:  caffeinate -i python train.py"""
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from g1_env import G1Push

N_ENVS = 8                                    # roughly your number of performance cores
STAGES = [(0.5, 8_000_000),                   # (push strength, steps): 50-100 N
          (0.75, 8_000_000),                  # 75-150 N
          (1.0, 14_000_000)]                  # 100-200 N


def make_env():
    return Monitor(G1Push())


if __name__ == "__main__":
    env = VecNormalize(SubprocVecEnv([make_env] * N_ENVS))
    model = PPO("MlpPolicy", env, n_steps=512, batch_size=1024, gamma=0.99,
                learning_rate=3e-4, policy_kwargs=dict(net_arch=[256, 256]),
                device="cpu", verbose=1)

    for i, (scale, steps) in enumerate(STAGES):
        print(f"\n=== Stage {i + 1}: push_scale = {scale} ===")
        env.env_method("set_push_scale", scale)
        model.learn(steps, reset_num_timesteps=(i == 0))
        for name in ("g1_push", f"g1_push_stage{i + 1}"):     # latest + a copy per stage
            model.save(name)
            env.save(name.replace("g1_push", "vecnorm") + ".pkl")
