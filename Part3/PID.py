import time
import numpy as np
import mujoco
import mujoco.viewer

# ---------------------------------------------------------------- settings
MODEL_PATH = "mujoco_menagerie/unitree_g1/scene.xml"

# Flip these to -1.0 if the correction pushes the wrong way (see step 4 of the guide).
PITCH_SIGN = -1.0
ROLL_SIGN = -1.0

# PID gains (outputs are ankle angle offsets in radians)
PITCH_GAINS = dict(kp=5.5, ki=0.1, kd=0.4)
ROLL_GAINS = dict(kp=5.5, ki=0.1, kd=0.4)

ENABLE_PID = False       # set False to watch the robot tip with no controller
PUSH_TIME = 2.0        # seconds; brief shove to test recovery (None to disable)
PUSH_FORCE = 30.0       # newtons, forward (+x)
PUSH_DURATION = 0.1     # seconds


# --------------------------------------------------------------------- PID
class PID:
    def __init__(self, kp, ki, kd, limit=0.5, i_limit=0.3):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.limit, self.i_limit = limit, i_limit
        self.integral = 0.0

    def __call__(self, error, error_rate, dt):
        self.integral += error * dt
        self.integral = max(-self.i_limit, min(self.i_limit, self.integral))  # anti-windup
        out = self.kp * error + self.ki * self.integral + self.kd * error_rate
        return max(-self.limit, min(self.limit, out))                         # clamp (rad)


# ------------------------------------------------------------------- setup
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)   # standing pose + standing targets
mujoco.mj_forward(model, data)
home = data.ctrl.copy()


def actuator_id(name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if i < 0:
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, k) for k in range(model.nu)]
        raise ValueError(f"Actuator '{name}' not found. Available: {names}")
    return i


def body_id(name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if i < 0:
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, k) for k in range(model.nbody)]
        raise ValueError(f"Body '{name}' not found. Available: {names}")
    return i


ankle_pitch = [actuator_id("left_ankle_pitch_joint"), actuator_id("right_ankle_pitch_joint")]
ankle_roll = [actuator_id("left_ankle_roll_joint"), actuator_id("right_ankle_roll_joint")]
pelvis = body_id("pelvis")

pitch_pid = PID(**PITCH_GAINS)
roll_pid = ROLL_GAINS and PID(**ROLL_GAINS)


# --------------------------------------------------------------- measuring
def tilt():
    """Small-angle pelvis pitch/roll and their rates (target is 0 for all)."""
    R = data.xmat[pelvis].reshape(3, 3)
    pitch = -R[2, 0]           # forward/back lean (about y)
    roll = R[2, 1]             # sideways lean (about x)
    roll_rate = data.qvel[3]   # angular velocity from the free joint
    pitch_rate = data.qvel[4]
    return pitch, pitch_rate, roll, roll_rate


# -------------------------------------------------------------------- loop
dt = model.opt.timestep
last_print = 0.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        pitch, pitch_rate, roll, roll_rate = tilt()

        data.ctrl[:] = home
        if ENABLE_PID:
            u_pitch = pitch_pid(0.0 - pitch, 0.0 - pitch_rate, dt)
            u_roll = roll_pid(0.0 - roll, 0.0 - roll_rate, dt)
            for a in ankle_pitch:
                data.ctrl[a] = home[a] + PITCH_SIGN * u_pitch
            for a in ankle_roll:
                data.ctrl[a] = home[a] + ROLL_SIGN * u_roll

        # Brief push to test recovery
        if PUSH_TIME is not None and PUSH_TIME < data.time < PUSH_TIME + PUSH_DURATION:
            data.xfrc_applied[pelvis, 0] = PUSH_FORCE
        else:
            data.xfrc_applied[pelvis, 0] = 0.0

        mujoco.mj_step(model, data)
        viewer.sync()

        # Print tilt once per second so you can see what the controller sees
        if data.time - last_print >= 1.0:
            last_print = data.time
            print(f"t={data.time:5.1f}s  pitch={np.degrees(pitch):6.2f} deg  roll={np.degrees(roll):6.2f} deg")

        time.sleep(dt)