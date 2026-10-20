import math, time
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/unitree_g1/scene.xml")
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)

home_ctrl = data.ctrl.copy()          # the standing targets from the keyframe

def act(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

left_hip = act("left_hip_pitch_joint")
right_hip = act("right_hip_pitch_joint")
left_knee = act("left_knee_joint")
right_knee = act("right_knee_joint")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = data.time
        data.ctrl[:] = home_ctrl                       # start from standing targets
        data.ctrl[left_hip] = home_ctrl[left_hip] + 0.5 * math.sin(8 * math.pi * 0.5 * t)   # swing left leg
        data.ctrl[right_hip] = home_ctrl[right_hip] + 0.5 * math.sin(8 * math.pi * 0.5 * t + math.pi)  # swing right leg
        data.ctrl[left_knee] = home_ctrl[left_knee] + 0.5 * math.sin(8 * math.pi * 0.5 * t + math.pi) # swing left knee
        data.ctrl[right_knee] = home_ctrl[right_knee] + 0.5 * math.sin(8 * math.pi * 0.5 * t)  # swing right knee
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)

        