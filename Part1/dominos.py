import time
import mujoco
import mujoco.viewer

N = 12          # number of dominoes
SPACING = 0.09  # distance between centers (m)

bodies = ""
for i in range(N):
    bodies += f"""
    <body name="d{i}" pos="{i * SPACING} 0 0.08">
      <freejoint/>
      <geom type="box" size="0.01 0.04 0.08" mass="0.05"
            rgba="0.9 0.3 0.2 1" friction="0.6 0.005 0.0001"/>
    </body>"""

xml = f"""
<mujoco>
  <option timestep="0.001"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"
          friction="0.8 0.005 0.0001"/>
    {bodies}
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Give the first domino a push (angular velocity about y)
data.qvel[4] = 20

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)