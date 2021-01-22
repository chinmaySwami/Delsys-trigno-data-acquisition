import mujoco_py
import os
import random
import sklearn
import pickle

# Load the trained regression models
regression_ps = pickle.load(open("D:/Chinmay/ML Pipeline/Trained model/mode_0_20201006-104041", "rb"))
regression_rud = pickle.load(open("D:/Chinmay/ML Pipeline/Trained model/mode_1_20201006-083222", "rb"))


mj_path, _ = mujoco_py.utils.discover_mujoco()
xml_path = os.path.join(mj_path, 'External Models/MPL/MPL/', 'arm_claw_ADL.xml')
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)
rend = mujoco_py.MjViewer(sim)
while True:
    rend.render()
    sim.step()
    fe = random.uniform(-100.0, 100.0)
    rud = random.uniform(-100.0, 100.0)
    sim.data.ctrl[5:7] = [rud, fe]
    print(sim.data.ctrl)
