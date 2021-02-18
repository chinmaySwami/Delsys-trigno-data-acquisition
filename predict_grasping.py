import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import sklearn
import pandas
import numpy as np
import pickle
import pytrigno
from scipy.signal import butter, lfilter_zi, lfilter
import time
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mujoco_py
from tensorflow import keras
import random
from collections import deque
from collections import Counter


def create_connection_imu(host):
    dev = pytrigno.TrignoAccel(channel_range=(0, 62), samples_per_read=1,
                               host=host)
    dev.check_sensor_n_type(sensor_number)
    dev.check_sensor_n_mode(sensor_number)
    dev.check_sensor_n_start_index(sensor_number)
    dev.check_sensor_n_auxchannel_count(sensor_number)
    dev.check_sensor_channel_unit(sensor_number, 1)
    dev.check_sensor_channel_unit(sensor_number, 4)
    dev.check_sensor_channel_unit(sensor_number, 7)
    return dev


def create_connection_EMG(host):
    dev = pytrigno.TrignoEMG(channel_range=(0, 15), samples_per_read=1,
                               host=host)
    dev.check_sensor_n_type(sensor_number)
    dev.check_sensor_n_mode(sensor_number)
    dev.check_sensor_n_start_index(sensor_number)
    dev.check_sensor_n_auxchannel_count(sensor_number)
    dev.check_sensor_channel_unit(sensor_number, 1)
    dev.check_sensor_channel_unit(sensor_number, 4)
    dev.check_sensor_channel_unit(sensor_number, 7)
    return dev


def create_butterworth_filter():
    nyq = 0.5 * sampling_frequency
    low = lowCutoff / nyq
    b = butter(filterOrderIMU, low, btype='low', output='ba')
    z = lfilter_zi(b[0], b[1])
    return b, z


def normalize_data(data):
    n_data = np.divide((data - avg_mean_training), avg_std_training)
    return n_data


def acquire_imu(zi):
    dev.start()
    print("Connection established::")
    while True:
        data = dev.read()
        # if data[12][0] != 0.0:
        #     za = 2
        data_s = np.multiply(data, scaling_array)
        data = np.concatenate((data_s[9:15], data_s[27:33], data_s[36:42], data_s[45:51],
                               data_s[54:60]), axis=0)
        filtered = np.zeros(30)
        for i in range(data.shape[0]):
            filtered[i], zi[i] = lfilter(b[0], b[1], data[i], zi=zi[i])
        imu_data.append(filtered)


def acquire_EMG(zi_emg):
    dev_EMG.start()
    print("Connection established::")
    while True:
        data = dev_EMG.read()
        filtered = np.zeros(16)
        for i in range(data.shape[0]):
            aew, asw = lfilter(b[0], b[1], data[i], zi=zi_emg[i])
        for i in range(data.shape[0]):
            filtered[i], zi_emg[i] = lfilter(b[0], b[1], data[i], zi=zi_emg[i])
        EMG_data.append(np.absolute(filtered))


# def iterative_gain(lst, prev_theta, prev_aug_theta, gain):
#     aug_lst = []
#     initiated_gain = False
#     for index in range(len(lst)):
#         if abs(prev_theta) <= abs(lst[index]):
#             # aug_val = df.at[index, 'Beta Dot (Degrees/sec)'] + (df.at[index, 'Beta Dot (Degrees/sec)'] * 0.15)
#             aug_val = lst[index] * gain
#             if abs(aug_val) <= 200.0:  # for PS 200, DTM
#                 aug_lst.append(aug_val)
#                 if gain <= 2:
#                     gain += 0.00008  # for PS :: 0.00008, DTM: 0.00025
#             else:
#                 aug_lst.append(aug_val - (aug_val - prev_aug_theta))
#         else:
#             gain -= 0.00008
#             if gain <= 1:
#                 gain = 1
#             aug_val = lst[index] * gain
#             # aug_lst.append(lst[index])
#             aug_lst.append(aug_val)
# 
#         prev_aug_theta = aug_lst[index]
#         prev_theta = lst[index]
#     hc_aug = lst * hc_val
# 
#     return aug_lst, hc_aug


def predict_theta_dot():
    while True:
        if imu_data:
            data_read = imu_data[-1]
            normalized_data = normalize_data(data_read)
            # 8 for 3 class classifier
            normalized_data = normalized_data / 1.5
            start = time.time()
            movement_class_prob = classifier.predict(normalized_data.reshape(1, 30))
            movement_class = np.argmax(movement_class_prob, axis=1)

            if movement_class == 1:
                if use_rf_rud:
                    predicted_theta_dot = rud_model.predict([normalized_data])
                else:
                    predicted_theta_dot = [random.choice([-4, 4])]
            elif movement_class == 0:
                if use_rf_ps:
                    predicted_theta_dot = ps_model.predict([normalized_data])
                else:
                    predicted_theta_dot = [random.choice([-5, 5])]
            else:
                predicted_theta_dot = [0]
            print("time to predict:\t", time.time() - start, "\t", normalized_data[3], "\t", movement_class_prob, "\t",
                  movement_class[0], "\t", predicted_theta_dot[0])
            theta_dot.append([movement_class[0], predicted_theta_dot[0]])


def predict_grasping():
    while True:
        if EMG_data:
            data = EMG_data[-1]
            data = np.multiply(data, scaling_array_emg)
            data = np.concatenate((np.array([data[1]]), data[3:5]), axis=0)
            data = data[2]
            print("Sum of Data: ", data)
            if abs(data) > 5:
                grasp_pred.append(1)
            else:
                grasp_pred.append(0)


def animate(i):
    if theta_dot:
        xss = theta_dot[-500:]
        ax.clear()
        # ay.clear()
        ax.plot(xss)
        # ay.plot(yss)


sampling_frequency = 148
filterOrderIMU = 2
lowCutoff = 1
avg_mean_training = np.array([-7557.074342, 238.9051397, -7004.522484, 1100.075402, 8357.125645, -3072.610687,
                              -9787.098475, 716.5740751, -2461.864722, 1468.178587, -412.4093905, -1145.956741,
                              -3408.054336, 992.7556896, 624.4395445, 23619.40505, 8885.855499, 1394.194849,
                              -4494.225019, -454.6410738, -1114.837898, 3616.154164, 7812.728116, 879.4170656,
                              4581.595772, -1112.281237, -1521.265704, 498.9367503, 8031.761407, 3475.425272])
avg_std_training = np.array([3899.100149, 514.3201803, 4127.585205, 23852.0515, 74641.24657, 30605.56797, 3006.187073,
                             644.0836957, 4982.276768, 19552.98379, 79514.72385, 32777.61901, 1990.689673, 830.3656838,
                             2630.36271, 35482.23785, 79879.04171, 16961.90462, 1422.30238, 412.355329, 3258.372477,
                             16624.68747, 35696.7495, 19250.84621, 1370.720671, 374.9487406, 3091.811698, 18509.75631,
                             19317.85871, 21869.72258])
scaling_array = np.array([9806.65, 9806.65, 9806.65, 1000, 1000, 1000, 1000, 1000, 1000,
                          9806.65, 9806.65, 9806.65, 1000, 1000, 1000, 1000, 1000, 1000,
                          9806.65, 9806.65, 9806.65, 1000, 1000, 1000, 1000, 1000, 1000,
                          9806.65, 9806.65, 9806.65, 1000, 1000, 1000, 1000, 1000, 1000,
                          9806.65, 9806.65, 9806.65, 1000, 1000, 1000, 1000, 1000, 1000,
                          9806.65, 9806.65, 9806.65, 1000, 1000, 1000, 1000, 1000, 1000,
                          9806.65, 9806.65, 9806.65, 1000, 1000, 1000, 1000, 1000, 1000])
scaling_array = scaling_array.reshape(-1, 1)
scaling_array_emg = np.array([100000.0, 100000.0, 100000.0, 100000.0, 100000.0, 100000.0, 100000.0, 100000.0, 100000.0,
                          100000.0, 100000.0, 100000.0, 100000.0, 100000.0, 100000.0, 100000.0])
# scaling_array_emg = scaling_array_emg.reshape(-1, 1)

imu_data = []
EMG_data = []
theta_dot = []
grasp_pred = []
grasp_status = False
sensor_number = 2
use_rf_ps = False
use_rf_rud = False
nn_path = 'D:/Chinmay/Wrist Control Offline/saved classifier/2_class_PSDTM'

classification_queue = deque()

b, z = create_butterworth_filter()
zi = {}
zi_emg = {}

# index number 31 in zi is for PS and 32 is for RUD
for i in range(33):
    zi[i] = z

for i in range(16):
    zi_emg[i] = z

# dev = create_connection_imu('localhost')
dev_EMG = create_connection_EMG('localhost')

print("Loading NN model: ")
classifier = keras.models.load_model(nn_path)
print("Model loaded successfully::")

if use_rf_rud:
    print("Loading regression model::")
    # For RUD
    rud_model = pickle.load(open("D:/Chinmay/ML Pipeline/Trained model/mode_1_20201006-083222", "rb"))
    rud_model.verbose = False
    rud_model.n_jobs = 1
    print("RUD Model loaded successfully::")

if use_rf_ps:
    # For PS
    start = time.time()
    ps_model = pickle.load(open("D:/Chinmay/ML Pipeline/Trained model/mode_0_20201006-104041", "rb"))
    ps_model.verbose = False
    ps_model.n_jobs = 1
    print("PS Model loaded successfully:: ", time.time()-start)


fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
# ay = fig.add_subplot(2, 1, 2)

try:
    # acquire_data_thread = threading.Thread(target=acquire_imu, args=(zi,))
    acquire_EMG_data_thread = threading.Thread(target=acquire_EMG, args=(zi_emg,))
    predict_grasp_thread = threading.Thread(target=predict_grasping)
    # predict_theta_thread = threading.Thread(target=predict_theta_dot)

    # acquire_data_thread.daemon = True
    acquire_EMG_data_thread.daemon = True
    predict_grasp_thread.daemon = True
    # predict_theta_thread.daemon = True

    # acquire_data_thread.start()
    acquire_EMG_data_thread.start()
    predict_grasp_thread.start()
    # predict_theta_thread.start()

    mj_path, _ = mujoco_py.utils.discover_mujoco()
    xml_path = os.path.join(mj_path, 'External Models/MPL/MPL/', 'arm_claw_ADL.xml')
    print(mj_path, xml_path)
    model = mujoco_py.load_model_from_path(xml_path)
    sim = mujoco_py.MjSim(model)
    rend = mujoco_py.MjViewer(sim)

    while True:
        if grasp_pred:
            mov_class = grasp_pred[-1]
            if mov_class:
                sim.data.ctrl[8] = -1.0
                sim.data.ctrl[7] = 1.0
            else:
                sim.data.ctrl[8] = 1.0
                sim.data.ctrl[7] = -1.0

            sim.step()
            rend.render()
    # ani = animation.FuncAnimation(fig, animate, fargs=(), interval=6)
    # plt.show()
except KeyboardInterrupt:
    dev.stop()
    dev_EMG.stop()
    exit(0)

# acquire_imu_predict()
# save_to_csv()
print("Finished")
