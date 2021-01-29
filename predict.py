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
import os


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


def predict_theta_dot():
    while True:
        if imu_data:
            data_read = imu_data[-1]
            normalized_data = normalize_data(data_read)
            start = time.time()
            predicted_theta_dot = rud_model.predict([normalized_data])
            # print("time to predict:\t", time.time() - start, "\t", data_read[3], "\t", normalized_data[3], "\t",
            #       predicted_theta_dot[0])
            theta_dot.append(predicted_theta_dot[0])


def animate(i):
    if theta_dot:
        xss = theta_dot[-500:]
        ax.clear()
        # ay.clear()
        ax.plot(xss)
        # ay.plot(yss)



# def animate_test(i):
#     if imu_data:
#         data_read = imu_data[-1]
#         normalized_data = normalize_data(data_read)
#         start = time.time()
#         predicted_theta_dot = rud_model.predict([normalized_data])
#         print("time to predict:\t", time.time() - start, "\t", data_read[3], "\t", normalized_data[3], "\t",
#               predicted_theta_dot[0])
#         theta_dot.append(predicted_theta_dot[0])
#         xss = theta_dot[-500:]
#         ax.clear()
#         # ay.clear()
#         ax.plot(xss)
#         # ay.plot(yss)


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
imu_data = []
theta_dot = []
sensor_number = 2

b, z = create_butterworth_filter()
zi={}
for i in range(31):
    zi[i] = z
dev = create_connection_imu('localhost')

print("Loading regression model::")
# For PS
# rud_model = pickle.load(open("D:/Chinmay/ML Pipeline/Trained model/mode_0_20201006-104041", "rb"))
# For RUD
rud_model = pickle.load(open("D:/Chinmay/ML Pipeline/Trained model/mode_1_20201006-083222", "rb"))
rud_model.verbose = False
rud_model.n_jobs = 1
print(rud_model.n_estimators, rud_model.max_depth, rud_model.max_features)
print("Model loaded successfully::")

fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
# ay = fig.add_subplot(2, 1, 2)

try:
    acquire_data_thread = threading.Thread(target=acquire_imu, args=(zi,))
    predict_theta_thread = threading.Thread(target=predict_theta_dot)
    acquire_data_thread.daemon = True
    predict_theta_thread.daemon = True

    acquire_data_thread.start()
    predict_theta_thread.start()

    mj_path, _ = mujoco_py.utils.discover_mujoco()
    xml_path = os.path.join(mj_path, 'External Models/MPL/MPL/', 'arm_claw_ADL.xml')
    model = mujoco_py.load_model_from_path(xml_path)
    sim = mujoco_py.MjSim(model)
    rend = mujoco_py.MjViewer(sim)

    while True:
        rend.render()
        sim.step()
        if theta_dot:
            print("Predicted theta dot: ", theta_dot[-1])
            sim.data.ctrl[5] = theta_dot[-1]

    # ani = animation.FuncAnimation(fig, animate, fargs=(), interval=6)
    # plt.show()

except KeyboardInterrupt:
    dev.stop()
    exit(0)

# acquire_imu_predict()
# save_to_csv()
print("Finished")
