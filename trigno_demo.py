import pytrigno
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
from scipy.signal import butter, lfilter_zi, lfilter
from sklearn.preprocessing import StandardScaler


def create_standard_scalar():
    scalar = StandardScaler()
    scalar.mean_ = np.array([804.68])
    scalar.var_ = np.array([67047282117])
    scalar.scale_ = np.array(np.sqrt(scalar.var_))
    return scalar


def create_butterworth_filter():
    nyq = 0.5 * sampling_frequency
    low = lowCutoff / nyq
    b = butter(filterOrderIMU, low, btype='low', output='ba')
    z = lfilter_zi(b[0], b[1])
    return b, z


def smooth_data(data, zi):
    #  Lowpass filter
    smoothed_data, z = lfilter(b[0], b[1], [data], zi=zi)
    zi = z
    return smoothed_data[0]


def check_accel(host):
    dev = pytrigno.TrignoAccel(channel_range=(0, 35), samples_per_read=1, host=host)
    dev.check_sensor_n_type(1)
    dev.check_sensor_n_mode(1)
    dev.check_sensor_n_start_index(1)
    dev.check_sensor_n_auxchannel_count(1)
    dev.check_sensor_channel_unit(2, 1)
    dev.check_sensor_channel_unit(2, 4)
    dev.check_sensor_channel_unit(2, 7)

    dev.start()
    print("Connection established::")
    while True:
        data = dev.read()
        ys.append(data)
        assert data.shape == (36, 1)
        print(data[10][0], "\t", data[13][0], "\t", data[28][0], "\t", data[32][0])
        # print("Frame No: ", i, "\t", data[10][0], "\t", data[13][0], "\t", data[28][0], "\t", data[32][0])
    dev.stop()


def normalize_data(data):
    n_data = (data - avg_mean_training)/avg_std_training
    return n_data


def create_connection_accel(host):
    dev = pytrigno.TrignoAccel(channel_range=(0, 35), samples_per_read=1, host=host)
    dev.check_sensor_n_type(sensor_number)
    dev.check_sensor_n_mode(sensor_number)
    dev.check_sensor_n_start_index(sensor_number)
    dev.check_sensor_n_auxchannel_count(sensor_number)
    dev.check_sensor_channel_unit(sensor_number, 1)
    dev.check_sensor_channel_unit(sensor_number, 4)
    dev.check_sensor_channel_unit(sensor_number, 7)
    return dev


def check_accel_thread(host):
    dev.check_sensor_n_type(2)
    dev.check_sensor_n_mode(2)
    dev.check_sensor_n_start_index(2)
    dev.check_sensor_n_auxchannel_count(2)
    for i in range(7):
        dev.check_sensor_channel_unit(2, i)

    dev.start()
    print("Connection established::")
    while True:
        data = dev.read()
        xs.append(data[0][0])
        ys.append(data[1][0])
        # zs.append(data[2][0])


def check_accel_thread_filtered(host, zi):
    dev.start()
    print("Connection established::")
    while True:
        datao = dev.read()
        if datao[12][0] != 0:
            za = 10
        data = np.multiply(datao, scaling_array)
        filtered = np.zeros(36)
        for i in range(data.shape[0]):
            filtered[i], zi[i] = lfilter(b[0], b[1],  data[i], zi=zi[i])
        norm = normalize_data(np.array(filtered))
        xs.append(datao[12])
        xf.append(filtered[12])
        xn.append(norm[12])
        ys.append(datao[27])
        yf.append(filtered[27])
        yn.append(norm[27])


def animate(i):
    if ys:
        # data = dev.read()
        # ys.append(data[1][0])
        # print(xs[-1], "\t", ys[-1], "\t", zs[-1])
        # ys = ys[-100:]
        # Draw x and y lists
        xss = xs[-500:]
        xsf = xf[-500:]
        xsn = xn[-500:]
        yss = ys[-500:]
        ysf = yf[-500:]
        ysn = yn[-500:]
        ax.clear()
        ay.clear()
        axf.clear()
        ayf.clear()
        axn.clear()
        ayn.clear()
        ax.plot(xss)
        ay.plot(yss)
        axf.plot(xsf)
        ayf.plot(ysf)
        axn.plot(xsn)
        ayn.plot(ysn)


fig = plt.figure()
xs = []
xf = []
xn = []
ys = []
yf = []
yn = []
scaling_array = np.array([[9806.65], [9806.65], [9806.65], [1000], [1000], [1000], [1000], [1000], [1000],
                          [9806.65], [9806.65], [9806.65], [1000], [1000], [1000], [1000], [1000], [1000],
                          [9806.65], [9806.65], [9806.65], [1000], [1000], [1000], [1000], [1000], [1000],
                          [9806.65], [9806.65], [9806.65], [1000], [1000], [1000], [1000], [1000], [1000]])
avg_mean_training = np.array([9026.581419, 902.5942718, 4676.531876, 1536.347365, 824.2234014, 2355.680627,
                              -7557.074342, 238.9051397, -7004.522484, 1100.075402, 8357.125645, -3072.610687,
                              -9787.098475, 716.5740751, -2461.864722, 1468.178587, -412.4093905, -1145.956741,
                              -3408.054336, 992.7556896, 624.4395445, 23619.40505, 8885.855499, 1394.194849,
                              -4494.225019, -454.6410738, -1114.837898, 3616.154164, 7812.728116, 879.4170656,
                              4581.595772, -1112.281237, -1521.265704, 498.9367503, 8031.761407, 3475.425272])

# avg_mean_training = avg_mean_training.reshape(-1, 1)
avg_std_training = np.array([3817.600932, 683.383692, 4721.688517, 15619.74755, 70935.28791, 72087.63885,
                             3899.100149, 514.3201803, 4127.585205, 23852.0515, 74641.24657, 30605.56797, 3006.187073,
                            644.0836957, 4982.276768, 19552.98379, 79514.72385, 32777.61901, 1990.689673, 830.3656838,
                            2630.36271, 35482.23785, 79879.04171, 16961.90462, 1422.30238, 412.355329, 3258.372477,
                            16624.68747, 35696.7495, 19250.84621, 1370.720671, 374.9487406, 3091.811698, 18509.75631,
                            19317.85871, 21869.72258])
# avg_std_training = avg_std_training.reshape(-1, 1)
sensor_number = 2
sampling_frequency = 148
filterOrderIMU = 2
lowCutoff = 1

b, z = create_butterworth_filter()
zi={}
for i in range(36):
    zi[i] = z

dev = create_connection_accel('localhost')

# standard_scalar = StandardScaler().fit(np.array([[0]]))
# standard_scalar.mean_ = np.array([804.68])
# standard_scalar.var_ = np.array([67047282117])
# standard_scalar.scale_ = np.array(np.sqrt(67047282117))

ax = fig.add_subplot(3, 2, 1)
ay = fig.add_subplot(3, 2, 2)
axf = fig.add_subplot(3, 2, 3)
ayf = fig.add_subplot(3, 2, 4)
axn = fig.add_subplot(3, 2, 5)
ayn = fig.add_subplot(3, 2, 6)

try:
    # acquire_data_thread = threading.Thread(target=check_accel_thread, args=('localhost', ))
    acquire_data_thread = threading.Thread(target=check_accel_thread_filtered, args=('localhost', zi, ))

    acquire_data_thread.daemon = True
    acquire_data_thread.start()

    ani = animation.FuncAnimation(fig, animate, fargs=(), interval=8)
    plt.show()

except KeyboardInterrupt:
    dev.stop()
    exit(0)
