import pytrigno
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
from scipy.signal import butter, filtfilt, lfilter_zi, lfilter
from sklearn.preprocessing import StandardScaler


def create_standard_scalar():
    scalar = StandardScaler()
    scalar.mean_ = np.array([804.68])
    scalar.var_ = np.array([32228.98001])
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
    dev = pytrigno.TrignoAccel(channel_range=(0, 35), samples_per_read=1,
                               host=host)
    dev.check_sensor_n_type(1)
    dev.check_sensor_n_mode(1)
    dev.check_sensor_n_start_index(1)
    dev.check_sensor_n_auxchannel_count(1)

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
    dev = pytrigno.TrignoAccel(channel_range=(0, 35), samples_per_read=1,
                               host=host)
    dev.check_sensor_n_type(1)
    dev.check_sensor_n_mode(1)
    dev.check_sensor_n_start_index(1)
    dev.check_sensor_n_auxchannel_count(1)
    return dev


def check_accel_thread(host):
    dev.check_sensor_n_type(1)
    dev.check_sensor_n_mode(1)
    dev.check_sensor_n_start_index(1)
    dev.check_sensor_n_auxchannel_count(1)

    dev.start()
    print("Connection established::")
    while True:
        data = dev.read()
        xs.append(data[0][0])
        ys.append(data[1][0])
        zs.append(data[2][0])


def check_accel_thread_filtered(host, scalar):
    dev.check_sensor_n_type(1)
    dev.check_sensor_n_mode(1)
    dev.check_sensor_n_start_index(1)
    dev.check_sensor_n_auxchannel_count(1)

    dev.start()
    print("Connection established::")
    while True:
        data = dev.read()
        filtered = smooth_data(data[12][0], zi)
        # norm = normalize_data(data[12][0])
        norm = scalar.transform(np.array([data[12]]))
        xs.append(data[12][0])
        ys.append(filtered)
        zs.append(norm)


def animate(i):
    if ys:
        # data = dev.read()
        # ys.append(data[1][0])
        print(xs[-1], "\t", ys[-1], "\t", zs[-1])
        # ys = ys[-100:]
        # Draw x and y lists
        xss = xs[-500:]
        yss = ys[-500:]
        zss = zs[-500:]
        ax.clear()
        ay.clear()
        az.clear()
        ax.plot(xss)
        ay.plot(yss)
        az.plot(zss)


fig = plt.figure()
xs = []
ys = []
zs = []

sampling_frequency = 2000
filterOrderIMU = 1
lowCutoff = 1
avg_mean_training = 1200.68
avg_std_training = 1000.0515

b, zi = create_butterworth_filter()
dev = create_connection_accel('localhost')
scalar = create_standard_scalar()

ax = fig.add_subplot(3, 1, 1)
ay = fig.add_subplot(3, 1, 2)
az = fig.add_subplot(3, 1, 3)

try:
    # acquire_data_thread = threading.Thread(target=check_accel_thread, args=('localhost', ))
    acquire_data_thread = threading.Thread(target=check_accel_thread_filtered, args=('localhost', scalar, ))

    acquire_data_thread.daemon = True
    acquire_data_thread.start()

    ani = animation.FuncAnimation(fig, animate, fargs=(), interval=10)
    plt.show()

except KeyboardInterrupt:
    dev.stop()
    exit(0)
