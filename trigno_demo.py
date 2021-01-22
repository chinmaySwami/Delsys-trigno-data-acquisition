import pytrigno
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
from scipy.signal import butter, filtfilt, lfilter_zi, lfilter


def create_butterworth_filter():
    nyq = 0.5 * sampling_frequency
    low = lowCutoff / nyq
    b = butter(filterOrderIMU, low, btype='low', output='ba')
    z = lfilter_zi(b[0], b[1])
    return b, z


def smooth_data(data, zi):
    #  Lowpass filter
    # smoothed_data, zi = lfilter(b[0], b[1], [data], zi=zi)

    # Bujtterworth Filter
    smoothed_data = filtfilt(b[0], b[1], [data], method="gust")

    zi = zi
    # smoothed_data = filtfilt(b[0], b[1], data, method='gust', irlen=2)
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


def check_accel_thread_filtered(host):
    dev.check_sensor_n_type(1)
    dev.check_sensor_n_mode(1)
    dev.check_sensor_n_start_index(1)
    dev.check_sensor_n_auxchannel_count(1)

    dev.start()
    print("Connection established::")
    while True:
        data = dev.read()
        xs.append(data[0][0])
        ys.append(smooth_data(data[0][0], zi))
        zs.append(data[2][0])


def animate(i):
    if ys:
        # data = dev.read()
        # ys.append(data[1][0])
        print(ys[-1])
        # ys = ys[-100:]
        # Draw x and y lists
        xss = xs[-200:]
        yss = ys[-200:]
        zss = zs[-200:]
        ax.clear()
        ay.clear()
        az.clear()
        ax.plot(xss)
        ay.plot(yss)
        az.plot(zss)


fig = plt.figure()
ax = fig.add_subplot(3, 1, 1)
ay = fig.add_subplot(3, 1, 2)
az = fig.add_subplot(3, 1, 3)
xs = []
ys = []
zs = []

sampling_frequency = 2000
filterOrderEMG = 4
filterOrderIMU = 1
lowCutoff = 1
avg_mean_training = []
avg_std_training = []

b, zi = create_butterworth_filter()
dev = create_connection_accel('localhost')

# Non filtered data
# acquire_data_thread = threading.Thread(target=check_accel_thread, args=('localhost', ))
# filtered data
acquire_data_thread = threading.Thread(target=check_accel_thread_filtered, args=('localhost', ))

acquire_data_thread.daemon = True
acquire_data_thread.start()

ani = animation.FuncAnimation(fig, animate, fargs=(), interval=10)
plt.show()



