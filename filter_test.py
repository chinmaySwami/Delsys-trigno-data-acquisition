from scipy.signal import butter, lfilter_zi, lfilter, firwin
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import numpy as np


def generate_data():
    while True:
        randomlist = random.sample(range(-10, 30), 2)
        print(randomlist)
        data.append(randomlist)


def create_butterworth_filter():
    nyq = 0.5 * sampling_frequency
    low = lowCutoff / nyq
    b = butter(filterOrderIMU, low, btype='low', output='ba')
    z = lfilter_zi(b[0], b[1])
    return b, z


def animate(i, z):
    datas = np.array(data[-1])
    y, z = lfilter(b[0], b[1], datas[:], zi=z)
    data_a.append(datas[0])
    data_b.append(datas[1])
    data_af.append(y[0])
    data_bf.append(y[-1])
    xss = data_a[-500:]
    yss = data_b[-500:]
    zss = data_af[-500:]
    aas = data_bf[-500:]
    ax.clear()
    ay.clear()
    az.clear()
    aa.clear()
    ax.plot(xss)
    ay.plot(yss)
    az.plot(zss)
    aa.plot(aas)


data_a = []
data_b = []
data_af = []
data_bf = []
data = []
sensor_number = 2
sampling_frequency = 148
filterOrderIMU = 3
lowCutoff = 1

# while True:
#     generate_data()

acquire_data_thread = threading.Thread(target=generate_data)
acquire_data_thread.daemon = True
acquire_data_thread.start()

b, zi = create_butterworth_filter()

fig = plt.figure()
ax = fig.add_subplot(2, 2, 1)
ay = fig.add_subplot(2, 2, 2)
az = fig.add_subplot(2, 2, 3)
aa = fig.add_subplot(2, 2, 4)

ani = animation.FuncAnimation(fig, animate, fargs=(zi,), interval=8)
plt.show()
