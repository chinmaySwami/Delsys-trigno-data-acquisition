import pytrigno
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import logging
import threading
import time


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


def create_connection_accl(host):
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


def save_to_csv():
    ys_np = []
    for lst in ys:
        ys_np.append(lst.flatten())
    ys_np = np.array(ys_np)
    np.savetxt("Tests/test.csv", ys_np, delimiter=",")
    print("Saved to csv successfully: ")


def animate(i, xs, ys):
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

dev = create_connection_accl('localhost')
acquire_data_thread = threading.Thread(target=check_accel_thread, args=('localhost', ))
acquire_data_thread.daemon = True
acquire_data_thread.start()

ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys,), interval=10)
plt.show()



