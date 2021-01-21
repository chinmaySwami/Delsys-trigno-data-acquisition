import numpy as np
import pytrigno


def create_connection_accl(host):
    dev = pytrigno.TrignoAccel(channel_range=(0, 63), samples_per_read=1,
                               host=host)
    dev.check_sensor_n_type(1)
    dev.check_sensor_n_mode(1)
    dev.check_sensor_n_start_index(1)
    dev.check_sensor_n_auxchannel_count(1)
    return dev


def acquire_imu():
    dev.start()
    print("Connection established::")
    try:
        while True:
            data = dev.read()
            imu_data.append(data)
            print(data[1][0], "\t", data[13][0], "\t", data[28][0], "\t", data[32][0])
    except KeyboardInterrupt:
        dev.stop()
        print('Data acquisition stopped')


def save_to_csv():
    ys_np = []
    for lst in imu_data:
        ys_np.append(lst.flatten())
    ys_np = np.array(ys_np)
    np.savetxt("Tests/test.csv", ys_np, delimiter=",")
    print("Saved to csv successfully: ")


imu_data = []
dev = create_connection_accl('localhost')
acquire_imu()
save_to_csv()
print("Finished")
