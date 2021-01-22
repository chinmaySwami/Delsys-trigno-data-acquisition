import sklearn
import pandas
import numpy as np
import pickle
import pytrigno


def create_connection_accl(host):
    dev = pytrigno.TrignoAccel(channel_range=(0, 63), samples_per_read=1,
                               host=host)
    dev.check_sensor_n_type(1)
    dev.check_sensor_n_mode(1)
    dev.check_sensor_n_start_index(1)
    dev.check_sensor_n_auxchannel_count(1)
    return dev


def acquire_imu_predict():
    dev.start()
    print("Connection established::")
    try:
        while True:
            data = dev.read()
            imu_data.append(np.concatenate((data[9:15], data[27:33], data[36:42],
                            data[45:51], data[54:60]), axis=0))
            pred_theta_dot = rud_model.predict([imu_data[-1].flatten()])
            print("predicted angular velocity:\t", pred_theta_dot[0])
    except KeyboardInterrupt:
        dev.stop()
        print('Data acquisition stopped')


print("Loading regression model::")
# For PS
rud_model = pickle.load(open("D:/Chinmay/ML Pipeline/Trained model/mode_0_20201006-104041", "rb"))
# For RUD
# rud_model = pickle.load(open("D:/Chinmay/ML Pipeline/Trained model/mode_1_20201006-083222", "rb"))
rud_model.verbose = False
print(rud_model.n_estimators, rud_model.max_depth, rud_model.max_features)
print("Model loaded successfully::")
imu_data = []
dev = create_connection_accl('localhost')
acquire_imu_predict()
# save_to_csv()
print("Finished")
