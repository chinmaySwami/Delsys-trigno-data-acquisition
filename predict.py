import sklearn
import pandas
import numpy as np
import pickle
import pytrigno
from scipy.signal import butter, lfilter, freqz,filtfilt


def create_connection_accl(host):
    dev = pytrigno.TrignoAccel(channel_range=(0, 63), samples_per_read=1,
                               host=host)
    dev.check_sensor_n_type(1)
    dev.check_sensor_n_mode(1)
    dev.check_sensor_n_start_index(1)
    dev.check_sensor_n_auxchannel_count(1)
    return dev


def smooth_data(data):
    nyq = 0.5 * sampling_frequency
    low = lowCutoff / nyq
    b = butter(filterOrderIMU, low, btype='low', output='ba')
    smoothed_data = filtfilt(b[0], b[1], data, method="gust")
    return smoothed_data


def normalize_data(data):
    n_data = (data - avg_mean_training)/avg_std_training
    return n_data


def acquire_imu_predict():
    dev.start()
    print("Connection established::")
    try:
        while True:
            data = dev.read()
            raw_imu_data = np.concatenate((data[9:15], data[27:33], data[36:42], data[45:51], data[54:60]), axis=0)
            filtered_imu_data = smooth_data(raw_imu_data)
            normalized_imu_data = normalize_data(filtered_imu_data)
            pred_theta_dot = rud_model.predict([normalized_imu_data[-1].flatten()])
            print("predicted angular velocity:\t", pred_theta_dot[0])
    except KeyboardInterrupt:
        dev.stop()
        print('Data acquisition stopped')


sampling_frequency = 2000
filterOrderEMG = 4
filterOrderIMU = 1
lowCutoff = 1
avg_mean_training = []
avg_std_training = []

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
