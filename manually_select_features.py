from lib.preprocessing_data import generate_npz
if __name__ == '__main__':
    # input_features = ["WIND_DIR", "PM1", "PM2.5"]
    input_features = ["MONTH", "YEAR", "WIND_SPEED", "RH", "PM1", "PM2.5"]
    # input_features = ["YEAR", "HOUR", "WIND_SPEED", "TEMP", "BAROMETER", "RADIATION", "PM10", "PM1", "PM2.5"]
    # input_features = ["RADIATION", "PM10", "PM1", "PM2.5"]
    # input_features = ["DAY", "HOUR", "WIND_DIR", "TEMP", "PM2.5"]
    for i in range(1, 7):
        generate_npz(input_features, 'data/csv/hanoi.csv', 'data/npz/new_training_mechanism/hanoi2.npz',
                'config/hanoi/new_training_mechanism_{}.yaml'.format(str(i)),
                'log/new_training_mechanism/')
