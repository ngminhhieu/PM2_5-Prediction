from lib.preprocessing_data import generate_npz
if __name__ == '__main__':
    input_features = ["WIND_DIR", "PM1", "PM2.5"]
    generate_npz(input_features, 'data/csv/hanoi.csv', 'data/npz/new_training_mechanism/hanoi.npz',
             'config/hanoi/new_training_mechanism_1.yaml',
             'log/new_training_mechanism/')