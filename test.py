import pandas as pd
import numpy as np

dataset = pd.read_csv("./data/csv/taiwan.csv", usecols=['AMB_TEMP', 'RH', 'WIND_DIREC', 'WIND_SPEED', 'PM10', 'PM2.5'])
dataset = dataset.to_numpy()
np.savez("./data/npz/taiwan/taiwan.npz", monitoring_data = dataset)