# from keras.models import Model
# from keras.layers import Input
# from keras.layers import LSTM
# from numpy import array
# # define model
# inputs1 = Input(shape=(3, 1))
# lstm1, state_h, state_c = LSTM(1, return_sequences=True, return_state=True)(inputs1)
# model = Model(inputs=inputs1, outputs=[lstm1, state_h, state_c])
# # define input data
# data = array([0.4, 0.2, 0.3]).reshape((1,3,1))
# # make and show prediction
# print(model.predict(data))

import pandas as pd
import numpy as np

df = pd.read_csv('data/csv/hanoi_data_full.csv', usecols=['WIND_SPEED', 'TEMP', 'RADIATION', 'PM10', 'PM2.5'])

df = df.groupby(np.arange(len(df.index))//24).mean()
df.to_csv('data/csv/hanoi_data_full_day.csv')
np.savez('data/npz/hanoi/hanoi_data_full_day.npz', monitoring_data = df)