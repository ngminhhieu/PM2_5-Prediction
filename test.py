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
import random
import pandas as pd
import numpy as np
# df = pd.read_csv('data/csv/hanoi_data_full.csv', usecols=['WIND_SPEED', 'TEMP', 'RADIATION', 'PM10', 'PM2.5'])

# df = df.groupby(np.arange(len(df.index))//24).mean()
# df.to_csv('data/csv/hanoi_data_full_day.csv')
# np.savez('data/npz/hanoi/hanoi_data_full_day.npz', monitoring_data = df)


# dataset =  pd.read_csv('data/csv/ga/dataset_train.csv')
# pivot = int(10 * len(dataset) / 100)
# for i in range(100000):
#     random_start_point = random.randint(0, len(dataset) - pivot)
#     tmp_dataset = dataset.iloc[random_start_point:random_start_point + pivot + 1000]
#     print(i)
#     if tmp_dataset.isnull().values.any():
#         print(random_start_point)
#         print(pivot)
#         print(tmp_dataset)

import datetime
currentTime = datetime.datetime.now()
print(str(currentTime))

import psutil
# gives a single float value
print(psutil.cpu_percent())
# gives an object with many fields
print(psutil.virtual_memory())
# you can convert that object to a dictionary 
print(dict(psutil.virtual_memory()._asdict()))
# you can have the percentage of used RAM
print(psutil.virtual_memory().percent)
# you can calculate percentage of available memory
print(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)



X = [['male', 'bs'], ['male', 'phd'], ['male', 'bs'], 
     ['male', 'phd'],['male', 'bs'], ['male', 'phd'], ['male', 'phd'],
     ['male', 'phd'], ['male', 'bs'], ['male', 'bs'], 
     ['male', 'bs'], ['male', 'phd'], ['male', 'phd']]

Y = ['good', 'good', 'good', 'good', 'good', 'good', 
    'well','well','well', 'good', 'good',
     'well', 'well']

test_data = [['male', 'phd'],['male', 'phd'],['male', 'bs']]
test_labels = ['good','well','well']

from sklearn.svm import SVC

#Support Vector Classifier
s_clf = SVC()
s_clf.fit(X,Y)
s_prediction = s_clf.predict(test_data)
print(s_prediction)