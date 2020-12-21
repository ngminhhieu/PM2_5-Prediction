import os
import pandas as pd
from lib import constant
from lib import utils_ga
from lib import utils_model
from lib import preprocessing_data
from model.supervisor import EncoderDecoder
import numpy as np
import csv

root_dir = './log/PM2.5/'
tmp = 0

for rootdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file == 'fitness_gen.csv':
            fitness_result = pd.read_csv(os.path.join(rootdir,file), header=None)
            fitness_gen = fitness_result.iloc[-2, 2]
            time = fitness_result.iloc[-1, 1]
            # fitness_gen = np.asarray(list(fitness_gen))
            # fitness_gen = fitness_gen[1:-1:3].astype(np.int)
            result = [rootdir, fitness_gen, time]
            with open("./result_ga.csv", 'a') as file:
                writer = csv.writer(file)
                writer.writerow(result)
            tmp += 1
            # input_features = []
            # for index, value in enumerate(fitness_gen, start=0):
            #     if value == 1:
            #         input_features.append(constant.hanoi_features[index])
            # preprocessing_data.generate_npz(
            #             input_features + ['PM2.5'],
            #             'data/csv/hanoi_data_full.csv', "data/npz/hanoi/final_after_ga_{}.npz".format(str(tmp)),
            #             'config/hanoi/final_after_ga_{}.yaml'.format(str(tmp)), 'log/PM2.5/final_after_ga_{}/GA/seq2seq/'.format(str(tmp)))
            # config = utils_ga.load_config('config/hanoi/final_after_ga_{}.yaml'.format(str(tmp)))
            # model = EncoderDecoder(is_training=True, **config)
            # training_time = model.train()
            # model = EncoderDecoder(is_training=False, **config)
            # mae = model.test()