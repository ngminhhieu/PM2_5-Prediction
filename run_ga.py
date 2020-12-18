import os
import pandas as pd
from lib import constant
from lib import utils_ga
from lib import utils_model
from lib import preprocessing_data
from model.supervisor import EncoderDecoder

root_dir = './log/PM2.5/'
tmp = 0
for rootdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file == 'fitness_gen.csv':
            tmp += 1
            fitness_result = pd.read_csv(os.path.join(rootdir,file), header=None)
            fitness_gen = fitness_result.iloc[-2, 2]
            input_features = []
            for index, value in enumerate(fitness_gen, start=0):
                if value == 1:
                    input_features.append(constant.hanoi_features[index])
            preprocessing_data.generate_npz(
                        input_features + ['PM2.5'],
                        'data/csv/hanoi_data_full.csv', "data/npz/hanoi/final_after_ga_{}.npz".format(str(tmp)),
                        'config/hanoi/final_after_ga_{}.yaml'.format(str(tmp)), 'log/PM2.5/final_after_ga_{}/GA/seq2seq/'.format(str(tmp)))
            config = utils_ga.load_config('config/hanoi/final_after_ga_{}.yaml'.format(str(tmp)))
            model = EncoderDecoder(is_training=True, **config)
            training_time = model.train()
            model = EncoderDecoder(is_training=False, **config)
            mae = model.test()
