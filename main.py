import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
import argparse
import sys
import numpy as np
import yaml
import random as rn
from model.supervisor import EncoderDecoder
from lib.GABinary import GA
from lib import utils_ga
from lib import constant
import tensorflow as tf
from tensorflow.python.keras import backend as K
from lib import preprocessing_data
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
# Allow run multiple scripts python main.py
K.set_session(tf.compat.v1.Session(config=config))
def checkGPU():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

def seed():
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(2)
    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(1)
    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see:
    # https://www.tensorflow.org/api_docs/python/tf/set_random_seed
    tf.compat.v1.set_random_seed(12345)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    checkGPU()
    seed()
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only',
                        default=False,
                        type=str,
                        help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config_file',
                        default=False,
                        type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--mode',
                        default='ga_seq2seq',
                        type=str,
                        help='Run mode.')
    parser.add_argument('--pc',
                        default=0.8,
                        type=float,
                        help='Probability of Crossover')
    parser.add_argument('--pm',
                        default=0.2,
                        type=float,
                        help='Probability of Mutation')
    parser.add_argument('--population',
                        default=30,
                        type=int,
                        help='Population size')
    parser.add_argument('--gen',
                        default=40,
                        type=int,
                        help='Number of generation')
    parser.add_argument('--select_best_only',
                        default=True,
                        type=str2bool,
                        help='Select best individuals only')
    parser.add_argument('--tmp',
                        default=0,
                        type=int,
                        help='Number of experiments')                    
    parser.add_argument('--percentage_split', default=10, type=int, help='')
    parser.add_argument('--percentage_back_test', default=0, type=int, help='')
    parser.add_argument('--split', default=True, type=str2bool, help='')
    parser.add_argument('--fixed', default=True, type=str2bool, help='')
    parser.add_argument('--shuffle', default=True, type=str2bool, help='')

    args = parser.parse_args()

    if args.mode == 'ga_seq2seq':
        log_path = "log/PM2.5/pc_{}-pm_{}-pop_{}-gen_{}-bestonly_{}-percensplit_{}-percenbacktest_{}-split_{}-fixed_{}-shuffle_{}/".format(
            str(args.pc), str(args.pm), str(args.population), str(args.gen),
            str(args.select_best_only), str(args.percentage_split),
            str(args.percentage_back_test), str(args.split), str(args.fixed),
            str(args.shuffle))

        ga = GA(args.percentage_split, args.percentage_back_test, args.split,
                args.fixed, args.shuffle, args.tmp)
        last_pop_fitness, fitness_gen = ga.evolution(total_feature=len(
            constant.hanoi_features),
                                        pc=args.pc,
                                        pm=args.pm,
                                        population_size=args.population,
                                        max_gen=args.gen,
                                        select_best_only=args.select_best_only,
                                        log_path=log_path)

        input_features = []
        for index, value in enumerate(fitness_gen, start=0):
            if value == 1:
                input_features.append(constant.hanoi_features[index])
        preprocessing_data.generate_npz(
                    input_features + ['PM2.5'],
                    'data/csv/hanoi_data_full.csv', "data/npz/hanoi/final_after_ga_{}.npz".format(str(args.tmp)),
                    'config/hanoi/final_after_ga_{}.yaml'.format(str(args.tmp)), 'log/PM2.5/final_after_ga_{}/GA/seq2seq/'.format(str(args.tmp)))
        config = utils_ga.load_config('config/hanoi/final_after_ga_{}.yaml'.format(str(args.tmp)))
        model = EncoderDecoder(is_training=True, **config)
        training_time = model.train()
        model = EncoderDecoder(is_training=False, **config)
        mae = model.test()
    elif args.mode == 'seq2seq_train':
        with open(args.config_file) as f:
            config = yaml.safe_load(f)
        model = EncoderDecoder(is_training=True, **config)
        model.train()
    elif args.mode == 'seq2seq_test':
        with open(args.config_file) as f:
            config = yaml.safe_load(f)
        model = EncoderDecoder(is_training=False, **config)
        model.test()
        model.plot_series()
    else:
        raise RuntimeError("Mode needs to be train/evaluate/test!")
