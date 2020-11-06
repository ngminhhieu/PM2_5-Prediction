import argparse
import os
import sys
import numpy as np
import yaml
import random as rn
from model.supervisor import EncoderDecoder
from lib.GABinary import GA
from lib import utils_ga
from lib import constant
from model.supervisor import EncoderDecoder
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
session = tf.compat.v1.Session(config=config)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#     try:
#         tf.config.experimental.set_virtual_device_configuration(
#             gpus[0], [
#                 tf.config.experimental.VirtualDeviceConfiguration(
#                     memory_limit=1024)
#             ])
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Virtual devices must be set before GPUs have been initialized
#         print(e)


def seed():
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(2)
    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    # rn.seed(12345)
    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see:
    # https://www.tensorflow.org/api_docs/python/tf/set_random_seed
    tf.compat.v1.set_random_seed(1234)


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
                        default=0.3,
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
    parser.add_argument('--percentage_split', default=10, type=int, help='')
    parser.add_argument('--percentage_back_test', default=0, type=int, help='')
    parser.add_argument('--split_training_data',
                        default=True,
                        type=str2bool,
                        help='')
    parser.add_argument('--fixed_splitted_data',
                        default=True,
                        type=str2bool,
                        help='')
    parser.add_argument('--shuffle_gen', default=True, type=str2bool, help='')

    args = parser.parse_args()

    if args.mode == 'ga_seq2seq':
        log_path = "log/PM2.5/pc_{}-pm_{}-pop_{}-gen_{}-bestonly_{}-percensplit_{}-percenbacktest_{}-split_{}-fixed_{}-shuffle_{}/".format(
            str(args.pc), str(args.pm), str(args.population), str(args.gen),
            str(args.select_best_only), str(args.percentage_split),
            str(args.percentage_back_test), str(args.split_training_data),
            str(args.fixed_splitted_data), str(args.shuffle_gen))

        ga = GA(args.percentage_split, args.percentage_back_test,
                args.split_training_data, args.fixed_splitted_data,
                args.shuffle_gen)
        last_pop_fitness = ga.evolution(total_feature=len(
            constant.hanoi_features),
                                        pc=args.pc,
                                        pm=args.pm,
                                        population_size=args.population,
                                        max_gen=args.gen,
                                        select_best_only=args.select_best_only,
                                        log_path=log_path)
        print(last_pop_fitness)
    elif args.mode == 'seq2seq_train':
        model = EncoderDecoder(is_training=True, **config)
        model.train()
    elif args.mode == 'seq2seq_test':
        model = EncoderDecoder(is_training=False, **config)
        model.test()
        model.plot_series()
    else:
        raise RuntimeError("Mode needs to be train/evaluate/test!")
