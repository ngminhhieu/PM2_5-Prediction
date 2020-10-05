import argparse
import os
import sys
import tensorflow as tf
import numpy as np
import yaml
import random as rn
from model.supervisor import EncoderDecoder
from lib.GABinary import evolution
from lib import utils_ga
from lib import constant
from model.supervisor import EncoderDecoder


def seed():
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(2)
    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(12345)
    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see:
    # https://www.tensorflow.org/api_docs/python/tf/set_random_seed
    tf.set_random_seed(1234)


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
                        default="true",
                        type=str,
                        help='Select best individuals only')
    args = parser.parse_args()
    flag_select_best_only=True

    if args.select_best_only=="true" or args.select_best_only=="True":
        flag_select_best_only=True
    else:
        flag_select_best_only=False
    # load config for seq2seq model
    if args.config_file != False:
        with open(args.config_file) as f:
            config = yaml.load(f)

    if args.mode == 'ga_seq2seq':
        log_path = "log/PM2.5/pc_{}-pm_{}-pop_{}-gen_{}-bestonly_{}/".format(
            str(args.pc), str(args.pm), str(args.population), str(args.gen),
            str(flag_select_best_only))
        evo = evolution(total_feature=len(constant.hanoi_features),
                        pc=args.pc,
                        pm=args.pm,
                        population_size=args.population,
                        max_gen=args.gen,
                        select_best_only=flag_select_best_only,
                        log_path=log_path)
        fitness = [evo["gen"], evo["fitness"]]
        utils_ga.write_log(path=log_path,
                           filename="result_binary.csv",
                           error=fitness)
    elif args.mode == 'seq2seq_train':
        model = EncoderDecoder(is_training=True, **config)
        model.train()
    elif args.mode == 'seq2seq_test':
        model = EncoderDecoder(is_training=False, **config)
        model.test()
        model.plot_series()
    else:
        raise RuntimeError("Mode needs to be train/evaluate/test!")
