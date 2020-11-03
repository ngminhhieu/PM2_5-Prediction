# ignore warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import random
from operator import itemgetter
import copy
import yaml
import math
from sklearn.metrics import mean_absolute_error
import pandas as pd

from lib import constant
from lib import utils_ga
from lib import preprocessing_data
from model.supervisor import EncoderDecoder
import numpy as np
from datetime import datetime

target_feature = ['PM2.5']
dataset = 'data/csv/hanoi_data.csv'
# output_dir = 'data/npz/hanoi/ga_hanoi.npz'
# config_path_ga = 'config/hanoi/ga_hanoi.yaml'
output_dir_npz = ''
config_path_ga = ''
seq2seq_path = ''


def get_input_features(gen_array):
    input_features = []
    for index, value in enumerate(gen_array, start=0):
        if value == 1:
            input_features.append(constant.hanoi_features[index])
    return input_features


def load_config():
    with open(config_path_ga) as f:
        config = yaml.load(f)
    return config


def fitness(gen_array):
    input_features = get_input_features(gen_array)
    preprocessing_data.generate_npz(input_features + target_feature, dataset,
                                    output_dir_npz, config_path_ga, seq2seq_path)
    config = load_config()
    # train
    model = EncoderDecoder(is_training=True, **config)
    training_time = model.train()

    # predict
    model = EncoderDecoder(is_training=False, **config)
    mae = model.test()
    return mae, np.sum(np.array(training_time))


def individual(total_feature):
    a = [0 for _ in range(total_feature)]
    for i in range(total_feature):
        r = random.random()
        if r < 0.5:
            a[i] = 1
    indi = {"gen": a, "fitness": 0, "time": 0}
    indi["fitness"], indi["time"] = fitness(indi["gen"])
    return indi


def crossover(father, mother, total_feature):
    cutA = random.randint(1, total_feature - 1)
    cutB = random.randint(1, total_feature - 1)
    while cutB == cutA:
        cutB = random.randint(1, total_feature - 1)
    start = min(cutA, cutB)
    end = max(cutA, cutB)
    child1 = {
        "gen": [0 for _ in range(total_feature)],
        "fitness": 0,
        "time": 0
    }
    child2 = {
        "gen": [0 for _ in range(total_feature)],
        "fitness": 0,
        "time": 0
    }

    child1["gen"][:start] = father["gen"][:start]
    child1["gen"][start:end] = mother["gen"][start:end]
    child1["gen"][end:] = father["gen"][end:]
    child1["fitness"], child1["time"] = fitness(child1["gen"])

    child2["gen"][:start] = mother["gen"][:start]
    child2["gen"][start:end] = father["gen"][start:end]
    child2["gen"][end:] = mother["gen"][end:]
    child2["fitness"], child1["time"] = fitness(child2["gen"])
    return child1, child2


def mutation(father, total_feature):
    a = copy.deepcopy(father["gen"])
    i = random.randint(0, total_feature - 1)
    if a[i] == 0:
        a[i] = 1
    else:
        a[i] = 0
    child = {"gen": a, "fitness": 0, "time": 0}
    child["fitness"], child["time"] = fitness(child["gen"])
    return child


def selection(popu, population_size, best_only=True):
    if best_only:
        new_list = sorted(popu, key=itemgetter("fitness"), reverse=False)
        return new_list[:population_size]
    else:
        n = math.floor(population_size / 2)
        temp = sorted(popu, key=itemgetter("fitness"), reverse=False)
        new_list = temp[:n]
        while len(new_list) < population_size:
            i = random.randint(n, len(temp) - 1)
            new_list.append(temp[i])
            temp.remove(temp[i])
        return new_list


def evolution(total_feature,
              pc=0.8,
              pm=0.2,
              population_size=50,
              max_gen=1000,
              select_best_only=True,
              log_path="log/GA/"):
    tmp_path = "ga_hanoi_pc_{}-pm_{}-pop_{}-gen_{}-bestonly_{}".format(
        str(pc), str(pm), str(population_size), str(max_gen),
        str(select_best_only))
    global output_dir_npz
    global config_path_ga
    global seq2seq_path
    output_dir_npz = 'data/npz/hanoi/{}.npz'.format(tmp_path)
    config_path_ga = 'config/hanoi/{}.yaml'.format(tmp_path)
    seq2seq_path = log_path + "seq2seq/"
    ga_log_path = log_path + "GA/"
    population = []
    first_training_time = 0
    start_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    for _ in range(population_size):
        indi = individual(total_feature=total_feature)
        population.append(indi)
        first_training_time += indi["time"]
    new_pop = sorted(population, key=itemgetter("fitness"), reverse=False)
    utils_ga.write_log(path=ga_log_path,
                       filename="fitness_gen.csv",
                       error=[start_time, population[0]["gen"], population[0]["fitness"], first_training_time])

    t = 1
    while t <= max_gen:
        training_time_gen = 0
        temp_population = []
        for i, _ in enumerate(population):
            r = random.random()
            if r < pc:
                j = random.randint(0, population_size - 1)
                while j == i:
                    j = random.randint(0, population_size - 1)
                f_child, m_child = crossover(population[i].copy(),
                                             population[j].copy(),
                                             total_feature=total_feature)
                temp_population.append(f_child)
                temp_population.append(m_child)
                training_time_gen += f_child["time"] + m_child["time"]
            if r < pm:
                off = mutation(population[i].copy(), total_feature=total_feature)
                temp_population.append(off)
                training_time_gen += off["time"]

        population = selection(population.copy() + temp_population, population_size,
                               select_best_only)

        pop_fitness = population[0]["fitness"]
        fitness = [
            t, population[0]["gen"], pop_fitness,
            training_time_gen
        ]
        utils_ga.write_log(path=ga_log_path,
                           filename="fitness_gen.csv",
                           error=fitness)
        print("t =", t, "fitness =", pop_fitness, "time =",
              training_time_gen)
        t = t + 1
    return pop_fitness
