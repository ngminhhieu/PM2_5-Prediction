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

class GA(object):
    def __init__(self, percentage_split=10, splitted_training_data=True, fixed_splitted_data=False, shuffle_gen=False):
        self.percentage_split = percentage_split
        self.target_feature = ['PM2.5']
        self.output_dir_npz = ''
        self.config_path_ga = ''
        self.seq2seq_path = ''
        self.shuffle_gen = shuffle_gen
        self.dataset_csv = 'data/csv/hanoi_data_full.csv'
        self.splitted_training_data = splitted_training_data
        self.fixed_splitted_data = fixed_splitted_data
        self.number_of_minor_dataset = self.split_data()

        self.gen = 1
        self.count_gen = 0


    def split_data(self):
        if self.splitted_training_data:
            dataset = pd.read_csv(self.dataset_csv)
            if 100%self.percentage_split == 0:
                number_of_minor_dataset = int(100/self.percentage_split)
            else:
                number_of_minor_dataset = int(100/self.percentage_split) + 1
            if self.fixed_splitted_data:
                pivot = int(self.percentage_split*len(dataset)/100)
                for i in range(number_of_minor_dataset-1):
                    # pd.read_csv(self.dataset_csv, skiprows = lambda x: x not in rows_to_keep)
                    tmp_dataset = dataset.iloc[i*pivot:(i+1)*pivot].copy()
                    tmp_dataset.to_csv('data/csv/ga/dataset_{}.csv'.format(i+1))

                tmp_dataset = dataset.iloc[pivot*(number_of_minor_dataset-1):]
                tmp_dataset.to_csv('data/csv/ga/dataset_{}.csv'.format(number_of_minor_dataset))

        return number_of_minor_dataset


    def fitness_shuffle_gen(self, gen_array):
        input_features = []
        for index, value in enumerate(gen_array, start=0):
            if value == 1:
                input_features.append(constant.hanoi_features[index])

        if self.splitted_training_data:
            dataset = pd.read_csv(self.dataset_csv)
            if self.fixed_splitted_data:
                random_number_dataset = random.randint(1, self.number_of_minor_dataset)
                preprocessing_data.generate_npz(input_features + self.target_feature, 'data/csv/ga/dataset_{}.csv'.format(random_number_dataset),
                                            self.output_dir_npz, self.config_path_ga, self.seq2seq_path)
            else:
                pivot = int(self.percentage_split*len(dataset)/100)
                random_start_point = random.randint(0, len(dataset) - pivot)
                tmp_dataset = dataset.iloc[random_start_point : random_start_point+pivot]
                tmp_dataset.to_csv('data/csv/ga/flex_shuffle_split_data.csv')
                preprocessing_data.generate_npz(input_features + self.target_feature, 'data/csv/ga/flex_shuffle_split_data.csv',
                                            self.output_dir_npz, self.config_path_ga, self.seq2seq_path)


        config = utils_ga.load_config(self.config_path_ga)
        # train
        model = EncoderDecoder(is_training=True, **config)
        training_time = model.train()

        # predict
        model = EncoderDecoder(is_training=False, **config)
        mae = model.test()
        return mae, np.sum(np.array(training_time))

    def fitness(self, gen_array, random_number_dataset):
        input_features = []
        for index, value in enumerate(gen_array, start=0):
            if value == 1:
                input_features.append(constant.hanoi_features[index])
        
        if self.splitted_training_data:
            dataset = pd.read_csv(self.dataset_csv)
            if self.fixed_splitted_data:
                preprocessing_data.generate_npz(input_features + self.target_feature, 'data/csv/ga/dataset_{}.csv'.format(str(random_number_dataset)),
                                            self.output_dir_npz, self.config_path_ga, self.seq2seq_path)
            else:
                pivot = int(self.percentage_split*len(dataset)/100)
                if self.gen != self.count_gen:
                    random_start_point = random.randint(0, len(dataset) - pivot)
                    tmp_dataset = dataset.iloc[random_start_point:random_start_point+pivot]
                    tmp_dataset.to_csv('data/csv/ga/flex_no_shuffle_split_data.csv')
                    self.count_gen = self.gen
                preprocessing_data.generate_npz(input_features + self.target_feature, 'data/csv/ga/flex_no_shuffle_split_data.csv',
                                            self.output_dir_npz, self.config_path_ga, self.seq2seq_path)
        
        config = utils_ga.load_config(self.config_path_ga)
        # train
        model = EncoderDecoder(is_training=True, **config)
        training_time = model.train()

        # predict
        model = EncoderDecoder(is_training=False, **config)
        mae = model.test()
        return mae, np.sum(np.array(training_time))

    def individual(self, total_feature):
        a = [0 for _ in range(total_feature)]
        for i in range(total_feature):
            r = random.random()
            if r < 0.5:
                a[i] = 1
        indi = {"gen": a, "fitness": 0, "time": 0}
        indi["fitness"], indi["time"] = self.fitness_shuffle_gen(indi["gen"])
        return indi


    def crossover(self, father, mother, total_feature, random_number_dataset):
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
        if self.shuffle_gen == False:
            child1["fitness"], child1["time"] = self.fitness(child1["gen"], random_number_dataset)
        else:
            child1["fitness"], child1["time"] = self.fitness_shuffle_gen(child1["gen"])

        child2["gen"][:start] = mother["gen"][:start]
        child2["gen"][start:end] = father["gen"][start:end]
        child2["gen"][end:] = mother["gen"][end:]
        if self.shuffle_gen == False:
            child2["fitness"], child1["time"] = self.fitness(child2["gen"], random_number_dataset)
        else:
            child2["fitness"], child1["time"] = self.fitness_shuffle_gen(child2["gen"])
        return child1, child2


    def mutation(self, father, total_feature, random_number_dataset):
        a = copy.deepcopy(father["gen"])
        i = random.randint(0, total_feature - 1)
        if a[i] == 0:
            a[i] = 1
        else:
            a[i] = 0
        child = {"gen": a, "fitness": 0, "time": 0}
        if self.shuffle_gen == False:
            child["fitness"], child["time"] = self.fitness(child["gen"], random_number_dataset)
        return child


    def selection(self, popu, population_size, best_only=True):
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


    def evolution(self, total_feature,
                pc=0.8,
                pm=0.2,
                population_size=50,
                max_gen=1000,
                select_best_only=True,
                log_path="log/GA/"):
        tmp_path = "ga_hanoi_pc_{}-pm_{}-pop_{}-gen_{}-bestonly_{}".format(
            str(pc), str(pm), str(population_size), str(max_gen),
            str(select_best_only))
        self.output_dir_npz = 'data/npz/hanoi/{}.npz'.format(tmp_path)
        self.config_path_ga = 'config/hanoi/{}.yaml'.format(tmp_path)
        self.seq2seq_path = log_path + "seq2seq/"
        ga_log_path = log_path + "GA/"
        population = []
        first_training_time = 0
        start_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        for _ in range(population_size):
            indi = self.individual(total_feature=total_feature)
            population.append(indi)
            first_training_time += indi["time"]
        new_pop = sorted(population, key=itemgetter("fitness"), reverse=False)
        utils_ga.write_log(path=ga_log_path,
                        filename="fitness_gen.csv",
                        error=[start_time, population[0]["gen"], population[0]["fitness"], first_training_time])
        while self.gen <= max_gen:
            if self.shuffle_gen == False:
                random_number_dataset = random.randint(1, self.number_of_minor_dataset)
            training_time_gen = 0
            temp_population = []
            for i, _ in enumerate(population):
                r = random.random()
                if r < pc:
                    j = random.randint(0, population_size - 1)
                    while j == i:
                        j = random.randint(0, population_size - 1)
                    f_child, m_child = self.crossover(population[i].copy(),
                                                population[j].copy(),
                                                total_feature, random_number_dataset)
                    temp_population.append(f_child)
                    temp_population.append(m_child)
                    training_time_gen += f_child["time"] + m_child["time"]
                if r < pm:
                    off = self.mutation(population[i].copy(), total_feature, random_number_dataset)
                    temp_population.append(off)
                    training_time_gen += off["time"]

            population = selection(population.copy() + temp_population, population_size,
                                select_best_only)

            pop_fitness = population[0]["fitness"]
            fitness = [
                self.gen, population[0]["gen"], pop_fitness,
                training_time_gen
            ]
            utils_ga.write_log(path=ga_log_path,
                            filename="fitness_gen.csv",
                            error=fitness)
            print("gen =", self.gen, "fitness =", pop_fitness, "time =", training_time_gen)
            self.gen = self.gen + 1
        return pop_fitness
