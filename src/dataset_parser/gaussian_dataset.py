import math
import numpy as np
from typing import List
from itertools import product
from dataclasses import dataclass
import matplotlib.colors as mcolors
from random import seed, shuffle, sample
from scipy.stats import multivariate_normal
from scipy.stats import norm as univariate_normal
from .common_functionality import CreateIterators, IteratorData, split_data

@dataclass
class GaussianInfo:
    class_lable: int
    attribute: List[int]
    mu: List
    sigma: List
    nv: multivariate_normal

class GaussianDataset:

    def __init__(self, dataset_name, **params):
        self.batch_size = params['batch_size']
        self.dataset_name = dataset_name
        self.dataset_size = params['dataset_size']
        self.return_numpy_array = params['return_numpy_array']

        self.train_split = 0.80
        self.valid_split = 0.25



    def generate_dataset(self):

        # Setup
        class_lable = [1, 0]
        attributes = [[0, 1], [0, 1, 2]]

        # Our gaussian is bounded in 4*4 space!
        mean_choices = list(np.arange(-4, 4, 0.5))

        all_gaussian_dict = {}
        for class_l in class_lable:
            for attr in product(*attributes):
                mu, sigma = [np.random.choice(mean_choices), -np.random.choice(mean_choices)], [[1, 0], [0, 1]]
                gaussian = GaussianInfo(class_lable=class_l, attribute=attr, mu=mu, sigma=sigma,
                                        nv=multivariate_normal(mean=mu, cov=sigma))
                all_gaussian_dict[str(list(attr))] = gaussian

        hidden_attribute_rate = [0.05, 0.55, 0.1, 0.1, 0.1, 0.1]
        rates = {
            0: [0.95, 0.05],  # [label0 = 0.95, lable1 = 0.05]
            1: [0.05, 0.95],
            2: [0.5, 0.5],
            3: [0.5, 0.5],
            4: [0.5, 0.5],
            5: [0.5, 0.5]
        }

        # Since there are two attriute - S1 - (A1,A2) and S2 - (A1', A2', A3')
        s_to_S = {
            0: [0, 0],
            1: [0, 1],
            2: [0, 2],
            3: [1, 0],
            4: [1, 1],
            5: [1, 2]
        }

        all_x = []
        all_y = []
        all_s = []
        for _ in range(self.dataset_size):
            s = np.random.choice(len(hidden_attribute_rate), size=1, replace=False, p=hidden_attribute_rate)[0]
            y = np.random.choice(2, size=1, replace=False, p=rates[s])[0]
            all_y.append(y)
            all_s.append(s_to_S[s])
            attribute = s_to_S[s]
            x = all_gaussian_dict[str(list(attribute))].nv.rvs(1)
            all_x.append(x)

        all_x = np.array(all_x)
        all_y = np.array(all_y)
        all_s = np.asarray(all_s)


        return all_x, all_y, all_s



    def run(self):
        """Orchestrates the whole process"""
        X,y,s = self.generate_dataset()

        # the dataset is shuffled so as to get a unique test set for each seed.
        index, test_index, dev_index = split_data(X.shape[0], train_split=self.train_split, valid_split=self.valid_split)
        X, y, s = X[index], y[index], s[index]
        train_X, train_y, train_s = X[:dev_index, :], y[:dev_index], s[:dev_index]
        valid_X, valid_y, valid_s = X[dev_index:test_index, :], y[dev_index:test_index], s[dev_index:test_index]
        test_X, test_y, test_s = X[test_index:, :], y[test_index:], s[test_index:]


        create_iterator = CreateIterators()
        iterator_data = IteratorData(
            train_X=train_X, train_y=train_y, train_s=train_s,
            dev_X=valid_X, dev_y=valid_y, dev_s=valid_s,
            test_X=test_X, test_y=test_y, test_s=test_s,
            batch_size=self.batch_size,
            do_standard_scalar_transformation=True
        )
        iterator_set, vocab, s_flatten_lookup = create_iterator.get_iterators(iterator_data)

        iterators = [iterator_set]  # If it was k-fold. One could append k iterators here.


        iterators.append(iterator_set)

        other_meta_data = {}
        other_meta_data['task'] = 'simple_classification'
        other_meta_data['dataset_name'] = self.dataset_name
        other_meta_data['number_of_main_task_label'] = len(np.unique(y))
        other_meta_data['number_of_aux_label_per_attribute'] = [len(np.unique(s[:,i])) for i in range(s.shape[1])]
        other_meta_data['input_dim'] = train_X.shape[1]
        other_meta_data['s_flatten_lookup'] = s_flatten_lookup
        if self.return_numpy_array:
            raw_data = {
                'train_X': train_X,
                'train_y': train_y,
                'train_s': train_s,
                'valid_X': valid_X,
                'valid_y': valid_y,
                'valid_s': valid_s,
                'test_X': test_X,
                'test_y': test_y,
                'test_s': test_s
            }
            other_meta_data['raw_data'] = raw_data

        return iterators, other_meta_data






