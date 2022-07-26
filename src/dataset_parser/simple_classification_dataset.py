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
from .simple_classification_dataset_helper import get_adult_multigroups_data

@dataclass
class GaussianInfo:
    class_lable: int
    attribute: List[int]
    mu: List
    sigma: List
    nv: multivariate_normal

class SimpleClassificationDataset:

    def __init__(self, dataset_name, **params):
        self.batch_size = params['batch_size']
        self.dataset_name = dataset_name
        self.return_numpy_array = params['return_numpy_array']

        if self.dataset_name == 'adult_multi_group':
            self.X, self.y, self.s = get_adult_multigroups_data()

        self.train_split = 0.80
        self.valid_split = 0.25

        self.y = (self.y + 1) / 2

        if len(np.unique(self.s)) == 2 and -1 in np.unique(self.s):
            self.s = (self.s + 1) / 2

    def run(self):
        """Orchestrates the whole process"""
        X,y,s = self.X, self.y, self.s

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






