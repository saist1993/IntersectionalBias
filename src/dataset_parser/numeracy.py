import math
import numpy as np
from typing import List
from pathlib import Path
from itertools import product
from dataclasses import dataclass
import matplotlib.colors as mcolors
from random import seed, shuffle, sample
from scipy.stats import multivariate_normal
from scipy.stats import norm as univariate_normal
from .common_functionality import CreateIterators, IteratorData, AugmentData, split_data




from sklearn.preprocessing import StandardScaler



class SimpleClassificationDataset:

    def __init__(self, dataset_name, **params):
        self.batch_size = params['batch_size']
        self.dataset_name = dataset_name
        self.return_numpy_array = params['return_numpy_array']
        self.max_number_of_generated_examples = params['max_number_of_generated_examples']
        self.seed = params['seed']
        self.dataset_location = Path(params['dataset_location'])

        self.train_split = 0.80
        self.valid_split = 0.25

        self.X, self.y, self.s = np.load(self.dataset_location / Path('encoding_cls.npy')), \
            np.load(self.dataset_location / Path('labels.npy')).squeeze(), \
            np.load(self.dataset_location / Path('s.npy'))

        self.per_group_label_number_of_examples = params['per_group_label_number_of_examples']

        self.max_number_of_generated_examples = params['max_number_of_generated_examples']
        self.mmd_augmentation_mechanism = params['mmd_augmentation_mechanism']

        if "augmented" in self.dataset_name:
            self.positive_gen_model = params['positive_gen_model']
            self.negative_gen_model = params['negative_gen_model']

    def run(self):
        """Orchestrates the whole process"""
        X,y,s = self.X, self.y, self.s

        # the dataset is shuffled so as to get a unique test set for each seed.
        index, test_index, dev_index = split_data(X.shape[0], train_split=self.train_split, valid_split=self.valid_split)
        X, y, s = X[index], y[index], s[index]
        train_X, train_y, train_s = X[:dev_index, :], y[:dev_index], s[:dev_index]
        valid_X, valid_y, valid_s = X[dev_index:test_index, :], y[dev_index:test_index], s[dev_index:test_index]
        test_X, test_y, test_s = X[test_index:, :], y[test_index:], s[test_index:]

        scaler = StandardScaler().fit(train_X)
        train_X = scaler.transform(train_X)
        valid_X = scaler.transform(valid_X)
        test_X = scaler.transform(test_X)





        if "augmented" in self.dataset_name:
            augment_data = AugmentData(self.dataset_name, train_X, train_y, train_s,
                                       np.unique(valid_s, axis=0),
                                       self.max_number_of_generated_examples,
                                       max_number_of_positive_examples=self.per_group_label_number_of_examples,
                                       max_number_of_negative_examples=self.per_group_label_number_of_examples,
                                       mmd_augmentation_mechanism=self.mmd_augmentation_mechanism,
                                       seed=self.seed,
                                       positive_gen_model=self.positive_gen_model,
                                       negative_gen_model=self.negative_gen_model
                                       )

            if self.mmd_augmentation_mechanism == 'only_generated_data':
                train_X_augmented, train_y_augmented, train_s_augmented = augment_data.run()

            else:
                train_X, train_y, train_s = augment_data.run()
                train_X_augmented, train_y_augmented, train_s_augmented = train_X, train_y, train_s

        else:
            train_X_augmented, train_y_augmented, train_s_augmented = train_X, train_y, train_s



        # this is where the data augmentation can take place.

        create_iterator = CreateIterators()
        iterator_data = IteratorData(
            train_X=train_X, train_y=train_y, train_s=train_s,
            dev_X=valid_X, dev_y=valid_y, dev_s=valid_s,
            test_X=test_X, test_y=test_y, test_s=test_s,
            batch_size=self.batch_size,
            do_standard_scalar_transformation=False
        )
        iterator_set, vocab, s_flatten_lookup = create_iterator.get_iterators(iterator_data)

        if "augmented" in self.dataset_name:
            if self.mmd_augmentation_mechanism == 'only_generated_data':
                scaler = iterator_set['scaler']
                # train_X_augmented = scaler.transform(train_X_augmented)
        else:
            # scaler = iterator_set['scaler']
            train_X_augmented = scaler.transform(train_X_augmented)


        iterator_set['train_X_augmented'] = train_X_augmented
        iterator_set['train_y_augmented'] = train_y_augmented
        iterator_set['train_s_augmented'] = train_s_augmented

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






