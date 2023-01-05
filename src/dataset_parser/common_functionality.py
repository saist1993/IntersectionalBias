import copy
import torch
import numpy as np
import pandas as pd
from scipy import optimize
from metrics import fairness_utils
from itertools import combinations
from typing import NamedTuple, List
from sklearn.preprocessing import StandardScaler
from utils.iterator import TextClassificationDataset, sequential_transforms


def split_data(dataset_size, train_split, valid_split):
    """
    :param dataset_size: The total size of the dataset
    :param train_split: The total train split, remaining acts as test split
    :param valid_split: size of validation split wrt to train split -> 100*train_split*validation_split
    :return: indexes

    The splits are: dataset[:dev_index], dataset[dev_index:test_index], dataset[test_index:]
    """
    index = np.random.permutation(dataset_size)
    test_index = int(train_split * dataset_size)
    dev_index = int(train_split * dataset_size) - int(train_split * dataset_size * valid_split)

    return index, test_index, dev_index


class IteratorData(NamedTuple):
    """Input data to create iterators"""
    train_X: np.asarray
    train_y: np.asarray
    train_s: np.asarray
    dev_X: np.asarray
    dev_y: np.asarray
    dev_s: np.asarray
    test_X: np.asarray
    test_y: np.asarray
    test_s: np.asarray
    batch_size: int = 512
    do_standard_scalar_transformation: bool = True


class CreateIterators:
    """ A general purpose iterators. Takes numpy matrix for train, dev, and test matrix and creates iterator."""

    def __init__(self):
        self.s_to_flattened_s = {}

    def collate(self, batch):
        labels, encoded_input, aux = zip(*batch)

        labels = torch.LongTensor(labels)
        aux_flattened = torch.LongTensor(self.get_flatten_s(aux))
        aux = torch.LongTensor(np.asarray(aux))
        lengths = torch.LongTensor([len(x) for x in encoded_input])
        encoded_input = torch.FloatTensor(np.asarray(encoded_input))

        input_data = {
            'labels': labels,
            'input': encoded_input,
            'lengths': lengths,
            'aux': aux,
            'aux_flattened': aux_flattened
        }

        return input_data

    def process_data(self, X, y, s, vocab):
        """raw data is assumed to be tokenized"""

        final_data = [(a, b, c) for a, b, c in zip(y, X, s)]

        label_transform = sequential_transforms()
        input_transform = sequential_transforms()
        aux_transform = sequential_transforms()

        transforms = (label_transform, input_transform, aux_transform)

        return TextClassificationDataset(final_data, vocab, transforms)

    def flatten_s(self, s: List[List]):
        for f in s:
            keys = self.s_to_flattened_s.keys()
            if tuple([int(i) for i in f]) not in keys:
                self.s_to_flattened_s[tuple([int(i) for i in f])] = len(keys)

    def get_flatten_s(self, s: List[List]):
        return [self.s_to_flattened_s[tuple([int(j) for j in i])] for i in s]

    def get_iterators(self, iterator_data: IteratorData):
        train_X, train_y, train_s, dev_X, \
            dev_y, dev_s, test_X, test_y, test_s, batch_size, do_standard_scalar_transformation = iterator_data
        self.flatten_s(train_s)  # need to add test as well as valid
        self.flatten_s(test_s)  # need to add test as well as valid
        self.flatten_s(dev_s)  # need to add test as well as valid
        if do_standard_scalar_transformation:
            scaler = StandardScaler().fit(train_X)
            train_X = scaler.transform(train_X)
            dev_X = scaler.transform(dev_X)
            test_X = scaler.transform(test_X)
        else:
            scalar = None

        vocab = {'<pad>': 1}  # no need of vocab in these dataset. It is there for code compatibility purposes.

        # need to add flatten here! And that too in the process itself!
        train_data = self.process_data(train_X, train_y, train_s, vocab=vocab)
        dev_data = self.process_data(dev_X, dev_y, dev_s, vocab=vocab)
        test_data = self.process_data(test_X, test_y, test_s, vocab=vocab)

        train_iterator = torch.utils.data.DataLoader(train_data,
                                                     batch_size,
                                                     shuffle=False,
                                                     collate_fn=self.collate
                                                     )

        dev_iterator = torch.utils.data.DataLoader(dev_data,
                                                   512,
                                                   shuffle=False,
                                                   collate_fn=self.collate
                                                   )

        test_iterator = torch.utils.data.DataLoader(test_data,
                                                    512,
                                                    shuffle=False,
                                                    collate_fn=self.collate
                                                    )

        iterator_set = {
            'train_iterator': train_iterator,
            'valid_iterator': dev_iterator,
            'test_iterator': test_iterator,
            'scalar': scaler
        }

        return iterator_set, vocab, self.s_to_flattened_s


class AugmentDataCommonFunctionality:
    @staticmethod
    def generate_combinations_only_leaf_node(s, k=1):
        all_s_combinations = []

        for i in combinations(range(len(s)), k):
            _temp = list(copy.deepcopy(s))
            for j in i:
                if _temp[j] == 1:
                    _temp[j] = 0.0
                else:
                    _temp[j] = 1.0
            all_s_combinations.append(tuple(_temp))

        return all_s_combinations

    @staticmethod
    def get_all_representation_positive_negative_seperate_only_leaf_node(df, s):
        s_abstract = AugmentDataCommonFunctionality.generate_combinations_only_leaf_node(list(s))
        s = tuple(s)
        s_abstract.insert(0, s)
        all_representation_positive = [df.loc[df['group_pattern'] == _s]['average_representation_positive'].item() for
                                       _s in s_abstract]
        all_representation_negative = [df.loc[df['group_pattern'] == _s]['average_representation_negative'].item() for
                                       _s in s_abstract]
        return all_representation_positive, all_representation_negative, s_abstract

    @staticmethod
    def get_all_representation(df, s):
        s_abstract = AugmentDataCommonFunctionality.get_all_representation_positive_negative_seperate_only_leaf_node(df,
                                                                                                                     list(
                                                                                                                         s))
        s_abstract.insert(0, s)
        all_representation = [df.loc[df['group_pattern'] == _s]['average_representation'].item() for _s in s_abstract]
        return all_representation

    @staticmethod
    def generate_mask(all_s, mask_pattern):
        keep_indices = []

        for index, i in enumerate(mask_pattern):
            if i != 'x':
                keep_indices.append(i == all_s[:, index])
            else:
                keep_indices.append(np.ones_like(all_s[:, 0], dtype='bool'))

        mask = np.ones_like(all_s[:, 0], dtype='bool')

        # mask = [np.logical_and(mask, i) for i in keep_indices]

        for i in keep_indices:
            mask = np.logical_and(mask, i)
        return mask

    @staticmethod
    def create_group_to_lambda_weight_seperate_positive_negative(other_meta_data):
        all_label, all_s, all_input = other_meta_data['raw_data']['train_y'], other_meta_data['raw_data']['train_s'], \
            other_meta_data['raw_data']['train_X']
        # all_possible_groups = fairness_utils.create_all_possible_groups(
        #     attributes=[list(np.unique(all_s[:, i])) for i in range(all_s.shape[1])])
        # all_leaf_group = [i for i in all_possible_groups if "x" not in i]
        all_unique_groups = np.unique(all_s, axis=0)
        all_leaf_group = np.unique(all_s, axis=0)

        row_header = ['group_pattern', 'size', 'label==1', 'label==0', 'average_representation',
                      'average_representation_positive', 'average_representation_negative']

        rows = []

        for unique_group in all_leaf_group:
            mask = AugmentDataCommonFunctionality.generate_mask(all_s, unique_group)
            size = np.sum(mask)
            train_1 = np.sum(all_label[mask] == 1)
            train_0 = np.sum(all_label[mask] == 0)
            positive_mask = np.logical_and(mask, all_label == 1)
            negative_mask = np.logical_and(mask, all_label == 0)
            average_representation_positive = np.mean(all_input[positive_mask], axis=0)
            average_representation_negative = np.mean(all_input[negative_mask], axis=0)
            average_representation = np.mean(all_input[mask], axis=0)
            rows.append([tuple(unique_group), size, train_1, train_0, average_representation,
                         average_representation_positive, average_representation_negative])

        df = pd.DataFrame(rows, columns=row_header)

        group_to_lambda_weights = {}
        '''
        group_to_lambda_weights: {
                (0,0,0): {
                1: [((0,0,1),2.1), ((0,1,0),2.0), ((1,0,0),0.1)],
                0: [((0,0,1),2.1), ((0,1,0),2.0), ((1,0,0),0.1)]
            }
            ....
            }
        '''
        for unique_group in all_unique_groups:
            unique_group = tuple(list(unique_group))
            all_representation_positive, all_representation_negative, s_abstract = \
                AugmentDataCommonFunctionality.get_all_representation_positive_negative_seperate_only_leaf_node(
                    df, unique_group)

            def find_weights(all_representation):
                # P = np.matrix([all_representation[1], all_representation[2], all_representation[3]])
                P = np.matrix(all_representation[1:])
                Ps = np.array(all_representation[0])

                def objective(x):
                    x = np.array([x])
                    res = Ps - np.dot(x, P)
                    return np.asarray(res).flatten()

                def main():
                    x = np.array([1 for i in range(len(P))] / np.sum([1 for i in range(len(P))]))
                    try:
                        final_lambda_weights = optimize.least_squares(objective, x).x
                        return final_lambda_weights
                    except ValueError:
                        print("something broke!")

                return main()

            if True:
                s_abstract_positive = []
                s_abstract_negative =  []

                new_all_representation_positive = []
                new_all_representation_negative = []

                for i,representation in zip(s_abstract[1:],all_representation_positive[1:]):
                    if np.sum(all_label[AugmentDataCommonFunctionality.generate_mask(all_s, i)] == 1) > 183:
                        new_all_representation_positive.append(representation)
                        s_abstract_positive.append(i)

                for i,representation in zip(s_abstract[1:],all_representation_negative[1:]):
                    if np.sum(all_label[AugmentDataCommonFunctionality.generate_mask(all_s, i)] == 0) > 200:
                        new_all_representation_negative.append(representation)
                        s_abstract_negative.append(i)

                s_abstract_positive.insert(0, s_abstract[0])
                s_abstract_negative.insert(0, s_abstract[0])
                new_all_representation_positive.insert(0, all_representation_positive[0])
                new_all_representation_negative.insert(0, all_representation_negative[0])

                key = tuple([int(i) for i in list(unique_group)])
                positive_weight = find_weights(new_all_representation_positive)
                negative_weight = find_weights(new_all_representation_negative)
                group_to_lambda_weights[key] = {
                    1: [(i, j) for i, j in zip(s_abstract_positive[1:], positive_weight)],
                    0: [(i, j) for i, j in zip(s_abstract_negative[1:], negative_weight)]
                }

        return group_to_lambda_weights


    @staticmethod
    def generate_examples(s, group_to_lambda_weights, number_of_examples, other_meta_data):
        positive_weights, negative_weights = group_to_lambda_weights[s][1], group_to_lambda_weights[s][0]
        all_input = other_meta_data['raw_data']['train_X']

        all_group_examples_label_1 = []
        all_group_examples_label_0 = []

        for leaf_node, weight in positive_weights:
            group_mask = AugmentDataCommonFunctionality.generate_mask(other_meta_data['raw_data']['train_s'], leaf_node)
            label_1_group_mask = np.logical_and(group_mask, other_meta_data['raw_data']['train_y'] == 1)
            label_1_examples = np.random.choice(np.where(label_1_group_mask == True)[0], size=number_of_examples,
                                                replace=True)

            all_group_examples_label_1.append(label_1_examples)

        for leaf_node, weight in negative_weights:
            group_mask = AugmentDataCommonFunctionality.generate_mask(other_meta_data['raw_data']['train_s'], leaf_node)
            label_0_group_mask = np.logical_and(group_mask, other_meta_data['raw_data']['train_y'] == 0)
            label_0_examples = np.random.choice(np.where(label_0_group_mask == True)[0], size=number_of_examples,
                                                replace=True)

            all_group_examples_label_0.append(label_0_examples)

        augmented_input_positive = np.sum([weight * all_input[indexes] for (group_info, weight), indexes
                                           in zip(positive_weights, all_group_examples_label_1)], axis=0)

        augmented_input_negative = np.sum([weight * all_input[indexes] for (group_info, weight), indexes
                                           in zip(negative_weights, all_group_examples_label_0)], axis=0)

        return augmented_input_positive, augmented_input_negative


class AugmentData:
    """A static data augmentation mechanism. Currently not very general purpose"""

    def __init__(self, dataset_name, X, y, s):
        self.dataset_name = dataset_name
        self.X, self.y, self.s = X, y, s

        # formating data in a specific way for legacy purpose!
        self.other_meta_data = {
            'raw_data': {
                'train_X': self.X,
                'train_y': self.y,
                'train_s': self.s
            }
        }

        self.common_func = AugmentDataCommonFunctionality()

    def run(self):
        if 'adult_multi_group' in self.dataset_name:
            train_X, train_y, train_s = self.data_augmentation_for_adult_multi_group()
        else:
            raise NotImplementedError

        return train_X, train_y, train_s

    def data_augmentation_for_adult_multi_group(self):

        group_to_lambda_weights = self.common_func.create_group_to_lambda_weight_seperate_positive_negative(
            self.other_meta_data)

        all_unique_groups = np.unique(self.other_meta_data['raw_data']['train_s'], axis=0)

        max_number_of_positive_examples = 1000
        max_number_of_negative_examples = 1000
        max_ratio_of_generated_examples = 1.0

        augmented_train_X, augmented_train_y, augmented_train_s = [], [], []

        for group in all_unique_groups:
            group_mask = self.common_func.generate_mask(self.other_meta_data['raw_data']['train_s'], group)
            label_1_group_mask = np.logical_and(group_mask, self.other_meta_data['raw_data']['train_y'] == 1)
            label_0_group_mask = np.logical_and(group_mask, self.other_meta_data['raw_data']['train_y'] == 0)
            total_positive_examples = np.sum(label_1_group_mask)
            total_negative_examples = np.sum(group_mask) - total_positive_examples

            def sub_routine(label_mask, total_examples, max_number_of_examples, example_type):
                if total_examples > max_number_of_examples:
                    index_of_selected_examples = np.random.choice(np.where(label_mask == True)[0],
                                                                  size=max_number_of_examples,
                                                                  replace=False)  # sample max number of positive examples

                    augmented_train_X.append(self.other_meta_data['raw_data']['train_X'][index_of_selected_examples])
                    augmented_train_y.append(self.other_meta_data['raw_data']['train_y'][index_of_selected_examples])
                    augmented_train_s.append(self.other_meta_data['raw_data']['train_s'][index_of_selected_examples])
                else:
                    number_of_examples_to_generate = int(min(max_number_of_examples - total_examples,
                                                             max_ratio_of_generated_examples * total_examples))
                    index_of_selected_examples = np.random.choice(np.where(label_mask == True)[0],
                                                                  size=max_number_of_examples - number_of_examples_to_generate,
                                                                  replace=True)  # sample remaining
                    # now generate remaining examples!
                    if example_type == 'positive':
                        augmented_input, _ = self.common_func.generate_examples(tuple(group), group_to_lambda_weights,
                                                                                number_of_examples_to_generate,
                                                                                self.other_meta_data)
                    elif example_type == 'negative':
                        _, augmented_input = self.common_func.generate_examples(tuple(group), group_to_lambda_weights,
                                                                                number_of_examples_to_generate,
                                                                                self.other_meta_data)
                    else:
                        raise NotImplementedError

                    augmented_train_X.append(
                        np.vstack((self.other_meta_data['raw_data']['train_X'][index_of_selected_examples],
                                   augmented_input)))

                    # this is a hack. All examples for this group would have same y and s and thus it works
                    index_of_selected_examples = np.random.choice(np.where(label_mask == True)[0],
                                                                  size=max_number_of_examples,
                                                                  replace=True)
                    augmented_train_y.append(self.other_meta_data['raw_data']['train_y'][index_of_selected_examples])
                    augmented_train_s.append(self.other_meta_data['raw_data']['train_s'][index_of_selected_examples])

            sub_routine(label_mask=label_1_group_mask, total_examples=total_positive_examples,
                        max_number_of_examples=max_number_of_positive_examples, example_type="positive")

            sub_routine(label_mask=label_0_group_mask, total_examples=total_negative_examples,
                        max_number_of_examples=max_number_of_negative_examples, example_type='negative')

        augmented_train_X = np.vstack(augmented_train_X)
        augmented_train_s = np.vstack(augmented_train_s)
        augmented_train_y = np.hstack(augmented_train_y)

        return augmented_train_X, augmented_train_y, augmented_train_s
