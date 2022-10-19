import torch
import numpy as np
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


    def flatten_s(self, s:List[List]):
        for f in s:
            keys = self.s_to_flattened_s.keys()
            if tuple([int(i) for i in f]) not in keys:
                self.s_to_flattened_s[tuple([int(i) for i in f])] = len(keys)


    def get_flatten_s(self, s:List[List]):
        return [self.s_to_flattened_s[tuple([int(j) for j in i])] for i in s]


    def get_iterators(self, iterator_data:IteratorData):
        train_X, train_y, train_s, dev_X, \
        dev_y, dev_s, test_X, test_y, test_s,batch_size,do_standard_scalar_transformation= iterator_data
        self.flatten_s(train_s) # need to add test as well as valid
        self.flatten_s(test_s) # need to add test as well as valid
        self.flatten_s(dev_s) # need to add test as well as valid
        if do_standard_scalar_transformation:
            scaler = StandardScaler().fit(train_X)
            train_X = scaler.transform(train_X)
            dev_X = scaler.transform(dev_X)
            test_X = scaler.transform(test_X)

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
            'test_iterator': test_iterator
        }

        return iterator_set, vocab, self.s_to_flattened_s