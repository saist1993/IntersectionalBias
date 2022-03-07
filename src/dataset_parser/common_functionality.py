import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from utils.iterator import TextClassificationDataset, sequential_transforms


class CreateIterators:
    """ A general purpose iterators. Takes numpy matrix for train, dev, and test matrix and creates iterator."""

    @staticmethod
    def collate(batch):
        labels, input, aux = zip(*batch)

        labels = torch.LongTensor(labels)
        aux = torch.LongTensor(aux)
        lengths = torch.LongTensor([len(x) for x in input])
        input = torch.FloatTensor(input)

        input_data = {
            'labels': labels,
            'input': input,
            'lengths': lengths,
            'aux': aux
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

    def get_iterators(self, train_X: np.asarray, train_y: np.asarray, train_s: np.asarray,
                         dev_X: np.asarray, dev_y: np.asarray, dev_s: np.asarray,
                         test_X: np.asarray, test_y: np.asarray, test_s: np.asarray,
                         batch_size: int = 512,
                         do_standard_scalar_transformation: bool = True,
                         fairness_iterator_method: str = 'custom_3'):
        if do_standard_scalar_transformation:
            scaler = StandardScaler().fit(train_X)
            train_X = scaler.transform(train_X)
            dev_X = scaler.transform(dev_X)
            test_X = scaler.transform(test_X)

        vocab = {'<pad>': 1}  # no need of vocab in these dataset. It is there for code compatibility purposes.

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
        }

        return iterator_set, vocab