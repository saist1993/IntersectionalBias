import torch
import config
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler
from transformers import PreTrainedTokenizer, PreTrainedModel
from .common_functionality import CreateIterators, IteratorData, AugmentData, create_mask
from utils.encode_text_via_lm import load_lm, batch_tokenize, encode_text_batch


class CleanDataAndSaveEncodings:
    """Cleans the given tsv file (context of twitter hate speech and them saves the encodings."""

    def __init__(self, data_path):
        self.data_path = data_path

    @staticmethod
    def remove_x_from_data(df: pd.DataFrame) -> pd.DataFrame:
        """Remove columns from dataframe which are not useful for our purposes
         as we don't consider that specific attribute of the tweet """
        del df['city']  #
        del df['state']
        return df.dropna()

    @staticmethod
    def read_data_from_path(data_path: Path, split: str) -> pd.DataFrame:
        """
        In multilingual twitter dataset x represents that no information was available
        :param data_path: the actual path to the dataset
        :param split: train/valid/test
        :return: dataframe with dropped data
        """
        assert split in ['train', 'valid', 'test'], "only splits available are train, valid, and test"
        df = pd.read_csv(str(data_path / Path(f'{split}.tsv')), sep='\t', na_values='x')
        return df

    @staticmethod
    def get_encoding_avg_and_cls(sentences: List[str], model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> \
            Tuple[
                np.array, np.array]:
        """
        Reads from the data frame and create sentence representation using BERT embeddings.
        :param sentences: Sentences which needs to be encoded.
        :param tokenizer: Pretrained tokenizer which will tokenize the sentences.
        :param model: Pretrained model which will be used for encoding.
        :return: encoded sentence representation.
        """

        _batch_size = 20  # changing this does not get any speedup on this dataset.
        batched_and_tokenized_sentences = batch_tokenize(tokenizer=tokenizer,
                                                         sentences=sentences,
                                                         batch_size=_batch_size)
        encoding_avg, encoding_cls = encode_text_batch(model=model,
                                                       batched_tokenized_sentences=batched_and_tokenized_sentences,
                                                       device=torch.device('cpu'))

        return encoding_avg, encoding_cls

    @staticmethod
    def save_encoding(data_path: Path, model_name: str, encoding_avg: np.array, encoding_cls: np.array,
                      split: str) -> None:
        """
        :param model_name: Name of the model which is used for encoding.
        This name will be appended in the saved numpy array.
        :param data_path: The path where the numpy arrays will be stored.
        :param encoding_avg: Encoded average version of the sentences.
        :param encoding_cls: Encoded cls version of the sentences.
        :param split: Split information train/test/valid.
        :return: None
        """
        np.save(str(data_path / Path(split + f'{model_name}_encoding_avg')),
                encoding_avg)
        np.save(str(data_path / Path(split + f'{model_name}_encoding_cls')),
                encoding_cls)

    def parse_and_save_encodings_hate_speech(self):
        """Orchestrates the parsing, encoding, and saving"""

        model_type = 'bert-base-uncased'

        # Step 1: Read the dataset and store it as dataframe.
        train_df = self.read_data_from_path(data_path=self.data_path, split='train')
        valid_df = self.read_data_from_path(data_path=self.data_path, split='valid')
        test_df = self.read_data_from_path(data_path=self.data_path, split='test')

        # Step 2: Drop the irrelevant columns.
        train_df = self.remove_x_from_data(df=train_df)
        valid_df = self.remove_x_from_data(df=valid_df)
        test_df = self.remove_x_from_data(df=test_df)

        # Step 3: Load a language model and Tokenizer.
        model, tokenizer = load_lm(model_type=model_type)

        # Step 4: Get the avg and cls encodings for all the splits.
        train_encoding_avg, train_encoding_cls = \
            self.get_encoding_avg_and_cls(sentences=train_df['text'].to_list(), model=model, tokenizer=tokenizer)
        valid_encoding_avg, valid_encoding_cls = \
            self.get_encoding_avg_and_cls(sentences=valid_df['text'].to_list(), model=model, tokenizer=tokenizer)
        test_encoding_avg, test_encoding_cls = \
            self.get_encoding_avg_and_cls(sentences=test_df['text'].to_list(), model=model, tokenizer=tokenizer)

        # Step 5: Save the encodings.
        self.save_encoding(data_path=self.data_path, model_name=model_type,
                           encoding_avg=train_encoding_avg, encoding_cls=train_encoding_cls, split='train')
        self.save_encoding(data_path=self.data_path, model_name=model_type,
                           encoding_avg=valid_encoding_avg, encoding_cls=valid_encoding_cls, split='valid')
        self.save_encoding(data_path=self.data_path, model_name=model_type,
                           encoding_avg=test_encoding_avg, encoding_cls=test_encoding_cls, split='test')


class DatasetTwitterHateSpeech:

    def __init__(self, dataset_name, **params):
        self.batch_size = params['batch_size']
        self.dataset_name = dataset_name
        self.dataset_location = Path(params['dataset_location'])
        self.lm_encoder_type = params['lm_encoder_type']
        self.lm_encoding_to_use = params['lm_encoding_to_use']  # use_cls, or use_avg
        self.return_numpy_array = params['return_numpy_array']
        self.dataset_helper = CleanDataAndSaveEncodings(data_path=self.dataset_location)
        self.train_df = None
        self.valid_df = None
        self.test_df = None
        self.train_encodings = None
        self.valid_encodings = None
        self.test_encodings = None
        self.per_group_label_number_of_examples = params['per_group_label_number_of_examples']

        self.max_number_of_generated_examples = params['max_number_of_generated_examples']

    def _check_if_encodings_are_present(self):
        location_avg = self.dataset_location / Path('train' + f'{self.lm_encoder_type}_encoding_avg.npy')
        location_cls = self.dataset_location / Path('train' + f'{self.lm_encoder_type}_encoding_cls.npy')
        return location_avg.is_file() and location_cls.is_file()

    def create_encodings(self):
        # Check if the encodings exists, and if not: run the parse and save encodings hate speech.
        if not self._check_if_encodings_are_present():
            self.dataset_helper.parse_and_save_encodings_hate_speech()

    def set_encodings(self):
        # check which kind of encoding to use.
        if self.lm_encoding_to_use == 'use_cls':
            encoding_type = 'cls'
        elif self.lm_encoding_to_use == 'use_avg':
            encoding_type = 'avg'
        else:
            raise NotImplementedError

        # Set the encodings.
        self.train_encodings = np.load(
            str(self.dataset_location / Path('train' + f'{self.lm_encoder_type}_encoding_{encoding_type}' + '.npy')))
        self.valid_encodings = np.load(
            str(self.dataset_location / Path('valid' + f'{self.lm_encoder_type}_encoding_{encoding_type}' + '.npy')))
        self.test_encodings = np.load(
            str(self.dataset_location / Path('test' + f'{self.lm_encoder_type}_encoding_{encoding_type}' + '.npy')))

    def set_dataframes(self):
        # Set the data frame.
        self.train_df = self.dataset_helper.remove_x_from_data(
            self.dataset_helper.read_data_from_path(data_path=self.dataset_location, split='train'))
        self.valid_df = self.dataset_helper.remove_x_from_data(
            self.dataset_helper.read_data_from_path(data_path=self.dataset_location, split='valid'))
        self.test_df = self.dataset_helper.remove_x_from_data(
            self.dataset_helper.read_data_from_path(data_path=self.dataset_location, split='test'))

    @staticmethod
    def get_private_attribute(df: pd.DataFrame) -> np.array:
        return np.asarray(df[['gender', 'age', 'country', 'ethnicity']].to_numpy(), dtype=int)

    @staticmethod
    def get_label(df: pd.DataFrame) -> np.array:
        return np.asarray(df['label'].to_numpy(), dtype=int)

    def run(self):
        """Orchestrates the whole process"""

        # Step1: Set the encodings and data frames.
        self.set_dataframes()
        self.create_encodings()
        self.set_encodings()

        # Step2: Get the data into numpy format.
        train_X, train_s, train_y = \
            self.train_encodings, self.get_private_attribute(self.train_df), self.get_label(self.train_df)

        valid_X, valid_s, valid_y = \
            self.valid_encodings, self.get_private_attribute(self.valid_df), self.get_label(self.valid_df)
        test_X, test_s, test_y = \
            self.test_encodings, self.get_private_attribute(self.test_df), self.get_label(self.test_df)

        if self.dataset_name == 'twitter_hate_speech_v1':
            train_s, valid_s, test_s = train_s[:, :1], valid_s[:, :1], test_s[:, :1]

        if self.dataset_name == 'twitter_hate_speech_v2':
            train_s, valid_s, test_s = train_s[:, :2], valid_s[:, :2], test_s[:, :2]

        if self.dataset_name == 'twitter_hate_speech_v3':
            train_s, valid_s, test_s = train_s[:, :3], valid_s[:, :3], test_s[:, :3]

        # now we add or remove some data!

        all_groups = np.unique(test_s, axis=0)
        all_labels = np.unique(test_y)

        minimum_group_size_test = 100
        minimum_group_size_valid = 100
        new_test_examples_index = []

        for group in all_groups:
            for label in all_labels:
                group_mask = create_mask(test_s, group)
                label_mask = test_y == label
                final_mask = np.logical_and(group_mask, label_mask)
                if np.sum(final_mask) < minimum_group_size_test:
                    # new_test_examples.append((group, label, minimum_group_size_test-np.sum(final_mask)))

                    train_group_mask = create_mask(train_s, group)
                    train_label_mask = train_y == label
                    final_train_mask = np.logical_and(train_group_mask, train_label_mask)


                    new_test_examples_index += np.random.choice(np.where(final_train_mask == True)[0],
                                                                size=minimum_group_size_test-np.sum(final_mask),
                                                                replace=False).tolist()

        test_X = np.vstack([test_X, train_X[new_test_examples_index]])
        test_y = np.hstack([test_y, train_y[new_test_examples_index]])
        test_s = np.vstack([test_s, train_s[new_test_examples_index]])

        train_X = np.delete(train_X, new_test_examples_index, axis=0)
        train_s = np.delete(train_s, new_test_examples_index, axis=0)
        train_y = np.delete(train_y, new_test_examples_index, axis=0)

        scaler = StandardScaler().fit(train_X)
        train_X = scaler.transform(train_X)
        valid_X = scaler.transform(valid_X)
        test_X = scaler.transform(test_X)

        if "augmented" in self.dataset_name:
            augment_data = AugmentData(self.dataset_name, train_X, train_y, train_s,
                                       self.max_number_of_generated_examples,
                                       max_number_of_positive_examples=self.per_group_label_number_of_examples,
                                       max_number_of_negative_examples=self.per_group_label_number_of_examples)
            train_X_augmented, train_y_augmented, train_s_augmented = augment_data.run()

        # Step3: Create iterators - This can be abstracted out to dataset iterators.
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
            iterator_set['scaler'] = scaler
            train_X_augmented = scaler.transform(train_X_augmented)
            iterator_set['train_X_augmented'] = train_X_augmented
            iterator_set['train_y_augmented'] = train_y_augmented
            iterator_set['train_s_augmented'] = train_s_augmented

        iterators = [iterator_set]  # If it was k-fold. One could append k iterators here.

        iterators.append(iterator_set)

        other_meta_data = {}
        other_meta_data['task'] = 'simple_classification'
        other_meta_data['dataset_name'] = self.dataset_name
        other_meta_data['number_of_main_task_label'] = len(np.unique(train_y))
        other_meta_data['number_of_aux_label_per_attribute'] = [len(np.unique(train_s[:, i])) for i in
                                                                range(train_s.shape[1])]
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
                'test_s': test_s,
            }
            other_meta_data['raw_data'] = raw_data

        return iterators, other_meta_data


def test_batch_shape_hate_speech():
    params = {
        'batch_size': 64,
        'dataset_location': config.hate_speech_dataset_path[0],
        'lm_encoder_type': 'bert-base-uncased',
        'lm_encoding_to_use': 'use_cls',
        'return_numpy_array': False
    }

    dataset_twitter_runner = DatasetTwitterHateSpeech('twitter_hate_speech', **params)
    iterators, other_meta_data = dataset_twitter_runner.run()
    print(iterators)
    for item in iterators[0]['train_iterator']:
        label_shape = item['labels'].shape
        input_shape = item['input'].shape
        aux_shape = item['aux'].shape

    assert label_shape == torch.Size([64]), "there is a shape mis match in label"
    assert input_shape == torch.Size([64, 768]), "there is a shape mis match in input"
    assert aux_shape == torch.Size([64, 4]), "there is a shape mis match in aux"


if __name__ == '__main__':

    params = {
        'batch_size': 64,
        'dataset_location': config.hate_speech_dataset_path[0],
        'lm_encoder_type': 'bert-base-uncased',
        'lm_encoding_to_use': 'use_cls',
        'return_numpy_array': False
    }

    dataset_twitter_runner = DatasetTwitterHateSpeech('twitter_hate_speech', **params)
    iterators, other_meta_data = dataset_twitter_runner.run()
    print(iterators)
    for item in iterators[0]['train_iterator']:
        print(item['labels'].shape)
        print(item['input'].shape)
        print(item['aux'].shape)
        break
