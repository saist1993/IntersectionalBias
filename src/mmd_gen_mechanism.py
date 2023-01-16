import copy
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from scipy import optimize
from functools import partial
from collections import Counter
import matplotlib.pyplot as plt
from dataclasses import dataclass
from itertools import combinations
from metrics import fairness_utils
from typing import Dict, Callable, Optional
from dataset_iterators import generate_data_iterators
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from utils.misc import resolve_device, set_seed, make_opt, CustomError
from training_loops.dro_and_erm import group_sampling_procedure_func, create_group, example_sampling_procedure_func


dataset_name = 'adult_multi_group'
batch_size = 512


class SimpleModelGenerator(nn.Module):
    """Fairgrad uses this as complex non linear model"""

    def __init__(self, input_dim):
        super().__init__()

        self.layer_1 = nn.Linear(input_dim, 128)
        self.layer_2 = nn.Linear(128, input_dim)
        self.relu = nn.ReLU()

    def forward(self, other_examples):
        final_output = torch.tensor(0.0, requires_grad=True)
        for group in other_examples:
            x = group['input']
            x = self.layer_1(x)
            x = self.relu(x)
            x = self.layer_2(x)
            final_output = final_output + x

        output = {
            'prediction': final_output,
            'adv_output': None,
            'hidden': x,  # just for compatabilit
            'classifier_hiddens': None,
            'adv_hiddens': None
        }

        return output

    @property
    def layers(self):
        return torch.nn.ModuleList([self.layer_1, self.layer_2])


class SimpleClassifier(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.layer_1 = nn.Linear(input_dim, 256)
        self.layer_2 = nn.Linear(256, output_dim)
        self.relu = nn.ReLU()

    def forward(self, params):
        input = params['input']
        x = self.layer_1(input)
        x = self.relu(x)
        output = self.layer_2(x)

        return output

    @property
    def layers(self):
        return torch.nn.ModuleList([self.layer_1, self.layer_2])


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss


class AuxilaryFunction:

    @staticmethod
    def generate_abstract_node(s, k=1):
        all_s_combinations = []

        for i in combinations(range(len(s)), k):
            _temp = list(copy.deepcopy(s))
            for j in i:
                _temp[j] = 'x'
            all_s_combinations.append(tuple(_temp))

        return all_s_combinations

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
    def size_of_group(other_meta_data):
        labels = [value for key, value in other_meta_data['s_flatten_lookup'].items()]
        reverse_flatten = {value: str(key) for key, value in other_meta_data['s_flatten_lookup'].items()}

        flattened_s = np.asarray(
            [other_meta_data['s_flatten_lookup'][tuple(i)] for i in other_meta_data['raw_data']['train_s']])
        mask_y_0 = other_meta_data['raw_data']['train_y'] == 0
        mask_y_1 = other_meta_data['raw_data']['train_y'] == 1
        s_y_0 = Counter(flattened_s[mask_y_0])
        s_y_1 = Counter(flattened_s[mask_y_1])

        return s_y_0, s_y_1

    @staticmethod
    def create_mask_with_x(data, condition):

        keep_indices = []

        for index, i in enumerate(condition):
            if i != 'x':
                keep_indices.append(i == data[:, index])
            else:
                keep_indices.append(np.ones_like(data[:, 0], dtype='bool'))

        mask = np.ones_like(data[:, 0], dtype='bool')

        # mask = [np.logical_and(mask, i) for i in keep_indices]

        for i in keep_indices:
            mask = np.logical_and(mask, i)
        return mask


    @staticmethod
    def custom_sample_data(group, all_label, all_input, all_aux, all_aux_flatten, number_of_positive_examples,
                           number_of_negative_examples):
        '''The group would have x - (1,1,X)'''

        group_mask = AuxilaryFunction.create_mask_with_x(data=all_aux, condition=group)
        positive_index = np.where(np.logical_and(all_label == 1, group_mask == True))[0]
        negative_index = np.where(np.logical_and(all_label == 0, group_mask == True))[0]
        negative_index = np.random.choice(negative_index, size=number_of_negative_examples, replace=True).tolist()
        positive_index = np.random.choice(positive_index, size=number_of_positive_examples, replace=True).tolist()

        batch_input_negative = {
            'labels': torch.LongTensor(all_label[negative_index]),
            'input': torch.FloatTensor(all_input[negative_index]),
            'aux': torch.LongTensor(all_aux[negative_index]),
            'aux_flattened': torch.LongTensor(all_aux_flatten[negative_index])
        }

        batch_input_positive = {
            'labels': torch.LongTensor(all_label[positive_index]),
            'input': torch.FloatTensor(all_input[positive_index]),
            'aux': torch.LongTensor(all_aux[positive_index]),
            'aux_flattened': torch.LongTensor(all_aux_flatten[positive_index])
        }

        return batch_input_negative, batch_input_positive

    @staticmethod
    def sample_batch(current_group):
        # other_leaf_group = [train_tilted_params.other_params['s_to_flattened_s'][i] for i in generate_combinations_only_leaf_node(flattened_s_to_s[current_group], k=1)]
        other_leaf_group = [i for i in AuxilaryFunction.generate_abstract_node(flattened_s_to_s[current_group], k=1)]

        examples_current_group, _ = example_sampling_procedure_func(
            train_tilted_params=train_tilted_params,
            group0=current_group,
            group1=None
        )

        index = int(train_tilted_params.other_params['batch_size'] / 2)

        negative_examples_current_group = {
            'labels': torch.LongTensor(examples_current_group['labels'][:index]),
            'input': torch.FloatTensor(examples_current_group['input'][:index]),
            'aux': torch.LongTensor(examples_current_group['aux'][:index]),
            'aux_flattened': torch.LongTensor(examples_current_group['aux_flattened'][:index])
        }

        positive_examples_current_group = {
            'labels': torch.LongTensor(examples_current_group['labels'][index:]),
            'input': torch.FloatTensor(examples_current_group['input'][index:]),
            'aux': torch.LongTensor(examples_current_group['aux'][index:]),
            'aux_flattened': torch.LongTensor(examples_current_group['aux_flattened'][index:])
        }

        examples_other_leaf_group_negative, examples_other_leaf_group_positive = [], []

        for group in other_leaf_group:
            #             examples, _ = example_sampling_procedure_func(
            #                 train_tilted_params=train_tilted_params,
            #                 group0=group,
            #                 group1=None
            #             )

            #             examples_other_leaf_group.append(examples)

            batch_input_negative, batch_input_positive = AuxilaryFunction.custom_sample_data(group=group,
                                                                            all_label=all_label,
                                                                            all_input=all_input,
                                                                            all_aux=all_aux,
                                                                            all_aux_flatten=all_aux_flatten,
                                                                            number_of_positive_examples=int(
                                                                                train_tilted_params.other_params[
                                                                                    'batch_size'] / 2),
                                                                            number_of_negative_examples=int(
                                                                                train_tilted_params.other_params[
                                                                                    'batch_size'] / 2))

            examples_other_leaf_group_negative.append(batch_input_negative)
            examples_other_leaf_group_positive.append(batch_input_positive)

        return negative_examples_current_group, positive_examples_current_group, examples_other_leaf_group_negative, examples_other_leaf_group_positive

class TestGeneratedSamples:

    def __init__(self, iterators, other_meta_data):
        self.iterators = iterators
        self.other_meta_data = other_meta_data
        self.single_attribute_reco_models = []
        self.all_attribute_reco_models = None
        self.number_of_attributes = other_meta_data['raw_data']['train_s'].shape[1]

    def run(self):
        self.create_and_learn_all_attribute_model()
        self.create_and_learn_single_attribute_models()

    def create_and_learn_single_attribute_models(self):
        number_of_attributes = other_meta_data['raw_data']['train_s'].shape[1]
        self.number_of_attributes = number_of_attributes
        for k in range(number_of_attributes):

            print(f"learning for attribute {k}")
            input_dim = other_meta_data['raw_data']['train_X'].shape[1]
            output_dim = 2
            single_attribute_classifier = SimpleClassifier(input_dim=input_dim, output_dim=output_dim)

            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(single_attribute_classifier.parameters(), lr=1e-3)

            for e in tqdm(range(10)):
                average_loss = []
                for items in (self.iterators[0]['train_iterator']):
                    optimizer.zero_grad()
                    prediction = single_attribute_classifier(items).squeeze()
                    attribute_label = items['aux'][:, k]
                    loss = loss_fn(prediction, attribute_label)
                    loss.backward()
                    optimizer.step()
                    average_loss.append(loss.data)

            with torch.no_grad():

                def common_sub_routine(iterator):
                    all_predictions = []
                    all_label = []
                    for items in iterator:
                        prediction = single_attribute_classifier(items)
                        all_predictions.append(prediction)
                        all_label.append(items['aux'][:, k])

                    all_predictions = np.vstack(all_predictions)
                    all_label = np.hstack(all_label)

                    return all_predictions, all_label

                all_predictions_train, all_label_train = common_sub_routine(iterators[0]['train_iterator'])
                all_predictions_test, all_label_test = common_sub_routine(iterators[0]['test_iterator'])

                print(f"{k} Balanced test accuracy: {balanced_accuracy_score(all_label_test, all_predictions_test.argmax(axis=1))}"
                      f" and unbalanced test accuracy: {accuracy_score(all_label_test, all_predictions_test.argmax(axis=1))}" )

            self.single_attribute_reco_models.append(single_attribute_classifier)

    def create_and_learn_all_attribute_model(self):
        input_dim = other_meta_data['raw_data']['train_X'].shape[1]
        output_dim = len(other_meta_data['s_flatten_lookup']) # this needs to change
        all_attribute_classifier = SimpleClassifier(input_dim=input_dim, output_dim=output_dim)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(all_attribute_classifier.parameters(), lr=1e-3)

        for e in tqdm(range(10)):
            average_loss = []
            for items in (self.iterators[0]['train_iterator']):
                optimizer.zero_grad()
                prediction = all_attribute_classifier(items).squeeze()
                attribute_label = items['aux_flattened']
                loss = loss_fn(prediction, attribute_label)
                loss.backward()
                optimizer.step()
                average_loss.append(loss.data)

        with torch.no_grad():

            def common_sub_routine(iterator):
                all_predictions = []
                all_label = []
                for items in iterator:
                    prediction = all_attribute_classifier(items)
                    all_predictions.append(prediction)
                    all_label.append(items['aux_flattened'])

                all_predictions = np.vstack(all_predictions)
                all_label = np.hstack(all_label)

                return all_predictions, all_label

            all_predictions_train, all_label_train = common_sub_routine(iterators[0]['train_iterator'])
            all_predictions_test, all_label_test = common_sub_routine(iterators[0]['test_iterator'])

            print(
                f"Balanced test accuracy: {balanced_accuracy_score(all_label_test, all_predictions_test.argmax(axis=1))}"
                f"and unbalanced test accuracy: {accuracy_score(all_label_test, all_predictions_test.argmax(axis=1))}")

        self.all_attribute_reco_models = all_attribute_classifier


    def prediction_over_generated_examples(self, generated_examples, gold_label):
        with torch.no_grad():
            final_accuracy = []
            for k in range(self.number_of_attributes):
                items['input'] = generated_examples
                output = self.single_attribute_reco_models[k](items)
                label = gold_label[:,k]
                final_accuracy.append((balanced_accuracy_score(label, output.argmax(axis=1)), accuracy_score(label, output.argmax(axis=1))))
            return final_accuracy




@dataclass
class TrainingLoopParameters:
    n_epochs: int
    model: torch.nn.Module
    iterators: Dict
    optimizer: torch.optim
    criterion: Callable
    device: torch.device
    use_wandb: bool
    other_params: Dict
    save_model_as: Optional[str]
    fairness_function: str
    unique_id_for_run: str


other_params = {}


iterator_params = {
    'batch_size': batch_size,
    'lm_encoder_type': 'bert-base-uncased',
    'lm_encoding_to_use': 'use_cls',
    'return_numpy_array': True,
    'dataset_size': 1000,
    'max_number_of_generated_examples' : 1000
}

iterators, other_meta_data = generate_data_iterators(dataset_name=dataset_name, **iterator_params)


tgs = TestGeneratedSamples(iterators=iterators, other_meta_data=other_meta_data)
tgs.run()

_labels, _input, _lengths, _aux, _aux_flattened = [], [], [], [], []
for items in (iterators[0]['train_iterator']):
    _labels.append(items['labels'])
    _input.append(items['input'])
    _lengths.append(items['lengths'])
    _aux.append(items['aux'])
    _aux_flattened.append(items['aux_flattened'])

all_label = np.hstack(_labels)
all_aux = np.vstack(_aux)
all_aux_flatten = np.hstack(_aux_flattened)
all_input = np.vstack(_input)

total_no_groups = len(np.unique(all_aux_flatten))
groups_matrix, global_weight, global_loss = create_group(total_no_groups, method="single_group")



train_tilted_params = TrainingLoopParameters(
    n_epochs=10,
    model=None,
    iterators=iterators,
    criterion=None,
    device=None,
    use_wandb=False,
    other_params=other_params,
    save_model_as=None,
    fairness_function='equal_odds',
    unique_id_for_run=None,
    optimizer=None
)


train_tilted_params.other_params['batch_size'] = 1000
train_tilted_params.other_params['all_aux'] = all_aux
train_tilted_params.other_params['all_label'] = all_label
train_tilted_params.other_params['all_input'] = all_input
train_tilted_params.other_params['number_of_iterations'] = 100
train_tilted_params.other_params['global_weight'] = global_weight
train_tilted_params.other_params['groups_matrix'] = groups_matrix
train_tilted_params.other_params['all_aux_flatten'] = all_aux_flatten
train_tilted_params.other_params['groups'] = [i for i in range(total_no_groups)]
train_tilted_params.other_params['example_sampling_procedure'] = 'equal_sampling'
train_tilted_params.other_params['group_sampling_procedure'] = 'random_single_group'
train_tilted_params.other_params['s_to_flattened_s'] = other_meta_data['s_flatten_lookup']







flattened_s_to_s = {value: key for key, value in train_tilted_params.other_params['s_to_flattened_s'].items()}



# model
input_dim = all_input.shape[1]
gen_model = SimpleModelGenerator(input_dim=input_dim)

# optimizer
opt_fn = partial(torch.optim.Adam)
optimizer = make_opt(gen_model, opt_fn, lr=0.001)

# mmd loss
mmd_loss = MMD_loss()
max_size = 500
aux_func = AuxilaryFunction()

for _ in range(50):
    total_loss = 0.0
    for i in tqdm(range(train_tilted_params.other_params['number_of_iterations'])):
        current_group, _ = group_sampling_procedure_func(
            train_tilted_params,
            global_weight,
            None
        )

        negative_examples_current_group, positive_examples_current_group, examples_other_leaf_group_negative, examples_other_leaf_group_positive = aux_func.sample_batch(
            current_group)

        optimizer.zero_grad()

        output_positive = gen_model(examples_other_leaf_group_positive)
        output_negative = gen_model(examples_other_leaf_group_negative)
        positive_loss = mmd_loss(positive_examples_current_group['input'], output_positive['prediction'])
        negative_loss = mmd_loss(negative_examples_current_group['input'], output_negative['prediction'])
        loss = positive_loss + negative_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.data
    print(total_loss / train_tilted_params.other_params['number_of_iterations'])

    balanced_accuracy, overall_accuracy = [], []
    for _, current_group in other_meta_data['s_flatten_lookup'].items():
        negative_examples_current_group, positive_examples_current_group, examples_other_leaf_group_negative, examples_other_leaf_group_positive = aux_func.sample_batch(
            current_group)

        output_positive = gen_model(examples_other_leaf_group_positive)
        output_negative = gen_model(examples_other_leaf_group_negative)

        final_accuracy_positive = tgs.prediction_over_generated_examples(generated_examples=output_positive['prediction'], gold_label=negative_examples_current_group['aux'])
        final_accuracy_negative = tgs.prediction_over_generated_examples(generated_examples=output_negative['prediction'], gold_label=negative_examples_current_group['aux'])


        # print(f"current_group: {current_group}, and positive accuracy {final_accuracy_positive}")
        # print(f"current_group: {current_group}, and negative accuracy {final_accuracy_negative}")

        balanced_accuracy += [i[0] for i  in final_accuracy_positive]
        balanced_accuracy += [i[0] for i  in final_accuracy_negative]

        overall_accuracy += [i[1] for i in final_accuracy_positive]
        overall_accuracy += [i[1] for i in final_accuracy_negative]

    print(np.mean(balanced_accuracy), np.mean(overall_accuracy))