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
from geomloss import SamplesLoss
from dataclasses import dataclass
from itertools import combinations
from metrics import fairness_utils
from typing import Dict, Callable, Optional
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from dataset_iterators import generate_data_iterators
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from utils.misc import resolve_device, set_seed, make_opt, CustomError
from training_loops.dro_and_erm import group_sampling_procedure_func, create_group, example_sampling_procedure_func


import warnings
warnings.filterwarnings("ignore")

dataset_name = 'adult_multi_group'
batch_size = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY)

# class SimpleModelGenerator(nn.Module):
#     """Fairgrad uses this as complex non linear model"""
#
#     def __init__(self, input_dim):
#         super().__init__()
#
#         self.layer_1 = nn.Linear(input_dim, 30)
#         self.layer_2 = nn.Linear(30, input_dim)
#         self.relu = nn.ReLU()
#
#     def forward(self, other_examples):
#         final_output = torch.tensor(0.0, requires_grad=True)
#         for group in other_examples:
#             x = group['input']
#             x = self.layer_1(x)
#             x = self.relu(x)
#             x = self.layer_2(x)
#             final_output = final_output + x
#
#
#
#
#         output = {
#             'prediction': final_output,
#             'adv_output': None,
#             'hidden': x,  # just for compatabilit
#             'classifier_hiddens': None,
#             'adv_hiddens': None
#         }
#
#         return output
#
#     @property
#     def layers(self):
#         return torch.nn.ModuleList([self.layer_1, self.layer_2])


class SimpleModelGenerator(nn.Module):
    """Fairgrad uses this as complex non linear model"""

    def __init__(self, input_dim):
        super().__init__()

        self.lambda_params = torch.nn.Parameter(torch.FloatTensor([0.33, 0.33, 0.33]))

    def forward(self, other_examples):
        final_output = torch.tensor(0.0, requires_grad=True)
        for param, group in zip(self.lambda_params, other_examples):
            x = group['input']
            final_output = final_output + x*param

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
        # other_leaf_group = [i for i in AuxilaryFunction.generate_combinations_only_leaf_node(flattened_s_to_s[current_group], k=1)]

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
        self.one_vs_all_clf = {}
        self.one_vs_all_clf = {}
        self.iterators = iterators
        self.all_attribute_reco_models = None
        self.other_meta_data = other_meta_data
        self.single_attribute_reco_models = []
        self.raw_data = other_meta_data['raw_data']
        self.s_flatten_lookup = other_meta_data['s_flatten_lookup']
        self.number_of_attributes = other_meta_data['raw_data']['train_s'].shape[1]

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


    def run(self):
        self.run_one_vs_all_classifier_group()
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

    def one_vs_all_classifier_group(self, group):
        mask_group_train_X = self.create_mask_with_x(data=self.raw_data['train_s'], condition=group)
        size = mask_group_train_X.sum()
        index_group_train_X = np.random.choice(np.where(mask_group_train_X == True)[0], size=size, replace=False)
        index_not_group_train_X = np.random.choice(np.where(mask_group_train_X == False)[0], size=size, replace=False)

        train_X_group, train_y_group = self.raw_data['train_X'][index_group_train_X], np.ones(size)
        train_X_not_group, train_y_not_group = self.raw_data['train_X'][index_not_group_train_X], np.zeros(size)

        clf = MLPClassifier(solver="adam", learning_rate_init=0.01, hidden_layer_sizes=(25, 5), random_state=1)

        X = np.vstack([train_X_group, train_X_not_group])
        y = np.hstack([train_y_group, train_y_not_group])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)

        clf.fit(X_train, y_train)

        print()
        y_pred = clf.predict(X_test)
        print(clf.score(X_train, y_train), accuracy_score(y_test, y_pred), balanced_accuracy_score(y_test, y_pred))

        return clf

    def run_one_vs_all_classifier_group(self):

        for group, group_id in self.s_flatten_lookup.items():
            print(f"current group: {group_id}")
            clf = self.one_vs_all_classifier_group(group=group)
            print("**************")
            self.one_vs_all_clf[group_id] = clf



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
    'max_number_of_generated_examples': 1000
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


group_size = {}

for g in train_tilted_params.other_params['groups']:
    group_mask = all_aux_flatten == g
    positive_label_mask = all_label == 1
    negative_label_mask = all_label == 0
    positive_group_size = np.sum(np.logical_and(group_mask, positive_label_mask))
    negative_group_size = np.sum(np.logical_and(group_mask, negative_label_mask))
    group_size[g] = [positive_group_size, negative_group_size]



flattened_s_to_s = {value: key for key, value in train_tilted_params.other_params['s_to_flattened_s'].items()}



# model
input_dim = all_input.shape[1]
gen_model_positive = SimpleModelGenerator(input_dim=input_dim)
gen_model_negative = SimpleModelGenerator(input_dim=input_dim)

# optimizer
# opt_fn = partial(torch.optim.Adam)
# optimizer_positive = make_opt(gen_model_positive, opt_fn, lr=0.001)
# optimizer_negative = make_opt(gen_model_negative, opt_fn, lr=0.001)
optimizer_positive = torch.optim.Adam(gen_model_positive.parameters(), lr=0.001)
optimizer_negative = torch.optim.Adam(gen_model_negative.parameters(), lr=0.001)

# mmd loss
mmd_loss = MMD_loss()
sample_loss = SamplesLoss(loss="laplacian".lower(), p=2)
max_size = 20000000000
aux_func = AuxilaryFunction()

for _ in range(5):
    total_loss_positive, total_loss_negative = 0.0, 0.0
    for i in tqdm(range(train_tilted_params.other_params['number_of_iterations'])):
        current_group, _ = group_sampling_procedure_func(
            train_tilted_params,
            global_weight,
            None
        )

        positive_size, negative_size = group_size[current_group]


        negative_examples_current_group, positive_examples_current_group, examples_other_leaf_group_negative, examples_other_leaf_group_positive = aux_func.sample_batch(
            current_group)

        if positive_size < max_size:
            optimizer_positive.zero_grad()
            output_positive = gen_model_positive(examples_other_leaf_group_positive)
            positive_loss = MMD(x=positive_examples_current_group['input'], y=output_positive['prediction'],
                                kernel='rbf')

                            # + MMD(x=examples_other_leaf_group_positive[0]['input'], y=output_positive['prediction'],
                            #     kernel='rbf') + MMD(x=examples_other_leaf_group_positive[1]['input'], y=output_positive['prediction'],
                            #     kernel='rbf') + MMD(x=examples_other_leaf_group_positive[2]['input'], y=output_positive['prediction'],
                            #     kernel='rbf')
            # positive_loss = sample_loss(positive_examples_current_group['input'], output_positive['prediction'])*0.0+\
            #                 sample_loss(examples_other_leaf_group_positive[0]['input'], output_positive['prediction'])\
            #                 + sample_loss(examples_other_leaf_group_positive[1]['input'], output_positive['prediction'])\
            #                  + + sample_loss(examples_other_leaf_group_positive[2]['input'], output_positive['prediction'])

            positive_loss.backward()
            total_loss_positive += positive_loss.data
            optimizer_positive.step()

        if negative_size < max_size:
            optimizer_negative.zero_grad()
            output_negative = gen_model_negative(examples_other_leaf_group_negative)
            negative_loss = MMD(x=negative_examples_current_group['input'], y=output_negative['prediction'],
                                kernel='rbf')*0.0
                            # MMD(x=examples_other_leaf_group_negative[0]['input'], y=output_negative['prediction'],
                            #     kernel='rbf') + MMD(x=examples_other_leaf_group_negative[1]['input'], y=output_negative['prediction'],
                            #     kernel='rbf') + MMD(x=examples_other_leaf_group_negative[2]['input'], y=output_negative['prediction'],
                            #     kernel='rbf')
            # negative_loss = sample_loss(negative_examples_current_group['input'], output_negative['prediction'])*0.0+\
            #                 sample_loss(examples_other_leaf_group_negative[0]['input'], output_negative['prediction']) \
            #                  + sample_loss(examples_other_leaf_group_negative[1]['input'], output_negative['prediction'])\
            #                  + sample_loss(examples_other_leaf_group_negative[2]['input'], output_negative['prediction'])



            negative_loss.backward()
            optimizer_negative.step()
            total_loss_negative += negative_loss.data
    print(total_loss_positive / train_tilted_params.other_params['number_of_iterations'])
    print(total_loss_negative / train_tilted_params.other_params['number_of_iterations'])

    balanced_accuracy, overall_accuracy, one_vs_all_accuracy = [], [], []
    for _, current_group in other_meta_data['s_flatten_lookup'].items():
        positive_size, negative_size = group_size[current_group]


        negative_examples_current_group, positive_examples_current_group, examples_other_leaf_group_negative, examples_other_leaf_group_positive = aux_func.sample_batch(
            current_group)

        if positive_size < max_size:

            output_positive = gen_model_positive(examples_other_leaf_group_positive)
            final_accuracy_positive = tgs.prediction_over_generated_examples(
                generated_examples=output_positive['prediction'], gold_label=negative_examples_current_group['aux'])
            balanced_accuracy += [i[0] for i in final_accuracy_positive]
            overall_accuracy += [i[1] for i in final_accuracy_positive]

        if negative_size < max_size:
            output_negative = gen_model_negative(examples_other_leaf_group_negative)
            final_accuracy_negative = tgs.prediction_over_generated_examples(generated_examples=output_negative['prediction'], gold_label=negative_examples_current_group['aux'])
            balanced_accuracy += [i[0] for i in final_accuracy_negative]
            overall_accuracy += [i[1] for i in final_accuracy_negative]

        acc1 = tgs.one_vs_all_clf[current_group].score(output_positive['prediction'].detach().numpy(),
                                                      np.ones(len(output_positive['prediction'])))
        acc2 = tgs.one_vs_all_clf[current_group].score(output_negative['prediction'].detach().numpy(),
                                                      np.ones(len(output_negative['prediction'])))

        print(f"group {current_group}: {np.mean([acc2, acc1])}")

        one_vs_all_accuracy.append(np.mean([acc2, acc1]))

    print(np.mean(overall_accuracy), np.max(overall_accuracy), np.min(overall_accuracy))
    print(np.mean(one_vs_all_accuracy), np.max(one_vs_all_accuracy), np.min(one_vs_all_accuracy))



torch.save(gen_model_positive.state_dict(), "gen_model_adult_positive.pth")
torch.save(gen_model_negative.state_dict(), "gen_model_adult_negative.pth")