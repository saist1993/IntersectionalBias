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
train_tilted_params.other_params['number_of_iterations']  = 100
train_tilted_params.other_params['global_weight'] = global_weight
train_tilted_params.other_params['groups_matrix'] = groups_matrix
train_tilted_params.other_params['all_aux_flatten'] = all_aux_flatten
train_tilted_params.other_params['groups'] = [i for i in range(total_no_groups)]
train_tilted_params.other_params['example_sampling_procedure'] = 'equal_sampling'
train_tilted_params.other_params['group_sampling_procedure'] = 'random_single_group'
train_tilted_params.other_params['s_to_flattened_s'] = other_meta_data['s_flatten_lookup']



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


def generate_abstract_node(s, k=1):
    all_s_combinations = []

    for i in combinations(range(len(s)),k):
        _temp = list(copy.deepcopy(s))
        for j in i:
            _temp[j] = 'x'
        all_s_combinations.append(tuple(_temp))

    return all_s_combinations



flattened_s_to_s = {value: key for key, value in train_tilted_params.other_params['s_to_flattened_s'].items()}



# model
input_dim = all_input.shape[1]
gen_model = SimpleModelGenerator(input_dim=input_dim)

# optimizer
opt_fn = partial(torch.optim.Adam)
optimizer = make_opt(gen_model, opt_fn, lr=0.001)

# mmd loss
mmd_loss = MMD_loss()



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


size_of_group_0, size_of_group_1 = size_of_group(other_meta_data)

max_size = 500
for _ in range(50):
    total_loss = 0.0
    for i in tqdm(range(train_tilted_params.other_params['number_of_iterations'])):
        current_group, _ = group_sampling_procedure_func(
            train_tilted_params,
            global_weight,
            None
        )

        other_leaf_group = [train_tilted_params.other_params['s_to_flattened_s'][i] for i in generate_combinations_only_leaf_node(flattened_s_to_s[current_group], k=1)]

        examples_current_group, _ = example_sampling_procedure_func(
            train_tilted_params=train_tilted_params,
            group0=current_group,
            group1=None
        )
        examples_other_leaf_group = []

        for group in other_leaf_group:
            examples, _ = example_sampling_procedure_func(
                train_tilted_params=train_tilted_params,
                group0=group,
                group1=None
            )

            examples_other_leaf_group.append(examples)

        optimizer.zero_grad()
        output = gen_model(examples_other_leaf_group)
        # loss = MMD(examples_current_group['input'], output['prediction'], 'rbf')
        loss = mmd_loss(examples_current_group['input'], output['prediction'])
        loss.backward()
        optimizer.step()

        total_loss += loss.data
    print(total_loss / train_tilted_params.other_params['number_of_iterations'])