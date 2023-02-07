import copy
import torch
import pickle
import numpy as np
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from mmd_utils import *
from scipy import optimize
from functools import partial
from collections import Counter
import matplotlib.pyplot as plt
from geomloss import SamplesLoss
from dataclasses import dataclass
from itertools import combinations
from metrics import fairness_utils
from typing import Dict, Callable, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from dataset_iterators import generate_data_iterators
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from utils.misc import resolve_device, set_seed, make_opt, CustomError
from training_loops.dro_and_erm import group_sampling_procedure_func, create_group, example_sampling_procedure_func

dataset_name = 'twitter_hate_speech'
batch_size = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np


# import mkl
# mkl.set_num_threads(3)

torch.set_num_threads(4)
torch.set_num_interop_threads(4)


import warnings
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")



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
    def sample_batch(current_group, other_meta_data, batch_sizes=512):

        # all_label_train = other_meta_data['raw_data']['train_y']
        # all_aux_train = other_meta_data['raw_data']['train_s']
        # all_input_train = other_meta_data['raw_data']['train_X']
        # all_aux_flatten_train = [other_meta_data['s_flatten_lookup'][tuple(i)] for i in all_aux_train]
        #
        # all_label_valid = other_meta_data['raw_data']['valid_y']
        # all_aux_valid = other_meta_data['raw_data']['valid_s']
        # all_input_valid = other_meta_data['raw_data']['valid_X']
        # all_aux_flatten_valid = [other_meta_data['s_flatten_lookup'][tuple(i)] for i in all_aux_valid]

        # other_leaf_group = [train_tilted_params.other_params['s_to_flattened_s'][i]
        # for i in generate_combinations_only_leaf_node(flattened_s_to_s[current_group], k=1)]
        other_leaf_group = [i for i in AuxilaryFunction.generate_abstract_node(flattened_s_to_s[current_group], k=1)]
        # other_leaf_group = [i for i in AuxilaryFunction.generate_
        # combinations_only_leaf_node(flattened_s_to_s[current_group], k=1)]
        #
        # examples_current_group, _ = example_sampling_procedure_func(
        #     train_tilted_params=train_tilted_params,
        #     group0=current_group,
        #     group1=None
        # )
        #
        # index = int(train_tilted_params.other_params['batch_size'] / 2)
        #
        # negative_examples_current_group = {
        #     'labels': torch.LongTensor(examples_current_group['labels'][:index]),
        #     'input': torch.FloatTensor(examples_current_group['input'][:index]),
        #     'aux': torch.LongTensor(examples_current_group['aux'][:index]),
        #     'aux_flattened': torch.LongTensor(examples_current_group['aux_flattened'][:index])
        # }
        #
        # positive_examples_current_group = {
        #     'labels': torch.LongTensor(examples_current_group['labels'][index:]),
        #     'input': torch.FloatTensor(examples_current_group['input'][index:]),
        #     'aux': torch.LongTensor(examples_current_group['aux'][index:]),
        #     'aux_flattened': torch.LongTensor(examples_current_group['aux_flattened'][index:])
        # }

        # reverse_lookup = {value: key for key, value in other_meta_data['s_flatten_lookup'].items()}

        # negative_examples_current_group, positive_examples_current_group = AuxilaryFunction.custom_sample_data(
        #     group=reverse_lookup[current_group], all_label=other_meta_data['raw_data']['valid_y'],
        #     all_input=other_meta_data['raw_data']['valid_X'], all_aux=other_meta_data['raw_data']['valid_s'],
        #     all_aux_flatten=np.asarray(
        #         [other_meta_data['s_flatten_lookup'][tuple(i)] for i in other_meta_data['raw_data']['valid_s']]),
        #     number_of_positive_examples=int(batch_size / 2), number_of_negative_examples=int(batch_size / 2))

        all_label = np.hstack([other_meta_data['raw_data']['valid_y'], other_meta_data['raw_data']['train_y']])
        all_aux = np.vstack([other_meta_data['raw_data']['valid_s'], other_meta_data['raw_data']['train_s']])
        all_input = np.vstack([other_meta_data['raw_data']['valid_X'], other_meta_data['raw_data']['train_X']])
        #
        # all_label = other_meta_data['raw_data']['train_y']
        # all_aux = other_meta_data['raw_data']['train_s']
        # all_input = other_meta_data['raw_data']['train_X']

        # all_label = other_meta_data['raw_data']['valid_y']
        # all_aux = other_meta_data['raw_data']['valid_s']
        # all_input = other_meta_data['raw_data']['valid_X']

        negative_examples_current_group, positive_examples_current_group = AuxilaryFunction.custom_sample_data(
            group=flattened_s_to_s[current_group], all_label=all_label,
            all_input=all_input, all_aux=all_aux,
            all_aux_flatten=np.asarray(
                [other_meta_data['s_flatten_lookup'][tuple(i)] for i in all_aux]),
            number_of_positive_examples=int(batch_sizes / 2), number_of_negative_examples=int(batch_sizes / 2))


        examples_other_leaf_group_negative, examples_other_leaf_group_positive = [], []

        for group in other_leaf_group:
            batch_input_negative, batch_input_positive = AuxilaryFunction.custom_sample_data(group=group,
                                                                                             all_label=other_meta_data[
                                                                                                 'raw_data']['train_y'],
                                                                                             all_input=other_meta_data[
                                                                                                 'raw_data']['train_X'],
                                                                                             all_aux=other_meta_data[
                                                                                                 'raw_data']['train_s'],
                                                                                             all_aux_flatten=np.asarray(
                                                                                                 [other_meta_data[
                                                                                                      's_flatten_lookup'][
                                                                                                      tuple(i)] for i in
                                                                                                  other_meta_data[
                                                                                                      'raw_data'][
                                                                                                      'train_s']]),
                                                                                             number_of_positive_examples=int(
                                                                                                 batch_sizes / 2),
                                                                                             number_of_negative_examples=int(
                                                                                                 batch_sizes / 2))

            examples_other_leaf_group_negative.append(batch_input_negative)
            examples_other_leaf_group_positive.append(batch_input_positive)

        return negative_examples_current_group, positive_examples_current_group, examples_other_leaf_group_negative, examples_other_leaf_group_positive

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


if __name__ == '__main__':
    other_params = {}

    iterator_params = {
        'batch_size': batch_size,
        'lm_encoder_type': 'bert-base-uncased',
        'lm_encoding_to_use': 'use_cls',
        'return_numpy_array': True,
        'dataset_size': 1000,
        'max_number_of_generated_examples': 1000,
        'per_group_label_number_of_examples': 1000,
        'mmd_augmentation_mechanism': "asd"
    }

    iterators, other_meta_data = generate_data_iterators(dataset_name=dataset_name, **iterator_params)

    tgs = TestGeneratedSamples(iterators=iterators, other_meta_data=other_meta_data)
    # tgs.run()

    # we need to remove data here

    scaler = StandardScaler().fit(other_meta_data['raw_data']['train_X'])
    other_meta_data['raw_data']['train_X'] = scaler.transform(other_meta_data['raw_data']['train_X'])
    other_meta_data['raw_data']['valid_X'] = scaler.transform(other_meta_data['raw_data']['valid_X'])


    original_meta_data = copy.deepcopy(other_meta_data)

    all_groups = np.unique(other_meta_data['raw_data']['train_s'], axis=0)
    size_of_groups = {tuple(group): np.sum(generate_mask(other_meta_data['raw_data']['train_s'], group)) for group in all_groups}

    if False:
        deleted_group = [0,1,0,0]
        group_to_remove_index = np.where(generate_mask(other_meta_data['raw_data']['train_s'], deleted_group))[0]
        print(f"deleted group is {deleted_group} and flat version"
              f" {other_meta_data['s_flatten_lookup'][tuple(deleted_group)]}")

        other_meta_data['raw_data']['train_X'] = np.delete(other_meta_data['raw_data']['train_X'] , group_to_remove_index, 0)
        other_meta_data['raw_data']['train_s'] = np.delete(other_meta_data['raw_data']['train_s'] , group_to_remove_index, 0)
        other_meta_data['raw_data']['train_y'] = np.delete(other_meta_data['raw_data']['train_y'] , group_to_remove_index, 0)

    #other groups -> [0,1,1,1] and [1,1,0,0]


    all_label_train = other_meta_data['raw_data']['train_y']
    all_aux_train = other_meta_data['raw_data']['train_s']
    all_input_train = other_meta_data['raw_data']['train_X']
    all_aux_flatten_train = [other_meta_data['s_flatten_lookup'][tuple(i)] for i in all_aux_train]

    all_label_valid = other_meta_data['raw_data']['valid_y']
    all_aux_valid = other_meta_data['raw_data']['valid_s']
    all_input_valid = other_meta_data['raw_data']['valid_X']
    all_aux_flatten_valid = [other_meta_data['s_flatten_lookup'][tuple(i)] for i in all_aux_valid]

    total_no_groups = len(np.unique(all_aux_flatten_valid))  # hopefully this also has all the groups
    # groups_matrix, global_weight, global_loss = create_group(total_no_groups, method="single_group")




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

    train_tilted_params.other_params['batch_size'] = batch_size
    train_tilted_params.other_params['number_of_iterations'] = 100
    train_tilted_params.other_params['global_weight'] = None
    train_tilted_params.other_params['groups_matrix'] = None
    train_tilted_params.other_params['groups'] = [i for i in range(total_no_groups)]
    train_tilted_params.other_params['example_sampling_procedure'] = 'equal_sampling'
    train_tilted_params.other_params['group_sampling_procedure'] = 'random_single_group'
    train_tilted_params.other_params['s_to_flattened_s'] = other_meta_data['s_flatten_lookup']

    flattened_s_to_s = {value: key for key, value in train_tilted_params.other_params['s_to_flattened_s'].items()}

    all_models = {}
    input_dim = all_input_valid.shape[1]
    # sigma_list = [1.0, 2.0, 4.0, 8.0, 16.0]
    sigma_list = [1.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    # sigma_list = [1.0, 5.0, 10.0, 20.0, 30.0]
    # sigma_list = [1.0, 10.0, 15.0, 20.0, 50.0]

    per_group_accuracy = [1.0 for i in train_tilted_params.other_params['groups']]



    for current_group_flat, current_group in flattened_s_to_s.items():
        gen_model_positive = SimpleModelGenerator(input_dim=input_dim, number_of_params=len(flattened_s_to_s[1]))
        gen_model_negative = SimpleModelGenerator(input_dim=input_dim, number_of_params=len(flattened_s_to_s[1]))

        optimizer_positive = torch.optim.Adam(gen_model_positive.parameters(), lr=0.1)
        optimizer_negative = torch.optim.Adam(gen_model_negative.parameters(), lr=0.1)

        all_models['a'] = {
            'gen_model_positive': gen_model_positive,
            'gen_model_negative': gen_model_negative,
            'optimizer_positive': optimizer_positive,
            'optimizer_negative': optimizer_negative
        }

        # all_models[current_group] = {
        #     'gen_model_positive': gen_model_positive,
        #     'gen_model_negative': gen_model_negative,
        #     'optimizer_positive': optimizer_positive,
        #     'optimizer_negative': optimizer_negative
        # }

        break







    # mmd_loss = MMD_loss()
    # sample_loss = SamplesLoss(loss="gaussian".lower(), p=2)
    # sample_loss = ot.sliced_wasserstein_distance
    max_size = 20000000000
    aux_func = AuxilaryFunction()
    worst_accuracy = 1.0

    for _ in range(100):
        total_loss_positive, total_loss_negative = 0.0, 0.0
        for i in tqdm(range(train_tilted_params.other_params['number_of_iterations'])):
            current_group = np.random.choice(train_tilted_params.other_params['groups'], p=per_group_accuracy/np.linalg.norm(per_group_accuracy,1), size=1)[0]

            # if current_group == train_tilted_params.other_params['s_to_flattened_s'][tuple(deleted_group)]:
            #     continue

            gen_model_positive, gen_model_negative, optimizer_positive, optimizer_negative = \
                all_models['a']['gen_model_positive'], \
                    all_models['a']['gen_model_negative'], \
                    all_models['a']['optimizer_positive'], \
                    all_models['a']['optimizer_negative']


            positive_size, negative_size = 0, 0

            negative_examples_current_group, positive_examples_current_group, \
                examples_other_leaf_group_negative, examples_other_leaf_group_positive = aux_func.sample_batch(
                current_group, other_meta_data, batch_sizes=batch_size)

            if positive_size < max_size:
                optimizer_positive.zero_grad()
                output_positive = gen_model_positive(examples_other_leaf_group_positive)

                positive_loss = mix_rbf_mmd2(positive_examples_current_group['input'],
                                             output_positive['prediction'],
                                             sigma_list=sigma_list)

                other_positive_loss = torch.sum(torch.tensor([mix_rbf_mmd2(examples['input'], output_positive['prediction'],
                                                                  sigma_list=sigma_list) for examples in
                                                              examples_other_leaf_group_positive], requires_grad=True))


                positive_loss = positive_loss + other_positive_loss



                positive_loss.backward()
                total_loss_positive += positive_loss.data
                optimizer_positive.step()

            if negative_size < max_size:
                optimizer_negative.zero_grad()
                output_negative = gen_model_negative(examples_other_leaf_group_negative)

                negative_loss = mix_rbf_mmd2(negative_examples_current_group['input'],
                                             output_negative['prediction'],
                                             sigma_list=sigma_list)

                other_negative_loss = torch.sum(
                    torch.tensor([mix_rbf_mmd2(examples['input'], output_negative['prediction'],
                                               sigma_list=sigma_list) for examples in
                                  examples_other_leaf_group_negative], requires_grad=True))


                #
                # negative_loss = MMD(x=negative_examples_current_group['input'], y=output_negative['prediction'],
                #                     kernel='rbf')
                # #
                # other_negative_loss = torch.sum(torch.tensor([MMD(x=examples['input'], y=output_negative['prediction'],
                #                                                   kernel='rbf') for examples in
                #                                               examples_other_leaf_group_negative],
                #                                              requires_grad=True))

                # negative_loss = sample_loss(negative_examples_current_group['input'], output_negative['prediction'])
                #
                # other_negative_loss = torch.sum(torch.tensor([sample_loss(examples['input'], output_positive['prediction']) for examples in
                #                                               examples_other_leaf_group_negative],
                #                                              requires_grad=True))

                # loss2 = MMD(x=torch.tensor(other_meta_data['raw_data']['train_X'][np.random.choice(np.where((other_meta_data['raw_data']['train_y'] == 0) == True)[0],
                #                                                           size=len(output_negative['prediction']), replace=False)], dtype=torch.float),
                #             y=output_negative['prediction'], kernel='rbf')

                negative_loss = negative_loss + other_negative_loss
                # negative_loss = sample_loss(negative_examples_current_group['input'], output_negative['prediction'])+\
                #                 sample_loss(examples_other_leaf_group_negative[0]['input'], output_negative['prediction']) \
                #                  + sample_loss(examples_other_leaf_group_negative[1]['input'], output_negative['prediction'])\
                #                  + sample_loss(examples_other_leaf_group_negative[2]['input'], output_negative['prediction'])

                negative_loss.backward()
                optimizer_negative.step()
                total_loss_negative += negative_loss.data


            # all_models['a']['gen_model_positive'], \
            #         all_models['a']['gen_model_negative'], \
            #         all_models['a']['optimizer_positive'], \
            #         all_models['a']['optimizer_negative'] = gen_model_positive, gen_model_negative, optimizer_positive, optimizer_negative

        print(total_loss_positive / train_tilted_params.other_params['number_of_iterations'])
        print(total_loss_negative / train_tilted_params.other_params['number_of_iterations'])

        balanced_accuracy, overall_accuracy, one_vs_all_accuracy = [], [], []

        all_generated_examples = []


        for flat_current_group, current_group in other_meta_data['s_flatten_lookup'].items():
            positive_size, negative_size = 0, 0

            gen_model_positive, gen_model_negative, optimizer_positive, optimizer_negative = \
                all_models['a']['gen_model_positive'], \
                    all_models['a']['gen_model_negative'], \
                    all_models['a']['optimizer_positive'], \
                    all_models['a']['optimizer_negative']

            # gen_model_positive, gen_model_negative, optimizer_positive, optimizer_negative = \
            #     all_models[flat_current_group]['gen_model_positive'], \
            #         all_models[flat_current_group]['gen_model_negative'], \
            #         all_models[flat_current_group]['optimizer_positive'], \
            #         all_models[flat_current_group]['optimizer_negative']

            negative_examples_current_group, positive_examples_current_group, examples_other_leaf_group_negative, examples_other_leaf_group_positive = aux_func.sample_batch(
                current_group, original_meta_data, batch_sizes=1024)

            if positive_size < max_size:
                output_positive = gen_model_positive(examples_other_leaf_group_positive)
                # final_accuracy_positive = tgs.prediction_over_generated_examples(
                #     generated_examples=output_positive['prediction'], gold_label=negative_examples_current_group['aux'])
                # balanced_accuracy += [i[0] for i in final_accuracy_positive]
                # overall_accuracy += [i[1] for i in final_accuracy_positive]

            if negative_size < max_size:
                output_negative = gen_model_negative(examples_other_leaf_group_negative)
                # final_accuracy_negative = tgs.prediction_over_generated_examples(generated_examples=output_negative['prediction'], gold_label=negative_examples_current_group['aux'])
                # balanced_accuracy += [i[0] for i in final_accuracy_negative]
                # overall_accuracy += [i[1] for i in final_accuracy_negative]

            # acc1 = tgs.one_vs_all_clf[current_group].score(output_positive['prediction'].detach().numpy(),
            #                                               np.ones(len(output_positive['prediction'])))
            # acc2 = tgs.one_vs_all_clf[current_group].score(output_negative['prediction'].detach().numpy(),
            #                                               np.ones(len(output_negative['prediction'])))
            #
            # print(f"group {current_group}: {np.mean([acc2, acc1])}")
            #
            # one_vs_all_accuracy.append(np.mean([acc2, acc1]))

            all_generated_examples.append(output_positive['prediction'])
            all_generated_examples.append(output_negative['prediction'])



        # print(np.mean(overall_accuracy), np.max(overall_accuracy), np.min(overall_accuracy))
        # print(np.mean(one_vs_all_accuracy), np.max(one_vs_all_accuracy), np.min(one_vs_all_accuracy))

        # for name, param in gen_model_positive.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        #
        # for name, param in gen_model_negative.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)

        all_generated_examples = torch.vstack(all_generated_examples).detach().numpy()
        real_examples = np.random.choice(range(len(original_meta_data['raw_data']['train_X'])),
                                         size=len(all_generated_examples), replace=False)
        real_examples = original_meta_data['raw_data']['train_X'][real_examples]
        generated_example_label = np.zeros(len(all_generated_examples))
        real_example_label = np.ones(len(all_generated_examples))

        X = np.vstack([all_generated_examples, real_examples])
        y = np.hstack([generated_example_label, real_example_label])

        clf = MLPClassifier(solver="adam", learning_rate_init=0.01, hidden_layer_sizes=(50, 20), random_state=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("****")
        print(clf.score(X_train, y_train), accuracy_score(y_test, y_pred), balanced_accuracy_score(y_test, y_pred))
        print("***")



        if balanced_accuracy_score(y_test, y_pred) < worst_accuracy:
            worst_accuracy = balanced_accuracy_score(y_test, y_pred)
            # torch.save(gen_model_positive.state_dict(), "dummy.pth")
            # torch.save(gen_model_negative.state_dict(), "dummy.pth")
            # pickle.dump(all_models, open(f"all_{dataset_name}.pt", 'wb'))
            # pickle.dump(clf, open(f"real_vs_fake_{dataset_name}.sklearn", 'wb'))




        # group wise accuracy

        average_accuracy = []


        for flat_current_group, current_group in other_meta_data['s_flatten_lookup'].items():
            positive_size, negative_size = 0, 0

            gen_model_positive, gen_model_negative, optimizer_positive, optimizer_negative = \
                all_models['a']['gen_model_positive'], \
                    all_models['a']['gen_model_negative'], \
                    all_models['a']['optimizer_positive'], \
                    all_models['a']['optimizer_negative']


            negative_examples_current_group, positive_examples_current_group, examples_other_leaf_group_negative, examples_other_leaf_group_positive = aux_func.sample_batch(
                current_group, original_meta_data, batch_sizes=1024)

            if positive_size < max_size:
                output_positive = gen_model_positive(examples_other_leaf_group_positive)

            if negative_size < max_size:
                output_negative = gen_model_negative(examples_other_leaf_group_negative)



            examples = np.vstack([output_positive['prediction'].detach().numpy(), output_negative['prediction'].detach().numpy()])
            y_test = np.zeros(len(examples))
            y_pred = clf.predict(examples)
            print(current_group, accuracy_score(y_test, y_pred), balanced_accuracy_score(y_test, y_pred))

            average_accuracy.append(accuracy_score(y_test, y_pred))

        per_group_accuracy = average_accuracy

        print(f"average accuracy is {np.mean(average_accuracy)}")
        if np.mean(average_accuracy) < worst_accuracy:
            worst_accuracy = np.mean(average_accuracy)
            # torch.save(gen_model_positive.state_dict(), "dummy.pth")
            # torch.save(gen_model_negative.state_dict(), "dummy.pth")
            pickle.dump(all_models, open(f"all_{dataset_name}.pt", 'wb'))
            pickle.dump(clf, open(f"real_vs_fake_{dataset_name}.sklearn", 'wb'))


