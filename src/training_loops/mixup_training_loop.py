import math
import torch
from tqdm.auto import tqdm
from numpy.random import beta
from collections import Counter
import torch.nn.functional as F
from itertools import combinations
from .common_functionality import *
from sklearn.metrics.pairwise import cosine_distances


def titled_sample_batch_sen_idx_with_y(all_input, all_label, all_aux, all_aux_flatten, batch_size, s):
    """
        This will sample batch size number of elements from input with given s!
    """
    index_s_0 = np.where(np.logical_and(all_aux_flatten==s, all_label==0) == True)[0]
    index_s_1 = np.where(np.logical_and(all_aux_flatten==s, all_label==1) == True)[0]
    relevant_index = np.random.choice(index_s_0, size=int(batch_size/2), replace=True).tolist()
    relevant_index = relevant_index + np.random.choice(index_s_1,size=int(batch_size/2), replace=True).tolist()

    # THIS IS DIFFERENT. IN ORIGINAL VERSION IT IS REPLACEMENT TRUE
    batch_input = {
        'labels': torch.LongTensor(all_label[relevant_index]),
        'input': torch.FloatTensor(all_input[relevant_index]),
        'aux': torch.LongTensor(all_aux[relevant_index]),
        'aux_flattened': torch.LongTensor(all_aux_flatten[relevant_index])
    }

    return batch_input

def generate_combinations(s, k =1):
    all_s_combinations = []

    for i in combinations(range(len(s)),k):
        _temp = copy.deepcopy(s)
        for j in i:
            _temp[j] = 'x'
        all_s_combinations.append(tuple(_temp))

    return all_s_combinations


def generate_possible_patterns(s0, s1):
    all_s0_combinations, all_s1_combinations = generate_combinations(s0,1), generate_combinations(s1,1)
    all_unique_s0_combination = list(set(all_s0_combinations) - set(all_s1_combinations))
    all_unique_s0_combination.append(tuple(s0))
    all_unique_s1_combination = list(set(all_s1_combinations) - set(all_s0_combinations))
    all_unique_s1_combination.append(tuple(s1))
    return all_unique_s0_combination, all_unique_s1_combination


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


def sample_batch_sen_idx_with_y(all_input, all_label, all_aux, all_aux_flatten, batch_size, s0, s1):
    """
        This will sample batch size number of elements from input with given s!
    """

    all_unique_s0_combination, all_unique_s1_combination = generate_possible_patterns(s0, s1)
    mask_s0 = np.logical_or.reduce([generate_mask(all_aux, mask_pattern) for mask_pattern in all_unique_s0_combination])
    mask_s1 = np.logical_or.reduce([generate_mask(all_aux, mask_pattern) for mask_pattern in all_unique_s1_combination])

    index_s0_0 = np.where(np.logical_and(mask_s0, all_label==0) == True)[0]
    index_s0_1 = np.where(np.logical_and(mask_s0, all_label==1) == True)[0]

    index_s1_0 = np.where(np.logical_and(mask_s1, all_label == 0) == True)[0]
    index_s1_1 = np.where(np.logical_and(mask_s1, all_label == 1) == True)[0]


    relevant_index_s0 = np.random.choice(index_s0_0, size=int(batch_size/2), replace=True).tolist()
    relevant_index_s0 = relevant_index_s0 + np.random.choice(index_s0_1,size=int(batch_size/2), replace=True).tolist()

    relevant_index_s1 = np.random.choice(index_s1_0,size=int(batch_size/2), replace=True).tolist()
    relevant_index_s1 = relevant_index_s1 + np.random.choice(index_s1_1,size=int(batch_size/2), replace=True).tolist()

    # THIS IS DIFFERENT. IN ORIGINAL VERSION IT IS REPLACEMENT TRUE
    batch_input_s0 = {
        'labels': torch.LongTensor(all_label[relevant_index_s0]),
        'input': torch.FloatTensor(all_input[relevant_index_s0]),
        'aux': torch.LongTensor(all_aux[relevant_index_s0]),
        'aux_flattened': torch.LongTensor(all_aux_flatten[relevant_index_s0])
    }

    batch_input_s1 = {
        'labels': torch.LongTensor(all_label[relevant_index_s1]),
        'input': torch.FloatTensor(all_input[relevant_index_s1]),
        'aux': torch.LongTensor(all_aux[relevant_index_s1]),
        'aux_flattened': torch.LongTensor(all_aux_flatten[relevant_index_s1])
    }

    return batch_input_s0, batch_input_s1

def sample_batch_sen_idx(all_input, all_label, all_aux, all_aux_flatten, batch_size, s):
    all_extra_combinations = generate_combinations(s, k=1)
    extra_mask_s = np.logical_or.reduce([generate_mask(all_aux, mask_pattern) for mask_pattern in all_extra_combinations])
    mask_s = generate_mask(all_aux,s)

    sample_from_extra = batch_size - np.sum(mask_s)

    if sample_from_extra <= 0:
        relevant_index = np.random.choice(np.where(mask_s == True)[0], size=batch_size, replace=True).tolist()
    else:
        relevant_index_1 = np.random.choice(np.where(mask_s == True)[0], size=np.sum(mask_s), replace=True).tolist()
        relevant_index_2 = np.random.choice(np.where(extra_mask_s == True)[0], size=sample_from_extra, replace=True).tolist()
        relevant_index = np.hstack([relevant_index_1, relevant_index_2])
        np.random.shuffle(relevant_index)

    batch_input = {
        'labels': torch.LongTensor(all_label[relevant_index]),
        'input': torch.FloatTensor(all_input[relevant_index]),
        'aux': torch.LongTensor(all_aux[relevant_index]),
        'aux_flattened': torch.LongTensor(all_aux_flatten[relevant_index])
    }

    return batch_input



def sample_batch_sen_idx_with_augmentation(all_input, all_label, all_aux, all_aux_flatten, batch_size, s):
    all_extra_combinations = generate_combinations(s, k=1)
    extra_mask_s =[generate_mask(all_aux, mask_pattern) for mask_pattern in all_extra_combinations] # this stores all the combinations
    mask_s = generate_mask(all_aux,s)

    relevant_index = np.random.choice(np.where(mask_s == True)[0], size=batch_size, replace=True).tolist()

    # now comes the augmentations
    each_split_size = int(batch_size/len(extra_mask_s)) # check this

    all_augmented_label_group1,all_augmented_label_group2, all_augmented_input, all_augmented_aux, all_augmented_flat_aux = [], [], [], [], []
    lam = np.random.beta(1, 1)

    for group2_mask in extra_mask_s:
        group1 = np.random.choice(np.where(mask_s == True)[0], size=each_split_size, replace=True).tolist()
        group2 = np.random.choice(np.where(group2_mask == True)[0], size=each_split_size, replace=True).tolist()

        augmented_input = lam*torch.FloatTensor(all_input[group1]) + (1-lam)*torch.FloatTensor(all_input[group2])
        all_augmented_input.append(augmented_input)
        # augmented_label = lam*torch.LongTensor(all_label[group1]) + (1-lam)*torch.LongTensor(all_label[group2]) # this is not a long tensor anymore
        all_augmented_label_group1.append(torch.LongTensor(all_label[group1]))
        all_augmented_label_group2.append(torch.LongTensor(all_label[group2]))
        all_augmented_aux.append(torch.LongTensor(all_aux[group2]))
        all_augmented_flat_aux.append(torch.LongTensor(all_aux_flatten[group2]))
    # stack them now
    all_augmented_label_group1 = torch.hstack(all_augmented_label_group1)
    all_augmented_label_group2 = torch.hstack(all_augmented_label_group2)
    all_aux = torch.vstack(all_augmented_aux)
    all_aux_flatten = torch.hstack(all_augmented_flat_aux)
    all_input = torch.vstack(all_augmented_input)

    batch_input = {
        'all_augmented_label_group1': torch.LongTensor(all_augmented_label_group1),
        'all_augmented_label_group2': torch.LongTensor(all_augmented_label_group2),
        'input': torch.FloatTensor(all_input),
        'labels': torch.LongTensor(all_augmented_label_group1),
        'aux': torch.LongTensor(all_aux),
        'aux_flattened': torch.LongTensor(all_aux_flatten),
        'lam': lam
    }

    return batch_input



def train_only_mixup_with_abstract_group(train_tilted_params:TrainParameters):

    global_weight = train_tilted_params.other_params['global_weight']
    global_loss = train_tilted_params.other_params['global_loss']
    tilt_t = train_tilted_params.other_params['titled_t']
    mixup_rg = train_tilted_params.other_params['mixup_rg']
    flattened_s_to_s = {value:key for key, value in train_tilted_params.other_params['s_to_flattened_s'].items()}

    model, optimizer, device, criterion = \
        train_tilted_params.model, train_tilted_params.optimizer, train_tilted_params.device, train_tilted_params.criterion
    model.train()
    track_output = []
    track_input = []

    for i in tqdm(range(train_tilted_params.other_params['number_of_iterations'])):
        s_group_0, s_group_1 = np.random.choice(train_tilted_params.other_params['groups'], 2, replace=False)
        s_group_0 = eval(flattened_s_to_s[s_group_0].replace('.', ','))
        s_group_1 = eval(flattened_s_to_s[s_group_1].replace('.', ','))
        if train_tilted_params.fairness_function == 'demographic_parity':
            items_group_0 = sample_batch_sen_idx(train_tilted_params.other_params['all_input'],
                                         train_tilted_params.other_params['all_label'],
                                         train_tilted_params.other_params['all_aux'],
                                         train_tilted_params.other_params['all_aux_flatten'],
                                         train_tilted_params.other_params['batch_size'],
                                         s_group_0)

            items_group_1 = sample_batch_sen_idx(train_tilted_params.other_params['all_input'],
                                                 train_tilted_params.other_params['all_label'],
                                                 train_tilted_params.other_params['all_aux'],
                                                 train_tilted_params.other_params['all_aux_flatten'],
                                                 train_tilted_params.other_params['batch_size'],
                                                 s_group_1)

        elif train_tilted_params.fairness_function == 'equal_odds' or \
                train_tilted_params.fairness_function == 'equal_opportunity':
            # group splits -
            # What we want is y=0,g=g0 and y=1,g=g0
            # here items_group_0 say with batch 500 -> first 250 are 0 label and next (last) 250 are 1 label
            items_group_0, items_group_1 = sample_batch_sen_idx_with_y(train_tilted_params.other_params['all_input'],
                                                 train_tilted_params.other_params['all_label'],
                                                 train_tilted_params.other_params['all_aux'],
                                                 train_tilted_params.other_params['all_aux_flatten'],
                                                 train_tilted_params.other_params['batch_size'],
                                                 s_group_0, s_group_1)
            # What we want is y=0,g=g1 and y=1,g=g1
            # here items_group_0 say with batch 500 -> first 250 are 0 label and next (last) 250 are 0 label
            # items_group_1 = sample_batch_sen_idx_with_y(train_tilted_params.other_params['all_input'],
            #                                      train_tilted_params.other_params['all_label'],
            #                                      train_tilted_params.other_params['all_aux'],
            #                                      train_tilted_params.other_params['all_aux_flatten'],
            #                                      train_tilted_params.other_params['batch_size'],
            #                                      s_group_1)
            # group split

            # class split

        else:
            raise NotImplementedError

        for key in items_group_0.keys():
            items_group_0[key] = items_group_0[key].to(train_tilted_params.device)

        for key in items_group_1.keys():
            items_group_1[key] = items_group_1[key].to(train_tilted_params.device)

        composite_items = {
            'input': torch.vstack([items_group_0['input'], items_group_1['input']]),
            'labels': torch.hstack([items_group_0['labels'], items_group_1['labels']]),
            'aux': torch.vstack([items_group_0['aux'], items_group_1['aux']]),
            'aux_flattened': torch.hstack([items_group_0['aux_flattened'], items_group_1['aux_flattened']])
        }

        optimizer.zero_grad()
        output = model(composite_items)
        loss = criterion(output['prediction'], composite_items['labels'])



        # Mix up
        #
        # if abs(sum(items_group_1['aux'][0]) - sum(items_group_0['aux'][0])) > 1:
        #     alpha = 1.0
        #     gamma = beta(alpha, alpha)
        # else:
        #     alpha = 1.0
        #     gamma = beta(alpha, alpha)
        alpha = 1.0
        gamma = beta(alpha, alpha)


        if train_tilted_params.fairness_function == 'demographic_parity':
            batch_x_mix = items_group_0['input'] * gamma + items_group_1['input'] * (1 - gamma)
            batch_x_mix = batch_x_mix.requires_grad_(True)
            output_mixup = model({'input': batch_x_mix})
            gradx = torch.autograd.grad(output_mixup['prediction'].sum(), batch_x_mix, create_graph=True)[
                0]  # may be .sum()

            batch_x_d = items_group_1['input'] - items_group_0['input']
            grad_inn = (gradx * batch_x_d).sum(1)
            E_grad = grad_inn.mean(0)
            if train_tilted_params.other_params['method'] == 'only_mixup_with_loss_group':
                loss_reg = torch.abs(E_grad)/torch.mean(loss[len(items_group_0['input']):])
            else:
                loss_reg = torch.abs(E_grad)

        elif train_tilted_params.fairness_function == 'equal_odds' or \
            train_tilted_params.fairness_function == 'equal_opportunity':
            split_index = int(train_tilted_params.other_params['batch_size']/2)
            if train_tilted_params.fairness_function == 'equal_odds':
                gold_labels = [0,1]
            elif train_tilted_params.fairness_function == 'equal_opportunity':
                gold_labels = [1]
            else:
                raise NotImplementedError
            loss_reg = 0
            for i in gold_labels:
                if i == 0:
                    index_start = 0
                    index_end = split_index
                elif i == 1:
                    index_start = split_index
                    index_end = -1
                else:
                    raise NotImplementedError("only support binary labels!")



                batch_x_mix = items_group_0['input'][index_start:index_end] * gamma + items_group_1['input'][index_start:index_end] * (1 - gamma)
                batch_x_mix = batch_x_mix.requires_grad_(True)
                output_mixup = model({'input':batch_x_mix})
                gradx = torch.autograd.grad(output_mixup['prediction'].sum(), batch_x_mix, create_graph=True)[0] # may be .sum()

                batch_x_d = items_group_1['input'][index_start:index_end] - items_group_0['input'][index_start:index_end]
                grad_inn = (gradx * batch_x_d).sum(1)
                E_grad = grad_inn.mean(0)
                if train_tilted_params.other_params['method'] == 'only_mixup_with_loss_group':
                    loss_reg = loss_reg +  torch.abs(E_grad) / torch.mean(loss[index_start:index_end])
                else:
                    loss_reg = loss_reg + torch.abs(E_grad)
                # loss_reg = loss_reg + torch.abs(E_grad)

        else:
            raise NotImplementedError

        loss = torch.mean(loss)
        loss = loss + mixup_rg*loss_reg
        loss.backward()
        optimizer.step()


        output['loss_batch'] = loss.item()
        track_output.append(output)
        track_input.append(composite_items)

    epoch_metric_tracker, loss = train_tilted_params.per_epoch_metric(track_output,
                                                                   track_input,
                                                                   train_tilted_params.fairness_function)
    return epoch_metric_tracker, loss, global_weight, global_loss


def train_only_tilted_erm_with_abstract_group(train_tilted_params:TrainParameters):

    global_weight = train_tilted_params.other_params['global_weight']
    global_loss = train_tilted_params.other_params['global_loss']
    tilt_t = train_tilted_params.other_params['titled_t']
    flattened_s_to_s = {value: key for key, value in train_tilted_params.other_params['s_to_flattened_s'].items()}


    model, optimizer, device, criterion = \
        train_tilted_params.model, train_tilted_params.optimizer, train_tilted_params.device, train_tilted_params.criterion
    model.train()
    track_output = []
    track_input = []

    for i in tqdm(range(train_tilted_params.other_params['number_of_iterations'])):
        s = np.random.choice(train_tilted_params.other_params['groups'], 1, p=global_weight)[0]
        s_flat = eval(flattened_s_to_s[s].replace('.', ','))

        items = sample_batch_sen_idx(train_tilted_params.other_params['all_input'],
                                     train_tilted_params.other_params['all_label'],
                                     train_tilted_params.other_params['all_aux'],
                                     train_tilted_params.other_params['all_aux_flatten'],
                                     train_tilted_params.other_params['batch_size'],
                                     s_flat)

        for key in items.keys():
            items[key] = items[key].to(train_tilted_params.device)

        optimizer.zero_grad()
        output = model(items)
        loss = torch.mean(criterion(output['prediction'], items['labels']))
        loss_without_backward = torch.clone(loss).detach()

        # tilt the loss
        # loss_r_b = torch.log(torch.mean(torch.exp(tao * loss_without_backward)))/tao


        global_loss[s] =  0.2 * torch.exp(tilt_t*loss_without_backward) + 0.8 * global_loss[s]

        # weights = torch.exp(tao*loss_without_backward - tao*global_loss[s])
        global_weight = global_loss / torch.sum(global_loss)
        # global_weight = global_loss
        # loss = torch.mean(weights*loss)
        loss.backward()
        optimizer.step()

        output['loss_batch'] = loss.item()
        track_output.append(output)
        track_input.append(items)



    epoch_metric_tracker, loss = train_tilted_params.per_epoch_metric(track_output,
                                                                   track_input,
                                                                   train_tilted_params.fairness_function)


    return epoch_metric_tracker, loss, global_weight, global_loss


def custom_criterion(prediction, all_augmented_label_group1, all_augmented_label_group2, lam, criterion):
    return lam * criterion(prediction, all_augmented_label_group1) + (1 - lam) * criterion(prediction, all_augmented_label_group2)


def generate_flat_output_custom(input_iterator, attribute_id=None):
    all_label = []
    all_s = []
    all_s_flatten = []
    all_input = []

    for batch_input in input_iterator:
        all_label.append(batch_input['labels'].numpy())
        all_s.append(batch_input['aux'].numpy())
        all_s_flatten.append(batch_input['aux_flattened'].numpy())
        all_input.append(batch_input['input'].numpy())

    all_label = np.hstack(all_label)
    all_s = np.vstack(all_s)
    all_s_flatten = np.hstack(all_s_flatten)
    all_input = np.vstack(all_input)

    return all_label, all_s, all_s_flatten, all_input


def generate_similarity_matrix(iterator, model, groups, reverse_groups):
    all_train_label, all_train_s, all_train_s_flatten, all_input = generate_flat_output_custom(
        iterator)

    all_unique_groups = np.unique(all_train_s, axis=0) # unique groups - [[0,0,0], [0,0,1], [0,1,1]]
    all_average_representation = {}



    for unique_group in all_unique_groups:
        mask = generate_mask(all_train_s, unique_group)
        current_input = all_input[mask]

        batch_input = {
            'input': torch.FloatTensor(current_input),
        }

        model_hidden = model(batch_input)['hidden']


        # average_representation = np.mean(all_input[mask], axis=0) # THIS IS INCORRECT. WE NEED MODEL OUTPUT AND NOT INPUT
        average_representation = torch.mean(model_hidden, axis=0).cpu().detach().numpy() # THIS IS INCORRECT. WE NEED MODEL OUTPUT AND NOT INPUT
        all_average_representation[str(unique_group).replace('.', '')] = average_representation

    # average representation = {str([0,0,1]): average_representation, str([0,1,1]): average_representation}
    distance_lookup = {}

    for unique_group in groups:
        distance = []
        unique_group_representation = all_average_representation[reverse_groups[unique_group].replace('.','')]
        for group in groups:
            distance.append(cosine_distances([unique_group_representation], [all_average_representation[reverse_groups[group].replace('.','')]])[0][0])
        distance_lookup[unique_group] = distance
    return distance_lookup

def train_only_mixup_based_on_distance(train_tilted_params:TrainParameters):
    mixup_rg = train_tilted_params.other_params['mixup_rg']
    model, optimizer, device, criterion = \
        train_tilted_params.model, train_tilted_params.optimizer, train_tilted_params.device, train_tilted_params.criterion

    track_output = []
    track_input = []

    global_weight = train_tilted_params.other_params['global_weight']
    global_loss = train_tilted_params.other_params['global_loss']

    flattened_s_to_s = {value: key for key, value in train_tilted_params.other_params['s_to_flattened_s'].items()}

    similarity_matrix = generate_similarity_matrix(train_tilted_params.other_params['valid_iterator'], model, train_tilted_params.other_params['groups'], flattened_s_to_s)
    # similarity_matrix = generate_similarity_matrix(train_tilted_params.iterator, model, train_tilted_params.other_params['groups'], flattened_s_to_s)
    # print([np.sum(value) for key, value in similarity_matrix.items()])

    for i in tqdm(range(train_tilted_params.other_params['number_of_iterations'])):
        s_group_0 = np.random.choice(train_tilted_params.other_params['groups'], 2, replace=False)[0]
        s_group_distance = similarity_matrix[s_group_0]
        s_group_1 = np.random.choice(train_tilted_params.other_params['groups'], 1, replace=False, p=s_group_distance/np.linalg.norm(s_group_distance, 1))[0]

        if train_tilted_params.fairness_function == 'demographic_parity':
            items_group_0 = sample_batch_sen_idx(train_tilted_params.other_params['all_input'],
                                                 train_tilted_params.other_params['all_label'],
                                                 train_tilted_params.other_params['all_aux'],
                                                 train_tilted_params.other_params['all_aux_flatten'],
                                                 train_tilted_params.other_params['batch_size'],
                                                 s_group_0)

            items_group_1 = sample_batch_sen_idx(train_tilted_params.other_params['all_input'],
                                                 train_tilted_params.other_params['all_label'],
                                                 train_tilted_params.other_params['all_aux'],
                                                 train_tilted_params.other_params['all_aux_flatten'],
                                                 train_tilted_params.other_params['batch_size'],
                                                 s_group_1)

        elif train_tilted_params.fairness_function == 'equal_odds' or \
                train_tilted_params.fairness_function == 'equal_opportunity':
            # group splits -
            # What we want is y=0,g=g0 and y=1,g=g0
            # here items_group_0 say with batch 500 -> first 250 are 0 label and next (last) 250 are 1 label
            items_group_0 = titled_sample_batch_sen_idx_with_y(train_tilted_params.other_params['all_input'],
                                                        train_tilted_params.other_params['all_label'],
                                                        train_tilted_params.other_params['all_aux'],
                                                        train_tilted_params.other_params['all_aux_flatten'],
                                                        train_tilted_params.other_params['batch_size'],
                                                        s_group_0)
            # What we want is y=0,g=g1 and y=1,g=g1
            # here items_group_0 say with batch 500 -> first 250 are 0 label and next (last) 250 are 0 label
            items_group_1 = titled_sample_batch_sen_idx_with_y(train_tilted_params.other_params['all_input'],
                                                        train_tilted_params.other_params['all_label'],
                                                        train_tilted_params.other_params['all_aux'],
                                                        train_tilted_params.other_params['all_aux_flatten'],
                                                        train_tilted_params.other_params['batch_size'],
                                                        s_group_1)
            # group split

            # class split

        else:
            raise NotImplementedError

        for key in items_group_0.keys():
            items_group_0[key] = items_group_0[key].to(train_tilted_params.device)

        for key in items_group_1.keys():
            items_group_1[key] = items_group_1[key].to(train_tilted_params.device)

        composite_items = {
            'input': torch.vstack([items_group_0['input'], items_group_1['input']]),
            'labels': torch.hstack([items_group_0['labels'], items_group_1['labels']]),
            'aux': torch.vstack([items_group_0['aux'], items_group_1['aux']]),
            'aux_flattened': torch.hstack([items_group_0['aux_flattened'], items_group_1['aux_flattened']])
        }

        optimizer.zero_grad()
        output = model(composite_items)
        loss = criterion(output['prediction'], composite_items['labels'])

        # Mix up
        #
        # if abs(sum(items_group_1['aux'][0]) - sum(items_group_0['aux'][0])) > 1:
        #     alpha = 1.0
        #     gamma = beta(alpha, alpha)
        # else:
        #     alpha = 1.0
        #     gamma = beta(alpha, alpha)
        alpha = 1.0
        gamma = beta(alpha, alpha)

        if train_tilted_params.fairness_function == 'demographic_parity':
            batch_x_mix = items_group_0['input'] * gamma + items_group_1['input'] * (1 - gamma)
            batch_x_mix = batch_x_mix.requires_grad_(True)
            output_mixup = model({'input': batch_x_mix})
            gradx = torch.autograd.grad(output_mixup['prediction'].sum(), batch_x_mix, create_graph=True)[
                0]  # may be .sum()

            batch_x_d = items_group_1['input'] - items_group_0['input']
            grad_inn = (gradx * batch_x_d).sum(1)
            E_grad = grad_inn.mean(0)
            if train_tilted_params.other_params['method'] == 'only_mixup_with_loss_group':
                loss_reg = torch.abs(E_grad) / torch.mean(loss[len(items_group_0['input']):])
            else:
                loss_reg = torch.abs(E_grad)

        elif train_tilted_params.fairness_function == 'equal_odds' or \
                train_tilted_params.fairness_function == 'equal_opportunity':
            split_index = int(train_tilted_params.other_params['batch_size'] / 2)
            if train_tilted_params.fairness_function == 'equal_odds':
                gold_labels = [0, 1]
            elif train_tilted_params.fairness_function == 'equal_opportunity':
                gold_labels = [1]
            else:
                raise NotImplementedError
            loss_reg = 0
            for i in gold_labels:
                if i == 0:
                    index_start = 0
                    index_end = split_index
                elif i == 1:
                    index_start = split_index
                    index_end = -1
                else:
                    raise NotImplementedError("only support binary labels!")

                batch_x_mix = items_group_0['input'][index_start:index_end] * gamma + items_group_1['input'][
                                                                                      index_start:index_end] * (
                                          1 - gamma)
                batch_x_mix = batch_x_mix.requires_grad_(True)
                output_mixup = model({'input': batch_x_mix})
                gradx = torch.autograd.grad(output_mixup['prediction'].sum(), batch_x_mix, create_graph=True)[
                    0]  # may be .sum()

                batch_x_d = items_group_1['input'][index_start:index_end] - items_group_0['input'][
                                                                            index_start:index_end]
                grad_inn = (gradx * batch_x_d).sum(1)
                E_grad = grad_inn.mean(0)
                if train_tilted_params.other_params['method'] == 'only_mixup_with_loss_group':
                    loss_reg = loss_reg + torch.abs(E_grad) / torch.mean(loss[index_start:index_end])
                else:
                    loss_reg = loss_reg + torch.abs(E_grad)
                # loss_reg = loss_reg + torch.abs(E_grad)

        else:
            raise NotImplementedError

        loss = torch.mean(loss)
        loss = loss + mixup_rg * loss_reg
        loss.backward()
        optimizer.step()

        output['loss_batch'] = loss.item()
        track_output.append(output)
        track_input.append(composite_items)

    epoch_metric_tracker, loss = train_tilted_params.per_epoch_metric(track_output,
                                                                      track_input,
                                                                      train_tilted_params.fairness_function)
    return epoch_metric_tracker, loss, global_weight, global_loss



def train_with_mixup_only_one_group_based_distance(train_tilted_params:TrainParameters):

    global_weight = train_tilted_params.other_params['global_weight']
    global_loss = train_tilted_params.other_params['global_loss']
    tilt_t = train_tilted_params.other_params['titled_t']
    mixup_rg = train_tilted_params.other_params['mixup_rg']


    model, optimizer, device, criterion = \
        train_tilted_params.model, train_tilted_params.optimizer, train_tilted_params.device, train_tilted_params.criterion
    model.train()
    track_output = []
    track_input = []

    flattened_s_to_s = {value: key for key, value in train_tilted_params.other_params['s_to_flattened_s'].items()}

    similarity_matrix = generate_similarity_matrix(train_tilted_params.other_params['valid_iterator'], model,
                                                   train_tilted_params.other_params['groups'], flattened_s_to_s)

    for i in tqdm(range(train_tilted_params.other_params['number_of_iterations'])):
        # s_group_0, s_group_1 = np.random.choice(train_tilted_params.other_params['groups'], 2, p=global_weight, replace=False)
        # s_group_0 = np.random.choice(train_tilted_params.other_params['groups'], 1, p=global_weight, replace=False)[0]
        s_group_0 = np.random.choice(train_tilted_params.other_params['groups'], 2, replace=False,  p=global_weight)[0]
        s_group_distance = similarity_matrix[s_group_0]
        s_group_1 = np.random.choice(train_tilted_params.other_params['groups'], 1, replace=False,
                                     p=s_group_distance / np.linalg.norm(s_group_distance, 1))[0]


        # break_flag = True
        # while break_flag:
        #     temp_global_weight = np.reciprocal(global_weight)
        #     temp_global_weight = temp_global_weight / np.linalg.norm(temp_global_weight, 1)# Invert all weights
        #     s_group_1 = np.random.choice(train_tilted_params.other_params['groups'], 1, p=temp_global_weight, replace=False)[0]
        #
        #     if s_group_1 != s_group_0:
        #         break_flag = False
        # s = F.gumbel_softmax(global_weight, tau=1/10, hard=True).nonzero()[0][0].item()

        if train_tilted_params.fairness_function == 'demographic_parity':
            items_group_0 = sample_batch_sen_idx(train_tilted_params.other_params['all_input'],
                                                 train_tilted_params.other_params['all_label'],
                                                 train_tilted_params.other_params['all_aux'],
                                                 train_tilted_params.other_params['all_aux_flatten'],
                                                 train_tilted_params.other_params['batch_size'],
                                                 s_group_0)

            items_group_1 = sample_batch_sen_idx(train_tilted_params.other_params['all_input'],
                                                 train_tilted_params.other_params['all_label'],
                                                 train_tilted_params.other_params['all_aux'],
                                                 train_tilted_params.other_params['all_aux_flatten'],
                                                 train_tilted_params.other_params['batch_size'],
                                                 s_group_1)

        elif train_tilted_params.fairness_function == 'equal_odds' or \
                train_tilted_params.fairness_function == 'equal_opportunity':
            # group splits -
            # What we want is y=0,g=g0 and y=1,g=g0
            # here items_group_0 say with batch 500 -> first 250 are 0 label and next (last) 250 are 1 label
            items_group_0 = titled_sample_batch_sen_idx_with_y(train_tilted_params.other_params['all_input'],
                                                        train_tilted_params.other_params['all_label'],
                                                        train_tilted_params.other_params['all_aux'],
                                                        train_tilted_params.other_params['all_aux_flatten'],
                                                        train_tilted_params.other_params['batch_size'],
                                                        s_group_0)
            # What we want is y=0,g=g1 and y=1,g=g1
            # here items_group_0 say with batch 500 -> first 250 are 0 label and next (last) 250 are 0 label
            items_group_1 = titled_sample_batch_sen_idx_with_y(train_tilted_params.other_params['all_input'],
                                                        train_tilted_params.other_params['all_label'],
                                                        train_tilted_params.other_params['all_aux'],
                                                        train_tilted_params.other_params['all_aux_flatten'],
                                                        train_tilted_params.other_params['batch_size'],
                                                        s_group_1)
            # group split

            # class split

        else:
            raise NotImplementedError

        for key in items_group_0.keys():
            items_group_0[key] = items_group_0[key].to(train_tilted_params.device)

        for key in items_group_1.keys():
            items_group_1[key] = items_group_1[key].to(train_tilted_params.device)

        # composite_items = {
        #     'input': torch.vstack([items_group_0['input'], items_group_1['input']]),
        #     'labels': torch.hstack([items_group_0['labels'], items_group_1['labels']]),
        #     'aux': torch.vstack([items_group_0['aux'], items_group_1['aux']]),
        #     'aux_flattened': torch.hstack([items_group_0['aux_flattened'], items_group_1['aux_flattened']])
        # }

        optimizer.zero_grad()
        output = model(items_group_0)
        loss = criterion(output['prediction'], items_group_0['labels'])
        loss_without_backward = torch.mean(torch.clone(loss).detach())
        # loss_without_backward_group_0 = torch.mean(loss_without_backward[:len(items_group_0['input'])])
        # loss_without_backward_group_1 = torch.mean(loss_without_backward[len(items_group_0['input']):])
        loss = torch.mean(loss)



        global_loss[s_group_0] =  0.2 * torch.exp(tilt_t*loss_without_backward) + 0.8 * global_loss[s_group_0]
        # global_loss[s_group_1] =  0.2 * torch.exp(tilt_t*loss_without_backward_group_1) + 0.8 * global_loss[s_group_1]

        # weights = torch.exp(tao*loss_without_backward - tao*global_loss[s])
        global_weight = global_loss / torch.sum(global_loss)
        # global_weight = global_loss
        # loss = torch.mean(weights*loss)


        # fair mixup now
        alpha = 1
        gamma = beta(alpha, alpha)

        if train_tilted_params.fairness_function == 'demographic_parity':
            batch_x_mix = items_group_0['input'] * gamma + items_group_1['input'] * (1 - gamma)
            batch_x_mix = batch_x_mix.requires_grad_(True)
            output_mixup = model({'input': batch_x_mix})
            gradx = torch.autograd.grad(output_mixup['prediction'].sum(), batch_x_mix, create_graph=True)[
                0]  # may be .sum()

            batch_x_d = items_group_1['input'] - items_group_0['input']
            grad_inn = (gradx * batch_x_d).sum(1)
            E_grad = grad_inn.mean(0)
            loss_reg = torch.abs(E_grad)

        elif train_tilted_params.fairness_function == 'equal_odds' or \
                train_tilted_params.fairness_function == 'equal_opportunity':
            split_index = int(train_tilted_params.other_params['batch_size'] / 2)
            if train_tilted_params.fairness_function == 'equal_odds':
                gold_labels = [0, 1]
            elif train_tilted_params.fairness_function == 'equal_opportunity':
                gold_labels = [1]
            else:
                raise NotImplementedError
            loss_reg = 0
            for i in gold_labels:
                if i == 0:
                    index_start = 0
                    index_end = split_index
                elif i == 1:
                    index_start = split_index
                    index_end = -1
                else:
                    raise NotImplementedError("only support binary labels!")

                batch_x_mix = items_group_0['input'][index_start:index_end] * gamma + items_group_1['input'][
                                                                                      index_start:index_end] * (
                                          1 - gamma)
                batch_x_mix = batch_x_mix.requires_grad_(True)
                output_mixup = model({'input': batch_x_mix})
                gradx = torch.autograd.grad(output_mixup['prediction'].sum(), batch_x_mix, create_graph=True)[
                    0]  # may be .sum()

                batch_x_d = items_group_1['input'][index_start:index_end] - items_group_0['input'][
                                                                            index_start:index_end]
                grad_inn = (gradx * batch_x_d).sum(1)
                E_grad = grad_inn.mean(0)
                loss_reg = loss_reg + torch.abs(E_grad)

        else:
            raise NotImplementedError


        loss = loss + mixup_rg*loss_reg



        loss.backward()
        optimizer.step()

        output['loss_batch'] = loss.item()
        track_output.append(output)
        track_input.append(items_group_0)



    epoch_metric_tracker, loss = train_tilted_params.per_epoch_metric(track_output,
                                                                   track_input,
                                                                   train_tilted_params.fairness_function)



    return epoch_metric_tracker, loss, global_weight, global_loss


def train_only_tilted_erm_with_mixup_augmentation(train_tilted_params:TrainParameters):

    global_weight = train_tilted_params.other_params['global_weight']
    global_loss = train_tilted_params.other_params['global_loss']
    tilt_t = train_tilted_params.other_params['titled_t']
    flattened_s_to_s = {value: key for key, value in train_tilted_params.other_params['s_to_flattened_s'].items()}


    model, optimizer, device, criterion = \
        train_tilted_params.model, train_tilted_params.optimizer, train_tilted_params.device, train_tilted_params.criterion
    model.train()
    track_output = []
    track_input = []




    for i in tqdm(range(train_tilted_params.other_params['number_of_iterations'])):
        s = np.random.choice(train_tilted_params.other_params['groups'], 1, p=global_weight)[0]
        s_flat = eval(flattened_s_to_s[s].replace('.', ','))

        items = sample_batch_sen_idx_with_augmentation(train_tilted_params.other_params['all_input'],
                                     train_tilted_params.other_params['all_label'],
                                     train_tilted_params.other_params['all_aux'],
                                     train_tilted_params.other_params['all_aux_flatten'],
                                     train_tilted_params.other_params['batch_size'],
                                     s_flat)

        for key in items.keys():
            if key == 'lam':
                continue
            items[key] = items[key].to(train_tilted_params.device)

        optimizer.zero_grad()
        output = model(items)
        loss = torch.mean(custom_criterion(output['prediction'], items['all_augmented_label_group1'], items['all_augmented_label_group2'], items['lam'], criterion))
        loss_without_backward = torch.clone(loss).detach()

        # tilt the loss
        # loss_r_b = torch.log(torch.mean(torch.exp(tao * loss_without_backward)))/tao


        global_loss[s] =  0.2 * torch.exp(tilt_t*loss_without_backward) + 0.8 * global_loss[s]

        # weights = torch.exp(tao*loss_without_backward - tao*global_loss[s])
        global_weight = global_loss / torch.sum(global_loss)
        # global_weight = global_loss
        # loss = torch.mean(weights*loss)
        loss.backward()
        optimizer.step()

        output['loss_batch'] = loss.item()
        track_output.append(output)
        track_input.append(items)



    epoch_metric_tracker, loss = train_tilted_params.per_epoch_metric(track_output,
                                                                   track_input,
                                                                   train_tilted_params.fairness_function)


    return epoch_metric_tracker, loss, global_weight, global_loss



