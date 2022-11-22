import math
import torch
import pandas as pd
from tqdm.auto import tqdm
from scipy import optimize
from numpy.random import beta
from collections import Counter
import torch.nn.functional as F
from itertools import combinations
from .common_functionality import *
from metrics import fairness_utils
from sklearn.metrics.pairwise import cosine_distances

def sample_batch_sen_idx_with_augmentation_with_lambda(all_input, all_label, all_aux, all_aux_flatten, batch_size, s, group_weight_lambda, scalar):
    all_extra_combinations = generate_combinations(s, k=1)
    extra_mask_s =[generate_mask(all_aux, mask_pattern) for mask_pattern in all_extra_combinations] # this stores all the combinations
    mask_s = generate_mask(all_aux,s)


    # now comes the augmentations

    all_group_examples = [np.random.choice(np.where(mask == True)[0], size=batch_size, replace=True).tolist()
                          for mask in extra_mask_s]
    all_group_examples_average_representation = [np.mean(all_input[group], axis=0) for group in all_group_examples]

    augmented_input =  torch.FloatTensor(np.sum([lambda_weight * all_input[group] for group, lambda_weight in zip(all_group_examples, group_weight_lambda)],
           axis=0))
    augmented_output = [torch.LongTensor(all_label[group]) for group in all_group_examples]

    batch_input = {
        'augmented_labels': augmented_output,
        'input': torch.FloatTensor(augmented_input),
        'labels': torch.LongTensor(torch.LongTensor(all_label[all_group_examples[0]])),
        'aux': torch.LongTensor(all_aux[all_group_examples[0]]),
        'aux_flattened': torch.LongTensor(all_aux_flatten[all_group_examples[0]]),
        'group_weight_lambda': group_weight_lambda
    }



    return batch_input, np.sum(mask_s)


def sample_batch_sen_idx_with_augmentation_with_lambda_custom_with_positive_and_negative_seperate\
                (all_input, all_label, all_aux, all_aux_flatten, batch_size, s, group_weight_lambda, scalar, input_s):
    all_extra_combinations = generate_combinations(s, k=1)
    extra_mask_s =[generate_mask(all_aux, mask_pattern) for mask_pattern in all_extra_combinations] # this stores all the combinations
    mask_s = generate_mask(all_aux,s)

    # positive and negative examples in input_s
    number_of_positive_examples = torch.sum(input_s['labels']).item()
    number_of_negative_examples = input_s['labels'].shape[0] - number_of_positive_examples

    all_group_examples_positive = []
    all_group_examples_negative = []

    for mask in extra_mask_s:
        mask_positive = np.logical_and(all_label == 1, mask == True)
        mask_negative = np.logical_and(all_label == 0, mask == True)
        positive_examples =  np.random.choice(np.where(mask_positive == True)[0], size=number_of_positive_examples, replace=True) # find somehow only positive example
        negative_examples =  np.random.choice(np.where(mask_negative == True)[0], size=number_of_negative_examples, replace=True) # find somehow only positive example
        all_group_examples_positive.append(positive_examples)
        all_group_examples_negative.append(negative_examples)
        # stack them
    # now comes the augmentations


    try:
        augmented_input_positive =  torch.FloatTensor(np.sum([lambda_weight[0] * all_input[group] for group, lambda_weight in zip(all_group_examples_positive, group_weight_lambda)],axis=0))
    except IndexError:
        print("here")
    augmented_input_negative =  torch.FloatTensor(np.sum([lambda_weight[1] * all_input[group] for group, lambda_weight in zip(all_group_examples_negative, group_weight_lambda)],axis=0))

    augmented_output_positive = [torch.LongTensor(all_label[group]) for group in all_group_examples_positive]
    augmented_output_negative = [torch.LongTensor(all_label[group]) for group in all_group_examples_negative]


    augmented_aux_flattened_positive = [torch.LongTensor(all_aux_flatten[group]) for group in all_group_examples_positive]
    augmented_aux_flattened_negative = [torch.LongTensor(all_aux_flatten[group]) for group in all_group_examples_negative]


    augmented_aux_positive = [torch.LongTensor(all_aux[group]) for group in all_group_examples_positive]
    augmented_aux_negative = [torch.LongTensor(all_aux[group]) for group in all_group_examples_negative]

    augmented_input = torch.vstack([augmented_input_positive, augmented_input_negative])
    augmented_aux = torch.vstack([augmented_aux_positive[0], augmented_aux_negative[0]])
    augmented_output = torch.hstack([augmented_output_positive[0], augmented_output_negative[0]])
    augmented_aux_flat = torch.hstack([augmented_aux_flattened_positive[0], augmented_aux_flattened_negative[0]])

    shuffle_index = torch.randperm(augmented_input.shape[0])

    batch_input = {
        'augmented_labels': torch.LongTensor(augmented_output)[shuffle_index],
        'input': torch.FloatTensor(augmented_input)[shuffle_index],
        'labels': torch.LongTensor(augmented_output)[shuffle_index],
        'aux': torch.LongTensor(augmented_aux)[shuffle_index],
        'aux_flattened': torch.LongTensor(augmented_aux_flat)[shuffle_index],
        'group_weight_lambda': group_weight_lambda
    }



    return batch_input, np.sum(mask_s)



def sample_batch_sen_idx_with_augmentation_with_lambda_custom(all_input, all_label, all_aux, all_aux_flatten, batch_size, s, group_weight_lambda, scalar, input_s):
    all_extra_combinations = generate_combinations(s, k=1)
    extra_mask_s =[generate_mask(all_aux, mask_pattern) for mask_pattern in all_extra_combinations] # this stores all the combinations
    mask_s = generate_mask(all_aux,s)


    # now comes the augmentations

    all_group_examples = [np.random.choice(np.where(mask == True)[0], size=batch_size, replace=True).tolist()
                          for mask in extra_mask_s]
    all_group_examples_average_representation = [np.mean(all_input[group], axis=0) for group in all_group_examples]
    average_representation_s = np.mean(input_s.detach().cpu().numpy(), axis=0)

    P = np.matrix(all_group_examples_average_representation)
    Ps = np.array(average_representation_s)

    def objective(x):
        x = np.array([x])
        res = Ps - np.dot(x, P)
        return np.asarray(res).flatten()

    def main():
        x = np.array([1 for i in range(len(all_aux[0]))] / np.sum([1 for i in range(len(all_aux[0]))]))
        final_lambda_weights = optimize.least_squares(objective, x).x
        return final_lambda_weights

    group_weight_lambda = main()


    augmented_input =  torch.FloatTensor(np.sum([lambda_weight * all_input[group] for group, lambda_weight in zip(all_group_examples, group_weight_lambda)],
           axis=0))
    augmented_output = [torch.LongTensor(all_label[group]) for group in all_group_examples]

    batch_input = {
        'augmented_labels': augmented_output,
        'input': torch.FloatTensor(augmented_input),
        'labels': torch.LongTensor(torch.LongTensor(all_label[all_group_examples[0]])),
        'aux': torch.LongTensor(all_aux[all_group_examples[0]]),
        'aux_flattened': torch.LongTensor(all_aux_flatten[all_group_examples[0]]),
        'group_weight_lambda': group_weight_lambda
    }



    return batch_input, np.sum(mask_s)

def sample_batch_sen_idx_custom(all_input, all_label, all_aux, all_aux_flatten, batch_size, s):
    """
        This will sample batch size number of elements from input with given s!
    """
    relevant_index = np.random.choice(np.where(all_aux_flatten==s)[0], size=batch_size, replace=True).tolist()
    # THIS IS DIFFERENT. IN ORIGINAL VERSION IT IS REPLACEMENT TRUE
    batch_input = {
        'labels': torch.LongTensor(all_label[relevant_index]),
        'input': torch.FloatTensor(all_input[relevant_index]),
        'aux': torch.LongTensor(all_aux[relevant_index]),
        'aux_flattened': torch.LongTensor(all_aux_flatten[relevant_index])
    }

    return batch_input


def custom_criterion_lambda_weights_old(prediction, augmented_labels, group_weight_lambda, criterion):
    temp = [weight*criterion(prediction, output) for output, weight in zip(augmented_labels, group_weight_lambda)]
    return torch.stack(temp, dim=0).sum(dim=0)


def custom_criterion_lambda_weights(prediction, augmented_labels, group_weight_lambda, criterion):
    label = torch.stack([weight*F.one_hot(output) for output, weight in zip(augmented_labels, group_weight_lambda)], dim=0).sum(dim=0)
    return criterion(prediction, label)


def train_only_tilted_erm_with_mixup_augmentation_lambda_weights(train_tilted_params:TrainParameters):

    global_weight = train_tilted_params.other_params['global_weight']
    global_loss = train_tilted_params.other_params['global_loss']
    tilt_t = train_tilted_params.other_params['titled_t']
    group_to_lambda_weight = train_tilted_params.other_params['group_to_lambda_weight']
    flattened_s_to_s = {value: key for key, value in train_tilted_params.other_params['s_to_flattened_s'].items()}


    model, optimizer, device, criterion = \
        train_tilted_params.model, train_tilted_params.optimizer, train_tilted_params.device, train_tilted_params.criterion
    model.train()
    track_output = []
    track_input = []




    for i in tqdm(range(train_tilted_params.other_params['number_of_iterations'])):
        s = np.random.choice(train_tilted_params.other_params['groups'], 1, p=global_weight)[0]
        s_flat = flattened_s_to_s[s]
        # s_flat = train_tilted_params.other_params['s_to_flattened_s'][s]

        items, _ = sample_batch_sen_idx_with_augmentation_with_lambda(train_tilted_params.other_params['all_input'],
                                     train_tilted_params.other_params['all_label'],
                                     train_tilted_params.other_params['all_aux'],
                                     train_tilted_params.other_params['all_aux_flatten'],
                                     train_tilted_params.other_params['batch_size'],
                                     s_flat,
                                     group_to_lambda_weight[s],
                                     train_tilted_params.other_params['scalar'])



        for key in items.keys():
            if key == 'group_weight_lambda' or key == 'augmented_labels':
                continue
            items[key] = items[key].to(train_tilted_params.device)

        optimizer.zero_grad()
        output = model(items)
        loss = torch.mean(custom_criterion_lambda_weights(output['prediction'], items['augmented_labels'], items['group_weight_lambda'], criterion))


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


def train_only_tilted_erm_with_mixup_augmentation_lambda_weights_v2(train_tilted_params:TrainParameters):

    global_weight = train_tilted_params.other_params['global_weight']
    global_loss = train_tilted_params.other_params['global_loss']
    tilt_t = train_tilted_params.other_params['titled_t']
    group_to_lambda_weight = train_tilted_params.other_params['group_to_lambda_weight']
    flattened_s_to_s = {value: key for key, value in train_tilted_params.other_params['s_to_flattened_s'].items()}


    model, optimizer, device, criterion = \
        train_tilted_params.model, train_tilted_params.optimizer, train_tilted_params.device, train_tilted_params.criterion
    model.train()
    track_output = []
    track_input = []




    for i in tqdm(range(train_tilted_params.other_params['number_of_iterations'])):
        s = np.random.choice(train_tilted_params.other_params['groups'], 1, p=global_weight)[0]
        s_flat = flattened_s_to_s[s]

        items_only_s = sample_batch_sen_idx_custom(train_tilted_params.other_params['all_input'],
                                                   train_tilted_params.other_params['all_label'],
                                                   train_tilted_params.other_params['all_aux'],
                                                   train_tilted_params.other_params['all_aux_flatten'],
                                                   train_tilted_params.other_params['batch_size'],
                                                   s)

        items, size_of_s = sample_batch_sen_idx_with_augmentation_with_lambda(train_tilted_params.other_params['all_input'],
                                     train_tilted_params.other_params['all_label'],
                                     train_tilted_params.other_params['all_aux'],
                                     train_tilted_params.other_params['all_aux_flatten'],
                                     train_tilted_params.other_params['batch_size'],
                                     s_flat,
                                     group_to_lambda_weight[s],
                                     train_tilted_params.other_params['scalar'],
                                     items_only_s['input'])



        for key in items.keys():
            if key == 'group_weight_lambda' or key == 'augmented_labels':
                continue
            items[key] = items[key].to(train_tilted_params.device)

        optimizer.zero_grad()
        output = model(items)
        output_only_s = model(items_only_s)
        # loss = (1-size_of_s*1.0/len(train_tilted_params.other_params['all_label']))*torch.mean(custom_criterion_lambda_weights(output['prediction'], items['augmented_labels'], items['group_weight_lambda'], criterion))

        loss = torch.mean(criterion(output_only_s['prediction'], items_only_s['labels']))
        loss += torch.mean(custom_criterion_lambda_weights(output['prediction'], items['augmented_labels'],
                                                          items['group_weight_lambda'], criterion))
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


def train_only_tilted_erm_with_mixup_augmentation_lambda_weights_v3(train_tilted_params:TrainParameters):

    global_weight = train_tilted_params.other_params['global_weight']
    global_loss = train_tilted_params.other_params['global_loss']
    tilt_t = train_tilted_params.other_params['titled_t']
    group_to_lambda_weight = train_tilted_params.other_params['group_to_lambda_weight']
    flattened_s_to_s = {value: key for key, value in train_tilted_params.other_params['s_to_flattened_s'].items()}


    model, optimizer, device, criterion = \
        train_tilted_params.model, train_tilted_params.optimizer, train_tilted_params.device, train_tilted_params.criterion
    model.train()
    track_output = []
    track_input = []




    for i in tqdm(range(train_tilted_params.other_params['number_of_iterations'])):
        s = np.random.choice(train_tilted_params.other_params['groups'], 1, p=global_weight)[0]
        s_flat = flattened_s_to_s[s]

        items_only_s = sample_batch_sen_idx_custom(train_tilted_params.other_params['all_input'],
                                                   train_tilted_params.other_params['all_label'],
                                                   train_tilted_params.other_params['all_aux'],
                                                   train_tilted_params.other_params['all_aux_flatten'],
                                                   train_tilted_params.other_params['batch_size'],
                                                   s)

        items, size_of_s = sample_batch_sen_idx_with_augmentation_with_lambda_custom(train_tilted_params.other_params['all_input'],
                                     train_tilted_params.other_params['all_label'],
                                     train_tilted_params.other_params['all_aux'],
                                     train_tilted_params.other_params['all_aux_flatten'],
                                     train_tilted_params.other_params['batch_size'],
                                     s_flat,
                                     group_to_lambda_weight[s],
                                     train_tilted_params.other_params['scalar'],
                                     items_only_s['input'])



        for key in items.keys():
            if key == 'group_weight_lambda' or key == 'augmented_labels':
                continue
            items[key] = items[key].to(train_tilted_params.device)

        optimizer.zero_grad()
        output = model(items)
        output_only_s = model(items_only_s)
        # loss = (1-size_of_s*1.0/len(train_tilted_params.other_params['all_label']))*torch.mean(custom_criterion_lambda_weights(output['prediction'], items['augmented_labels'], items['group_weight_lambda'], criterion))

        loss = torch.mean(criterion(output_only_s['prediction'], items_only_s['labels']))
        loss += torch.mean(custom_criterion_lambda_weights(output['prediction'], items['augmented_labels'],
                                                          items['group_weight_lambda'], criterion))
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


def train_only_tilted_erm_with_mixup_augmentation_lambda_weights_v4(train_tilted_params:TrainParameters):

    global_weight = train_tilted_params.other_params['global_weight']
    global_loss = train_tilted_params.other_params['global_loss']
    tilt_t = train_tilted_params.other_params['titled_t']
    group_to_lambda_weight = train_tilted_params.other_params['group_to_lambda_weight']
    flattened_s_to_s = {value: key for key, value in train_tilted_params.other_params['s_to_flattened_s'].items()}


    model, optimizer, device, criterion = \
        train_tilted_params.model, train_tilted_params.optimizer, train_tilted_params.device, train_tilted_params.criterion
    model.train()
    track_output = []
    track_input = []




    for i in tqdm(range(train_tilted_params.other_params['number_of_iterations'])):
        s = np.random.choice(train_tilted_params.other_params['groups'], 1, p=global_weight)[0]
        s_flat = flattened_s_to_s[s]

        items_only_s = sample_batch_sen_idx_custom(train_tilted_params.other_params['all_input'],
                                                   train_tilted_params.other_params['all_label'],
                                                   train_tilted_params.other_params['all_aux'],
                                                   train_tilted_params.other_params['all_aux_flatten'],
                                                   train_tilted_params.other_params['batch_size'],
                                                   s)

        items, size_of_s = sample_batch_sen_idx_with_augmentation_with_lambda_custom_with_positive_and_negative_seperate(train_tilted_params.other_params['all_input'],
                                     train_tilted_params.other_params['all_label'],
                                     train_tilted_params.other_params['all_aux'],
                                     train_tilted_params.other_params['all_aux_flatten'],
                                     train_tilted_params.other_params['batch_size'],
                                     s_flat,
                                     group_to_lambda_weight[s],
                                     train_tilted_params.other_params['scalar'],
                                     items_only_s)



        for key in items.keys():
            if key == 'group_weight_lambda' or key == 'augmented_labels':
                continue
            items[key] = items[key].to(train_tilted_params.device)

        optimizer.zero_grad()
        output = model(items)
        output_only_s = model(items_only_s)
        # loss = (1-size_of_s*1.0/len(train_tilted_params.other_params['all_label']))*torch.mean(custom_criterion_lambda_weights(output['prediction'], items['augmented_labels'], items['group_weight_lambda'], criterion))

        loss = torch.mean(criterion(output_only_s['prediction'], items_only_s['labels']))
        loss += torch.mean(criterion(output['prediction'], items['augmented_labels']))
        loss_without_backward = torch.clone(loss).detach()



        # tilt the loss
        # loss_r_b = torch.log(torch.mean(torch.exp(tao * loss_without_backward)))/tao


        global_loss[s] =  0.2 * torch.exp(tilt_t*loss_without_backward) + 0.8 * global_loss[s]

        # weights = torch.exp(tao*loss_without_backward - tao*global_loss[s])
        global_weight = global_loss / torch.sum(global_loss)
        # global_weight = global_loss
        # loss = torch.mean(weights*loss)
        loss = loss
        loss.backward()
        optimizer.step()

        output['loss_batch'] = loss.item()
        track_output.append(output)
        track_input.append(items)



    epoch_metric_tracker, loss = train_tilted_params.per_epoch_metric(track_output,
                                                                   track_input,
                                                                   train_tilted_params.fairness_function)


    return epoch_metric_tracker, loss, global_weight, global_loss
