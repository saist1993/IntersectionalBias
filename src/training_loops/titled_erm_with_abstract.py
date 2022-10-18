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

def sample_batch_sen_idx_with_augmentation_with_lambda(all_input, all_label, all_aux, all_aux_flatten, batch_size, s, group_weight_lambda):
    all_extra_combinations = generate_combinations(s, k=1)
    extra_mask_s =[generate_mask(all_aux, mask_pattern) for mask_pattern in all_extra_combinations] # this stores all the combinations
    mask_s = generate_mask(all_aux,s)


    # now comes the augmentations

    all_group_examples = [np.random.choice(np.where(mask == True)[0], size=batch_size, replace=True).tolist()
                          for mask in extra_mask_s]
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



    return batch_input


def custom_criterion_lambda_weights(prediction, augmented_labels, group_weight_lambda, criterion):
    temp = [weight*criterion(prediction, output) for output, weight in zip(augmented_labels, group_weight_lambda)]
    return torch.stack(temp, dim=0).sum(dim=0)



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
        s_flat = eval(flattened_s_to_s[s].replace('.', ','))

        items = sample_batch_sen_idx_with_augmentation_with_lambda(train_tilted_params.other_params['all_input'],
                                     train_tilted_params.other_params['all_label'],
                                     train_tilted_params.other_params['all_aux'],
                                     train_tilted_params.other_params['all_aux_flatten'],
                                     train_tilted_params.other_params['batch_size'],
                                     s_flat,
                                     group_to_lambda_weight[s])

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