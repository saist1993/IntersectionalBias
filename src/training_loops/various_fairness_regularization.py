import torch
from tqdm.auto import tqdm
from numpy.random import beta
from .common_functionality import *




def erm_super_group_with_simplified_fairness_loss(train_tilted_params:TrainParameters):
    mixup_rg = train_tilted_params.other_params['mixup_rg']
    global_weight = train_tilted_params.other_params['global_weight']
    global_loss = train_tilted_params.other_params['global_loss']
    model, optimizer, device, criterion = \
        train_tilted_params.model, train_tilted_params.optimizer, train_tilted_params.device, train_tilted_params.criterion
    track_output = []
    track_input = []


    for i in tqdm(range(train_tilted_params.other_params['number_of_iterations'])):
        flag = True
        while flag:
            s_group_0, s_group_1 = eval(
                np.random.choice(train_tilted_params.other_params['groups_matrix'].reshape(1, -1)[0], 1, replace=False,
                                 p=global_weight.reshape(1, -1)[0])[0])
            if s_group_1 != s_group_0:
                flag = False

        # items_group_0 = sample_batch_sen_idx(train_tilted_params.other_params['all_input'],
        #                                      train_tilted_params.other_params['all_label'],
        #                                      train_tilted_params.other_params['all_aux'],
        #                                      train_tilted_params.other_params['all_aux_flatten'],
        #                                      train_tilted_params.other_params['batch_size'],
        #                                      s_group_0)
        #
        # items_group_1 = sample_batch_sen_idx(train_tilted_params.other_params['all_input'],
        #                                      train_tilted_params.other_params['all_label'],
        #                                      train_tilted_params.other_params['all_aux'],
        #                                      train_tilted_params.other_params['all_aux_flatten'],
        #                                      train_tilted_params.other_params['batch_size'],
        #                                      s_group_1)

        items_group_0, items_group_1 = sample_data(train_tilted_params, s_group_0, s_group_1)

        for key in items_group_0.keys():
            items_group_0[key] = items_group_0[key].to(train_tilted_params.device)

        for key in items_group_1.keys():
            items_group_1[key] = items_group_1[key].to(train_tilted_params.device)



        optimizer.zero_grad()
        output_group_0 = model(items_group_0)
        output_group_1 = model(items_group_1)
        loss_group_0 = criterion(output_group_0['prediction'], items_group_0['labels'])
        loss_group_1 = criterion(output_group_1['prediction'], items_group_1['labels'])

        # if train_tilted_params.fairness_function == 'equal_opportunity':
        #
        #     fairness_reg = simplified_fairness_loss(fairness_function=train_tilted_params.fairness_function,
        #                                   loss = torch.hstack([loss_group_0, loss_group_1]),
        #                                   preds = torch.vstack([output_group_0['prediction'], output_group_1['prediction']]),
        #                                   aux = torch.hstack([items_group_0['aux_flattened'], items_group_1['aux_flattened']]),
        #                                   group1_pattern = s_group_0,
        #                                   group2_pattern = s_group_1,
        #                                   label = torch.hstack([items_group_0['labels'], items_group_1['labels']]))
        # else:
        #     fairness_reg = torch.abs(torch.mean(loss_group_0) - torch.mean(loss_group_1))

        fairness_reg = simplified_fairness_loss(fairness_function=train_tilted_params.fairness_function,
                                                loss=torch.hstack([loss_group_0, loss_group_1]),
                                                preds=torch.vstack(
                                                    [output_group_0['prediction'], output_group_1['prediction']]),
                                                aux=torch.hstack(
                                                    [items_group_0['aux_flattened'], items_group_1['aux_flattened']]),
                                                group1_pattern=s_group_0,
                                                group2_pattern=s_group_1,
                                                label=torch.hstack([items_group_0['labels'], items_group_1['labels']]))

        # fairness_reg = new_mixup_sub_routine(train_tilted_params, items_group_0, items_group_1, model)

        loss = torch.mean(loss_group_0) + torch.mean(loss_group_1) + mixup_rg*fairness_reg

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            output = model(items_group_0)
            track_output.append(output)
            track_input.append(items_group_0)
        output['loss_batch'] = loss.item()
        track_output.append(output)
        track_input.append(items_group_0)

    epoch_metric_tracker, loss = train_tilted_params.per_epoch_metric(track_output,
                                                                      track_input,
                                                                      train_tilted_params.fairness_function)

    return epoch_metric_tracker, loss, global_weight, global_loss





def train_only_group_dro_super_group_with_simplified_fairness_loss(train_tilted_params:TrainParameters):
    global_loss = train_tilted_params.other_params['global_loss'] # This tracks the actual loss
    global_weight = train_tilted_params.other_params['global_weight'] # Weights of each examples based on simple count
    # global_weight never gets updated.
    tilt_t = train_tilted_params.other_params['titled_t'] # This should be small. In order of magnitude of 0.01
    mixup_rg = train_tilted_params.other_params['mixup_rg'] # This should be small. In order of magnitude of 0.01
    method = train_tilted_params.other_params['method'] # This should be small. In order of magnitude of 0.01

    model, optimizer, device, criterion = \
        train_tilted_params.model, train_tilted_params.optimizer, train_tilted_params.device, train_tilted_params.criterion
    model.train()
    track_output = []
    track_input = []
    group_tracker = [0 for _ in range(len(train_tilted_params.other_params['groups']))]
    flattened_s_to_s = {value: key for key, value in train_tilted_params.other_params['s_to_flattened_s'].items()}

    total_reg = 0.0
    total_loss = 0.0
    for i in tqdm(range(train_tilted_params.other_params['number_of_iterations'])):
        split_index = int(train_tilted_params.other_params['batch_size'] / 2)
        s_group_0, s_group_1 = eval(
            np.random.choice(train_tilted_params.other_params['groups_matrix'].reshape(1, -1)[0], 1, replace=False,
                             p=global_weight.reshape(1, -1)[0])[0])

        # items_group_0, items_group_1 = sample_data(train_tilted_params, s_group_0, s_group_1)
        # items_group_0 = sample_batch_sen_idx(train_tilted_params.other_params['all_input'],
        #                              train_tilted_params.other_params['all_label'],
        #                              train_tilted_params.other_params['all_aux'],
        #                              train_tilted_params.other_params['all_aux_flatten'],
        #                              train_tilted_params.other_params['batch_size'],
        #                              s_group_0)
        #
        # items_group_1 = sample_batch_sen_idx(train_tilted_params.other_params['all_input'],
        #                                      train_tilted_params.other_params['all_label'],
        #                                      train_tilted_params.other_params['all_aux'],
        #                                      train_tilted_params.other_params['all_aux_flatten'],
        #                                      train_tilted_params.other_params['batch_size'],
        #                                      s_group_1)

        items_group_0, items_group_1 = sample_data(train_tilted_params, s_group_0, s_group_1)

        group_tracker[s_group_0] += 1
        group_tracker[s_group_1] += 1

        optimizer.zero_grad()

        output_group_0 = model(items_group_0)
        output_group_1 = model(items_group_1)

        loss_group_0 = criterion(output_group_0['prediction'], items_group_0['labels'])
        loss_group_1 = criterion(output_group_1['prediction'], items_group_1['labels'])


        loss_reg = simplified_fairness_loss(fairness_function=train_tilted_params.fairness_function,
                                                loss=torch.hstack([loss_group_0, loss_group_1]),
                                                preds=torch.vstack(
                                                    [output_group_0['prediction'], output_group_1['prediction']]),
                                                aux=torch.hstack(
                                                    [items_group_0['aux_flattened'], items_group_1['aux_flattened']]),
                                                group1_pattern=s_group_0,
                                                group2_pattern=s_group_1,
                                                label=torch.hstack([items_group_0['labels'], items_group_1['labels']]))

        loss = (torch.mean(loss_group_0) + torch.mean(loss_group_1)) / 2.0
        total_loss += loss.data
        # loss = loss
        global_loss[s_group_0, s_group_1] = global_loss[s_group_0, s_group_1] * torch.exp(tilt_t*loss.data)
        global_loss = global_loss/(global_loss.sum())
        loss = global_loss[s_group_0, s_group_1]*loss + mixup_rg * loss_reg
        loss.backward()
        optimizer.step()

        total_reg += loss_reg.item()

        output_group_0['loss_batch'] = torch.mean(loss_group_0).item()
        track_output.append(output_group_0)
        track_input.append(items_group_0)

    epoch_metric_tracker, loss = train_tilted_params.per_epoch_metric(track_output,
                                                                      track_input,
                                                                      train_tilted_params.fairness_function)

    print(total_loss, total_reg)

    return epoch_metric_tracker, loss, global_weight, global_loss





def new_mixup_sub_routine(train_tilted_params:TrainParameters, items_group_0, items_group_1, model, gamma=None):
    alpha = 1.0
    if not gamma:
        gamma = beta(alpha, alpha)

    if train_tilted_params.fairness_function == 'demographic_parity':
        raise NotImplementedError

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
            mask_item_group_0 = items_group_0['labels'] == i
            mask_item_group_1 = items_group_1['labels'] == i

            batch_x_mix = torch.mean(items_group_0['input'][mask_item_group_0], axis=0).unsqueeze(0) * gamma + \
                          torch.mean(items_group_1['input'][mask_item_group_1], axis=0).unsqueeze(0) * (1 - gamma)    # this is the point wise addition which forces this equal 1s,0s representation
            batch_x_mix = batch_x_mix.requires_grad_(True)
            model.eval()
            output_mixup = model({'input': batch_x_mix})
            model.train()
            gradx = torch.autograd.grad(output_mixup['prediction'].sum(), batch_x_mix, create_graph=True)[
                0]  # may be .sum()

            batch_x_d = torch.mean(items_group_1['input'][mask_item_group_1], axis=0).unsqueeze(0) - torch.mean(items_group_0['input'][mask_item_group_0], axis=0).unsqueeze(0)
            grad_inn = (gradx * batch_x_d).sum(1)
            E_grad = grad_inn.mean(0)
            loss_reg = loss_reg + torch.abs(E_grad)

    else:
        raise NotImplementedError


    return loss_reg