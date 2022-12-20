import torch
from tqdm.auto import tqdm
from .common_functionality import *

def simplified_fairness_loss(fairness_function, loss, preds, aux, group1_pattern, group2_pattern, label):
    group1_mask = aux == group1_pattern # where group1 exists
    group2_mask = aux == group2_pattern # where group2 exists

    if fairness_function == 'demographic_parity':
        preds_mask = torch.argmax(preds,1) == 1 # label does not matter here.
        group1_loss = loss[torch.logical_and(preds_mask, group1_mask)]
        group2_loss = loss[torch.logical_and(preds_mask, group2_mask)]
        return torch.abs(torch.mean(group1_loss)-torch.mean(group2_loss))
    elif fairness_function == 'equal_odds' or fairness_function == 'equal_opportunity':
        label_mask_1 = label == 1
        label_mask_0 = label == 0
        final_loss = []


        # true positive rate
        # final_loss = torch.tensor(0.0, requires_grad=True)
        numerator_label = 1
        preds_mask = torch.logical_and(torch.argmax(preds, 1) == numerator_label, label_mask_1)
        group1_loss = loss[torch.logical_and(preds_mask, group1_mask)]
        group2_loss = loss[torch.logical_and(preds_mask, group2_mask)]
        # final_loss.append(torch.abs(torch.mean(group1_loss) - torch.mean(group2_loss)))
        reg_loss = torch.abs(torch.mean(group1_loss) - torch.mean(group2_loss))

        if fairness_function == 'equal_odds':
            # false positive rate
            numerator_label = 0
            preds_mask = torch.logical_and(torch.argmax(preds, 1) == numerator_label, label_mask_0)
            group1_loss = loss[torch.logical_and(preds_mask, group1_mask)]
            group2_loss = loss[torch.logical_and(preds_mask, group2_mask)]
            # final_loss.append(torch.abs(torch.mean(group1_loss) - torch.mean(group2_loss)))
            # reg_loss = torch.max(reg_loss, torch.abs(torch.mean(group1_loss) - torch.mean(group2_loss)))
            reg_loss +=  torch.abs(torch.mean(group1_loss) - torch.mean(group2_loss))

        return reg_loss




        # if fairness_function == 'equal_opportunity':
        #     final_label_masks = [(1,label_mask_1)]
        # elif fairness_function == 'equal_odds':
        #     final_label_masks = [(0,label_mask_0), (1,label_mask_1)]
        # else:
        #     raise NotImplementedError
        #
        # for actual_label, label_mask in final_label_masks:
        #     preds_mask = torch.logical_and(torch.argmax(preds,1) == actual_label, label_mask)
        #     group1_loss = loss[torch.logical_and(preds_mask, group1_mask)]
        #     group2_loss = loss[torch.logical_and(preds_mask, group2_mask)]
        #     final_loss.append(torch.abs(torch.mean(group1_loss) - torch.mean(group2_loss)))
        #
        # if len(final_loss) == 1:
        #     return final_loss[0]
        # else:
        #     # return torch.max(torch.tensor(final_loss))
        #     return torch.sum(torch.tensor(final_loss))
    else:
        raise NotImplementedError


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
        loss = loss + mixup_rg * loss_reg
        global_loss[s_group_0, s_group_1] = global_loss[s_group_0, s_group_1] * torch.exp(tilt_t*loss.data)
        global_loss = global_loss/(global_loss.sum())
        loss = global_loss[s_group_0, s_group_1]*loss
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



