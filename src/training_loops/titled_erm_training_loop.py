import math
import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
from collections import Counter
from .common_functionality import *




def train(train_tilted_params:TrainParameters):

    global_weight = train_tilted_params.other_params['global_weight']
    global_loss = train_tilted_params.other_params['global_loss']
    tilt_t = train_tilted_params.other_params['titled_t']
    tao = 0.00001
    gamma = train_tilted_params.other_params['gamma']

    model, optimizer, device, criterion = \
        train_tilted_params.model, train_tilted_params.optimizer, train_tilted_params.device, train_tilted_params.criterion
    model.train()
    track_output = []
    track_input = []

    for i in tqdm(range(train_tilted_params.other_params['number_of_iterations'])):

        # s = np.random.choice(train_tilted_params.other_params['groups'], 1, p=global_weight)[0]
        s = F.gumbel_softmax(global_weight, tau=1/10, hard=True).nonzero()[0][0].item()

        if train_tilted_params.fairness_function == 'equal_opportunity':
            items = sample_batch_sen_idx(train_tilted_params.other_params['all_input'],
                                         train_tilted_params.other_params['all_label'],
                                         train_tilted_params.other_params['all_aux'],
                                         train_tilted_params.other_params['all_aux_flatten'],
                                         train_tilted_params.other_params['batch_size'],
                                         s)
        else:
            raise NotImplementedError

        for key in items.keys():
            items[key] = items[key].to(train_tilted_params.device)

        optimizer.zero_grad()
        output = model(items)
        loss = criterion(output['prediction'], items['labels'])
        loss_without_backward = torch.clone(loss).detach()

        # tilt the loss
        loss_r_b = torch.log(torch.mean(torch.exp(tao * loss_without_backward)))/tao


        global_loss[s] = (1/tao)*( (1 - gamma) * torch.exp(tao*global_loss[s]) + gamma * torch.exp(loss_r_b*tao))

        weights = torch.exp(tao*loss_without_backward - tao*global_loss[s])
        # global_weight = global_loss / np.linalg.norm(global_loss, 1)
        # global_weight = global_loss
        loss = torch.mean(weights*loss)
        loss.backward()
        optimizer.step()

        output['loss_batch'] = loss.item()
        track_output.append(output)
        track_input.append(items)



    epoch_metric_tracker, loss = train_tilted_params.per_epoch_metric(track_output,
                                                                   track_input,
                                                                   train_tilted_params.fairness_function)

    print(global_weight)
    print(global_loss)

    return epoch_metric_tracker, loss, global_weight, global_loss


def test(train_parameters: TrainParameters):
    """Trains the model for one epoch"""
    model, optimizer, device, criterion, mode = \
        train_parameters.model, train_parameters.optimizer, train_parameters.device, train_parameters.criterion, train_parameters.mode


    model.eval()
    track_output = []


    for items in tqdm(train_parameters.iterator):

        # Change the device of all the tensors!
        for key in items.keys():
            items[key] = items[key].to(device)


        with torch.no_grad():
            output = model(items)
            loss = torch.mean(criterion(output['prediction'], items['labels']))  # As reduction is None.


        # Save all batch stuff for further analysis
        output['loss_batch'] = loss.item()
        track_output.append(output)

    # Calculate all per-epoch metric by sending outputs and the inputs
    epoch_metric_tracker, loss = train_parameters.per_epoch_metric(track_output,
                                                                   train_parameters.iterator,
                                                                   train_parameters.fairness_function)
    return epoch_metric_tracker, loss


def sample_batch_sen_idx(all_input, all_label, all_aux, all_aux_flatten, batch_size, s):
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


def training_loop(training_loop_parameters: TrainingLoopParameters):
    '''

    :param training_loop_parameters:
    :return:
    This training loop would look different than all others. As this requires access to dataset based on s

    input_data = {
            'labels': labels,
            'input': encoded_input,
            'lengths': lengths,
            'aux': aux,
            'aux_flattened': aux_flattened
        }


    '''
    logger = logging.getLogger(training_loop_parameters.unique_id_for_run)
    output = {}
    all_train_eps_metrics = []
    all_test_eps_metrics = []
    all_valid_eps_metrics = []
    best_test_accuracy = 0.0
    best_eopp = 1.0

    # Housekeeping because we cannot use a simple iter which is returned by the dataset_parser!
    _labels, _input, _lengths, _aux, _aux_flattened = [], [], [], [], []
    for items in (training_loop_parameters.iterators[0]['train_iterator']):
        _labels.append(items['labels'])
        _input.append(items['input'])
        _lengths.append(items['lengths'])
        _aux.append(items['aux'])
        _aux_flattened.append(items['aux_flattened'])

    all_label = np.hstack(_labels)
    all_aux = np.vstack(_aux)
    all_aux_flatten = np.hstack(_aux_flattened)
    all_input = np.vstack(_input)
    size_of_training_dataset = len(all_label)

    total_no_groups = len(np.unique(all_aux_flatten))

    counts = Counter(all_aux_flatten)
    # size_of_each_group = [counts[i] for i in range(total_no_groups)]

    # Now comes the traing loop and evaluation loop
    # global_weight = np.full(total_no_groups, 1.0/total_no_groups)
    # global_loss = np.full(total_no_groups, 1.0/total_no_groups)

    global_weight = torch.tensor([math.log(counts[i])/training_loop_parameters.other_params['titled_t'] for i in range(total_no_groups)])
    global_loss = torch.tensor([math.log(counts[i])/training_loop_parameters.other_params['titled_t'] for i in range(total_no_groups)])
    print(global_weight, global_loss)
    groups = [i for i in range(total_no_groups)]


    for ep in range(training_loop_parameters.n_epochs):



        training_loop_parameters.other_params['number_of_iterations'] = int(size_of_training_dataset/training_loop_parameters.other_params['batch_size'])
        training_loop_parameters.other_params['global_weight'] = global_weight
        training_loop_parameters.other_params['global_loss'] = global_loss
        training_loop_parameters.other_params['groups'] = groups
        training_loop_parameters.other_params['all_label'] = all_label
        training_loop_parameters.other_params['all_aux'] = all_aux
        training_loop_parameters.other_params['all_aux_flatten'] = all_aux_flatten
        training_loop_parameters.other_params['all_input'] = all_input

        train_parameters = TrainParameters(
            model=training_loop_parameters.model,
            iterator=training_loop_parameters.iterators[0]['train_iterator'],
            optimizer=training_loop_parameters.optimizer,
            criterion=training_loop_parameters.criterion,
            device=training_loop_parameters.device,
            other_params=training_loop_parameters.other_params,
            per_epoch_metric=per_epoch_metric,
            mode='train',
            fairness_function=training_loop_parameters.fairness_function)

        train_epoch_metric, loss, global_weight, global_loss = train(train_parameters)
        log_epoch_metric(logger, start_message='train', epoch_metric=train_epoch_metric, epoch_number=ep, loss=loss)
        if training_loop_parameters.use_wandb:
            log_and_plot_data(epoch_metric=train_epoch_metric, loss=loss, train=True)
        all_train_eps_metrics.append(train_epoch_metric)

        # Valid split
        valid_parameters = TrainParameters(
            model=training_loop_parameters.model,
            iterator=training_loop_parameters.iterators[0]['valid_iterator'],
            optimizer=training_loop_parameters.optimizer,
            criterion=training_loop_parameters.criterion,
            device=training_loop_parameters.device,
            other_params=training_loop_parameters.other_params,
            per_epoch_metric=per_epoch_metric,
            mode='evaluate',
            fairness_function=training_loop_parameters.fairness_function)

        valid_epoch_metric, loss = test(valid_parameters)
        log_epoch_metric(logger, start_message='valid', epoch_metric=valid_epoch_metric, epoch_number=ep, loss=loss)

        if training_loop_parameters.use_wandb:
            log_and_plot_data(epoch_metric=valid_epoch_metric, loss=loss, train=True)
        all_valid_eps_metrics.append(train_epoch_metric)

        # test split
        test_parameters = TrainParameters(
            model=training_loop_parameters.model,
            iterator=training_loop_parameters.iterators[0]['test_iterator'],
            optimizer=training_loop_parameters.optimizer,
            criterion=training_loop_parameters.criterion,
            device=training_loop_parameters.device,
            other_params=training_loop_parameters.other_params,
            per_epoch_metric=per_epoch_metric,
            mode='evaluate',
            fairness_function=training_loop_parameters.fairness_function)

        test_epoch_metric, loss = test(test_parameters)
        if training_loop_parameters.use_wandb:
            log_and_plot_data(epoch_metric=test_epoch_metric, loss=loss, train=False)
        log_epoch_metric(logger, start_message='test', epoch_metric=test_epoch_metric, epoch_number=ep, loss=loss)
        all_test_eps_metrics.append(test_epoch_metric)

        logger.info("end of epoch block")

    output['all_train_eps_metric'] = all_train_eps_metrics
    output['all_test_eps_metric'] = all_test_eps_metrics
    output['all_valid_eps_metric'] = all_valid_eps_metrics
    index = find_best_model(output)
    output['best_model_index'] = index

    return output
