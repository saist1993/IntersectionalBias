# This file will simplify and make things more clear.
import torch
import numpy as np
from tqdm.auto import tqdm
from .titled_erm_training_loop import test

from .common_functionality import *

'''
parameters till now 

    - distance_mechanism
        - dynamic
        - static
    - group_sampling_procedure 
                - supergroup
                - randomgroup
                - distance
                - supergroup_and_distance
    - example_sampling_procedure
                - equal_sampling
                - random_sampling
    - fairness_regularization_procedure
                - mixup
                
    - integrate_reg_loss - True/False
    - loss_rg_weight
    
    possible methods 
    
    dro_supergroup_equal_sampling
    
    [OptimizationMechanism][SamplingProcedure][DistanceMechanism][ExampleSamplingProcedure][FairnessReg][IntegratedReg]
    
    
'''


def group_sampling_procedure_func(train_tilted_params, global_weight, similarity_matrix=None):
    '''
    there are 4 kinds of sampling procedure
    - Random (G1, G2) - Both are independent
    - Distance (Random G1, Distance from G1)
    - SuperGroup
    - SuperGroup with distance
    '''

    group_sampling_procedure = train_tilted_params.other_params['group_sampling_procedure']

    if group_sampling_procedure == 'super_group':
        # @TODO: write an assert stating that global weight should be of specific kind.
        s_group_0, s_group_1 = eval(
            np.random.choice(train_tilted_params.other_params['groups_matrix'].reshape(1, -1)[0], 1, replace=False,
                             p=global_weight.reshape(1, -1)[0])[0])
    elif group_sampling_procedure == 'random_group':
        s_group_0, s_group_1 = np.random.choice(train_tilted_params.other_params['groups'], 2, p=global_weight)
    elif group_sampling_procedure == 'random_single_group':
        s_group_0 = np.random.choice(train_tilted_params.other_params['groups'], 1, p=global_weight)[0]
        s_group_1 = None
    elif group_sampling_procedure == 'distance_group':
        assert similarity_matrix != None
        s_group_0 = np.random.choice(train_tilted_params.other_params['groups'], 1, p=global_weight)[0]
        s_group_distance = similarity_matrix[s_group_0]
        s_group_1 = np.random.choice(train_tilted_params.other_params['groups'], 1, replace=False,
                                     p=s_group_distance / np.linalg.norm(s_group_distance, 1))[0]
    elif group_sampling_procedure == 'super_group_and_distance':
        raise NotImplementedError
    else:
        raise NotImplementedError("currently supports: supergroup, distance, random, supergroup_and_distance")

    return s_group_0, s_group_1


def example_sampling_procedure_func(train_tilted_params, group0, group1):
    '''

    '''
    example_sampling_procedure = train_tilted_params.other_params['example_sampling_procedure']
    if example_sampling_procedure == 'equal_sampling':
        items_group_0, items_group_1 = \
            sample_data(train_tilted_params=train_tilted_params, s_group_0=group0, s_group_1=group1)
    elif example_sampling_procedure == 'random_sampling':
        items_group_0, items_group_1 = \
            randomly_sample_data(train_tilted_params=train_tilted_params, s_group_0=group0, s_group_1=group1)
    else:
        raise NotImplementedError

    return items_group_0, items_group_1


def fairness_regularization_procedure_func(train_tilted_params, items_group_0, items_group_1, model,
                                           other_params={'gamma': None}):
    '''

    '''
    fairness_regularization_procedure = train_tilted_params.other_params['fairness_regularization_procedure']
    if fairness_regularization_procedure == 'mixup':
        return mixup_sub_routine(train_tilted_params=train_tilted_params,
                                 items_group_0=items_group_0,
                                 items_group_1=items_group_1,
                                 model=model,
                                 gamma=other_params['gamma'])
    else:
        return None


def update_loss_and_global_loss_dro(train_tilted_params, s_group_0, s_group_1, loss_group_0, loss_group_1, loss_rg,
                                    global_loss):
    '''Function depends on the group sampling mechanism'''
    group_sampling_procedure = train_tilted_params.other_params['group_sampling_procedure']
    tilt_t = train_tilted_params.other_params['titled_t']
    integrate_reg_loss = train_tilted_params.other_params['integrate_reg_loss']
    loss_rg_weight = train_tilted_params.other_params['loss_rg_weight']

    if group_sampling_procedure == 'super_group' or group_sampling_procedure == 'super_group_and_distance':
        loss = torch.mean(loss_group_0) + torch.mean(loss_group_1)  # calculate the total loss

        if integrate_reg_loss and (loss_rg is not None):
            global_loss[s_group_0, s_group_1] = global_loss[s_group_0, s_group_1] * torch.exp(
                tilt_t * (loss + (loss_rg * loss_rg_weight)).data)
            global_loss = global_loss / (global_loss.sum())
            loss = (loss + (loss_rg * loss_rg_weight)) * global_loss[s_group_0, s_group_1]
        else:
            global_loss[s_group_0, s_group_1] = global_loss[s_group_0, s_group_1] * torch.exp(
                tilt_t * loss.data)
            global_loss = global_loss / (global_loss.sum())
            loss = loss * global_loss[s_group_0, s_group_1]
            if loss_rg is not None:
                loss = loss + (loss_rg * loss_rg_weight)

    elif group_sampling_procedure == 'random_group' or group_sampling_procedure == 'distance_group':

        if integrate_reg_loss and (loss_rg is not None):
            global_loss[s_group_0] = global_loss[s_group_0] * torch.exp(
                tilt_t * (torch.mean(loss_group_0) + (loss_rg * loss_rg_weight)).data)
            global_loss = global_loss / (global_loss.sum())
            loss = (torch.mean(loss_group_0) + (loss_rg * loss_rg_weight)) * global_loss[s_group_0]

            global_loss[s_group_1] = global_loss[s_group_1] * torch.exp(
                tilt_t * (torch.mean(loss_group_1) + (loss_rg * loss_rg_weight)).data)
            global_loss = global_loss / (global_loss.sum())
            loss = loss + (torch.mean(loss_group_1) + (loss_rg * loss_rg_weight)) * global_loss[s_group_1]
        else:
            global_loss[s_group_0] = global_loss[s_group_0] * torch.exp(
                tilt_t * torch.mean(loss_group_0).data)
            global_loss = global_loss / (global_loss.sum())
            loss = torch.mean(loss_group_0) * global_loss[s_group_0]

            global_loss[s_group_1] = global_loss[s_group_1] * torch.exp(
                tilt_t * torch.mean(loss_group_1).data)
            global_loss = global_loss / (global_loss.sum())
            loss = loss + torch.mean(loss_group_1) * global_loss[s_group_1]
            if loss_rg is not None:
                loss = loss + (loss_rg * loss_rg_weight)



    elif group_sampling_procedure == 'random_single_group':
        if integrate_reg_loss and (loss_rg is not None):
            global_loss[s_group_0] = global_loss[s_group_0] * torch.exp(
                tilt_t * (torch.mean(loss_group_0) + (loss_rg * loss_rg_weight)).data)
            global_loss = global_loss / (global_loss.sum())
            loss = (torch.mean(loss_group_0) + (loss_rg * loss_rg_weight)) * global_loss[s_group_0]
        else:
            global_loss[s_group_0] = global_loss[s_group_0] * torch.exp(
                tilt_t * torch.mean(loss_group_0).data)
            global_loss = global_loss / (global_loss.sum())
            loss = torch.mean(loss_group_0) * global_loss[s_group_0]

            if loss_rg is not None:
                loss = loss + (loss_rg * loss_rg_weight)

    else:
        raise NotImplementedError()

    return global_loss, loss


def dro_optimization_procedure(train_tilted_params):
    global_loss = train_tilted_params.other_params['global_loss']  # This tracks the actual loss
    # Gives weights to each group. In case of supergroup then two group shares the same weight
    # Note that global_weight never gets updated.
    global_weight = train_tilted_params.other_params['global_weight']

    model, optimizer, device, criterion = \
        train_tilted_params.model, train_tilted_params.optimizer, train_tilted_params.device, train_tilted_params.criterion
    model.train()

    group_sampling_procedure = train_tilted_params.other_params['group_sampling_procedure']

    if "distance" in group_sampling_procedure:
        flattened_s_to_s = {value: key for key, value in train_tilted_params.other_params['s_to_flattened_s'].items()}
        similarity_matrix = generate_similarity_matrix(train_tilted_params.other_params['valid_iterator'], model,
                                                       train_tilted_params.other_params['groups'], flattened_s_to_s,
                                                       distance_mechanism=train_tilted_params.other_params[
                                                           'distance_mechanism'])
    else:
        similarity_matrix = None

    track_output = []
    track_input = []

    for i in tqdm(range(train_tilted_params.other_params['number_of_iterations'])):
        s_group_0, s_group_1 = group_sampling_procedure_func(
            train_tilted_params=train_tilted_params,
            global_weight=global_weight,
            similarity_matrix=similarity_matrix
        )

        items_group_0, items_group_1 = example_sampling_procedure_func(
            train_tilted_params=train_tilted_params,
            group0=s_group_0,
            group1=s_group_1
        )

        if group_sampling_procedure == 'random_single_group':
            assert s_group_1 is None
            assert items_group_1 is None

        optimizer.zero_grad()
        output_group_0 = model(items_group_0)
        loss_group_0 = criterion(output_group_0['prediction'], items_group_0['labels'])

        if s_group_1:
            output_group_1 = model(items_group_1)
            loss_group_1 = criterion(output_group_1['prediction'], items_group_1['labels'])
        else:
            loss_group_1 = None

        loss_rg = fairness_regularization_procedure_func(train_tilted_params=train_tilted_params,
                                                         items_group_0=items_group_0,
                                                         items_group_1=items_group_1,
                                                         model=model,
                                                         other_params={'gamma': None})

        global_loss, loss = update_loss_and_global_loss_dro(train_tilted_params=train_tilted_params,
                                                            s_group_0=s_group_0,
                                                            s_group_1=s_group_1,
                                                            loss_group_0=loss_group_0,
                                                            loss_group_1=loss_group_1,
                                                            loss_rg=loss_rg,
                                                            global_loss=global_loss)

        loss.backward()
        optimizer.step()

        output_group_0['loss_batch'] = loss_group_0.item()  # handel this better!
        track_output.append(output_group_0)
        track_input.append(items_group_0)

    epoch_metric_tracker, loss = train_tilted_params.per_epoch_metric(track_output,
                                                                      track_input,
                                                                      train_tilted_params.fairness_function)

    return epoch_metric_tracker, loss, global_weight, global_loss


def erm_optimization_procedure(train_tilted_params):
    # Gives weights to each group. In case of supergroup then two group shares the same weight
    # Note that global_weight never gets updated.
    global_weight = train_tilted_params.other_params['global_weight']

    model, optimizer, device, criterion = \
        train_tilted_params.model, train_tilted_params.optimizer, train_tilted_params.device, train_tilted_params.criterion
    model.train()

    group_sampling_procedure = train_tilted_params.other_params['group_sampling_procedure']

    if "distance" in group_sampling_procedure:
        flattened_s_to_s = {value: key for key, value in train_tilted_params.other_params['s_to_flattened_s'].items()}
        similarity_matrix = generate_similarity_matrix(train_tilted_params.other_params['valid_iterator'], model,
                                                       train_tilted_params.other_params['groups'], flattened_s_to_s,
                                                       distance_mechanism=train_tilted_params.other_params[
                                                           'distance_mechanism'])
    else:
        similarity_matrix = None

    track_output = []
    track_input = []

    for i in tqdm(range(train_tilted_params.other_params['number_of_iterations'])):
        s_group_0, s_group_1 = group_sampling_procedure_func(
            train_tilted_params=train_tilted_params,
            global_weight=global_weight,
            similarity_matrix=similarity_matrix
        )

        items_group_0, items_group_1 = example_sampling_procedure_func(
            train_tilted_params=train_tilted_params,
            group0=s_group_0,
            group1=s_group_1
        )

        if group_sampling_procedure == 'random_single_group':
            assert s_group_1 is None
            assert items_group_1 is None

        optimizer.zero_grad()
        output_group_0 = model(items_group_0)
        loss_group_0 = criterion(output_group_0['prediction'], items_group_0['labels'])

        if s_group_1:
            output_group_1 = model(items_group_1)
            loss_group_1 = criterion(output_group_1['prediction'], items_group_1['labels'])
        else:
            loss_group_1 = None

        loss_rg = fairness_regularization_procedure_func(train_tilted_params=train_tilted_params,
                                                         items_group_0=items_group_0,
                                                         items_group_1=items_group_1,
                                                         model=model,
                                                         other_params={'gamma': None})

        loss = torch.mean(loss_group_0)
        if loss_group_1:
            loss = loss + torch.mean(loss_group_1)

        if loss_rg:
            loss = loss + loss_rg * train_tilted_params.other_params['loss_rg_weight']

        loss.backward()
        optimizer.step()

        output_group_0['loss_batch'] = loss_group_0.item()  # handel this better!
        track_output.append(output_group_0)
        track_input.append(items_group_0)

    epoch_metric_tracker, loss = train_tilted_params.per_epoch_metric(track_output,
                                                                      track_input,
                                                                      train_tilted_params.fairness_function)

    return epoch_metric_tracker, loss, global_weight, None


def create_group(total_no_groups, method="super_group"):
    if method == "super_group":
        global_weight = np.full((total_no_groups, total_no_groups), 1.0 / (total_no_groups * total_no_groups))
        global_loss = np.full((total_no_groups, total_no_groups), 1.0 / (total_no_groups * total_no_groups))
        return global_weight, global_loss
    elif method == 'single_group':
        weights = np.asarray([1 / total_no_groups for i in range(total_no_groups)])

        # global_weight = size_of_each_group / (np.linalg.norm(size_of_each_group, 1))
        global_weight = weights / (np.linalg.norm(weights, 1))
        global_loss = torch.tensor(weights / np.linalg.norm(weights, 1))
        return global_weight, global_loss
    else:
        raise NotImplementedError


def orchestrator(training_loop_parameters: TrainingLoopParameters):
    logger = logging.getLogger(training_loop_parameters.unique_id_for_run)
    output = {}
    all_train_eps_metrics = []
    all_test_eps_metrics = []
    all_valid_eps_metrics = []

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

    method = training_loop_parameters.other_params['method']
    total_no_groups = len(np.unique(all_aux_flatten))

    # set group sampling proecdure
    if "super_group_and_distance" in method:
        group_sampling_procedure = "super_group_and_distance"
        global_weight, global_loss = create_group(total_no_groups, method="super_group")
    elif "super_group" in method:
        group_sampling_procedure = "super_group"
        global_weight, global_loss = create_group(total_no_groups, method="super_group")
    elif "random_group" in method:
        group_sampling_procedure = "random_group"
        global_weight, global_loss = create_group(total_no_groups, method="single_group")
    elif "random_single_group" in method:
        group_sampling_procedure = "random_single_group"
        global_weight, global_loss = create_group(total_no_groups, method="single_group")
    else:
        raise NotImplementedError("no group sampling procedure specified")

    # set example sampling procedure
    if "random_sampling" in method:
        example_sampling_procedure = "random_sampling"
    elif "equal_sampling" in method:
        example_sampling_procedure = "equal_sampling"
    else:
        raise NotImplementedError("no example sampling procedure specified")

    # set distance based mechanism
    if "static_distance" in method:
        distance_mechanism = "static_distance"
    elif "dynamic_distance" in method:
        distance_mechanism = "dynamic_distance"
    else:
        raise NotImplementedError("no distance based procedure specified")

    # set fairness regularizer
    if "mixup_regularizer" in method:
        fairness_regularization_procedure = 'mixup'
    else:
        fairness_regularization_procedure = 'none'

    if "erm" in method:
        optimization_procedure = "erm"
        procedure = erm_optimization_procedure
    elif "dro" in method:
        optimization_procedure = "dro"
        procedure = dro_optimization_procedure
    else:
        raise NotImplementedError("no optimization procedure specified")

    training_loop_parameters.other_params['optimization_procedure'] = optimization_procedure
    training_loop_parameters.other_params['distance_mechanism'] = distance_mechanism
    training_loop_parameters.other_params['example_sampling_procedure'] = example_sampling_procedure
    training_loop_parameters.other_params['group_sampling_procedure'] = group_sampling_procedure
    training_loop_parameters.other_params['fairness_regularization_procedure'] = fairness_regularization_procedure

    for ep in range(training_loop_parameters.n_epochs):

        logger.info("start of epoch block  ")

        training_loop_parameters.other_params['number_of_iterations'] = int(
            size_of_training_dataset / training_loop_parameters.other_params['batch_size'])
        training_loop_parameters.other_params['global_weight'] = global_weight
        training_loop_parameters.other_params['global_loss'] = global_loss
        training_loop_parameters.other_params['all_label'] = all_label
        training_loop_parameters.other_params['all_aux'] = all_aux
        training_loop_parameters.other_params['all_aux_flatten'] = all_aux_flatten
        training_loop_parameters.other_params['all_input'] = all_input
        training_loop_parameters.other_params['valid_iterator'] = training_loop_parameters.iterators[0][
            'valid_iterator']
        training_loop_parameters.other_params['scalar'] = training_loop_parameters.iterators[0]['scalar']
        training_loop_parameters.other_params['train_iterator'] = training_loop_parameters.iterators[0][
            'train_iterator']

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

        train_epoch_metric, loss, global_weight, global_loss = procedure(train_parameters)

        valid_parameters = TrainParameters(
            model=training_loop_parameters.model,
            iterator=training_loop_parameters.iterators[0]['train_iterator'],
            optimizer=training_loop_parameters.optimizer,
            criterion=training_loop_parameters.criterion,
            device=training_loop_parameters.device,
            other_params=training_loop_parameters.other_params,
            per_epoch_metric=per_epoch_metric,
            mode='evaluate',
            fairness_function=training_loop_parameters.fairness_function)

        valid_epoch_metric, loss = test(valid_parameters)

        log_epoch_metric(logger, start_message='train', epoch_metric=valid_epoch_metric, epoch_number=ep, loss=loss)

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

    return output
