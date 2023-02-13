# This file will simplify and make things more clear.
import torch
import random
import numpy as np
from tqdm.auto import tqdm

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


def group_sampling_procedure_func(train_tilted_params, global_weight, similarity_matrix=None):
    '''
    there are 4 kinds of sampling procedure
    - Random (G1, G2) - Both are independent
    - Distance (Random G1, Distance from G1)
    - SuperGroup
    - SuperGroup with distance
    '''

    group_sampling_procedure = train_tilted_params.other_params['group_sampling_procedure']

    if group_sampling_procedure == 'super_group_and_distance':
        # This is a hack as the global weight never gets updated. If in the future global
        # weight gets updated. This will break
        assert similarity_matrix != None
        s_group_0 = np.random.choice(train_tilted_params.other_params['groups'], 1, p=global_weight)[0]
        s_group_distance = similarity_matrix[s_group_0]
        flag = True
        while flag:
            s_group_1 = np.random.choice(train_tilted_params.other_params['groups'], 1, replace=False,
                                         p=s_group_distance / np.linalg.norm(s_group_distance, 1))[0]
            if s_group_1 != s_group_0:
                flag = False
    elif group_sampling_procedure == 'super_group':
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
        # s_group_0 = np.random.choice(train_tilted_params.other_params['groups'], 2, p=global_weight)[0]
        s_group_0 = np.random.choice(train_tilted_params.other_params['groups'], 2, replace=False)[0]
        s_group_distance = similarity_matrix[s_group_0]
        s_group_1 = np.random.choice(train_tilted_params.other_params['groups'], 1, replace=False,
                                     p=s_group_distance / np.linalg.norm(s_group_distance, 1))[0]
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
    loss_rg_weight = train_tilted_params.other_params['mixup_rg']
    positive_index = int(train_tilted_params.other_params['batch_size']/2)
    if train_tilted_params.other_params['update_only_via_reg']:
        update_only_via_reg = 0.0
    else:
        update_only_via_reg = 1.0


    if group_sampling_procedure == 'super_group' or group_sampling_procedure == 'super_group_and_distance':
        # loss = torch.mean(loss_group_0[positive_index:]) + torch.mean(loss_group_1[positive_index:])  # calculate the total loss
        loss = torch.mean(loss_group_0) + torch.mean(loss_group_1)  # calculate the total loss

        if integrate_reg_loss and (loss_rg is not None):
            global_loss[s_group_0, s_group_1] = global_loss[s_group_0, s_group_1] * torch.exp(
                tilt_t * (loss*update_only_via_reg + (loss_rg * loss_rg_weight)).data)
            global_loss = global_loss / (global_loss.sum())
            loss = (torch.mean(loss_group_0) + torch.mean(loss_group_1) + (loss_rg * loss_rg_weight)) * global_loss[s_group_0, s_group_1]
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
                tilt_t * (torch.mean(loss_group_0)*update_only_via_reg + (loss_rg * loss_rg_weight)).data)
            # global_loss = global_loss / (global_loss.sum())
            # loss = (torch.mean(loss_group_0) + (loss_rg * loss_rg_weight)) * global_loss[s_group_0]

            global_loss[s_group_1] = global_loss[s_group_1] * torch.exp(
                tilt_t * (torch.mean(loss_group_1) + (loss_rg * loss_rg_weight)).data)
            global_loss = global_loss / (global_loss.sum())
            loss = (torch.mean(loss_group_0) + (loss_rg * loss_rg_weight)) * global_loss[s_group_0] + (torch.mean(loss_group_1) + (loss_rg * loss_rg_weight)) * global_loss[s_group_1]
        else:
            global_loss[s_group_0] = global_loss[s_group_0] * torch.exp(
                tilt_t * torch.mean(loss_group_0).data)
            # global_loss = global_loss / (global_loss.sum())
            # loss = torch.mean(loss_group_0) * global_loss[s_group_0]

            global_loss[s_group_1] = global_loss[s_group_1] * torch.exp(
                tilt_t * torch.mean(loss_group_1).data)
            global_loss = global_loss / (global_loss.sum())
            loss = torch.mean(loss_group_0) * global_loss[s_group_0] + torch.mean(loss_group_1) * global_loss[s_group_1]
            if loss_rg is not None:
                loss = loss + (loss_rg * loss_rg_weight)



    elif group_sampling_procedure == 'random_single_group':
        if integrate_reg_loss and (loss_rg is not None):
            global_loss[s_group_0] = global_loss[s_group_0] * torch.exp(
                tilt_t * (torch.mean(loss_group_0)*update_only_via_reg + (loss_rg * loss_rg_weight)).data)
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
                                                           'distance_mechanism'],
                                                       train_tilted_params=train_tilted_params)
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
        else:
            assert s_group_1 is not None
            assert items_group_1 is not None

        optimizer.zero_grad()
        output_group_0 = model(items_group_0)
        loss_group_0 = criterion(output_group_0['prediction'], items_group_0['labels'])

        if s_group_1 is not None:
            output_group_1 = model(items_group_1)
            loss_group_1 = criterion(output_group_1['prediction'], items_group_1['labels'])
        else:
            loss_group_1 = None



        def sub_routine():
            loss_rg = torch.tensor(0.0, requires_grad=True)
            k = 1
            for _ in range(k):

                loss_rg = loss_rg + fairness_regularization_procedure_func(train_tilted_params=train_tilted_params,
                                                                 items_group_0=items_group_0,
                                                                 items_group_1=items_group_1,
                                                                 model=model,
                                                                 other_params={'gamma': None})

            loss_rg = loss_rg/k

            return loss_rg

        loss_rg = sub_routine()



        global_loss, loss = update_loss_and_global_loss_dro(train_tilted_params=train_tilted_params,
                                                            s_group_0=s_group_0,
                                                            s_group_1=s_group_1,
                                                            loss_group_0=loss_group_0,
                                                            loss_group_1=loss_group_1,
                                                            loss_rg=loss_rg,
                                                            global_loss=global_loss)

        loss.backward()
        optimizer.step()

        output_group_0['loss_batch'] = torch.mean(loss_group_0).item()  # handel this better!
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

    all_groups = []

    # all_label_augmented, all_aux_augmented, all_input_augmented, all_aux_flatten_augmented = train_tilted_params.other_params['all_label_augmented'],\
    #     train_tilted_params.other_params['all_aux_augmented'],\
    #     train_tilted_params.other_params['all_input_augmented'], train_tilted_params.other_params['all_aux_flatten_augmented']


    # all_label, all_aux, all_input, all_aux_flatten =  train_tilted_params.other_params['all_label'],\
    #     train_tilted_params.other_params['all_aux'],\
    #     train_tilted_params.other_params['all_input'], train_tilted_params.other_params['all_aux_flatten']

    for i in tqdm(range(train_tilted_params.other_params['number_of_iterations'])):
        #
        # if False:    # bool(random.getrandbits(1)), i%20 != 0
        #     train_tilted_params.other_params['all_label'] = all_label_augmented
        #     train_tilted_params.other_params['all_aux'] = all_aux_augmented
        #     train_tilted_params.other_params['all_aux_flatten'] = np.asarray(all_aux_flatten_augmented)
        #     train_tilted_params.other_params['all_input'] = all_input_augmented
        # # else:
        # #     train_tilted_params.other_params['all_label'] = all_label
        # #     train_tilted_params.other_params['all_aux'] = all_aux
        # #     train_tilted_params.other_params['all_aux_flatten'] = np.asarray(all_aux_flatten)
        # #     train_tilted_params.other_params['all_input'] = all_input


        s_group_0, s_group_1 = group_sampling_procedure_func(
            train_tilted_params=train_tilted_params,
            global_weight=global_weight,
            similarity_matrix=similarity_matrix
        )

        all_groups.append(s_group_0)

        items_group_0, items_group_1 = example_sampling_procedure_func(
            train_tilted_params=train_tilted_params,
            group0=s_group_0,
            group1=s_group_1
        )

        if train_tilted_params.other_params['use_mixup_augmentation']:
            items_group_0, items_group_1 = \
                augment_current_data_via_mixup(train_tilted_params, s_group_0, s_group_1, items_group_0, items_group_1)

        if group_sampling_procedure == 'random_single_group':
            assert s_group_1 is None
            assert items_group_1 is None



        #####TheOriginalStartsHere############
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
        if loss_group_1 is not None:
            loss = torch.mean(loss + torch.mean(loss_group_1))

        if loss_rg:
            loss = loss + loss_rg * train_tilted_params.other_params['mixup_rg']






        #####TheOriginalEndsHere############

        # composite_items = {
        #     'input': torch.vstack([items_group_0['input'], items_group_1['input']]),
        #     'labels': torch.hstack([items_group_0['labels'], items_group_1['labels']]),
        #     'aux': torch.vstack([items_group_0['aux'], items_group_1['aux']]),
        #     'aux_flattened': torch.hstack([items_group_0['aux_flattened'], items_group_1['aux_flattened']])
        # }
        #
        # optimizer.zero_grad()
        # output = model(composite_items)
        # loss = criterion(output['prediction'], composite_items['labels'])
        #
        # loss = torch.mean(loss)
        # loss_rg = fairness_regularization_procedure_func(train_tilted_params=train_tilted_params,
        #                                                  items_group_0=items_group_0,
        #                                                  items_group_1=items_group_1,
        #                                                  model=model,
        #                                                  other_params={'gamma': None})
        # loss = loss + loss_rg*train_tilted_params.other_params['mixup_rg']

        ## Augmented Ends Here ###



        loss.backward()
        optimizer.step()

        output_group_0['loss_batch'] = torch.mean(loss).item()  # handel this better!
        track_output.append(output_group_0)
        track_input.append(items_group_0)

    all_groups = np.unique(all_groups)

    epoch_metric_tracker, loss = train_tilted_params.per_epoch_metric(track_output,
                                                                      track_input,
                                                                      train_tilted_params.fairness_function)

    return epoch_metric_tracker, loss, global_weight, None


def create_group(total_no_groups, method="super_group"):
    if method == "super_group":
        global_weight = np.full((total_no_groups, total_no_groups), 1.0 / (total_no_groups * total_no_groups))
        global_loss = np.full((total_no_groups, total_no_groups), 1.0 / (total_no_groups * total_no_groups))
        groups_matrix = np.asarray([[str((i, j)) for i in range(total_no_groups)] for j in range(total_no_groups)])
        return groups_matrix, global_weight, global_loss
    elif method == 'single_group':
        weights = np.asarray([1 / total_no_groups for i in range(total_no_groups)])

        # global_weight = size_of_each_group / (np.linalg.norm(size_of_each_group, 1))
        global_weight = weights / (np.linalg.norm(weights, 1))
        global_loss = torch.tensor(weights / np.linalg.norm(weights, 1))
        return None, global_weight, global_loss
    else:
        raise NotImplementedError



def orchestrator(training_loop_parameters: TrainingLoopParameters):
    """
    methods
    baseline - erm_random_single_group_random_sampling

    """
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


    # add augmnetation information



    # set group sampling proecdure
    if "super_group_and_distance" in method:
        group_sampling_procedure = "super_group_and_distance"
        # This is a hack which works because global weight does not get updated.
        _, _, global_loss = create_group(total_no_groups, method="super_group")
        groups_matrix, global_weight, _ = create_group(total_no_groups, method="single_group")
    elif "super_group" in method:
        group_sampling_procedure = "super_group"
        groups_matrix, global_weight, global_loss = create_group(total_no_groups, method="super_group")
    elif "distance_group" in method:
        group_sampling_procedure = "distance_group"
        groups_matrix, global_weight, global_loss = create_group(total_no_groups, method="single_group")
    elif "random_group" in method:
        group_sampling_procedure = "random_group"
        groups_matrix, global_weight, global_loss = create_group(total_no_groups, method="single_group")
    elif "random_single_group" in method:
        group_sampling_procedure = "random_single_group"
        groups_matrix, global_weight, global_loss = create_group(total_no_groups, method="single_group")
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
    elif "miixup_distance" in method:
        distance_mechanism = "miixup_distance"
    else:
        distance_mechanism = "static_distance"

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

    if "integrate_reg_loss" in method:
        integrate_reg_loss = True
    else:
        integrate_reg_loss = False

    if "update_only_via_reg" in method:
        update_only_via_reg = True
    else:
        update_only_via_reg = False


    if "mixup_generated_and_real_data" in method:
        use_mixup_augmentation = True
        all_input_augmented = training_loop_parameters.iterators[0]['train_X_augmented']
        all_label_augmented = training_loop_parameters.iterators[0]['train_y_augmented']
        all_aux_augmented = training_loop_parameters.iterators[0]['train_s_augmented']
        training_loop_parameters.other_params['all_label_augmented'] = all_label_augmented
        training_loop_parameters.other_params['all_aux_augmented'] = all_aux_augmented
        training_loop_parameters.other_params['all_aux_flatten_augmented'] = \
            [training_loop_parameters.other_params['s_to_flattened_s'][tuple(i)]
             for i in training_loop_parameters.other_params['all_aux_augmented']]
        training_loop_parameters.other_params['all_input_augmented'] = all_input_augmented
    else:
        use_mixup_augmentation = False


    training_loop_parameters.other_params['groups_matrix'] = groups_matrix
    training_loop_parameters.other_params['distance_mechanism'] = distance_mechanism
    training_loop_parameters.other_params['integrate_reg_loss'] = integrate_reg_loss
    training_loop_parameters.other_params['update_only_via_reg'] = update_only_via_reg
    training_loop_parameters.other_params['optimization_procedure'] = optimization_procedure
    training_loop_parameters.other_params['use_mixup_augmentation'] = use_mixup_augmentation
    training_loop_parameters.other_params['group_sampling_procedure'] = group_sampling_procedure
    training_loop_parameters.other_params['example_sampling_procedure'] = example_sampling_procedure
    training_loop_parameters.other_params['fairness_regularization_procedure'] = fairness_regularization_procedure

    for ep in range(training_loop_parameters.n_epochs):

        logger.info("start of epoch block  ")

        # training_loop_parameters.other_params['number_of_iterations'] = int(
        #     size_of_training_dataset / training_loop_parameters.other_params['batch_size'])

        training_loop_parameters.other_params['number_of_iterations'] = 25

        training_loop_parameters.other_params['global_weight'] = global_weight
        training_loop_parameters.other_params['global_loss'] = global_loss
        training_loop_parameters.other_params['all_label'] = all_label
        training_loop_parameters.other_params['all_aux'] = all_aux
        training_loop_parameters.other_params['all_aux_flatten'] = all_aux_flatten
        training_loop_parameters.other_params['all_input'] = all_input









        training_loop_parameters.other_params['valid_iterator'] = training_loop_parameters.iterators[0][
            'valid_iterator']
        training_loop_parameters.other_params['scaler'] = training_loop_parameters.iterators[0]['scaler']
        training_loop_parameters.other_params['train_iterator'] = training_loop_parameters.iterators[0][
            'train_iterator']
        training_loop_parameters.other_params['groups'] = [i for i in range(total_no_groups)]


        if "only_generated_data" in method:   #"only_generated_data" in method

            all_input_augmented = training_loop_parameters.iterators[0]['train_X_augmented']
            all_label_augmented = training_loop_parameters.iterators[0]['train_y_augmented']
            all_aux_augmented = training_loop_parameters.iterators[0]['train_s_augmented']
            training_loop_parameters.other_params['all_label_augmented'] = all_label_augmented
            training_loop_parameters.other_params['all_aux_augmented'] = all_aux_augmented
            training_loop_parameters.other_params['all_aux_flatten_augmented'] = \
                [training_loop_parameters.other_params['s_to_flattened_s'][tuple(i)]
                 for i in training_loop_parameters.other_params['all_aux_augmented']]
            training_loop_parameters.other_params['all_input_augmented'] = all_input_augmented


        if "train_on_only_generated_data" in method:

            training_loop_parameters.other_params['all_label'] = all_label_augmented
            training_loop_parameters.other_params['all_aux'] = all_aux_augmented
            training_loop_parameters.other_params['all_aux_flatten'] = np.asarray([training_loop_parameters.other_params['s_to_flattened_s'][tuple(i)]
             for i in training_loop_parameters.other_params['all_aux_augmented']])
            training_loop_parameters.other_params['all_input'] = all_input_augmented


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
