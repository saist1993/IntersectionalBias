# This will dynamically increase or decrease examples based on some notion of fairness.
import statistics

import numpy as np
from tqdm.auto import tqdm


from .common_functionality import *
from typing import NamedTuple, List
from fairgrad.torch import CrossEntropyLoss as fairgrad_CrossEntropyLoss
from utils.iterator import TextClassificationDataset, sequential_transforms


class CreateIterators:
    """ A general purpose iterators. Takes numpy matrix for train, dev, and test matrix and creates iterator."""

    def __init__(self, s_to_flattened_s):
        self.s_to_flattened_s = s_to_flattened_s

    def collate(self, batch):
        labels, encoded_input, aux, generated_encoded_input = zip(*batch)

        labels = torch.LongTensor(labels)
        aux_flattened = torch.LongTensor(self.get_flatten_s(aux))
        aux = torch.LongTensor(np.asarray(aux))
        lengths = torch.LongTensor([len(x) for x in encoded_input])
        encoded_input = torch.FloatTensor(np.asarray(encoded_input))
        generated_encoded_input = torch.FloatTensor(np.asarray(generated_encoded_input))

        input_data = {
            'labels': labels,
            'input': encoded_input,
            'generated_input': generated_encoded_input,
            'lengths': lengths,
            'aux': aux,
            'aux_flattened': aux_flattened
        }

        return input_data

    def process_data(self, X, y, s, generated_train_X, vocab):
        """raw data is assumed to be tokenized"""

        final_data = [(a, b, c, d) for a, b, c, d in zip(y, X, s, generated_train_X)]

        label_transform = sequential_transforms()
        input_transform = sequential_transforms()
        aux_transform = sequential_transforms()
        generated_data_transform = sequential_transforms()

        transforms = (label_transform, input_transform, aux_transform, generated_data_transform)

        return TextClassificationDataset(final_data, vocab, transforms)

    def flatten_s(self, s: List[List]):
        for f in s:
            keys = self.s_to_flattened_s.keys()
            if tuple([int(i) for i in f]) not in keys:
                self.s_to_flattened_s[tuple([int(i) for i in f])] = len(keys)

    def get_flatten_s(self, s: List[List]):
        return [self.s_to_flattened_s[tuple([int(j) for j in i])] for i in s]

    def get_iterators(self, train_X, train_s, train_y, generated_train_X, batch_size):
        train_X, train_s, train_y, generated_train_X = train_X, train_s, train_y, generated_train_X

        vocab = {'<pad>': 1}  # no need of vocab in these dataset. It is there for code compatibility purposes.

        # need to add flatten here! And that too in the process itself!
        train_data = self.process_data(train_X, train_y, train_s, generated_train_X, vocab=vocab)

        train_iterator = torch.utils.data.DataLoader(train_data,
                                                     batch_size,
                                                     shuffle=False,
                                                     collate_fn=self.collate
                                                     )

        return train_iterator


def get_subset_of_data(X, y, model):
    preds = torch.argmax(model({"input": torch.FloatTensor(np.asarray(X))})['prediction'], axis=1)
    return torch.eq(preds, torch.LongTensor(y)).detach().numpy()


def resample_dataset_older(all_X, all_s, all_y, size_of_each_group, model=None):
    resampled_X, resampled_s, resampled_y, size = [], [], [], {}
    all_unique_s = np.unique(all_s, axis=0)
    all_unique_label = np.unique(all_y)

    for s in all_unique_s:
        for label in all_unique_label:
            # sample number_of_examples_per_group with s and label from the dataset
            mask_s = generate_mask(all_s=all_s, mask_pattern=s)
            mask_label = all_y == label
            final_mask = np.logical_and(mask_s, mask_label)
            index = np.where(final_mask)[0]
            if model:
                subset_mask = get_subset_of_data(all_X[final_mask], all_y[final_mask], model)
                index = index[subset_mask == True]
            index_of_selected_examples = np.random.choice(index,
                                                          size=size_of_each_group[tuple(s.tolist())][label],
                                                          replace=True)

            resampled_X.append(all_X[index_of_selected_examples])
            resampled_s.append(all_s[index_of_selected_examples])
            resampled_y.append(all_y[index_of_selected_examples])

    resampled_X = np.vstack(resampled_X)
    resampled_s = np.vstack(resampled_s)
    resampled_y = np.hstack(resampled_y)

    # shuffle the samples here
    index = np.arange(len(resampled_X))
    np.random.shuffle(index)

    return resampled_X[index], resampled_s[index], resampled_y[index]


def resample_dataset(all_X, all_s, all_y, size_of_each_group, model=None):
    resampled_X, resampled_s, resampled_y, size = [], [], [], {}
    all_unique_s = np.unique(all_s, axis=0)
    all_unique_label = np.unique(all_y)

    for s in all_unique_s:
        for label in all_unique_label:
            # sample number_of_examples_per_group with s and label from the dataset
            mask_s = generate_mask(all_s=all_s, mask_pattern=s)
            mask_label = all_y == label
            final_mask = np.logical_and(mask_s, mask_label)
            index = np.where(final_mask)[0]
            if model:
                subset_mask = get_subset_of_data(all_X[final_mask], all_y[final_mask], model)
                index = index[subset_mask == True]
            # index are all the possible examples one can use.
            group_size, all_members_of_the_group = size_of_each_group[tuple(s.tolist())][label]

            if len(all_members_of_the_group) > group_size:
                k = len(all_members_of_the_group) - group_size
                all_members_of_the_group = all_members_of_the_group[k:]
                size_of_each_group[tuple(s.tolist())][label] = [group_size, all_members_of_the_group]
            else:
                # need to add examples
                k = group_size - len(all_members_of_the_group)
                all_members_of_the_group.extend(np.random.choice(index,
                                                          size=k,
                                                          replace=True))

                size_of_each_group[tuple(s.tolist())][label] = [group_size, all_members_of_the_group]

            index_of_selected_examples = np.random.choice(all_members_of_the_group,
                                                          size=size_of_each_group[tuple(s.tolist())][label][0],
                                                          replace=True)

            resampled_X.append(all_X[index_of_selected_examples])
            resampled_s.append(all_s[index_of_selected_examples])
            resampled_y.append(all_y[index_of_selected_examples])

    resampled_X = np.vstack(resampled_X)
    resampled_s = np.vstack(resampled_s)
    resampled_y = np.hstack(resampled_y)


    return resampled_X, resampled_s, resampled_y, size_of_each_group

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


def train(train_parameters: TrainParameters):
    model, optimizer, device, criterion, mode = \
        train_parameters.model, train_parameters.optimizer, train_parameters.device, train_parameters.criterion, train_parameters.mode

    adversarial_lambda = train_parameters.other_params['adversarial_lambda']

    model.train()

    for items in tqdm(train_parameters.iterator):
        for key in items.keys():
            items[key] = items[key].to(device)



        if "mixup_generated_and_real_data" in train_parameters.other_params['method']:
            alpha = 1.0
            gamma = beta(alpha, alpha)
            items['input'] = gamma*items['input'] + (1-gamma)*items['generated_input']

        optimizer.zero_grad()
        output = model(items)
        loss = criterion(output['prediction'], items['labels'], items['aux_flattened'], mode='train')
        loss = torch.mean(loss)
        adv_loss = torch.mean(criterion(output['adv_outputs'][0], items['aux_flattened']))
        loss = loss + adversarial_lambda*adv_loss

        # the adversarial part comes now.
        loss.backward()
        optimizer.step()


def update_size(train_other_info, valid_other_info, size_of_groups):
    def get_median(other_info):
        all_groups_accuracy = [value[-1] for key, value in other_info.items()]
        all_groups_accuracy.extend([value[-2] for key, value in other_info.items()])
        return statistics.median(all_groups_accuracy)

    train_accuracy_median = get_median(train_other_info)
    valid_accuracy_median = get_median(valid_other_info)

    train_accuracy_median = 0.80
    valid_accuracy_median = 0.80

    for group, value in size_of_groups.items():
        for label, current_size in size_of_groups[group].items():
            current_size = current_size[0]
            if label == 1:
                current_accuracy = train_other_info[group][-2]
            else:
                current_accuracy = train_other_info[group][-1]

            if current_accuracy > valid_accuracy_median:
                updated_size = max(1,current_size - 50)
            else:
                updated_size = current_size + 50
            size_of_groups[group][label] = [updated_size, size_of_groups[group][label][1]]

    return size_of_groups


def orchestrator(training_loop_parameters: TrainingLoopParameters):
    ##### Get the original data from the iterators #####
    _labels, _input, _lengths, _aux, _aux_flattened = [], [], [], [], []
    for items in (training_loop_parameters.iterators[0]['train_iterator']):
        _labels.append(items['labels'])
        _input.append(items['input'])
        _lengths.append(items['lengths'])
        _aux.append(items['aux'])
        _aux_flattened.append(items['aux_flattened'])

    original_train_label = np.hstack(_labels)
    original_train_aux = np.vstack(_aux)
    # original_train_aux_flatten = np.hstack(_aux_flattened) # We will not use flat_s anywhere.
    original_train_input = np.vstack(_input)

    # we have access to generated data!
    generated_train_label, generated_train_aux, generated_train_input = \
        training_loop_parameters.iterators[0]['train_y_augmented'], \
            training_loop_parameters.iterators[0]['train_s_augmented'],\
            training_loop_parameters.iterators[0]['train_X_augmented']



    all_unique_s = np.unique(original_train_aux, axis=0)
    all_unique_label = np.unique(original_train_label)

    size_of_each_group = {}
    for s in all_unique_s:
        size_of_each_group[tuple(s.tolist())] = {}
        for label in all_unique_label:
            size_of_each_group[tuple(s.tolist())][label] = [training_loop_parameters.other_params['per_group_label_number_of_examples'],
                                                            []]  # [size of group, all current members of the group]


    gen_size_of_each_group = copy.deepcopy(size_of_each_group)

    '''
        Steps to Follow. A rough procedure 

        - At the begining resample the dataset so that it refelects the number of samples for each group 
        - Create an iterator which iterates over this dataset. 
        - Send this resampled iterator through the system and get the output 
    '''

    resampled_X, resampled_s, resampled_y, size_of_each_group = resample_dataset(all_X=original_train_input,
                                                             all_s=original_train_aux,
                                                             all_y=original_train_label,
                                                             size_of_each_group=size_of_each_group)

    gen_resampled_X, gen_resampled_s, gen_resampled_y, gen_size_of_each_group = resample_dataset(all_X=generated_train_input,
                                                             all_s=generated_train_aux,
                                                             all_y=generated_train_label,
                                                             size_of_each_group=gen_size_of_each_group)

    fairness_related_info = {
        'y_train': resampled_y,
        's_train': np.asarray([training_loop_parameters.other_params['s_to_flattened_s'][tuple(i)] for i in resampled_s]),
        'fairness_measure': training_loop_parameters.fairness_function,
        'fairness_rate': 0.01,
        'epsilon': 0.0
    }

    training_loop_parameters.criterion = fairgrad_CrossEntropyLoss(reduction='none', **fairness_related_info)


    index = np.arange(len(resampled_X))
    np.random.shuffle(index)

    resampled_X, resampled_s, resampled_y = resampled_X[index], resampled_s[index], resampled_y[index]
    gen_resampled_X, gen_resampled_s, gen_resampled_y = gen_resampled_X[index], gen_resampled_s[index], gen_resampled_y[index]


    create_iterators = CreateIterators(s_to_flattened_s=training_loop_parameters.other_params['s_to_flattened_s'])
    resampled_iterator = create_iterators.get_iterators(train_X=resampled_X, train_s=resampled_s, train_y=resampled_y,
                                                        generated_train_X=gen_resampled_X,
                                                        batch_size=training_loop_parameters.other_params['batch_size'])

    logger = logging.getLogger(training_loop_parameters.unique_id_for_run)
    output = {}
    all_train_eps_metrics = []
    all_test_eps_metrics = []
    all_valid_eps_metrics = []

    for ep in range(training_loop_parameters.n_epochs):
        logger.info("start of epoch block  ")

        train_parameters = TrainParameters(
            model=training_loop_parameters.model,
            iterator=resampled_iterator,
            optimizer=training_loop_parameters.optimizer,
            criterion=training_loop_parameters.criterion,
            device=training_loop_parameters.device,
            other_params=training_loop_parameters.other_params,
            per_epoch_metric=per_epoch_metric,
            mode='train',
            fairness_function=training_loop_parameters.fairness_function,
            iterator_set=None)

        train(train_parameters=train_parameters)

        valid_parameters = TrainParameters(
            model=training_loop_parameters.model,
            iterator=training_loop_parameters.iterators[0]['train_iterator'],
            optimizer=training_loop_parameters.optimizer,
            criterion=training_loop_parameters.criterion,
            device=training_loop_parameters.device,
            other_params=training_loop_parameters.other_params,
            per_epoch_metric=per_epoch_metric,
            mode='evaluate',
            fairness_function=training_loop_parameters.fairness_function,
            iterator_set=None)

        train_epoch_metric, loss = test(valid_parameters)

        # log_epoch_metric(logger, start_message='train', epoch_metric=train_epoch_metric, epoch_number=ep, loss=loss)

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
            fairness_function=training_loop_parameters.fairness_function,
            iterator_set=None)

        valid_epoch_metric, loss = test(valid_parameters)
        log_epoch_metric(logger, start_message='valid', epoch_metric=valid_epoch_metric, epoch_number=ep, loss=loss)

        size_of_each_group = \
            update_size(train_other_info=train_epoch_metric.other_info, valid_other_info=valid_epoch_metric.other_info,
                        size_of_groups=size_of_each_group)

        gen_size_of_each_group = update_size(train_other_info=train_epoch_metric.other_info, valid_other_info=valid_epoch_metric.other_info,
                        size_of_groups=gen_size_of_each_group)


        # if ep%5 == 0:
        #
        #     for layer in training_loop_parameters.model.children():
        #         if hasattr(layer, 'reset_parameters'):
        #             layer.reset_parameters()

        # resampled_X, resampled_s, resampled_y, size_of_each_group = resample_dataset(all_X=original_train_input,
        #                                                                              all_s=original_train_aux,
        #                                                                              all_y=original_train_label,
        #                                                                              size_of_each_group=size_of_each_group,
        #                                                                              model=training_loop_parameters.model)

        # gen_resampled_X, gen_resampled_s, gen_resampled_y, gen_size_of_each_group = resample_dataset(
        #     all_X=generated_train_input,
        #     all_s=generated_train_aux,
        #     all_y=generated_train_label,
        #     size_of_each_group=gen_size_of_each_group,
        #     model=training_loop_parameters.model)

        # index = np.arange(len(resampled_X))
        # np.random.shuffle(index)
        #
        # resampled_X, resampled_s, resampled_y = resampled_X[index], resampled_s[index], resampled_y[index]
        # gen_resampled_X, gen_resampled_s, gen_resampled_y = gen_resampled_X[index], gen_resampled_s[index], \
        # gen_resampled_y[index]



        resampled_iterator = create_iterators.get_iterators(train_X=resampled_X, train_s=resampled_s,
                                                            train_y=resampled_y, generated_train_X=gen_resampled_X,
                                                            batch_size=training_loop_parameters.other_params[
                                                                'batch_size'])

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
            fairness_function=training_loop_parameters.fairness_function,
            iterator_set=None)

        test_epoch_metric, loss = test(test_parameters)
        if training_loop_parameters.use_wandb:
            log_and_plot_data(epoch_metric=test_epoch_metric, loss=loss, train=False)
        log_epoch_metric(logger, start_message='test', epoch_metric=test_epoch_metric, epoch_number=ep, loss=loss)
        all_test_eps_metrics.append(test_epoch_metric)

        # print(size_of_each_group)

        logger.info("end of epoch block")

    output['all_train_eps_metric'] = all_train_eps_metrics
    output['all_test_eps_metric'] = all_test_eps_metrics
    output['all_valid_eps_metric'] = all_valid_eps_metrics

    return output
