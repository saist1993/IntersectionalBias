import torch
import numpy as np
from main import *
from tqdm.auto import tqdm
from fairgrad.torch import CrossEntropyLoss
from training_loops.common_functionality import *


def generate_flat_outputs(model, iterator, criterion, attribute_id=None):
    track_output = []
    track_input = []
    for items in tqdm(iterator):
        output = model(items)
        loss = torch.mean(criterion(output['prediction'], items['labels']))
        output['loss_batch'] = loss.item()
        track_output.append(output)
        track_input.append(items)

    # Flattening the results
    all_label = []
    all_prediction = []
    all_loss = []
    all_s = []
    all_s_flatten = []

    for batch_output, batch_input in zip(track_output, track_input):
        all_prediction.append(batch_output['prediction'].detach().numpy())
        all_loss.append(batch_output['loss_batch'])
        all_label.append(batch_input['labels'].numpy())
        all_s.append(batch_input['aux'].numpy())
        if attribute_id is not None:
            all_s_flatten.append(batch_input['aux'][:, attribute_id].numpy())
        else:
            all_s_flatten.append(batch_input['aux_flattened'].numpy())

    all_prediction = np.vstack(all_prediction).argmax(1)
    all_label = np.hstack(all_label)
    all_s = np.vstack(all_s)
    all_s_flatten = np.hstack(all_s_flatten)

    return all_prediction, all_label, all_s, all_s_flatten


def smoothed_empirical_estimate_rate_parity(preds, labels, mask_aux, use_tpr=True):
    new_preds = preds[mask_aux]
    new_labels = labels[mask_aux]
    if use_tpr:
        new_mask = (new_preds == 1) & (new_labels == 1)
        numerator = np.count_nonzero(new_mask == 1)
        new_mask = new_labels == 1
        denominator = np.count_nonzero(new_mask)
    else:
        new_mask = (new_preds == 1) & (new_labels == 0)
        numerator = np.count_nonzero(new_mask == 1)
        new_mask = new_labels == 0
        denominator = np.count_nonzero(new_mask)

    # find the number of instance with S=s and label=1

    prob = (numerator + 0.01) / (denominator * 1.0 + 0.01 + 0.01)
    return prob


def calculate_equal_odds(all_prediction, all_label, all_s, all_unique_aux, use_tpr):
    all_probs = []
    for s in all_unique_aux:
        repeat_aux = np.tile(s,(all_s.shape[0],1))
        mask_aux = (repeat_aux == all_s).sum(axis=1) == all_s.shape[1]
        prob = smoothed_empirical_estimate_rate_parity(all_prediction, all_label, mask_aux, use_tpr=use_tpr)
        all_probs.append(prob)
    max_prob = max(all_probs)
    min_prob = min(all_probs)
    eps = np.log(max_prob / min_prob)
    return eps, all_probs


def run(model, iterators, criterion):
    all_prediction, all_label, all_s, all_s_flatten = generate_flat_outputs(model, iterators[0]['test_iterator'], criterion)

    eps_tpr, eps_fpr = 0.0, 0.0
    eps_tpr_prob, eps_fpr_prob = [], []

    all_unique_aux = np.unique(all_s, axis=0)

    for _ in range(1000):
        index_select = np.random.choice(len(all_prediction), len(all_prediction), replace=True)
        eps_tpr_, eps_tpr_prob_ = calculate_equal_odds(all_prediction[index_select], all_label[index_select],
                                                    all_s[index_select], all_unique_aux, True)
        eps_fpr_, eps_fpr_prob_ = calculate_equal_odds(all_prediction[index_select], all_label[index_select],
                                                    all_s[index_select], all_unique_aux, False)
        eps_tpr += eps_tpr_
        eps_fpr += eps_fpr_
        eps_tpr_prob.append(eps_tpr_prob_)
        eps_fpr_prob.append(eps_fpr_prob_)

    print(max(eps_tpr / 1000, eps_fpr / 1000))

    for i, j, k in zip(np.mean(eps_tpr_prob, axis=0), np.mean(eps_fpr_prob, axis=0), all_unique_aux):
        print(i, j, k)



if __name__ == '__main__':
    # All parameters we would need to change eventually
    model_save_path = '../saved_models/equal_odds/unconstrained_adult_multi_group.pt'
    dataset_name = 'adult_multi_group'
    method = 'unconstrained'
    fairness_function = 'equal_odds'
    seed = 10
    model_name = 'simple_non_linear'
    device = torch.device('cpu')

    set_seed(seed)


    iterator_params = {
        'batch_size': 64,
        'lm_encoder_type': 'bert-base-uncased',
        'lm_encoding_to_use': 'use_cls',
        'return_numpy_array': True,
        'dataset_size': 1000
    }
    iterators, other_meta_data = generate_data_iterators(dataset_name=dataset_name, **iterator_params)

    # Step 2 - Create and load the model
    other_meta_data['attribute_id'] = None

    # Create model
    model = get_model(method=method, model_name=model_name, other_meta_data=other_meta_data, device=device)

    # Load model
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    # create a loss function
    criterion = CrossEntropyLoss()

    run(model, iterators, criterion)