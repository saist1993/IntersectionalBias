import numpy as np
from itertools import product
from typing import List
from sklearn.metrics import confusion_matrix


def create_mask(data, condition):
    """
    :param data: np.array
    :param condition: a row of that numpy array
    :return:
    """
    # Step1: Find all occurances of x.
    dont_care_indices = [i for i, x in enumerate(condition) if str(x).lower() == "x"]

    # Step2: replace all occurances of x with 0. Here 0 is just arbitary as we don't care about these indices
    # However, it is necessary as creating mask requires only 0,1.
    updated_condition = []
    for index, c in enumerate(condition):
        if index in dont_care_indices:
            updated_condition.append(0)
        else:
            updated_condition.append(c)

    # Step3: Create the mask
    _mask = data == updated_condition

    # Step3: Iterate over the column of the mask and ignore all those columns which had x initially.
    mask = []
    for index, i in enumerate(range(data.shape[1])):
        if index not in dont_care_indices:
            mask.append(_mask[:, i])

    # Step4: if the mask is empty, it mean all columns are ignore. In that case, create a dummy mask with all True
    if not mask:
        mask = [np.full(data.shape[0], True, dtype=bool)]
    # Step4: reduce the mask.
    mask = np.logical_and.reduce(mask)
    return mask


def create_all_possible_groups(number_of_attributes: int):
    return [i for i in product([0, 1, 'x'], repeat=number_of_attributes)]


def get_gerrymandering_groups(groups: List):

    tuple_to_remove = tuple(['x' for _ in range(len(groups[0]))])
    all_index = [i for i in range(len(groups))]
    index = None
    try:
        index = groups.index(tuple_to_remove)
    except ValueError:
        pass
    if index:
        groups = [value for i, value in enumerate(groups) if i!= index]
        all_index = [i for i, value in enumerate(groups) if i != index]
    return groups, all_index


def get_independent_groups(groups:List):
    '''All x's apart from one place'''

    number_of_attribute = len(groups[1])
    independent_groups = []
    all_index = []
    for index, group in enumerate(groups):
        if type(group) == np.ndarray:
            group = group.tolist()
        if group.count('x') == number_of_attribute - 1:
            independent_groups.append(group)
            all_index.append(index)
    return independent_groups, all_index


def get_intersectional_groups(groups: List):
    '''No x's in any place'''

    intersectional_groups = []
    all_index = []
    for index, group in enumerate(groups):
        if type(group) == np.ndarray:
            group = group.tolist()
        if group.count('x') == 0:
            intersectional_groups.append(group)
            all_index.append(index)
    return intersectional_groups, all_index

def calculate_accuracy_classification(predictions, labels):
    # top_predictions = predictions.argmax(1)
    correct = (predictions == labels).sum()
    accuracy = correct*1.0/ labels.shape[0]
    return accuracy

def calculate_true_positive_rate(predictions, labels):
    tn, fp, fn, tp = confusion_matrix(y_true=labels, y_pred=predictions).ravel()
    return tp/(tp + fn)


def accuracy_parity(prediction, label, mask):
    """Calculates accuracy over given mask"""
    prediction = prediction[mask]
    label = label[mask]
    return  calculate_accuracy_classification(prediction, label)


def true_positive_rate(prediction, label, mask):
    """Calculates accuracy over given mask"""
    prediction = prediction[mask]
    label = label[mask]
    return  calculate_true_positive_rate(prediction, label)


def accuracy_parity_over_groups(prediction, label, all_possible_groups_mask):
    return [accuracy_parity(prediction, label, group_mask) for group_mask in all_possible_groups_mask]


def true_positive_rate_over_groups(prediction, label, all_possible_groups_mask):
    return [true_positive_rate(prediction, label, group_mask) for group_mask in all_possible_groups_mask]