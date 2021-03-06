from dataclasses import dataclass
from itertools import product
from typing import List, Optional

import numpy as np


# def create_mask(data, condition):
#     """
#     :param data: np.array
#     :param condition: a row of that numpy array
#     :return:
#     """
#     # Step1: Find all occurances of x.
#     dont_care_indices = [i for i, x in enumerate(condition) if str(x).lower() == "x"]
#
#     # Step2: replace all occurances of x with 0. Here 0 is just arbitary as we don't care about these indices
#     # However, it is necessary as creating mask requires only 0,1.
#     updated_condition = []
#     for index, c in enumerate(condition):
#         if index in dont_care_indices:
#             updated_condition.append(0)
#         else:
#             updated_condition.append(c)
#
#     # Step3: Create the mask
#     _mask = data == updated_condition
#
#     # Step3: Iterate over the column of the mask and ignore all those columns which had x initially.
#     mask = []
#     for index, i in enumerate(range(data.shape[1])):
#         if index not in dont_care_indices:
#             mask.append(_mask[:, i])
#
#     # Step4: if the mask is empty, it mean all columns are ignore. In that case, create a dummy mask with all True
#     if not mask:
#         mask = [np.full(data.shape[0], True, dtype=bool)]
#     # Step4: reduce the mask.
#     mask = np.logical_and.reduce(mask)
#
#     new_mask = create_mask_V2(data, condition)
#
#     if not np.array_equal(mask, new_mask):
#         print("Heheh")
#     return mask


def create_mask(data, condition):

    keep_indices = []

    for index, i in enumerate(condition):
        if i != 'x':
            keep_indices.append(i == data[:, index])
        else:
            keep_indices.append(np.ones_like(data[:,0], dtype='bool'))
    
    mask = np.ones_like(data[:,0], dtype='bool')

    # mask = [np.logical_and(mask, i) for i in keep_indices]

    for i in keep_indices:
        mask = np.logical_and(mask, i)
    return mask

# def create_all_possible_groups(number_of_attributes: int):
#     return [i for i in product([0, 1, 'x'], repeat=number_of_attributes)]


def create_all_possible_groups(attributes: List[list]):
    attributes = [attribute + ['x'] for attribute in attributes]
    return list(product(*attributes))

def get_gerrymandering_groups(groups: List):
    tuple_to_remove = tuple(['x' for _ in range(len(groups[0]))])
    all_index = [i for i in range(len(groups))]
    index = None
    try:
        index = groups.index(tuple_to_remove)
    except ValueError:
        pass
    if index:
        groups = [value for i, value in enumerate(groups) if i != index]
        all_index = [i for i, value in enumerate(groups) if i != index]
    return groups, all_index


def get_independent_groups(groups: List):
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
    accuracy = correct * 1.0 / labels.shape[0]
    return accuracy


def get_groups(all_possible_groups, type_of_group: str):
    if type_of_group == 'gerrymandering':
        return get_gerrymandering_groups(all_possible_groups)
    elif type_of_group == 'independent':
        return get_independent_groups(all_possible_groups)
    elif type_of_group == 'intersectional':
        return get_intersectional_groups(all_possible_groups)
    else:
        raise NotImplementedError


@dataclass
class FairnessMetricAnalytics:
    """

    abs_average -> (1/total_no_groups)( \sum abs (TPR_overall - TPR_current))
    rms_average -> (1/total_no_groups)( \root \square (TPR_overall - TPR_current))

    abs_max -> max( \iterate \abs (TPR_overall - TPR_current))
    abs_min -> min ( \iterate \abs (TPR_overall - TPR_current))
    rms_max -> max ( \iterate \root \square (TPR_overall - TPR_current)) # Is same as abs_max so not included
    rms_min -> min ( \iterate \root \square (TPR_overall - TPR_current)) # Is same as abs_min so not included

    """
    metric_over_whole_dataset: float
    maximum: float
    minimum: float
    average: float
    standard_dev: float
    abs_average: float
    abs_average_std: float
    rms_average: float
    rms_average_std: float
    abs_max: float
    abs_min: float


@dataclass
class FairnessMetricTracker:
    weighted_gerrymandering: FairnessMetricAnalytics
    weighted_independent: FairnessMetricAnalytics
    weighted_intersectional: FairnessMetricAnalytics

    unweighted_gerrymandering: FairnessMetricAnalytics
    unweighted_independent: FairnessMetricAnalytics
    unweighted_intersectional: FairnessMetricAnalytics


def generate_fairness_metric_analytics(results: list, metric_over_whole_dataset):
    difference_between_overall_and_current = [r - metric_over_whole_dataset for r in results]
    fairness_metric_analytics = FairnessMetricAnalytics(
        metric_over_whole_dataset=metric_over_whole_dataset,
        maximum=max(results),
        minimum=min(results),
        average=np.mean(results),
        standard_dev=np.std(results),
        abs_average=np.mean([abs(i) for i in difference_between_overall_and_current]),
        abs_average_std=np.std([abs(i) for i in difference_between_overall_and_current]),
        rms_average=np.sqrt(np.mean([i * i for i in difference_between_overall_and_current])),
        rms_average_std=np.sqrt(np.std([i * i for i in difference_between_overall_and_current])),
        abs_max=np.max([abs(i) for i in difference_between_overall_and_current]),
        abs_min=np.min([abs(i) for i in difference_between_overall_and_current]),
    )
    return fairness_metric_analytics


class FairnessTemplateClass:
    def __init__(self, prediction: np.asarray, label: np.asarray, aux: np.asarray,
                 all_possible_groups: List, all_possible_groups_mask: List, other_meta_data: Optional[dict] = None):
        self.prediction = prediction
        self.label = label
        self.aux = aux
        self.other_meta_data = other_meta_data

        # Find all possible groups, as all types of groups (gerrymandering, intersectional) are its subset.
        self.all_possible_groups = all_possible_groups
        self.all_possible_groups_mask = all_possible_groups_mask

        self.all_true_mask = np.asarray([True for _ in range(len(self.all_possible_groups_mask[0]))])

    def run(self):
        # does the actual heavy lifting!
        pass
