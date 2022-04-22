# This is going to be a orchastrator for the fairness/ other metric
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from .accuracy_parity import AccuracyParity
from .fairness_utils import FairnessMetricTracker
from metrics import fairness_utils


@dataclass
class EpochMetricTracker():
    accuracy: float
    balanced_accuracy: float
    accuracy_parity: FairnessMetricTracker


class CalculateEpochMetric:
    def __init__(self, prediction: np.asarray, label: np.asarray, aux: np.asarray,
                 other_meta_data: Optional[dict] = None):
        self.prediction = prediction
        self.label = label
        self.aux = aux
        self.other_meta_data = other_meta_data

        # Find all possible groups, as all types of groups (gerrymandering, intersectional) are its subset.
        self.all_possible_groups = fairness_utils.create_all_possible_groups(number_of_attributes=aux.shape[1])
        self.all_possible_groups_mask = [fairness_utils.create_mask(data=aux, condition=group) for group in
                                         self.all_possible_groups]

    def run(self):
        accuracy = fairness_utils.calculate_accuracy_classification(predictions=self.prediction,
                                                                    labels=self.label)

        balanced_accuracy = balanced_accuracy_score(self.label, self.prediction)

        accuracy_parity_metric = AccuracyParity(self.prediction, self.label, self.aux,
                                                self.all_possible_groups, self.all_possible_groups_mask,
                                                self.other_meta_data).run()

        epoch_metric = EpochMetricTracker(accuracy=accuracy, balanced_accuracy=balanced_accuracy,
                                          accuracy_parity=accuracy_parity_metric)

        return epoch_metric
