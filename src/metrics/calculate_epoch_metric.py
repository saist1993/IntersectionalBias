# This is going to be a orchastrator for the fairness/ other metric
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from .eps_fairness import EpsFairness, EPSFairnessMetric
from .accuracy_parity import AccuracyParity
from .true_postivie_rate import TruePositiveRateParity
from .fairness_utils import FairnessMetricTracker
from metrics import fairness_utils


@dataclass
class EpochMetricTracker():
    accuracy: float
    balanced_accuracy: float
    accuracy_parity: FairnessMetricTracker
    tpr_parity: FairnessMetricTracker
    eps_fairness: Dict[str, EPSFairnessMetric]
    loss: Optional[float] = None
    epoch_number: Optional[int] = None


class CalculateEpochMetric:
    def __init__(self, prediction: np.asarray, label: np.asarray, aux: np.asarray,
                 other_meta_data: Optional[dict] = None):
        self.prediction = prediction
        self.label = label
        self.aux = aux
        self.other_meta_data = other_meta_data

        # Find all possible groups, as all types of groups (gerrymandering, intersectional) are its subset.
        # self.all_possible_groups = fairness_utils.create_all_possible_groups(number_of_attributes=aux.shape[1])

        try:
            self.all_possible_groups = self.other_meta_data['all_possible_groups']
            self.all_possible_groups_mask = self.other_meta_data['all_possible_groups_mask']
        except KeyError:
            self.all_possible_groups = \
                fairness_utils.create_all_possible_groups \
                    (attributes=[list(np.unique(self.aux[:, i])) for i in range(aux.shape[1])])
            self.all_possible_groups_mask = [fairness_utils.create_mask(data=aux, condition=group) for group in
                                             self.all_possible_groups]

    def run(self):


        accuracy = fairness_utils.calculate_accuracy_classification(predictions=self.prediction,
                                                                    labels=self.label)

        balanced_accuracy = balanced_accuracy_score(self.label, self.prediction)

        if self.other_meta_data['adversarial_output'] is not None:
            adv_output = self.other_meta_data['adversarial_output']
            if type(adv_output) is not list:
                adv_accuracy = fairness_utils.calculate_accuracy_classification(predictions=adv_output,
                                                                            labels=self.other_meta_data['aux_flattened'])
            else:
                adv_accuracy = []
                for index in range(len(adv_output)):
                    adv_accuracy .append(fairness_utils.calculate_accuracy_classification(predictions=adv_output[index],
                                                                                    labels=self.aux[:,index]))
                adv_accuracy = np.mean(adv_accuracy)
            print(f"ADV accuracy is ********{adv_accuracy}*******************")

        if self.other_meta_data['no_fairness']:
            epoch_metric = EpochMetricTracker(accuracy=accuracy, balanced_accuracy=balanced_accuracy,
                                              accuracy_parity=None, tpr_parity=None,
                                              eps_fairness=None)

            # Is it possible to calculate equal odds here?
            return epoch_metric

        # accuracy_parity_metric = AccuracyParity(self.prediction, self.label, self.aux,
        #                                         self.all_possible_groups, self.all_possible_groups_mask,
        #                                         self.other_meta_data).run()

        accuracy_parity_metric = None

        # tpr_parity_metric = TruePositiveRateParity(self.prediction, self.label, self.aux,
        #                                         self.all_possible_groups, self.all_possible_groups_mask,
        #                                         self.other_meta_data).run()
        tpr_parity_metric = None

        eps_fairness_metric = EpsFairness(self.prediction, self.label, self.aux,
                                                self.all_possible_groups, self.all_possible_groups_mask,
                                                self.other_meta_data).run() # this is a dictionary of metric

        epoch_metric = EpochMetricTracker(accuracy=accuracy, balanced_accuracy=balanced_accuracy,
                                         accuracy_parity=accuracy_parity_metric, tpr_parity=tpr_parity_metric,
                                         eps_fairness=eps_fairness_metric)

        return epoch_metric
