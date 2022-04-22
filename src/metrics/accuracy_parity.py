from typing import List, Optional

import numpy as np

import fairness_utils


class AccuracyParity(fairness_utils.FairnessTemplateClass):

    def __init__(self, prediction: np.asarray, label: np.asarray, aux: np.asarray,
                 all_possible_groups: List, all_possible_groups_mask: List, other_meta_data: Optional[dict] = None):
        super().__init__(prediction, label, aux,
                         all_possible_groups, all_possible_groups_mask, other_meta_data)

    @staticmethod
    def accuracy_parity(prediction, label):
        """Calculates accuracy over given mask"""
        return fairness_utils.calculate_accuracy_classification(prediction, label)

    def accuracy_parity_over_groups(self, prediction, label, all_possible_groups_mask):
        return [self.accuracy_parity(prediction[group_mask], label[group_mask]) for group_mask in
                all_possible_groups_mask]

    def run(self):
        # calculate the macro, micro, and weighted accuracy

        # unbalanced_accuracy = fairness_utils.calculate_accuracy_classification(predictions=self.prediction,
        #                                                                     labels=self.label)

        # # removing all true mask, as this corresponds to overall_accuracy -> accuracy over the whole dataset
        per_group_accuracy = [fairness_utils.calculate_accuracy_classification(predictions=self.prediction[mask],
                                                                               labels=self.label[mask])
                              for mask in self.all_possible_groups_mask if not np.array_equal(mask, self.all_true_mask)]

        per_group_size = [len(mask) for mask in self.all_possible_groups_mask if
                          not np.array_equal(mask, self.all_true_mask)]

        # balanced_accuracy = balanced_accuracy_score(self.label, self.prediction)

        weighted_accuracy = np.average(per_group_accuracy, weights=per_group_size)
        unweighted_accuracy = np.average(per_group_accuracy)  # check if this is same as overall accuracy!

        def get_analytics(type_of_group, overall_accuracy):
            _, group_index = fairness_utils.get_groups(self.all_possible_groups, type_of_group)
            group_wise_accuracy = np.asarray(per_group_accuracy)[group_index]
            return fairness_utils.generate_fairness_metric_analytics(group_wise_accuracy,
                                                                     overall_accuracy)

        weighted_gerrymandering = get_analytics('gerrymandering', weighted_accuracy)
        unweighted_gerrymandering = get_analytics('gerrymandering', unweighted_accuracy)

        weighted_independent = get_analytics('independent', weighted_accuracy)
        unweighted_independent = get_analytics('independent', unweighted_accuracy)

        weighted_intersectional = get_analytics('intersectional', weighted_accuracy)
        unweighted_intersectional = get_analytics('intersectional', unweighted_accuracy)

        fairness_metric_tracker = fairness_utils.FairnessMetricTracker(weighted_gerrymandering=weighted_gerrymandering,
                                                                       weighted_independent=weighted_independent,
                                                                       weighted_intersectional=weighted_intersectional,
                                                                       unweighted_gerrymandering=unweighted_gerrymandering,
                                                                       unweighted_independent=unweighted_independent,
                                                                       unweighted_intersectional=unweighted_intersectional,
                                                                       )

        return fairness_metric_tracker
