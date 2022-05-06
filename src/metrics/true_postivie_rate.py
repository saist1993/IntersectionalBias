
import numpy as np
from typing import List, Optional
from metrics import fairness_utils
from sklearn.metrics import confusion_matrix


class TruePositiveRateParity(fairness_utils.FairnessTemplateClass):

    def __init__(self, prediction: np.asarray, label: np.asarray, aux: np.asarray,
                 all_possible_groups: List, all_possible_groups_mask: List, other_meta_data: Optional[dict] = None):
        super().__init__(prediction, label, aux,
                         all_possible_groups, all_possible_groups_mask, other_meta_data)

    @staticmethod
    def calculate_true_postive_rate(predictions, labels):
        """Calculates accuracy over given mask"""
        tn, fp, fn, tp = confusion_matrix(y_true=labels, y_pred=predictions).ravel()
        return tp/(tp + fn)


    def true_postivie_rates_over_groups(self, prediction, label, all_possible_groups_mask):
        return [self.calculate_true_postive_rate(prediction[group_mask], label[group_mask]) for group_mask in
                all_possible_groups_mask]

    def run(self):
        per_group_tpr = [self.calculate_true_postive_rate(predictions=self.prediction[mask],
                                                                               labels=self.label[mask])
                              for mask in self.all_possible_groups_mask if not np.array_equal(mask, self.all_true_mask)]

        per_group_size = [len(self.prediction[mask]) for mask in self.all_possible_groups_mask if
                          not np.array_equal(mask, self.all_true_mask)]

        overall_tpr = self.calculate_true_postive_rate(predictions=self.prediction, labels=self.label)
        weighted_tpr = np.average(per_group_tpr, weights=per_group_size)  # same as overall accuracy!
        unweighted_tpr = np.average(per_group_tpr)  # check if this is same as overall accuracy!

        # print(f"**overall tpr**{overall_tpr}**weighted tpr**{weighted_tpr}**unweighted tpr**{unweighted_tpr}")

        def get_analytics(type_of_group, overall_tpr):
            _, group_index = fairness_utils.get_groups(self.all_possible_groups, type_of_group)
            group_wise_accuracy = np.asarray(per_group_tpr)[group_index]
            return fairness_utils.generate_fairness_metric_analytics(group_wise_accuracy,
                                                                     overall_tpr)

        weighted_gerrymandering = get_analytics('gerrymandering', weighted_tpr)
        unweighted_gerrymandering = get_analytics('gerrymandering', unweighted_tpr)

        weighted_independent = get_analytics('independent', weighted_tpr)
        unweighted_independent = get_analytics('independent', unweighted_tpr)

        weighted_intersectional = get_analytics('intersectional', weighted_tpr)
        unweighted_intersectional = get_analytics('intersectional', unweighted_tpr)

        fairness_metric_tracker = fairness_utils.FairnessMetricTracker(weighted_gerrymandering=weighted_gerrymandering,
                                                                       weighted_independent=weighted_independent,
                                                                       weighted_intersectional=weighted_intersectional,
                                                                       unweighted_gerrymandering=unweighted_gerrymandering,
                                                                       unweighted_independent=unweighted_independent,
                                                                       unweighted_intersectional=unweighted_intersectional,
                                                                       )

        return fairness_metric_tracker
