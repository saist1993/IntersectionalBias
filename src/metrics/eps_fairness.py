from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from metrics import fairness_utils


class EstimateProbability:

    def __init__(self):
        self.alpha = 0.01
        self.beta = 0.01

    def empirical_estimate_demographic_parity(self, preds, labels, mask):
        '''
            mask gives us access to specific attribute - s.
            empirical_estimate = number_of_instance_with_s_and_preds_1 / number_of_instance_with_s

            labels are not useful in this case, as we don't rely on the underlying data stastics!
        '''
        new_preds = preds[mask]
        new_labels = preds[mask]  # this gives us all s

        # find number of instances with label = 1
        number_of_instance_with_s_and_y_1 = np.count_nonzero(new_preds == 1)

        return number_of_instance_with_s_and_y_1 / len(new_preds) * 1.0

    def smoothed_empirical_estimate_demographic_parity(self, preds, labels, masks):
        '''
            mask gives us access to specific attribute - s.
            empirical_estimate = number_of_instance_with_s_and_preds_1 / number_of_instance_with_s

            labels are not useful in this case, as we don't rely on the underlying data stastics!
        '''
        all_probs = []
        for mask in masks:
            new_preds = preds[mask]
            new_labels = labels[mask]  # this gives us all s

            # find number of instances with label = 1
            number_of_instance_with_s_and_y_1 = np.count_nonzero(new_preds == 1)

            prob = (number_of_instance_with_s_and_y_1 + self.alpha) / (len(new_preds) * 1.0 + self.alpha + self.beta)
            all_probs.append(prob)

        max_prob = max(all_probs)
        min_prob = min(all_probs)
        return np.log(max_prob / min_prob)

    def simple_bayesian_estimate_demographic_parity(self, preds, labels, masks):

        all_eps = []
        number_of_simulation = range(1000)
        for _ in number_of_simulation:
            all_probs = []
            for mask in masks:
                new_preds = preds[mask]
                new_labels = labels[mask]  # this gives us all s

                # find number of instances with preds = 1
                n_1_s = np.count_nonzero(new_preds == 1)
                n_s = len(new_preds)
                prob = np.random.beta(n_1_s + (1 / 3.0), n_s + (1 / 3.0) - n_1_s, size=1)
                all_probs.append(prob)

            max_prob = max(all_probs)
            min_prob = min(all_probs)
            eps = np.log(max_prob / min_prob)
            all_eps.append(eps)
        return np.mean(all_eps)

    def smoothed_empirical_estimate_rate_parity(self, preds, labels, masks, use_tpr=True):
        '''
            - if use_tpr is False, then return fpr
            mask gives us access to specific attribute - s.
            empirical_estimate = number_of_instance_with_s_and_preds_1_and_label_1 + alpha /
             number_of_instance_with_s_and_lable_1 + alpha + beta

            labels are not useful in this case, as we don't rely on the underlying data stastics!
        '''
        all_probs = []
        for mask in masks:
            new_preds = preds[mask]
            new_labels = labels[mask]  # this gives us all s

            # find the number of instances with pred=1, and S=s and label=1 - @TODO: Test this!
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

            prob = (numerator + self.alpha) / (denominator * 1.0 + self.alpha + self.beta)
            all_probs.append(prob)

        max_prob = max(all_probs)
        min_prob = min(all_probs)
        return np.log(max_prob / min_prob)

    def simple_bayesian_estimate_rate_parity(self, preds, labels, masks, use_tpr=True):

        all_eps = []
        number_of_simulation = range(1000)
        for _ in number_of_simulation:
            all_probs = []
            for mask in masks:
                new_preds = preds[mask]
                new_labels = labels[mask]  # this gives us all s

                # find number of instances with label = 1 and pred = 1
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

                prob = np.random.beta(numerator + (1 / 3.0), denominator + (1 / 3.0) - numerator, size=1)
                all_probs.append(prob)

            max_prob = max(all_probs)
            min_prob = min(all_probs)
            eps = np.log(max_prob / min_prob)
            all_eps.append(eps)
        return np.mean(all_eps)


@dataclass
class EPSFairnessMetric:
    gerrymandering_smoothed: float
    independent_smoothed: float
    intersectional_smoothed: float
    gerrymandering_simple_bayesian: float
    independent_simple_bayesian: float
    intersectional_simple_bayesian: float


# class EPSFairnessMetricMeta:
#     demographic_parity: EPSFairnessMetric
#     equal_opportunity: EPSFairnessMetric
#     equal_odds: EPSFairnessMetric

class EpsFairness(fairness_utils.FairnessTemplateClass):

    def __init__(self, prediction: np.asarray, label: np.asarray, aux: np.asarray,
                 all_possible_groups: List, all_possible_groups_mask: List, other_meta_data: Optional[dict] = None):
        super().__init__(prediction, label, aux,
                         all_possible_groups, all_possible_groups_mask, other_meta_data)

        self.fairness_mode = other_meta_data['fairness_mode']  # It is a list

        self.es = EstimateProbability()

    def demographic_parity(self, group_masks, robust_estimation_method='smoothed_empirical_estimate'):
        """
        Following steps will be done
            -> Calculate p(y_hat=1/S=s) for all s
            -> find the ratio with the maximum value and then return that eps!
        :return:
        """

        if robust_estimation_method == 'smoothed_empirical_estimate':
            eps = self.es.smoothed_empirical_estimate_demographic_parity(self.prediction, self.label, group_masks)
        elif robust_estimation_method == 'simple_bayesian_estimate':
            eps = self.es.simple_bayesian_estimate_demographic_parity(self.prediction, self.label, group_masks)
        else:
            raise NotImplementedError
        return eps

    def tpr_parity(self, group_masks, robust_estimation_method='smoothed_empirical_estimate'):
        """
        - This measure is also called equal opportunity with y=1 being the default!
        - We need to estimate p(y_hat=1/y=1,S=s) for all s
        :param group_masks:
        :param robust_estimation_method:
        :return:
        """

        if robust_estimation_method == 'smoothed_empirical_estimate':
            eps = self.es.smoothed_empirical_estimate_rate_parity(self.prediction, self.label, group_masks,
                                                                  use_tpr=True)
        elif robust_estimation_method == 'simple_bayesian_estimate':
            eps = self.es.simple_bayesian_estimate_rate_parity(self.prediction, self.label, group_masks, use_tpr=True)
        else:
            raise NotImplementedError
        return eps

    def fpr_parity(self, group_masks, robust_estimation_method='smoothed_empirical_estimate'):
        """
        - This measure is also called equal opportunity with y=1 being the default!
        - We need to estimate p(y_hat=1/y=1,S=s) for all s
        :param group_masks:
        :param robust_estimation_method:
        :return:
        """

        if robust_estimation_method == 'smoothed_empirical_estimate':
            eps = self.es.smoothed_empirical_estimate_rate_parity(self.prediction, self.label, group_masks,
                                                                  use_tpr=False)
        elif robust_estimation_method == 'simple_bayesian_estimate':
            eps = self.es.simple_bayesian_estimate_rate_parity(self.prediction, self.label, group_masks, use_tpr=False)
        else:
            raise NotImplementedError
        return eps

    def equal_odds(self, group_masks, robust_estimation_method='smoothed_empirical_estimate'):
        if robust_estimation_method == 'smoothed_empirical_estimate':
            eps_fpr = self.es.smoothed_empirical_estimate_rate_parity(self.prediction, self.label, group_masks,
                                                                      use_tpr=False)
            eps_tpr = self.es.smoothed_empirical_estimate_rate_parity(self.prediction, self.label, group_masks,
                                                                      use_tpr=True)
        elif robust_estimation_method == 'simple_bayesian_estimate':
            eps_fpr = self.es.simple_bayesian_estimate_rate_parity(self.prediction, self.label, group_masks,
                                                                   use_tpr=False)
            eps_tpr = self.es.simple_bayesian_estimate_rate_parity(self.prediction, self.label, group_masks,
                                                                   use_tpr=True)
        else:
            raise NotImplementedError
        return max(eps_fpr, eps_tpr)

    def run(self):
        """
        For now this works on binary setting. Not sure how to extend this for other settings!
        :return:
        """

        def get_analytics(type_of_group, robust_estimation_method, fairness_mode='demographic_parity'):
            _, group_index = fairness_utils.get_groups(self.all_possible_groups, type_of_group)
            if fairness_mode == 'demographic_parity':
                return self.demographic_parity(np.asarray(self.all_possible_groups_mask)[group_index],
                                               robust_estimation_method)
            elif fairness_mode == 'equal_opportunity':
                return self.tpr_parity(np.asarray(self.all_possible_groups_mask)[group_index],
                                       robust_estimation_method)
            elif fairness_mode == 'equal_odds':
                return self.equal_odds(np.asarray(self.all_possible_groups_mask)[group_index],
                                       robust_estimation_method)
            else:
                raise NotImplementedError

        fairness_metrics = {}

        for fairness_mode in self.fairness_mode:
            """
            Just intersectional works as eps of intersectional will always be greater or equal to 
            gerrymandering group and independent group
            """
            # fairness_mode_gerrymandering_smoothed = get_analytics('gerrymandering',
            #                                                            'smoothed_empirical_estimate',
            #                                                            fairness_mode)
            fairness_mode_intersectional_smoothed = get_analytics('intersectional',
                                                                       'smoothed_empirical_estimate',
                                                                       fairness_mode)
            # fairness_mode_independent_smoothed = get_analytics('independent',
            #                                                         'smoothed_empirical_estimate',
            #                                                         fairness_mode)

            # fairness_mode_gerrymandering_simple_bayesian = get_analytics('gerrymandering',
            #                                                                   'simple_bayesian_estimate',
            #                                                                   fairness_mode)
            fairness_mode_intersectional_simple_bayesian = get_analytics('intersectional',
                                                                              'simple_bayesian_estimate',
                                                                              fairness_mode)
            # fairness_mode_independent_simple_bayesian = get_analytics('independent',
            #                                                                'simple_bayesian_estimate',
            #                                                                fairness_mode)

            fairness_metric_tracker = EPSFairnessMetric(
                gerrymandering_smoothed=0.0,
                independent_smoothed=0.0,
                intersectional_smoothed=fairness_mode_intersectional_smoothed,
                gerrymandering_simple_bayesian=0.0,
                independent_simple_bayesian=0.0,
                intersectional_simple_bayesian=fairness_mode_intersectional_simple_bayesian
            )

            fairness_metrics[fairness_mode] = fairness_metric_tracker

        return fairness_metrics
