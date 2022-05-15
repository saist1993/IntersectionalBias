from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from metrics import fairness_utils


class EstimateProbability:

    def __init__(self):
        self.alpha = 0.01
        self.beta = 0.01
        self.number_of_bootstrap_dataset = 1000

    @staticmethod
    def get_confidence_interval(eps, percentile=95.0):
        try:
            return [np.percentile(eps, [100.0 - percentile])[0], np.percentile(eps, [percentile])[0] ]
        except:
            print("here")

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

    def calculate_smoothed_empirical_estimate_demographic_parity(self, preds, labels, mask):
        new_preds = preds[mask]
        new_labels = labels[mask]  # this gives us all s

        # find number of instances with label = 1
        number_of_instance_with_s_and_y_1 = np.count_nonzero(new_preds == 1)

        prob = (number_of_instance_with_s_and_y_1 + self.alpha) / (len(new_preds) * 1.0 + self.alpha + self.beta)

        return prob

    def smoothed_empirical_estimate_demographic_parity(self, preds, labels, masks):
        '''
            mask gives us access to specific attribute - s.
            empirical_estimate = number_of_instance_with_s_and_preds_1 / number_of_instance_with_s

            labels are not useful in this case, as we don't rely on the underlying data stastics!
        '''
        all_probs = []
        for mask in masks:
            prob = self.calculate_smoothed_empirical_estimate_demographic_parity(preds, labels, mask)
            all_probs.append(prob)

        max_prob = max(all_probs)
        min_prob = min(all_probs)
        return [np.log(max_prob / min_prob), [0.0, 0.0]]

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

        return [np.mean(all_eps), self.get_confidence_interval(all_eps)]

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
        return [np.log(max_prob / min_prob), [0.0, 0.0]]

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
        return [np.mean(all_eps), self.get_confidence_interval(all_eps)]

    def bootstrap_estimate_demographic_parity(self, preds, labels, masks):
        """ Similar to bagging. Performed via smoothing"""
        all_eps = []
        for _ in range(self.number_of_bootstrap_dataset):
            index_select = np.random.choice(len(preds), len(preds), replace=True)

            eps, _ = self.smoothed_empirical_estimate_demographic_parity(preds[index_select], labels[index_select],
                                                                      [m[index_select] for m in masks])
            all_eps.append(eps)
        # np.percentile(all_eps, [97.5])
        return [np.mean(all_eps), self.get_confidence_interval(all_eps)]

    def bootstrap_estimate_rate_parity(self, preds, labels, masks, use_tpr=True):
        """ Similar to bagging. Performed via smoothing"""
        all_eps = []
        for _ in range(self.number_of_bootstrap_dataset):
            index_select = np.random.choice(len(preds), len(preds), replace=True)

            eps, _ = self.smoothed_empirical_estimate_rate_parity(preds[index_select], labels[index_select],
                                                               [m[index_select] for m in masks], use_tpr=use_tpr)
            all_eps.append(eps)

        return [np.mean(all_eps), self.get_confidence_interval(all_eps)]


@dataclass
class EPSFairnessMetric:
    intersectional_smoothed: List
    intersectional_simple_bayesian: List
    intersectional_bootstrap: List
    intersectional_smoothed_bias_amplification: float = 0.0
    intersectional_simple_bayesian_bias_amplification: float = 0.0
    intersectional_bootstrap_bias_amplification:float = 0.0


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
        elif robust_estimation_method == 'bootstrap_empirical_estimate':
            eps = self.es.bootstrap_estimate_demographic_parity(self.prediction, self.label, group_masks)
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

        elif robust_estimation_method == 'bootstrap_empirical_estimate':
            eps = self.es.bootstrap_estimate_rate_parity(self.prediction, self.label, group_masks, use_tpr=True)

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

        elif robust_estimation_method == 'bootstrap_empirical_estimate':
            eps = self.es.bootstrap_estimate_rate_parity(self.prediction, self.label, group_masks, use_tpr=False)

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
        elif robust_estimation_method == 'bootstrap_empirical_estimate':
            eps_fpr = self.es.bootstrap_estimate_rate_parity(self.prediction, self.label, group_masks, use_tpr=False)
            eps_tpr = self.es.bootstrap_estimate_rate_parity(self.prediction, self.label, group_masks, use_tpr=True)

        else:
            raise NotImplementedError

        index = np.argmax([eps_fpr[0], eps_tpr[0]])
        return [eps_fpr, eps_tpr][index]

    def run(self):
        """
        For now this works on binary setting. Not sure how to extend this for other settings!
        :return:
        """

        def get_analytics(type_of_group, robust_estimation_method, fairness_mode='demographic_parity',
                          bias_amplification_mode=False):

            if bias_amplification_mode:
                self.original_prediction = deepcopy(self.prediction)
                self.prediction = self.label

            _, group_index = fairness_utils.get_groups(self.all_possible_groups, type_of_group)
            if fairness_mode == 'demographic_parity':
                output = self.demographic_parity(np.asarray(self.all_possible_groups_mask)[group_index],
                                                 robust_estimation_method)
            elif fairness_mode == 'equal_opportunity':
                output = self.tpr_parity(np.asarray(self.all_possible_groups_mask)[group_index],
                                         robust_estimation_method)
            elif fairness_mode == 'equal_odds':
                output = self.equal_odds(np.asarray(self.all_possible_groups_mask)[group_index],
                                         robust_estimation_method)
            else:
                raise NotImplementedError

            if bias_amplification_mode:
                self.prediction = self.original_prediction

            return output

        fairness_metrics = {}

        for fairness_mode in self.fairness_mode:
            """
            Just intersectional works as eps of intersectional will always be greater or equal to 
            gerrymandering group and independent group
            """

            fairness_mode_intersectional_smoothed = get_analytics('intersectional',
                                                                  'smoothed_empirical_estimate',
                                                                  fairness_mode)
            fairness_mode_intersectional_simple_bayesian = get_analytics('intersectional',
                                                                         'simple_bayesian_estimate',
                                                                         fairness_mode)
            fairness_mode_intersectional_bootstrap = get_analytics('intersectional',
                                                                   'bootstrap_empirical_estimate',
                                                                   fairness_mode)

            if fairness_mode == 'demographic_parity':
                """We also calculate the bias amplification ratio"""
                fairness_mode_intersectional_smoothed_bias_amplification = get_analytics('intersectional',
                                                                                         'smoothed_empirical_estimate',
                                                                                         fairness_mode,
                                                                                         True)
                fairness_mode_intersectional_simple_bayesian_bias_amplification = get_analytics('intersectional',
                                                                                                'simple_bayesian_estimate',
                                                                                                fairness_mode,
                                                                                                True)

                fairness_mode_intersectional_bootstrap_bias_amplification = get_analytics('intersectional',
                                                                       'bootstrap_empirical_estimate',
                                                                       fairness_mode,
                                                                       True)

                intersectional_smoothed_bias_amplification = fairness_mode_intersectional_smoothed[0] -\
                                                             fairness_mode_intersectional_smoothed_bias_amplification[0]

                intersectional_simple_bayesian_bias_amplification = fairness_mode_intersectional_simple_bayesian[0] - \
                                                                    fairness_mode_intersectional_simple_bayesian_bias_amplification[0]

                intersectional_bootstrap_bias_amplification = fairness_mode_intersectional_bootstrap[0] - \
                                                              fairness_mode_intersectional_bootstrap_bias_amplification[0]

            else:
                intersectional_smoothed_bias_amplification = 0
                intersectional_simple_bayesian_bias_amplification = 0
                intersectional_bootstrap_bias_amplification = 0

            fairness_metric_tracker = EPSFairnessMetric(
                intersectional_smoothed=fairness_mode_intersectional_smoothed,
                intersectional_simple_bayesian=fairness_mode_intersectional_simple_bayesian,
                intersectional_bootstrap=fairness_mode_intersectional_bootstrap,
                intersectional_smoothed_bias_amplification=intersectional_smoothed_bias_amplification,
                intersectional_simple_bayesian_bias_amplification=intersectional_simple_bayesian_bias_amplification,
                intersectional_bootstrap_bias_amplification=intersectional_bootstrap_bias_amplification,
            )

            fairness_metrics[fairness_mode] = fairness_metric_tracker

        return fairness_metrics
