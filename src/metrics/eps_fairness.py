import numpy as np
from typing import List, Optional
from metrics import fairness_utils
from dataclasses import dataclass

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
        new_labels = preds[mask] # this gives us all s

        # find number of instances with label = 1
        number_of_instance_with_s_and_y_1 = np.count_nonzero(new_preds==1)

        return number_of_instance_with_s_and_y_1/len(new_preds)*1.0

    def smoothed_empirical_estimate_demographic_parity(self, preds, labels, masks):
        '''
            mask gives us access to specific attribute - s.
            empirical_estimate = number_of_instance_with_s_and_preds_1 / number_of_instance_with_s

            labels are not useful in this case, as we don't rely on the underlying data stastics!
        '''
        all_probs = []
        for mask in masks:
            new_preds = preds[mask]
            new_labels = preds[mask]  # this gives us all s

            # find number of instances with label = 1
            number_of_instance_with_s_and_y_1 = np.count_nonzero(new_preds == 1)

            prob = (number_of_instance_with_s_and_y_1 + self.alpha) / (len(new_preds) * 1.0 + self.alpha + self.beta)
            all_probs.append(prob)

        max_prob = max(all_probs)
        min_prob = min(all_probs)
        return np.log(max_prob/min_prob)

    def simple_bayesian_estimate_demographic_parity(self, preds, labels, masks):

        all_eps = []
        number_of_simulation = range(1000)
        for _ in number_of_simulation:
            all_probs = []
            for mask in masks:
                new_preds = preds[mask]
                new_labels = preds[mask]  # this gives us all s

                # find number of instances with preds = 1
                n_1_s = np.count_nonzero(new_preds == 1)
                n_s = len(new_preds)
                prob = np.random.beta(n_1_s + (1/3.0), n_s + (1/3.0)- n_1_s, size=1)
                all_probs.append(prob)

            max_prob = max(all_probs)
            min_prob = min(all_probs)
            eps = np.log(max_prob / min_prob)
            all_eps.append(eps)
        return np.mean(all_eps)


    def smoothed_empirical_estimate_tpr_parity(self, preds, labels, masks):
        '''
            mask gives us access to specific attribute - s.
            empirical_estimate = number_of_instance_with_s_and_preds_1_and_label_1 + alpha /
             number_of_instance_with_s_and_lable_1 + alpha + beta

            labels are not useful in this case, as we don't rely on the underlying data stastics!
        '''
        all_probs = []
        for mask in masks:
            new_preds = preds[mask]
            new_labels = preds[mask]  # this gives us all s

            # find the number of instances with pred=1, and S=s and label=1 - @TODO: Test this!
            new_mask = (new_preds == 1) & (new_labels == 1)
            number_of_instance_with_s_and_y_1_and_pred_1 = np.count_nonzero(new_mask == 1)

            # find the number of instance with S=s and label=1
            new_mask = new_labels == 1
            number_of_instance_with_s_and_y_1 = np.count_nonzero(new_mask)

            prob = (number_of_instance_with_s_and_y_1_and_pred_1 + self.alpha) / (len(number_of_instance_with_s_and_y_1)
                                                                                  * 1.0 + self.alpha + self.beta)
            all_probs.append(prob)

        max_prob = max(all_probs)
        min_prob = min(all_probs)
        return np.log(max_prob/min_prob)

    def simple_bayesian_estimate_tpr_parity(self, preds, labels, masks):

        all_eps = []
        number_of_simulation = range(1000)
        for _ in number_of_simulation:
            all_probs = []
            for mask in masks:
                new_preds = preds[mask]
                new_labels = preds[mask]  # this gives us all s

                # find number of instances with label = 1 and pred = 1
                new_mask = (new_preds == 1) & (new_labels == 1)
                n_1_s_y_1 = np.count_nonzero(new_mask == 1)

                new_mask = new_labels == 1
                n_s_y_1 = np.count_nonzero(new_mask)




                prob = np.random.beta(n_1_s_y_1 + (1/3.0), n_s_y_1 + (1/3.0)- n_1_s_y_1, size=1)
                all_probs.append(prob)

            max_prob = max(all_probs)
            min_prob = min(all_probs)
            eps = np.log(max_prob / min_prob)
            all_eps.append(eps)
        return np.mean(all_eps)


@dataclass
class EPSFairnessMetric:
    gerrymandering_dp_smoothed: float
    independent_dp_smoothed: float
    intersectional_dp_smoothed: float
    gerrymandering_dp_simple_bayesian: float
    independent_dp_simple_bayesian: float
    intersectional_dp_simple_bayesian: float


class EpsFairness(fairness_utils.FairnessTemplateClass):

    def __init__(self, prediction: np.asarray, label: np.asarray, aux: np.asarray,
                 all_possible_groups: List, all_possible_groups_mask: List, other_meta_data: Optional[dict] = None):
        super().__init__(prediction, label, aux,
                         all_possible_groups, all_possible_groups_mask, other_meta_data)

        self.fairness_mode = other_meta_data['fairness_mode']   # It is a list

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
        return eps

    def tpr_parity(self, group_masks, robust_estimation_method='smoothed_empirical_estimate'):
        """
        - This measure is also called equal opportunity with y=1 being the default!
        - We need to estimate p(y_hat=1/y=1,S=s) for all s
        :param group_masks:
        :param robust_estimation_method:
        :return:
        """



    def run(self):
        """
        For now this works on binary setting. Not sure how to extend this for other settings!
        :return:
        """

        def get_analytics_demographic_parity(type_of_group, robust_estimation_method):
            _, group_index = fairness_utils.get_groups(self.all_possible_groups, type_of_group)
            return self.demographic_parity(np.asarray(self.all_possible_groups_mask)[group_index],
                                           robust_estimation_method)


        fairness_metrics = {}

        for fairness_mode in self.fairness_mode:
            if fairness_mode == 'demographic_parity':

                demographic_parity_gerrymandering_smoothed = get_analytics_demographic_parity('gerrymandering',
                                                                                     'smoothed_empirical_estimate')
                demographic_parity_intersectional_smoothed = get_analytics_demographic_parity('intersectional',
                                                                                     'smoothed_empirical_estimate')
                demographic_parity_independent_smoothed = get_analytics_demographic_parity('independent',
                                                                                  'smoothed_empirical_estimate')

                demographic_parity_gerrymandering_simple_bayesian = get_analytics_demographic_parity('gerrymandering',
                                                                                     'simple_bayesian_estimate')
                demographic_parity_intersectional_simple_bayesian = get_analytics_demographic_parity('intersectional',
                                                                                     'simple_bayesian_estimate')
                demographic_parity_independent_simple_bayesian = get_analytics_demographic_parity('independent',
                                                                                  'simple_bayesian_estimate')

                fairness_metric_tracker = EPSFairnessMetric(
                                                            gerrymandering_dp_smoothed=demographic_parity_gerrymandering_smoothed,
                                                            independent_dp_smoothed=demographic_parity_independent_smoothed,
                                                            intersectional_dp_smoothed=demographic_parity_intersectional_smoothed,
                                                            gerrymandering_dp_simple_bayesian=demographic_parity_gerrymandering_simple_bayesian,
                                                            independent_dp_simple_bayesian=demographic_parity_independent_simple_bayesian,
                                                            intersectional_dp_simple_bayesian=demographic_parity_intersectional_simple_bayesian
                                                            )

                fairness_metrics['demographic_parity'] = fairness_metric_tracker

            else:
                raise NotImplementedError

        return fairness_metrics

