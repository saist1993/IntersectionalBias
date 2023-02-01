
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Callable, Optional



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class TrainingLoopParameters:
    n_epochs: int
    model: torch.nn.Module
    iterators: Dict
    optimizer: torch.optim
    criterion: Callable
    device: torch.device
    use_wandb: bool
    other_params: Dict
    save_model_as: Optional[str]
    fairness_function: str
    unique_id_for_run: str



def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":

        bandwidth_range = [1, 10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY)

# class SimpleModelGenerator(nn.Module):
#     """Fairgrad uses this as complex non linear model"""
#
#     def __init__(self, input_dim, number_of_params=None):
#         super().__init__()
#
#         self.layer_1 = nn.Linear(input_dim, input_dim, bias=False)
#         # self.layer_2 = nn.Linear(125, 50)
#         # self.layer_3 = nn.Linear(50, input_dim)
#         # self.leaky_relu = nn.LeakyReLU()
#
#         # if number_of_params == 3:
#         #     self.lambda_params = torch.nn.Parameter(torch.FloatTensor([0.33, 0.33, 0.33]))
#         # elif number_of_params == 4:
#         #     self.lambda_params = torch.nn.Parameter(torch.FloatTensor([0.25, 0.25, 0.25, 0.25]))
#
#     def forward(self, other_examples):
#         final_output = torch.tensor(0.0, requires_grad=True)
#         for group in other_examples:
#             x = group['input']
#             x = self.layer_1(x)
#             # x = self.leaky_relu(x)
#             # x = self.layer_2(x)
#             # x = self.leaky_relu(x)
#             # x = self.layer_3(x)
#             final_output = final_output + x
#
#
#
#
#         output = {
#             'prediction': final_output,
#             'adv_output': None,
#             'hidden': x,  # just for compatabilit
#             'classifier_hiddens': None,
#             'adv_hiddens': None
#         }
#
#         return output
#
#     # @property
#     # def layers(self):
#     #     return torch.nn.ModuleList([self.layer_1, self.layer_2, self.layer_3])


class SimpleModelGenerator(nn.Module):
    """Fairgrad uses this as complex non linear model"""

    def __init__(self, input_dim, number_of_params=3):
        super().__init__()

        if number_of_params == 3:
            self.lambda_params = torch.nn.Parameter(torch.FloatTensor([0.33, 0.33, 0.33]))
        elif number_of_params == 4:
            self.lambda_params = torch.nn.Parameter(torch.FloatTensor([0.25, 0.25, 0.25, 0.25]))

        self.more_lambda_params = [torch.nn.Parameter(torch.FloatTensor(torch.ones(input_dim))) for i in
                                   range(len(self.lambda_params))]

    def forward(self, other_examples):
        final_output = torch.tensor(0.0, requires_grad=True)
        for param, group, more_params in zip(self.lambda_params, other_examples, self.more_lambda_params):
            x = group['input']
            final_output = final_output + (x*more_params)*param

        output = {
            'prediction': final_output,
            'adv_output': None,
            'hidden': x,  # just for compatabilit
            'classifier_hiddens': None,
            'adv_hiddens': None
        }

        return output

    @property
    def layers(self):
        return torch.nn.ModuleList([self.layer_1, self.layer_2])




# class SimpleModelGenerator(nn.Module):
#     """Fairgrad uses this as complex non linear model"""
#
#     def __init__(self, input_dim, number_of_params=3):
#         super().__init__()
#
#         if number_of_params == 3:
#             self.lambda_params = torch.nn.Parameter(torch.FloatTensor([0.33, 0.33, 0.33]))
#         elif number_of_params == 4:
#             self.lambda_params = torch.nn.Parameter(torch.FloatTensor([0.25, 0.25, 0.25, 0.25]))
#
#         self.more_lambda_params = torch.nn.Parameter(torch.FloatTensor(torch.ones(input_dim)))
#
#     def forward(self, other_examples):
#         final_output = torch.tensor(0.0, requires_grad=True)
#         for param, group in zip(self.lambda_params, other_examples):
#             x = group['input']
#             final_output = final_output + (x*self.more_lambda_params)*param
#
#         output = {
#             'prediction': final_output,
#             'adv_output': None,
#             'hidden': x,  # just for compatabilit
#             'classifier_hiddens': None,
#             'adv_hiddens': None
#         }
#
#         return output
#
#     @property
#     def layers(self):
#         return torch.nn.ModuleList([self.layer_1, self.layer_2])

class SimpleClassifier(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.layer_1 = nn.Linear(input_dim, 256)
        self.layer_2 = nn.Linear(256, output_dim)
        self.relu = nn.ReLU()

    def forward(self, params):
        input = params['input']
        x = self.layer_1(input)
        x = self.relu(x)
        output = self.layer_2(x)

        return output

    @property
    def layers(self):
        return torch.nn.ModuleList([self.layer_1, self.layer_2])




class TestGeneratedSamples:

    def __init__(self, iterators, other_meta_data):
        self.one_vs_all_clf = {}
        self.one_vs_all_clf = {}
        self.iterators = iterators
        self.all_attribute_reco_models = None
        self.other_meta_data = other_meta_data
        self.single_attribute_reco_models = []
        self.raw_data = other_meta_data['raw_data']
        self.s_flatten_lookup = other_meta_data['s_flatten_lookup']
        self.number_of_attributes = other_meta_data['raw_data']['train_s'].shape[1]

    @staticmethod
    def create_mask_with_x(data, condition):

        keep_indices = []

        for index, i in enumerate(condition):
            if i != 'x':
                keep_indices.append(i == data[:, index])
            else:
                keep_indices.append(np.ones_like(data[:, 0], dtype='bool'))

        mask = np.ones_like(data[:, 0], dtype='bool')

        # mask = [np.logical_and(mask, i) for i in keep_indices]

        for i in keep_indices:
            mask = np.logical_and(mask, i)
        return mask


    def run(self):
        self.run_one_vs_all_classifier_group()
        self.create_and_learn_single_attribute_models()

    def create_and_learn_single_attribute_models(self):
        number_of_attributes = other_meta_data['raw_data']['train_s'].shape[1]
        self.number_of_attributes = number_of_attributes
        for k in range(number_of_attributes):

            print(f"learning for attribute {k}")
            input_dim = other_meta_data['raw_data']['train_X'].shape[1]
            output_dim = 2
            single_attribute_classifier = SimpleClassifier(input_dim=input_dim, output_dim=output_dim)

            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(single_attribute_classifier.parameters(), lr=1e-3)

            for e in tqdm(range(10)):
                average_loss = []
                for items in (self.iterators[0]['train_iterator']):
                    optimizer.zero_grad()
                    prediction = single_attribute_classifier(items).squeeze()
                    attribute_label = items['aux'][:, k]
                    loss = loss_fn(prediction, attribute_label)
                    loss.backward()
                    optimizer.step()
                    average_loss.append(loss.data)

            with torch.no_grad():

                def common_sub_routine(iterator):
                    all_predictions = []
                    all_label = []
                    for items in iterator:
                        prediction = single_attribute_classifier(items)
                        all_predictions.append(prediction)
                        all_label.append(items['aux'][:, k])

                    all_predictions = np.vstack(all_predictions)
                    all_label = np.hstack(all_label)

                    return all_predictions, all_label

                all_predictions_train, all_label_train = common_sub_routine(iterators[0]['train_iterator'])
                all_predictions_test, all_label_test = common_sub_routine(iterators[0]['test_iterator'])

                print(f"{k} Balanced test accuracy: {balanced_accuracy_score(all_label_test, all_predictions_test.argmax(axis=1))}"
                      f" and unbalanced test accuracy: {accuracy_score(all_label_test, all_predictions_test.argmax(axis=1))}" )

            self.single_attribute_reco_models.append(single_attribute_classifier)

    def create_and_learn_all_attribute_model(self):
        input_dim = other_meta_data['raw_data']['train_X'].shape[1]
        output_dim = len(other_meta_data['s_flatten_lookup']) # this needs to change
        all_attribute_classifier = SimpleClassifier(input_dim=input_dim, output_dim=output_dim)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(all_attribute_classifier.parameters(), lr=1e-3)

        for e in tqdm(range(10)):
            average_loss = []
            for items in (self.iterators[0]['train_iterator']):
                optimizer.zero_grad()
                prediction = all_attribute_classifier(items).squeeze()
                attribute_label = items['aux_flattened']
                loss = loss_fn(prediction, attribute_label)
                loss.backward()
                optimizer.step()
                average_loss.append(loss.data)

        with torch.no_grad():

            def common_sub_routine(iterator):
                all_predictions = []
                all_label = []
                for items in iterator:
                    prediction = all_attribute_classifier(items)
                    all_predictions.append(prediction)
                    all_label.append(items['aux_flattened'])

                all_predictions = np.vstack(all_predictions)
                all_label = np.hstack(all_label)

                return all_predictions, all_label

            all_predictions_train, all_label_train = common_sub_routine(iterators[0]['train_iterator'])
            all_predictions_test, all_label_test = common_sub_routine(iterators[0]['test_iterator'])

            print(
                f"Balanced test accuracy: {balanced_accuracy_score(all_label_test, all_predictions_test.argmax(axis=1))}"
                f"and unbalanced test accuracy: {accuracy_score(all_label_test, all_predictions_test.argmax(axis=1))}")

        self.all_attribute_reco_models = all_attribute_classifier

    def one_vs_all_classifier_group(self, group):
        mask_group_train_X = self.create_mask_with_x(data=self.raw_data['train_s'], condition=group)
        size = mask_group_train_X.sum()
        index_group_train_X = np.random.choice(np.where(mask_group_train_X == True)[0], size=size, replace=False)
        index_not_group_train_X = np.random.choice(np.where(mask_group_train_X == False)[0], size=size, replace=False)

        train_X_group, train_y_group = self.raw_data['train_X'][index_group_train_X], np.ones(size)
        train_X_not_group, train_y_not_group = self.raw_data['train_X'][index_not_group_train_X], np.zeros(size)

        clf = MLPClassifier(solver="adam", learning_rate_init=0.01, hidden_layer_sizes=(25, 5), random_state=1)

        X = np.vstack([train_X_group, train_X_not_group])
        y = np.hstack([train_y_group, train_y_not_group])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        print(clf.score(X_train, y_train), accuracy_score(y_test, y_pred), balanced_accuracy_score(y_test, y_pred))

        return clf

    def run_one_vs_all_classifier_group(self):

        for group, group_id in self.s_flatten_lookup.items():
            print(f"current group: {group_id}")
            clf = self.one_vs_all_classifier_group(group=group)
            print("**************")
            self.one_vs_all_clf[group_id] = clf
            self.one_vs_all_clf[group] = clf

        pickle.dump(self.one_vs_all_clf, open("adult_one_vs_all_clf.sklearn", 'wb'))



    def prediction_over_generated_examples(self, generated_examples, gold_label):
        with torch.no_grad():
            final_accuracy = []
            for k in range(self.number_of_attributes):
                items['input'] = generated_examples
                output = self.single_attribute_reco_models[k](items)
                label = gold_label[:,k]
                final_accuracy.append((balanced_accuracy_score(label, output.argmax(axis=1)), accuracy_score(label, output.argmax(axis=1))))
            return final_accuracy






#!/usr/bin/env python
# encoding: utf-8


import torch

min_var_est = 1e-8


# Consider linear time MMD with a linear kernel:
# K(f(x), f(y)) = f(x)^Tf(y)
# h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)
#             = [f(x_i) - f(y_i)]^T[f(x_j) - f(y_j)]
#
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def linear_mmd2(f_of_X, f_of_Y):
    loss = 0.0
    delta = f_of_X - f_of_Y
    loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
    return loss


# Consider linear time MMD with a polynomial kernel:
# K(f(x), f(y)) = (alpha*f(x)^Tf(y) + c)^d
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def poly_mmd2(f_of_X, f_of_Y, d=2, alpha=1.0, c=2.0):
    K_XX = (alpha * (f_of_X[:-1] * f_of_X[1:]).sum(1) + c)
    K_XX_mean = torch.mean(K_XX.pow(d))

    K_YY = (alpha * (f_of_Y[:-1] * f_of_Y[1:]).sum(1) + c)
    K_YY_mean = torch.mean(K_YY.pow(d))

    K_XY = (alpha * (f_of_X[:-1] * f_of_Y[1:]).sum(1) + c)
    K_XY_mean = torch.mean(K_XY.pow(d))

    K_YX = (alpha * (f_of_Y[:-1] * f_of_X[1:]).sum(1) + c)
    K_YX_mean = torch.mean(K_YX.pow(d))

    return K_XX_mean + K_YY_mean - K_XY_mean - K_YX_mean


def _mix_rbf_kernel(X, Y, sigma_list):
    assert(X.size(0) == Y.size(0))
    m = X.size(0)

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K += torch.exp(-gamma * exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)


def mix_rbf_mmd2(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


def mix_rbf_mmd2_and_ratio(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


################################################################################
# Helper functions to compute variances based on kernel matrices
################################################################################


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)    # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e

    Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m))

    return mmd2


def _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    mmd2, var_est = _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=const_diagonal, biased=biased)
    loss = mmd2 / torch.sqrt(torch.clamp(var_est, min=min_var_est))
    return loss, mmd2, var_est


def _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)    # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
        sum_diag2_X = sum_diag2_Y = m * const_diagonal**2
    else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)
        sum_diag2_X = diag_X.dot(diag_X)
        sum_diag2_Y = diag_Y.dot(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e
    K_XY_sums_1 = K_XY.sum(dim=1)                     # K_{XY} * e

    Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

    Kt_XX_2_sum = (K_XX ** 2).sum() - sum_diag2_X      # \| \tilde{K}_XX \|_F^2
    Kt_YY_2_sum = (K_YY ** 2).sum() - sum_diag2_Y      # \| \tilde{K}_YY \|_F^2
    K_XY_2_sum  = (K_XY ** 2).sum()                    # \| K_{XY} \|_F^2

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m))

    var_est = (
        2.0 / (m**2 * (m - 1.0)**2) * (2 * Kt_XX_sums.dot(Kt_XX_sums) - Kt_XX_2_sum + 2 * Kt_YY_sums.dot(Kt_YY_sums) - Kt_YY_2_sum)
        - (4.0*m - 6.0) / (m**3 * (m - 1.0)**3) * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 4.0*(m - 2.0) / (m**3 * (m - 1.0)**2) * (K_XY_sums_1.dot(K_XY_sums_1) + K_XY_sums_0.dot(K_XY_sums_0))
        - 4.0*(m - 3.0) / (m**3 * (m - 1.0)**2) * (K_XY_2_sum) - (8 * m - 12) / (m**5 * (m - 1)) * K_XY_sum**2
        + 8.0 / (m**3 * (m - 1.0)) * (
            1.0 / m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
            - Kt_XX_sums.dot(K_XY_sums_1)
            - Kt_YY_sums.dot(K_XY_sums_0))
        )
    return mmd2, var_est

