
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

        bandwidth_range = [10, 15, 20, 50]
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
#         self.layer_1 = nn.Linear(input_dim, 125)
#         self.layer_2 = nn.Linear(125, 50)
#         self.layer_3 = nn.Linear(50, input_dim)
#         self.leaky_relu = nn.LeakyReLU()
#
#         if number_of_params == 3:
#             self.lambda_params = torch.nn.Parameter(torch.FloatTensor([0.33, 0.33, 0.33]))
#         elif number_of_params == 4:
#             self.lambda_params = torch.nn.Parameter(torch.FloatTensor([0.25, 0.25, 0.25, 0.25]))
#
#     def forward(self, other_examples):
#         final_output = torch.tensor(0.0, requires_grad=True)
#         for group, params in zip(other_examples, self.lambda_params):
#             x = group['input']
#             x = self.layer_1(x)
#             x = self.leaky_relu(x)
#             x = self.layer_2(x)
#             x = self.leaky_relu(x)
#             x = self.layer_3(x)
#             final_output = final_output + x*params
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

    def forward(self, other_examples):
        final_output = torch.tensor(0.0, requires_grad=True)
        for param, group in zip(self.lambda_params, other_examples):
            x = group['input']
            final_output = final_output + x*param

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

