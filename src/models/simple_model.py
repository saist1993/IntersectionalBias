import torch
import torch.nn as nn


class SimpleNonLinear(nn.Module):
    """Fairgrad uses this as complex non linear model"""
    def __init__(self, params):
        super().__init__()

        input_dim = params['model_arch']['encoder']['input_dim']
        output_dim = params['model_arch']['encoder']['output_dim']

        self.layer_1 = nn.Linear(input_dim, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.layer_3 = nn.Linear(64, 32)
        self.layer_out = nn.Linear(32, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.batchnorm3 = nn.BatchNorm1d(32)

    def forward(self, params):
        x = params['input']
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)  # This does not exists in Michael.

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        z = self.layer_3(x)
        z = self.batchnorm3(z)
        z = self.relu(z)
        z = self.dropout(z)

        z = self.layer_out(z)

        output = {
            'prediction': z,
            'adv_output': None,
            'hidden': x,  # just for compatabilit
            'classifier_hiddens': None,
            'adv_hiddens': None
        }

        return output

    @property
    def layers(self):
        return torch.nn.ModuleList([self.layer_1, self.layer_2, self.layer_3, self.layer_out])

# class SimpleNonLinear(nn.Module):
#     """Fairgrad uses this as complex non linear model"""
#     def __init__(self, params):
#         super().__init__()
#
#         input_dim = params['model_arch']['encoder']['input_dim']
#         output_dim = params['model_arch']['encoder']['output_dim']
#
#         self.layer_1 = nn.Linear(input_dim, 128)
#         self.layer_2 = nn.Linear(128, 64)
#         self.layer_2a = nn.Linear(64, 64)
#         self.layer_2b = nn.Linear(64, 64)
#         self.layer_3 = nn.Linear(64, 32)
#         self.layer_out = nn.Linear(32, output_dim)
#
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p=0.2)
#         self.batchnorm1 = nn.BatchNorm1d(128)
#         self.batchnorm2 = nn.BatchNorm1d(64)
#         self.batchnorm2a = nn.BatchNorm1d(64)
#         self.batchnorm2b = nn.BatchNorm1d(64)
#         self.batchnorm3 = nn.BatchNorm1d(32)
#
#     def forward(self, params):
#         x = params['input']
#         x = self.layer_1(x)
#         x = self.batchnorm1(x)
#         x = self.relu(x)
#         x = self.dropout(x)  # This does not exists in Michael.
#
#         x = self.layer_2(x)
#         x = self.batchnorm2(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#
#         x = self.layer_2a(x)
#         x = self.batchnorm2a(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#
#         x = self.layer_2b(x)
#         x = self.batchnorm2b(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#
#         z = self.layer_3(x)
#         z = self.batchnorm3(z)
#         z = self.relu(z)
#         z = self.dropout(z)
#
#         z = self.layer_out(z)
#
#         output = {
#             'prediction': z,
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
#         return torch.nn.ModuleList([self.layer_1, self.layer_2, self.layer_3, self.layer_out])

class SimpleLinear(nn.Module):
    def __init__(self, params):
        super().__init__()
        input_dim = params['model_arch']['encoder']['input_dim']
        output_dim = params['model_arch']['encoder']['output_dim']
        self.encoder = nn.Linear(input_dim, output_dim)
        # self.encoder.apply(initialize_parameters) # This is not used by Michael

    def forward(self, params):
        text = params['input']
        prediction = self.encoder(text)

        output = {
            'prediction': prediction,
            'adv_output': None,
            'hidden': prediction,  # just for compatability
            'classifier_hiddens': None,
            'adv_hiddens': None
            # 'second_adv_output': second_adv_output
        }

        return output

    @property
    def layers(self):
        return torch.nn.ModuleList([self.encoder])


class SimpleNonLinearMoreComplex(nn.Module):
    """Fairgrad uses this as complex non linear model"""

    def __init__(self, params):
        super().__init__()

        input_dim = params['model_arch']['encoder']['input_dim']
        output_dim = params['model_arch']['encoder']['output_dim']

        self.layer_1 = nn.Linear(input_dim, 256)
        self.layer_2 = nn.Linear(256, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_4 = nn.Linear(64, 32)
        self.layer_out = nn.Linear(32, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)
        self.batchnorm4 = nn.BatchNorm1d(32)

    def forward(self, params):
        x = params['input']
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_4(x)
        x = self.batchnorm4(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        output = {
            'prediction': x,
            'adv_output': None,
            'hidden': x,  # just for compatabilit
            'classifier_hiddens': None,
            'adv_hiddens': None
        }

        return output

    @property
    def layers(self):
        return torch.nn.ModuleList([self.layer_1, self.layer_2, self.layer_3, self.layer_4, self.layer_out])