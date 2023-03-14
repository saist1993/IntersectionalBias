import torch
import torch.nn as nn
from torch.autograd import Function


class GradReverse(Function):
    """
        Torch function used to invert the sign of gradients (to be used for argmax instead of argmin)
        Usage:
            x = GradReverse.apply(x) where x is a tensor with grads.

        Copied from here: https://github.com/geraltofrivia/mytorch/blob/0ce7b23ff5381803698f6ca25bad1783d21afd1f/src/mytorch/utils/goodies.py#L39
    """

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def initialize_parameters(m):
    if isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.05, 0.05)
    elif isinstance(m, nn.LSTM):
        for n, p in m.named_parameters():
            if 'weight_ih' in n:
                i, f, g, o = p.chunk(4)
                nn.init.xavier_uniform_(i)
                nn.init.xavier_uniform_(f)
                nn.init.xavier_uniform_(g)
                nn.init.xavier_uniform_(o)
            elif 'weight_hh' in n:
                i, f, g, o = p.chunk(4)
                nn.init.orthogonal_(i)
                nn.init.orthogonal_(f)
                nn.init.orthogonal_(g)
                nn.init.orthogonal_(o)
            elif 'bias' in n:
                i, f, g, o = p.chunk(4)
                nn.init.zeros_(i)
                nn.init.ones_(f)
                nn.init.zeros_(g)
                nn.init.zeros_(o)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        try:
            nn.init.zeros_(m.bias)
        except AttributeError:
            pass


class SimpleNonLinear(nn.Module):
    def __init__(self, params):
        super().__init__()

        input_dim = params['model_arch']['encoder']['input_dim']
        output_dim = params['model_arch']['encoder']['output_dim']
        # Generalizing this to n-adversarial
        adv_dims = params['model_arch']['adv']['output_dim']  # List with n adversarial output!

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        self.adversarials = []
        for adv_dim in adv_dims:
            self.adversarials.append(nn.Sequential(
                nn.Linear(128, 64),
                # nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(64, 32),
                # nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(32, adv_dim)
            ))

        self.adversarials = nn.ModuleList(self.adversarials)

        self.task_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, output_dim)
        )

    def forward(self, params):
        x = params['input']
        encoder_output = self.encoder(x)
        adv_output = [adv(GradReverse.apply(encoder_output)) for adv in self.adversarials]
        task_classifier_output = self.task_classifier(encoder_output)

        output = {
            'prediction': task_classifier_output,
            'adv_outputs': adv_output,
            'hidden': encoder_output,  # just for compatabilit
            'classifier_hiddens': None,
            'adv_hiddens': None
        }

        return output

    @property
    def layers(self):
        return torch.nn.ModuleList([self.encoder, self.adversarials, self.task_classifier])

    def reset_task_classifier(self):
        self.task_classifier.apply(initialize_parameters)
