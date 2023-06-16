import torch
from torch import nn


def calculate_Lm(prob, d):
    # Calcolo del termine -d log(Dm(Fm(x)))
    term1 = -d * torch.log(prob)
    # Calcolo del termine (1 - d) log(1 - Dm(Fm(x)))
    term2 = -(1 - d) * torch.log(1 - prob)
    # Calcolo della formula Lm
    Lm = term1 + term2
    return Lm


def lossWM2A(pred_discriminators, label):
    loss = 0
    modalities = pred_discriminators.keys()
    for m in modalities: # sommo loss dalle varie modalità
        prob_m = pred_discriminators[m]
        label_m = label[m]
        Lm = calculate_Lm(prob_m, label_m)
        loss += Lm
    return loss


class WM2A(nn.Module):
    def __init__(self, num_channels, num_features):
        super().__init__(WM2A)
        self.num_channels = num_channels
        self.num_features = num_features
        self.fc = nn.Linear(num_channels * num_features, 2)
        self.softmax = nn.Softmax()

    def forward(self, input):
        # Ricavo dimensione del batch
        shape = input.size()
        num_batch = shape[0]

        # Eseguo classificazione
        x = input.view(num_batch, -1)  # batch_size x 5 x 1024
        x = self.fc(x)
        output = self.softmax(x)
        return output
