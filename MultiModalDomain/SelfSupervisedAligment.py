import torch
from torch import nn


def lossSSA(pred, label):
    # Moltiplicazione componente per componente
    result = torch.mul(label, torch.log(pred))
    # sommo su tutte le componenti
    result = result.sum(dim=1)
    return result


class ClassifierSSA(nn.Module):
    def __init__(self, modalities, num_channels, num_features):
        super().__init__(ClassifierSSA)
        self.modalities = modalities
        self.num_modalities = len(modalities)
        self.num_channels = num_channels
        self.num_features = num_features
        self.fc = nn.Linear(self.num_modalities * num_channels * num_features, 2)
        self.softmax = nn.Softmax()

    def forward(self, input):  # input è il batch senza nessuna modifica
        shape = input.size()
        num_batch = shape[0]  # number of samples in batch
        # creating pseudo-samples and pseudo-labels


        tensor = None
        # concateno i tensori in un unico tensore
        for m in self.modalities:
            if tensor is None:
                tensor = input[m]  # batch_size x 5 x 1024
            else:
                tensor = torch.cat((tensor, input[m]), dim=2)

        x = tensor.view(num_batch, -1)  # batch_size x 5 x (num_modalities*1024)
        x = self.fc(x)
        output = self.softmax(x)
        return output

    def generatingPseudoSampling(self, input):
        shape = input.size()
        num_batch = shape[0]  # number of samples in batch
        for k in range(num_batch):  # for each batch
            random_idx = torch.randint(num_batch, size=(self.num_modalities,))

