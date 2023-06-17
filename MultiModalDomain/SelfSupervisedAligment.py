import torch
from torch import nn


def lossSSA(pred, label):
    # Moltiplicazione componente per componente
    result = torch.mul(label, torch.log(pred))
    # sommo su tutte le componenti
    result = result.sum(dim=1)
    return result


def generatingSamples(input):
    """
    Genera gli pseudo campioni con modalità non allineate
    input: batch_size x num_modalities x 5 x 1024
    """
    shape = input.size()
    num_batch = shape[0]  # number of samples in batch
    num_modalities = shape[1]
    num_segment = shape[2]
    num_features = shape[3]
    # Genero campioni disallineati
    notAlignedSamples = None
    for k in range(num_batch):  # for each batch
        are_identical = True
        while are_identical:
            random_idx = torch.randint(num_batch, size=(num_modalities,))
            are_identical = torch.all(torch.eq(random_idx, random_idx[0]))
        sample = torch.zeros(size=[num_modalities, num_segment, num_features])
        for m in range(num_modalities):
            sample[m, :, :] = input[random_idx[m], m, :, :]
        sample = sample.view(1, num_modalities, num_segment, num_features)
        if notAlignedSamples is None:
            notAlignedSamples = sample
        else:
            notAlignedSamples = torch.cat((notAlignedSamples, sample), dim=0)
        pseudoLabelNotAligned = torch.zeros(num_batch, )
        alignedSamples = input
        pseudoLabelAligned = torch.zeros(num_batch, )
        samples = torch.cat((notAlignedSamples, alignedSamples), dim=0)  # 2*num_batch x num_modalities x 5 x 1024
        labels = torch.cat((pseudoLabelNotAligned, pseudoLabelAligned))  # 2*num_batch

    return samples, labels


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
        x = input.view(num_batch, -1)
        x = self.fc(x)
        output = self.softmax(x)
        return output
