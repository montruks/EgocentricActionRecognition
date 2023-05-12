import torch
import torch.nn as nn
from models.TRNmodule import RelationModuleMultiScale
import torch.optim as optim
import numpy as np


class TemporalAveragePooling(nn.Module):
    def __init__(self, dim=1):
        super(TemporalAveragePooling, self).__init__()
        self.dim = dim

    def forward(self, x):
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, dim=0)
        return torch.mean(x, dim=1)


class FullLayerClassifierTAP(nn.Module):
    def __init__(self, num_features, num_classes):
        super(FullLayerClassifierTAP, self).__init__()
        self.TempAvgLayer = TemporalAveragePooling()
        self.fc1 = nn.Linear(num_features, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.TempAvgLayer(x)
        x = self.fc1(x)
        x = self.softmax(x)
        return x


class FullLayerClassifierTRN(nn.Module):
    def __init__(self, num_features, num_classes, num_frames):
        super(FullLayerClassifierTRN, self).__init__()
        self.RMS = RelationModuleMultiScale(num_features, num_features, num_frames)
        self.fc1 = nn.Linear((num_frames-1)*num_features, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, dim=0)
        x = self.RMS(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.softmax(x)
        return x


def trainLoopFLC(model, trainloader, criterion, optimizer, epochs=10):
    # Allenamento del modello
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):  # loop sui dati di training
            inputs, labels = data  # recupera i dati e le etichette
            optimizer.zero_grad()  # azzerare i gradienti
            outputs = model(inputs)  # eseguire l'inferenza
            loss = criterion(outputs, labels)  # calcolare la perdita
            loss.backward()  # calcolare i gradienti
            optimizer.step()  # aggiornare i pesi
            running_loss += loss.item()  # tenere traccia della perdita cumulativa

            # stampa le statistiche di allenamento
            if i % 30 == 29:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


def testLoopFLC(model, testloader, criterion):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()  # Imposta il modello in modalità di valutazione
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testloader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    return test_loss, correct / len(testloader.dataset)
