from models import ClassifierTempAverage as CTA
from SavedFeatureDataset import SavedFeatureDataset as SFD
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.functional as F
# optimizer = optim.Adam(model.parameters(), lr=0.001)


def train(dataloader, model, criterion, optimizer, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(dataloader)))

    print('Finished Training')


test_dataset = SFD(dataType='D1', train=False)
classifier = CTA.Classifier(num_classes=7, num_features=1024)
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

train(dataloader=DataLoader(dataset=test_dataset, batch_size=64, shuffle=True), model=classifier, criterion=loss, optimizer=optimizer,
      num_epochs=1)