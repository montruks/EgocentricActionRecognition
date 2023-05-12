from models import FullLayerClassifierTAP
from models import FullLayerClassifierTRN
from models import trainLoopFLC
from models import testLoopFLC
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


train_dataset = SFD(dataType='D1', train=True)
trainloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
print('Lunghezza Dataset di training:', len(trainloader.dataset))
classifier = FullLayerClassifierTRN(num_features=1024, num_classes=8, num_frames=5)
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)
trainLoopFLC(model=classifier, trainloader=trainloader, criterion=loss, optimizer=optimizer, epochs=10)
print("Finito train")

test_dataset = SFD(dataType='D1', train=False)
testloader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)
print('Lunghezza Dataset di training:', len(testloader.dataset))
test_loss, correct = testLoopFLC(model=classifier, testloader=testloader, criterion=loss)

# train(dataloader=DataLoader(dataset=test_dataset, batch_size=64, shuffle=True), model=classifier, criterion=loss,
# optimizer=optimizer, num_epochs=1)
