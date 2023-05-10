import torch
from torch import nn
from models.TAPmodule import TemporalAveragePooling as TAP


class Classifier(nn.Module):
    def __init__(self, num_classes, num_features):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.temp_avg_pool = TAP()
        self.fc1 = nn.Linear(num_features, 128, dtype=torch.float64)
        self.fc2 = nn.Linear(128, 64, dtype=torch.float64)
        self.fc3 = nn.Linear(64, num_classes, dtype=torch.float64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.pred_prob = None

    def classifier(self, x):
        x = self.temp_avg_pool(x)
        x = torch.from_numpy(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        x = nn.functional.softmax(x, dim=1)
        self.pred_prob = x
        output, predicted = torch.max(x.data, 1)
        return predicted

    def forward(self, x):
        return self.classifier(x), {}
