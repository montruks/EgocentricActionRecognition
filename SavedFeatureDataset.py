import pickle
import torch.utils.data as data
import pandas as pd
import platform


class SavedFeatureDataset(data.Dataset):
    def __init__(self, dataType, train=False):
        if platform.node() == 'LAPTOP-DANIELE':
            path_label = 'C:/Users/39334/Desktop/Poli/EgocentricActionRecognition/train_val'
            path_feature = 'C:/Users/39334/Desktop/Poli/EgocentricActionRecognition/saved_features'
        elif platform.node() == 'PC_Montrucchio':
            path_label = 'C:/Users/matte/Desktop/Workspace/EgocentricActionRecognition/train_val'
            path_feature = 'C:/Users/matte/Desktop/Workspace/EgocentricActionRecognition/saved_features'
        if train:
            path_label = path_label + '/' + dataType + '_' + 'train.pkl'
            path_feature = path_feature + '/' + 'saved_feat_I3D_' + 'data' + '_train.pkl'
        else:
            path_feature = path_feature + '/' + 'saved_feat_I3D_' + dataType + '_test.pkl'
            path_label = path_label + '/' + dataType + '_' + 'test.pkl'
        f_label = open(path_label, 'rb')
        f_feature = open(path_feature, 'rb')
        df1 = pickle.load(f_label)
        df2 = pickle.load(f_feature)
        label = df1['verb_class']
        self.complete_dataframe = pd.DataFrame(df2['features'])
        self.complete_dataframe['label'] = label
        self.data = self.complete_dataframe['features_RGB']
        self.target = self.complete_dataframe['label']

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return self.data.loc[index], self.target[index]