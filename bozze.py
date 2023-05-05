import pickle
import numpy as np
import copy


# f = open('C:/Users/39334/Desktop/Poli/EgocentricActionRecognition/saved_features/saved_feat_I3D_D1_test.pkl', 'rb')
# feature_pkl = pickle.load(f)
# temp_feature_pkl = copy.deepcopy(feature_pkl)  # empty dictionary
#


def temporalAveragePooling(path):
    f = open(path, 'rb')
    feature_pkl = pickle.load(f)
    temp_feature_pkl = copy.deepcopy(feature_pkl)
    for k in range(0, len(feature_pkl['features'])):
        feature = feature_pkl['features'][k]['features_RGB']
        # Temporal Pooling Average
        temp_feature_pkl['features'][k]['features_RGB'] = np.sum(feature, axis=0)
    return temp_feature_pkl
