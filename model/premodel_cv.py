import numpy as np
from numpy import load
import tensorflow
from tensorflow.keras import backend as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from sklearn.svm import OneClassSVM
from sklearn.utils import shuffle
from math import sqrt, log, exp
import statistics
from datetime import datetime, timedelta
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from scipy import stats
import sys
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from generator import DataGenerator
from sklearn.metrics import confusion_matrix

labels={}

total_samples=1792766

X=np.zeros(total_samples)

kf = KFold(n_splits=10)

foldcnt=1

for trains, tests in kf.split(X):
    print("Fold ", foldcnt)
    np.save("split"+str(foldcnt)+"_train.npy", trains)
    np.save("split"+str(foldcnt)+"_test.npy", tests)
    foldcnt=foldcnt+1
   
