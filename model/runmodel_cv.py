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

# Parameters
params = {'dim': (16,1250),
          'batch_size': 256,
          'n_classes': 4,
          'shuffle': False}

# Datasets

train_arr=[]
test_arr=[]

y_all1=np.load('/home/basu9/tuheeg2/data/y_all1.npy')
y_all2=np.load('/home/basu9/tuheeg2/data/y_all2.npy')
y_all3=np.load('/home/basu9/tuheeg2/data/y_all3.npy')
y_all4=np.load('/home/basu9/tuheeg2/data/y_all4.npy')
y_all5=np.load('/home/basu9/tuheeg2/data/y_all5.npy')
y_all6=np.load('/home/basu9/tuheeg2/data/y_all6.npy')
y_all7=np.load('/home/basu9/tuheeg2/data/y_all7.npy')
y_all=np.concatenate((y_all1,y_all2,y_all3,y_all4,y_all5,y_all6,y_all7), axis=0)

labels={}

cnt=0

print(len(y_all))
total_samples=1792766

X=np.zeros(total_samples)

kf = KFold(n_splits=10)

#partition=int(total_samples*0.9)

cnt0=0
cnt1=0
cnt2=0
cnt3=0

y_all_new=[]
for i in range(1,total_samples+1):
    k=0
    if list(y_all[cnt])==[0.0,0.0]:
        k=0
        cnt0=cnt0+1
    if list(y_all[cnt])==[0.0,1.0]:
        k=1
        cnt1=cnt1+1
    if list(y_all[cnt])==[1.0,0.0]:
        k=2
        cnt2=cnt2+1
    if list(y_all[cnt])==[1.0,1.0]:
        k=3
        cnt3=cnt3+1
    y_all_new.append(k)
    labels['sample'+str(i)]=k
    cnt=cnt+1

y_all_new=np.array(y_all_new)

#partition index to total_samples-1 index
#samples numbered from partition+1 to total_samples
#y_all_test.append(k)

print("Class Counts:",cnt0,cnt1,cnt2,cnt3)

fold=1

trains=np.load("split"+str(fold)+"_train.npy") 
tests=np.load("split"+str(fold)+"_test.npy")

for train in trains:
    train_arr.append('sample'+str(train+1))
for test in tests:
    test_arr.append('sample'+str(test+1))

partition = {'train': train_arr, 'validation': test_arr}
#labels = {'id-1': 0, 'id-2': 1, 'id-3': 2, 'id-4': 1}

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

# Design model
model = Sequential()
model.add(LSTM(128, return_sequences=True))
model.add(Dense(25, activation='relu'))
model.add(LSTM(128, return_sequences=True))
model.add(Dense(25, activation='relu'))
model.add(LSTM(128))
model.add(Dense(25, activation='relu'))
model.add(Dense(4, activation='softmax'))
opti = Adam(learning_rate=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opti, metrics=['accuracy'])

# Train model on dataset
model.fit_generator(generator=training_generator,validation_data=validation_generator,use_multiprocessing=True,workers=4,epochs=25,verbose=2)
#y_train=model.predict_generator(training_generator)
#print(len(y_train))
y_pred=model.predict_generator(validation_generator)
#for i in y_pred[-256:]:
#    print(i)
y_pred = np.argmax(y_pred, axis=1)
#print(len(y_pred))
#print(confusion_matrix(y_all_test,y_pred[:len(y_all_test)]))
y_all_test=y_all_new[tests]
unique, counts = np.unique(y_all_test[:len(y_pred)], return_counts=True)
print(unique,counts)
print(confusion_matrix(y_all_test[:len(y_pred)],y_pred))
#print(confusion_matrix(validation_generator.classes, y_pred))
#train_acc=model.history.history['accuracy']
#val_acc=model.history.history['val_accuracy']
#print(train_acc)
#print(val_acc)
