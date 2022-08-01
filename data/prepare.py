import numpy as np
from numpy import load
from numpy import save
from scipy import stats

numchannels=16
cnt=0

#range is based on number of loaded samples
for j in range(1,8):

    X_all=np.load("X_all"+str(j)+".npy")
    y_all=np.load("y_all"+str(j)+".npy")
    
    X_all_train=X_all[:int(0.8*len(X_all))]
    X_all_test=X_all[int(0.8*len(X_all)):]

    for q in range(numchannels):
        X_all_train[:,q]=stats.zscore(X_all_train[:,q], axis=None)
        X_all_test[:,q]=stats.zscore(X_all_test[:,q], axis=None)

    pos=0
    train_names=[]
    print("Normalizing done...")

    for i in X_all_train:
        cnt=cnt+1
        np.save("dataset2/sample"+str(cnt)+".npy",X_all_train[pos])
        pos=pos+1
        train_names.append("sample"+str(cnt))

    pos1=0
    test_names=[]

    for i in X_all_test:
        cnt=cnt+1
        np.save("dataset2/sample"+str(cnt)+".npy",X_all_test[pos1])
        pos1=pos1+1
        test_names.append("sample"+str(cnt))

    print(cnt,pos)
