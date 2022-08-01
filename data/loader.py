import mne
from pathlib import Path
import numpy as np
from sklearn.utils import shuffle

def generate_samples(t1,t2,file): #returns total samples (*256) given 2 times in seconds
    raw = mne.io.read_raw_edf(file, verbose='error')
    t1=round(t1,4)
    t2=round(t2,4)
    data,times=raw[:,round(t1*raw.info['sfreq'],4):round(t2*raw.info['sfreq'],4)]
    data=data[:16]
    data=np.array(data)
    return data

def generate_5_samples(t1,t2,file): #generate samples of 5 seconds specifically
    print(t1,t2,file)
    all_5_samples=[]
    raw = mne.io.read_raw_edf(file, verbose='error')
    times=np.arange(t1,t2-(5.0-1.0))
    for i in times:
        if round(i,4)+5.0<t2:
            all_5_samples.append(generate_samples(round(i,4),round(i,4)+5.0,file))
    all_5_samples=np.array(all_5_samples)
    print(all_5_samples.shape)
    return all_5_samples

PATH2 = "/home/basu9/tuheeg2/tusz/edf_resampled/train"
X_all=[]
y_all=[]
count=0
cnt=0

patients=np.load("patients.npy")
patient_meds=np.load("medications.npy", allow_pickle=True).item()

patient_list=sorted(list(patient_meds.keys()))
#adjust loading data based on computational resources
#patient_list=patient_list[:200]
#patient_list=patient_list[200:400]
#patient_list=patient_list[400:600]
#patient_list=patient_list[600:700]
#patient_list=patient_list[700:800]
#patient_list=patient_list[800:900]
patient_list=patient_list[900:]

for file2 in sorted(Path(PATH2).glob('**/*.edf')):
    file3=str(file2).split("/")
    file4=file3[-1].split(".edf")[0]
    file5=file4.split("_t")[0]
    if file5 in patient_list:
        cnt=cnt+1
        raw = mne.io.read_raw_edf(file2, verbose='error')
        common_channels=['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF']
        if raw.info['ch_names'][:16]==common_channels and raw.info['sfreq']==250.0:
            data,times=raw[:,:]
            label=np.array([0,0],dtype='float32')
            if 'keppra' in patient_meds[file5] and 'dilantin' in patient_meds[file5]:
                label=np.array([1,1],dtype='float32')
            elif 'keppra' in patient_meds[file5]:
                label=np.array([0,1],dtype='float32')
            elif 'dilantin' in patient_meds[file5]:
                label=np.array([1,0],dtype='float32')
            else:
                label=np.array([0,0],dtype='float32')
            start=0.0
            end=data.shape[1]/250.0
            print(file4)
            print(label)
            if end-start>=5.0:
                samples=generate_5_samples(start,end,file2)
                for s in samples:
                    X_all.append(s)
                    y_all.append(label)

print(cnt)
print(len(patient_meds.keys()))
X_all=np.array(X_all)
y_all=np.array(y_all)

X_all,y_all=shuffle(X_all,y_all,random_state=0)

print(X_all.shape)
print(y_all.shape)
#np.save("X_all1.npy",X_all)
#np.save("y_all1.npy",y_all)
#np.save("X_all2.npy",X_all)
#np.save("y_all2.npy",y_all)
#np.save("X_all3.npy",X_all)
#np.save("y_all3.npy",y_all)
#np.save("X_all4.npy",X_all)
#np.save("y_all4.npy",y_all)
#np.save("X_all5.npy",X_all)
#np.save("y_all5.npy",y_all)
#np.save("X_all6.npy",X_all)
#np.save("y_all6.npy",y_all)
np.save("X_all7.npy",X_all)
np.save("y_all7.npy",y_all)
