import mne
from pathlib import Path
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
stop_words = set(stopwords.words('english'))
ext_phrases = ["medications","sedation","mg","others","other","unknown","none"]
PATH1 = "/home/basu9/tuheeg2/tusz/edf/train"
patient_ids=[]
patient_meds={}
for file1 in sorted(Path(PATH1).glob('**/*.txt')):
    file2=str(file1).split("/")
    file3=file2[-1].split(".txt")[0]
    print(file3)
    readfile1 = open(str(file1), 'r', encoding = "ISO-8859-1")
    flag=0
    medicines=[]
    for line in readfile1:
        if line.startswith("MEDICATIONS"):
            if line.strip()!="MEDICATIONS:":
                medicines.append(line.strip())
                flag=1
    if len(medicines)!=0 and flag==1:
        patient_meds[file3]=medicines
        patient_ids.append(file3)
print(patient_ids)
tokens_all=[]
for k in patient_meds:
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(patient_meds[k][0])
    tokens1 = [w for w in tokens if w.isalpha()]
    tokens2 = [w for w in tokens1 if not w.lower() in stop_words]
    tokens3 = [w.lower() for w in tokens2 if not w.lower() in ext_phrases]
    print(k,tokens3)
    patient_meds[k]=tokens3 ###dictionary with original patient medications
    for i in tokens3:
        tokens_all.append(i)
fdist = FreqDist(word.lower() for word in tokens_all)
print(fdist.most_common(30))

PATH2 = "/home/basu9/tuheeg2/tusz/edf_resampled/train"
X_all=[]
y_all=[]
count=0
cnt=0
all_patients=[]
for file2 in sorted(Path(PATH2).glob('**/*.edf')):
    file3=str(file2).split("/")
    file4=file3[-1].split(".edf")[0]
    file5=file4.split("_t")[0]
    if file5 in patient_meds.keys():
        print(file4)
        all_patients.append(file5)
        cnt=cnt+1
print(len(patient_meds.keys()))

np.save("patients.npy",all_patients)
np.save("medications.npy",patient_meds)

