import pickle as cPickle

import numpy as np

from sklearn.model_selection import train_test_split

np.random.seed(42)


def unpickle(file):
    fo = open(file, 'rb')
    data = cPickle.load(fo, encoding='latin1')
    fo.close()
    return data


for b in range(1, 6):
    data_batch = unpickle("data/cifar-10-batches-py/data_batch_" + str(b))
    if b == 1:
        X_train = data_batch["data"]
        y_train = np.array(data_batch["labels"])
    else:
        X_train = np.append(X_train, data_batch["data"], axis=0)
        y_train = np.append(y_train, data_batch["labels"], axis=0)
        
data_batch = unpickle("data/cifar-10-batches-py/test_batch")
X_test = data_batch["data"]
y_test = np.array(data_batch["labels"])

classes = unpickle("data/cifar-10-batches-py/batches.meta")["label_names"]

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

subsample_rate = 0.02

X_train, _, y_train, _ = train_test_split(X_train, y_train, stratify=y_train, train_size=subsample_rate, random_state=42)
X_test, _, y_test, _ = train_test_split(X_test, y_test, stratify=y_test, train_size=subsample_rate, random_state=42)

print('Data Curation Complete')


def getTrainData():
    global X_train, y_train
    return X_train, y_train


def getTestData():
    global X_test, y_test
    return X_test, y_test


def getClasses():
    global classes
    return classes
