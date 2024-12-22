# model_training.py

from sklearn.model_selection import train_test_split
from sklearn import svm

def split_data(data, features, target):
    X = data[features]
    y = data[target]
    return train_test_split(X, y, test_size=0.2, random_state=1234)

def train_svm_model(X_train, y_train):
    svm_clf = svm.SVC(kernel='linear', C=1)
    svm_clf.fit(X_train, y_train)
    return svm_clf
