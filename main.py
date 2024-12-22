# main.py

from data_preprocessing import load_data, preprocess_data
from eda import pairplot, correlation_heatmap, histograms_and_boxplots
from model_training import split_data, train_svm_model
from evaluation import evaluate_model

# Load and preprocess data
file_path = "E:\\Projects\\heart\\phpgNaXZe.arff"
data = load_data(file_path)
data = preprocess_data(data)

# Perform EDA
pairplot(data)
correlation_heatmap(data)
histograms_and_boxplots(data)

# Train and evaluate the model
features = ['sbp', 'tobacco', 'ldl', 'adiposity', 'famhist', 'type', 'obesity', 'alcohol', 'age']
X_train, X_test, y_train, y_test = split_data(data, features, 'chd')
svm_model = train_svm_model(X_train, y_train)
evaluate_model(svm_model, X_test, y_test)
