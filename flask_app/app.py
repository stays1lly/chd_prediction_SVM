# flask_app/app.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request

import pandas as pd
from data_preprocessing import load_data, preprocess_data
from model_training import split_data, train_svm_model
from evaluation import evaluate_model

app = Flask(__name__)

# Load and preprocess data (done once at the start)
file_path = os.path.join(os.path.dirname(__file__), '../phpgNaXZe.arff')
data = load_data(file_path)
data = preprocess_data(data)

# Train the model
features = ['sbp', 'tobacco', 'ldl', 'adiposity', 'famhist', 'type', 'obesity', 'alcohol', 'age']
X_train, X_test, y_train, y_test = split_data(data, features, 'chd')
svm_model = train_svm_model(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect user input from the form
        input_data = [
            float(request.form['sbp']),
            float(request.form['tobacco']),
            float(request.form['ldl']),
            float(request.form['adiposity']),
            int(request.form['famhist']),  # 0 or 1
            float(request.form['type']),
            float(request.form['obesity']),
            float(request.form['alcohol']),
            float(request.form['age'])
        ]

        # Convert input to DataFrame for model prediction
        input_df = pd.DataFrame([input_data], columns=features)
        
        # Make prediction
        prediction = svm_model.predict(input_df)[0]

        # Convert the prediction to a human-readable format
        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

        return render_template('result.html', prediction=result)

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
