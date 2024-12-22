# Heart Disease Prediction with Support Vector Machines 

This project is a machine learning application for predicting the likelihood of heart disease based on various health parameters. It includes a Flask web interface for easy interaction and user input. I have used Gradio for better user interface as well.

## Overview
This project predicts the likelihood of heart disease using machine learning. It preprocesses health data, trains an SVM (Support Vector Machine) classifier, and provides a user-friendly web interface using Flask and Gradio.

## Features
- Data preprocessing: Handling missing values, encoding categorical variables, and scaling numeric features.
- Exploratory Data Analysis (EDA): Visualizing relationships in data and feature correlations.
- SVM-based classification for heart disease prediction.
- Web interface to input data and display predictions.

## Technologies Used
- **Programming Language**: Python
- **Frameworks**: Flask, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Data Handling**: Pandas
- **Frontend**: HTML, CSS
- **Deployment**: Flask server, Gradio (can be hosted on platforms like Heroku or AWS)


---

## Setup and Installation

### Prerequisites
- Python 3.8 or later installed.
- `pip` package manager installed.

### Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/heart-disease-prediction.git
   cd heart-disease-prediction
2. **Install dependencies**:
      ```bash
      pip install -r requirements.txt
3. **Run the Flask app**:
     ```bash
     python app.py
4. **Access the web interface**: Open your browser and go to http://127.0.0.1:5000.
