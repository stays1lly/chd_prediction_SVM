# data_preprocessing.py

import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def load_data(file_path):
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    return df

def preprocess_data(df):
    # Define column names
    columns = ['sbp', 'tobacco', 'ldl', 'adiposity', 'famhist', 'type', 'obesity', 'alcohol', 'age', 'chd']
    df.columns = columns
    
    # Handle categorical variables
    encoder = LabelEncoder()
    df['famhist'] = encoder.fit_transform(df['famhist'])
    df['chd'] = encoder.fit_transform(df['chd'])
    
    # Scale numeric features
    scaler = MinMaxScaler(feature_range=(0, 100))
    numeric_cols = ['sbp', 'tobacco', 'ldl', 'adiposity', 'obesity', 'alcohol', 'age']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df
