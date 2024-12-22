# eda.py

import seaborn as sns
import matplotlib.pyplot as plt

def pairplot(data):
    sns.pairplot(data, diag_kind='kde', hue='chd', palette='coolwarm')
    plt.show()

def correlation_heatmap(data):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.show()

def histograms_and_boxplots(data):
    data.plot(kind='hist', figsize=(10, 5), bins=20, alpha=0.7)
    plt.title('Histogram of Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(visible=True, linestyle='--', alpha=0.5)
    plt.show()
    
    data.plot(kind='box', figsize=(10, 6),
              color=dict(boxes='DarkGreen', whiskers='DarkOrange',
                         medians='DarkBlue', caps='Gray'))
    plt.xticks(rotation=45)
    plt.grid(visible=True, linestyle='--', alpha=0.5)
    plt.show()
