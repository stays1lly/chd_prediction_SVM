# evaluation.py

from sklearn.metrics import confusion_matrix, accuracy_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    recall = cm[0][0] / (cm[0][0] + cm[0][1])
    precision = cm[0][0] / (cm[0][0] + cm[1][1])
    
    print("Confusion Matrix:\n", cm)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}, Precision: {precision:.4f}")
    
    return y_pred, cm, accuracy, recall, precision
