
import gradio as gr
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import seaborn as sns

# Sample model training function with confusion matrix
def train_model():
    # Generate synthetic data with more variability
    np.random.seed(42)  # For reproducibility
    data = {
        'sbp': np.random.normal(130, 20, 1000),
        'tobacco': np.random.normal(5, 2.5, 1000),
        'ldl': np.random.normal(3, 1, 1000),
        'adiposity': np.random.normal(25, 7, 1000),
        'famhist': np.random.choice([0, 1], 1000),
        'type': np.random.normal(24, 8, 1000),
        'obesity': np.random.normal(30, 10, 1000),
        'alcohol': np.random.normal(10, 7, 1000),
        'age': np.random.normal(50, 15, 1000),
        'chd': np.random.choice([0, 1], 1000)
    }
    df = pd.DataFrame(data)

    # Preprocess data
    X = df.drop(columns='chd')
    y = df['chd']
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Split data and train Random Forest model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Check accuracy (for validation, usually not displayed in the app)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return model, scaler, conf_matrix

# Train the model and get scaler and confusion matrix
model, scaler, conf_matrix = train_model()

# Function to make predictions and visualize risk
def predict_risk(sbp, tobacco, ldl, adiposity, famhist, type, obesity, alcohol, age):
    # Prepare input data
    input_data = pd.DataFrame({
        'sbp': [sbp],
        'tobacco': [tobacco],
        'ldl': [ldl],
        'adiposity': [adiposity],
        'famhist': [1 if famhist == "Yes" else 0],
        'type': [type],
        'obesity': [obesity],
        'alcohol': [alcohol],
        'age': [age]
    })

    # Scale input data
    input_data_scaled = scaler.transform(input_data)
    
    # Predict risk
    risk_probability = model.predict_proba(input_data_scaled)[0][1]  # Probability of heart disease
    risk_percentage = round(risk_probability * 100, 2)
    risk_text = f"Heart Disease Risk: {risk_percentage}%"

    # Generate gauge plot for visualization
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.barh(["Heart Disease Risk"], [risk_percentage], color="coral", height=0.3)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Risk Percentage")
    ax.set_title(f"Risk Level: {risk_percentage}%")
    
    # Save plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    
    # Convert BytesIO buffer to PIL Image for Gradio compatibility
    img = Image.open(buf)
    
    # Create confusion matrix heatmap
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
    ax_cm.set_xlabel("Predicted Label")
    ax_cm.set_ylabel("True Label")
    ax_cm.set_title("Confusion Matrix")
    
    # Save confusion matrix plot to a BytesIO object
    buf_cm = BytesIO()
    fig_cm.savefig(buf_cm, format="png")
    buf_cm.seek(0)
    plt.close(fig_cm)
    
    # Convert confusion matrix buffer to PIL Image for Gradio compatibility
    img_cm = Image.open(buf_cm)
    
    return risk_text, img, img_cm

# Define Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Heart Disease Risk Prediction")
    gr.Markdown("Enter your details to predict the risk of heart disease and view a risk visualization.")
    
    # Input fields
    sbp = gr.Number(label="Systolic Blood Pressure (mm Hg)", value=130, precision=1)
    tobacco = gr.Number(label="Tobacco (g/day)", value=3.0, precision=2)
    ldl = gr.Number(label="LDL Cholesterol (mmol/L)", value=3.0, precision=2)
    adiposity = gr.Number(label="Adiposity", value=25.0, precision=1)
    famhist = gr.Radio(["Yes", "No"], label="Family History of Heart Disease", value="Yes")
    type = gr.Number(label="Type A Behavior", value=24.0, precision=1)
    obesity = gr.Number(label="Obesity", value=30.0, precision=1)
    alcohol = gr.Number(label="Alcohol Consumption (ml/day)", value=10.0, precision=1)
    age = gr.Number(label="Age (years)", value=50, precision=1)
    
    # Prediction output and visualization
    output_text = gr.Textbox(label="Risk Prediction")
    output_image = gr.Image(label="Risk Visualization")
    output_conf_matrix = gr.Image(label="Confusion Matrix")

    # Button to trigger prediction
    submit_button = gr.Button("Predict Heart Disease Risk")
    submit_button.click(
        predict_risk,
        inputs=[sbp, tobacco, ldl, adiposity, famhist, type, obesity, alcohol, age],
        outputs=[output_text, output_image, output_conf_matrix]
    )

demo.launch(share=True)
