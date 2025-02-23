from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Use RandomForest instead of TensorFlow

app = Flask(__name__)

# Load dataset
dataset = pd.read_csv('cancer.csv')
x = dataset.drop(columns=["diagnosis(1=m, 0=b)"])
y = dataset["diagnosis(1=m, 0=b)"]

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Define AI Model (Using Random Forest instead of TensorFlow)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)  # Train the model

# Flask Route for Home Page
@app.route('/')
def home():
    return render_template('index.html', features=x.columns)

# Route to Handle Predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = [float(request.form[feature]) for feature in x.columns]
        user_input = np.array(user_input).reshape(1, -1)
        prediction = model.predict(user_input)
        result = "Malignant (Cancer)" if prediction[0] == 1 else "Benign (Non-Cancer)"
        return render_template('index.html', features=x.columns, prediction=result)
    except Exception as e:
        return render_template('index.html', features=x.columns, error=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
