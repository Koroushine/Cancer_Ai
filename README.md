# Cancer_Ai
Just an Ai
This Flask-based AI web application predicts whether a tumor is Malignant (Cancerous) or Benign (Non-Cancerous) based on user-input medical parameters. The application uses a Random Forest Classifier for classification.

Project Structure
app.py - The main Flask application that loads the dataset, trains the AI model and handles predictions.
cancer.csv - The dataset containing medical parameters and labels for training the model.
index.html - The frontend interface allows users to enter input values and receive predictions.

How It Works
The dataset is loaded and preprocessed.
The Random Forest Classifier is trained on the data.
The web UI allows users to input medical parameters.
The model predicts whether the tumor is Malignant or Benign.

Installation & Usage
1. Install Dependencies
Make sure you have Python installed. Then, install the required libraries:

pip install flask pandas numpy scikit-learn
2. Run the Application
Start the Flask server by running:

python app.py
The app will be available at http://127.0.0.1:5000/.

3. Using the Web Interface
Enter the required medical parameters in the form.
Click the Predict button to get the result.
Technology Stack
Backend: Flask, Scikit-Learn
Frontend: HTML, CSS
Machine Learning Model: Random Forest Classifier

Future Improvements

Enhance UI design.
Allow users to upload medical reports for automated processing.
Deploy on a cloud platform for global access.
