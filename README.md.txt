Chronic Kidney Disease (CKD) Predictor
Project Overview
The CKD Predictor is a web application built using Flask and Machine Learning that predicts the likelihood of Chronic Kidney Disease based on patient health parameters.
It uses a Random Forest Classifier trained on a publicly available CKD dataset.

Folder Structure
CKD_Predictor/
│-- app.py → Flask backend
│-- model_training.py → ML model training script
│-- model.pkl → Saved trained model
│-- input_features.pkl → List of model input features
│-- templates/
│ ├── index.html → Input form
│ ├── result.html → Prediction result page
│-- static/ → CSS/JS files (if any)
│-- requirements.txt → Required dependencies
│-- ckd.csv → Dataset used for training

Installation & Setup
Clone the repository
git clone https://github.com/your-username/CKD_Predictor.git
cd CKD_Predictor

Create a virtual environment
python -m venv .venv

Activate the environment

Windows: .venv\Scripts\activate

Mac/Linux: source .venv/bin/activate

Install dependencies
pip install -r requirements.txt

How to Run
Train the model (if not already trained):
python model_training.py

Start the Flask app:
python app.py

Open your browser and visit:
http://127.0.0.1:5000/

How it Works
User enters 24 health parameters (BP, Hemoglobin, Sugar levels, etc.) in the web form.

The model processes these inputs and predicts:
✅ "You are healthy"
⚠️ "CKD Detected"

The backend is powered by a Random Forest Classifier.

Dataset
Source: UCI Machine Learning Repository – Chronic Kidney Disease Dataset

Size: 400 records, 24 features + target label

Target Variable: CKD (1) or Not CKD (0)

Future Improvements
Deploy on Heroku or Render

Add more disease prediction modules

Improve model accuracy with hyperparameter tuning

Enhance UI for better user experience

License
This project is licensed under the MIT License – feel free to use and modify it.

Author
Rutuja Bhusare