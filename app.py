from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and the exact feature list it expects
model = joblib.load('model1.pkl')  # Your trained RandomForestClassifier
input_features = joblib.load('input_features.pkl')  # List of feature names

# Mapping for categorical values
encoding_map = {
    'normal': 1, 'abnormal': 0,
    'present': 1, 'notpresent': 0,
    'yes': 1, 'no': 0,
    'good': 1, 'poor': 0,
    'male': 1, 'female': 0
}

# Medical reference ranges for validation
medical_ranges = {
    'age': (0, 120),
    'bp': (50, 250),  # Blood pressure (mmHg)
    'sg': (1.000, 1.040),  # Specific gravity
    'al': (0, 5),  # Albumin
    'su': (0, 5),  # Sugar
    'bgr': (50, 500),  # Blood glucose random (mg/dL)
    'bu': (5, 150),  # Blood urea (mg/dL)
    'sc': (0.1, 15.0),  # Serum creatinine (mg/dL)
    'sod': (100, 160),  # Sodium (mEq/L)
    'pot': (2.0, 7.0),  # Potassium (mEq/L)
    'hemo': (3.0, 20.0),  # Hemoglobin (g/dL)
    'pcv': (10, 60),  # Packed cell volume (%)
    'wc': (2000, 20000),  # White blood cell count (cells/mm³)
    'rc': (2.0, 8.0)  # Red blood cell count (millions/mm³)
}


@app.route('/')
def home():
    """Render the main input form page"""
    return render_template('index1.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission and return prediction results"""
    try:
        input_data = []
        user_inputs = {}  # Store original user inputs for display

        # Loop over the expected input features in order
        for feature in input_features:
            val = request.form.get(feature)

            if val is None or val.strip() == '':
                return render_template('result.html',
                                       error_message=f"⚠️ Missing required input: {feature}",
                                       prediction=None)

            val = val.strip().lower()
            user_inputs[feature] = val  # Store original value

            # Validate numerical values against medical ranges
            if feature in medical_ranges:
                try:
                    num_val = float(val)
                    min_val, max_val = medical_ranges[feature]
                    if not (min_val <= num_val <= max_val):
                        return render_template('result.html',
                                               error_message=f"⚠️ {feature} value {val} is outside plausible medical range ({min_val}-{max_val})",
                                               prediction=None)
                except ValueError:
                    return render_template('result.html',
                                           error_message=f"⚠️ Invalid numerical input for '{feature}': {val}",
                                           prediction=None)

            # Convert categorical values using encoding_map
            if val in encoding_map:
                input_data.append(encoding_map[val])
            else:
                try:
                    input_data.append(float(val))
                except ValueError:
                    return render_template('result.html',
                                           error_message=f"⚠️ Invalid input for '{feature}': {val}",
                                           prediction=None)

        # Ensure the input has exactly the shape the model expects
        final_input = np.array(input_data).reshape(1, -1)

        # Predict using the model
        prediction = model.predict(final_input)[0]
        probability = model.predict_proba(final_input)[0][1]  # Probability of CKD

        # Prepare context for template
        context = {
            'prediction': prediction,
            'probability': round(probability * 100, 1),
            'age': float(user_inputs.get('age', 0)),
            'htn': user_inputs.get('htn', 'no'),
            'dm': user_inputs.get('dm', 'no'),
            'cad': user_inputs.get('cad', 'no'),
            'appet': user_inputs.get('appet', 'good'),
            'pe': user_inputs.get('pe', 'no'),
            'ane': user_inputs.get('ane', 'no'),
            'sc': float(user_inputs.get('sc', 0)),
            'bgr': float(user_inputs.get('bgr', 0)),
            'hemo': float(user_inputs.get('hemo', 0)),
            'bp': float(user_inputs.get('bp', 0)),
            'rbc': user_inputs.get('rbc', 'normal'),
            'pc': user_inputs.get('pc', 'normal'),
            'pcc': user_inputs.get('pcc', 'notpresent'),
            'ba': user_inputs.get('ba', 'notpresent'),
            'al': float(user_inputs.get('al', 0))
        }

        return render_template('result.html', **context)

    except Exception as e:
        return render_template('result.html',
                               error_message=f"⚠️ System error occurred: {str(e)}",
                               prediction=None)


if __name__ == '__main__':
    app.run(debug=True)