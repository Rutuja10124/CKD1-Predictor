import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv('ckd.csv')

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Replace text values with numeric for categorical fields
replace_dict = {
    'normal': 0, 'abnormal': 1,
    'present': 1, 'notpresent': 0,
    'yes': 1, 'no': 0,
    'good': 0, 'poor': 1
}
df.replace(replace_dict, inplace=True)

# Encode target column
if 'classification' in df.columns:
    df['classification'] = df['classification'].apply(lambda x: 1 if 'ckd' in str(x).lower() else 0)

# Select final 24 features
feature_columns = [
    'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu',
    'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad',
    'appet', 'pe', 'ane'
]

# Ensure columns exist and convert numeric where needed
for col in feature_columns:
    if col not in df.columns:
        raise ValueError(f"Missing column in dataset: {col}")
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing values
df.dropna(subset=feature_columns + ['classification'], inplace=True)

# Split data
X = df[feature_columns]
y = df['classification']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"✅ Model trained with accuracy: {accuracy:.2f}")

# Save model and feature list
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('input_features.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

print("🎉 model.pkl and input_features.pkl saved with 24 features.")
