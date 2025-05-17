import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import joblib
# Load the data
df = pd.read_csv('../../data/heart_failure_clinical_records_dataset.csv')

# Preprocess the data
x = df.drop(columns=['DEATH_EVENT'])
y = df['DEATH_EVENT']
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the model
model = XGBClassifier(n_estimators=100, learning_rate=0.1)
model.fit(x_train, y_train)

# Save the model
joblib.dump(model, 'heart_failure_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully.")
