from flask import Flask, request, render_template
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the pre-trained models
heart_model = joblib.load('heart_disease_model.pkl')
heart_scaler = joblib.load('scaler.pkl')
fraud_model = joblib.load('fraud_detection_model.pkl')

# Define the expected feature names for heart disease model
heart_expected_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Define the expected feature names for fraud detection model
fraud_expected_features = [
    'User_ID', 'Transaction_Amount', 'Transaction_Type', 
    'Time_of_Transaction', 'Device_Used', 'Location', 
    'Previous_Fraudulent_Transactions', 'Account_Age', 
    'Number_of_Transactions_Last_24H', 'Payment_Method'
]

# Initialize a LabelEncoder for categorical features
location_encoder = LabelEncoder()

# Example: Fit the encoder with possible location values (you should use the same values as in your training data)
location_encoder.fit(["New York", "San Francisco", "Los Angeles", "Chicago", "Miami"])

# Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Heart Disease Demo Page
@app.route('/demo_heart', methods=['GET', 'POST'])
def demo_heart():
    if request.method == 'POST':
        # Get the data from the form
        data = request.form.to_dict()
        
        # Create a DataFrame with the expected features
        input_data = pd.DataFrame([data], columns=heart_expected_features)
        
        # Convert all values to numeric
        input_data = input_data.apply(pd.to_numeric)
        
        # Preprocess the input data using the saved scaler
        input_data_scaled = heart_scaler.transform(input_data)
        
        # Make a prediction
        prediction = heart_model.predict(input_data_scaled)
        
        # Return the result
        if prediction[0] == 1:
            result = "High risk of heart disease"
        else:
            result = "Low risk of heart disease"
        
        return render_template('demo_heart.html', prediction_text=result)
    
    return render_template('demo_heart.html')

# Fraud Detection Demo Page
@app.route('/demo_fraud', methods=['GET', 'POST'])
def demo_fraud():
    if request.method == 'POST':
        # Get the data from the form
        data = request.form.to_dict()
        
        # Create a DataFrame with the expected features
        input_data = pd.DataFrame([data], columns=fraud_expected_features)
        
        # Convert categorical features to numeric
        input_data['Location'] = location_encoder.transform([input_data['Location'].iloc[0]])
        input_data['Transaction_Type'] = input_data['Transaction_Type'].astype(int)
        input_data['Device_Used'] = input_data['Device_Used'].astype(int)
        input_data['Payment_Method'] = input_data['Payment_Method'].astype(int)
        
        # Convert all values to numeric
        input_data = input_data.apply(pd.to_numeric)
        
        # Make a prediction
        prediction = fraud_model.predict(input_data)
        
        # Return the result
        if prediction[0] == 1:
            result = "Fraudulent"
        else:
            result = "Not Fraudulent"
        
        return render_template('demo_fraud.html', prediction_text=result)
    
    return render_template('demo_fraud.html')

if __name__ == '__main__':
    app.run(debug=True)