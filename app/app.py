from flask import Flask, request, render_template
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

heart_model = joblib.load('heart_disease_model.pkl')
heart_scaler = joblib.load('scaler.pkl')
fraud_model = joblib.load('fraud_detection_model.pkl')

heart_expected_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

fraud_expected_features = [
    'User_ID', 'Transaction_Amount', 'Transaction_Type', 
    'Time_of_Transaction', 'Device_Used', 'Location', 
    'Previous_Fraudulent_Transactions', 'Account_Age', 
    'Number_of_Transactions_Last_24H', 'Payment_Method'
]

location_encoder = LabelEncoder()

location_encoder.fit(["New York", "San Francisco", "Los Angeles", "Chicago", "Miami"])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/demo_heart', methods=['GET', 'POST'])
def demo_heart():
    if request.method == 'POST':
        data = request.form.to_dict()
        
        input_data = pd.DataFrame([data], columns=heart_expected_features)
        
        input_data = input_data.apply(pd.to_numeric)
        
        input_data_scaled = heart_scaler.transform(input_data)
        
        prediction = heart_model.predict(input_data_scaled)

        if prediction[0] == 1:
            result = "High risk of heart disease"
        else:
            result = "Low risk of heart disease"
        
        return render_template('demo_heart.html', prediction_text=result)
    
    return render_template('demo_heart.html')

@app.route('/demo_fraud', methods=['GET', 'POST'])
def demo_fraud():
    if request.method == 'POST':
        data = request.form.to_dict()

        input_data = pd.DataFrame([data], columns=fraud_expected_features)

        input_data['Location'] = location_encoder.transform([input_data['Location'].iloc[0]])
        input_data['Transaction_Type'] = input_data['Transaction_Type'].astype(int)
        input_data['Device_Used'] = input_data['Device_Used'].astype(int)
        input_data['Payment_Method'] = input_data['Payment_Method'].astype(int)

        input_data = input_data.apply(pd.to_numeric)

        prediction = fraud_model.predict(input_data)

        if prediction[0] == 1:
            result = "Fraudulent"
        else:
            result = "Not Fraudulent"
        
        return render_template('demo_fraud.html', prediction_text=result)
    
    return render_template('demo_fraud.html')

if __name__ == '__main__':
    app.run(debug=True)