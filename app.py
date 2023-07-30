from flask import Flask, render_template, request
import pickle
import pandas as pd

# Load the trained Random Forest Regressor model using pickle
with open('random_forest_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = request.form.to_dict()
    # Convert categorical values to numerical representations using one-hot encoding
    fuel_type = features['Fuel_Type_Petrol']
    seller_type = features['Seller_Type_Individual']
    transmission = features['Transmission_Mannual']

    # Create a new DataFrame with one-hot encoded features
    input_data = pd.DataFrame({
        'Present_Price': [float(features['Present_Price'])],
        'Kms_Driven': [int(features['Kms_Driven'])],
        'Owner': [int(features['Owner'])],
        'Fuel_Type_Diesel': [1 if fuel_type == 'Diesel' else 0],
        'Fuel_Type_Petrol': [1 if fuel_type == 'Petrol' else 0],
        'Seller_Type_Individual': [1 if seller_type == 'Individual' else 0],
        'Transmission_Mannual': [1 if transmission == 'Mannual' else 0],
        'no_year': [2023 - int(features['Year'])]  # Set default 'Current_Year' to 2023
    })

    prediction = model.predict(input_data)
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text='Predicted Selling Price: {} lakhs'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
