# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get feature inputs from the form
    features = [float(request.form['fixed_acidity']),
                float(request.form['volatile_acidity']),
                float(request.form['chlorides']),
                float(request.form['total_sulfur_dioxide']),
                float(request.form['density']),
                float(request.form['pH']),
                float(request.form['sulphates']),
                float(request.form['alcohol'])]

    # Convert features to numpy array and reshape for prediction
    features = np.array(features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)

    # Return result
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)