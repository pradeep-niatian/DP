from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model/diabetes_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    data = np.array([features])
    prediction = model.predict(data)
    return render_template('result.html', prediction="Positive" if prediction[0] else "Negative")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)