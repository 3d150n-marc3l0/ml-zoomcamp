import joblib
import requests
from flask import Flask, request, jsonify

# Cargar el modelo y el DictVectorizer
model = joblib.load('model1.bin')
dv = joblib.load('dv.bin')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos del request
    data = request.json

    # Transformar los datos utilizando el DictVectorizer
    X = dv.transform(data)

    # Realizar la predicción
    predictions_proba = model.predict_proba(X)
    predictions = predictions_proba[::,1]

    # Retornar la predicción como respuesta JSON
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)