import sys
import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

# Configurar el sistema para usar utf-8
sys.stdout.reconfigure(encoding='utf-8')

# Crear la aplicación Flask
app = Flask(__name__)

# Cargar los modelos entrenados
logreg_model = pickle.load(open('models/pipeline_logreg.pkl', 'rb'))
rnn_model = load_model('models/rnn_model.keras')
with open('models/preprocessor.pkl', 'rb') as file:
    preprocessor = pickle.load(file)

# Cargar el dataset para extraer las opciones categóricas
df = pd.read_csv('df_no_outliers.csv', encoding='utf-8')  # Usar codificación utf-8

# Identificar columnas categóricas y sus clases únicas
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
categorical_options = {col: df[col].dropna().unique().tolist() for col in categorical_features}


@app.route('/')
def home():
    return render_template(
        'index.html',
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        categorical_options=categorical_options
    )


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    model_type = data.pop('model')
    input_data = pd.DataFrame([data])

    if model_type == 'logreg':
        for col in input_data.columns:
            if col in categorical_features:
                input_data[col] = input_data[col].astype(str)
            elif col in numerical_features:
                input_data[col] = input_data[col].astype(float)
        prediction = logreg_model.predict(input_data)
        predicted_class = int(prediction[0])

    elif model_type == 'rnn':
        preprocessed_data = preprocessor.transform(input_data)
        if hasattr(preprocessed_data, "toarray"):
            preprocessed_data = preprocessed_data.toarray()
        features = preprocessed_data.astype(float).reshape(1, 1, -1)
        prediction = rnn_model.predict(features)
        predicted_class = int(prediction[0][0] > 0.5)

    return jsonify({'Modelo': model_type, 'Predicción': predicted_class})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
