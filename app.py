import sys
import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template

# Configurar el sistema para usar utf-8
sys.stdout.reconfigure(encoding='utf-8')

# Crear la aplicación Flask
app = Flask(__name__)

# Cargar el modelo entrenado (solo el de regresión logística)
logreg_model = pickle.load(open('models/pipeline_logreg.pkl', 'rb'))

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
    try:
        # Obtener los datos del formulario
        data = request.form.to_dict()

        # Convertir datos a DataFrame
        input_data = pd.DataFrame([data])

        # Preprocesar datos
        for col in input_data.columns:
            if col in categorical_features:
                input_data[col] = input_data[col].astype(str)
            elif col in numerical_features:
                input_data[col] = input_data[col].astype(float)

        # Realizar predicción con el modelo de regresión logística
        prediction = logreg_model.predict(input_data)
        predicted_class = int(prediction[0])

        return jsonify({'Modelo': 'logreg', 'Predicción': predicted_class})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
