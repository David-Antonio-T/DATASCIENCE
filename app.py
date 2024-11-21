import sys
import io
import os
import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Desactiva el uso de GPU en TensorFlow


# Set encoding explicitly
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Crear la aplicación Flask
app = Flask(__name__)

# Cargar los modelos entrenados
logreg_model = pickle.load(open('models/pipeline_logreg.pkl', 'rb'), encoding='latin1')
rnn_model = load_model('models/rnn_model.keras')

# Cargar preprocessor con encoding especifico
with open('models/preprocessor.pkl', 'rb') as file:
    preprocessor = pickle.load(file, encoding='latin1')

# Al cargar el dataset
df = pd.read_csv('df_no_outliers.csv', encoding='utf-8')

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

        # Extraer y eliminar el modelo seleccionado del diccionario
        model_type = data.pop('model')

        # Convertir datos a DataFrame con manejo de codificación
        input_data = pd.DataFrame([data])
        
        # Convertir todos los valores a strings y decodificar
        input_data = input_data.applymap(lambda x: str(x).encode('utf-8', errors='ignore').decode('utf-8'))

        if model_type == 'logreg':
            # Convertir categóricas a string y numéricas a float
            for col in input_data.columns:
                if col in categorical_features:
                    input_data[col] = input_data[col].astype(str)
                elif col in numerical_features:
                    input_data[col] = input_data[col].astype(float)

            # Predicción con regresión logística
            prediction = logreg_model.predict(input_data)
            predicted_class = int(prediction[0])

        elif model_type == 'rnn':
            # Preprocesar el input_data con el preprocesador
            preprocessed_data = preprocessor.transform(input_data)

            # Convertir a matriz densa si es dispersa
            if hasattr(preprocessed_data, "toarray"):
                preprocessed_data = preprocessed_data.toarray()

            # Redimensionar para la red neuronal
            features = preprocessed_data.astype(float).reshape(1, 1, -1)

            # Predicción con la red neuronal
            prediction = rnn_model.predict(features)
            predicted_class = int(prediction[0][0] > 0.5)

        return jsonify({'Modelo': model_type, 'Predicción': predicted_class})

    except Exception as e:
        # Registrar el error detallado
        print(f"Error en la predicción: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
