from flask import Flask, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf

# =========================
# CONFIG
# =========================
MODEL_PATH = "model_c2f.tflite"

app = Flask(__name__)
CORS(app)  # Habilita CORS para que tu frontend pueda consultar la API

# =========================
# LOAD TFLITE MODEL
# =========================
# Es importante manejar el error si el archivo .tflite no existe todavía
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Modelo TFLite cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")

# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return "Flask + TFLite API running"

@app.route("/predict/<float:celsius>", methods=["GET"])
def predict_get(celsius):
    try:
        # PREPARE INPUT (shape: 1x1)
        input_data = np.array([[celsius]], dtype=np.float32)
        
        # Ejecutar inferencia
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        
        # Obtener resultado
        output = interpreter.get_tensor(output_details[0]["index"])
        fahrenheit = float(output[0][0])
        
        return jsonify({
            "celsius": celsius,
            "fahrenheit": fahrenheit,
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # Usamos host="0.0.0.0" para que sea accesible en tu red local o Render
    app.run(debug=True, port=5000)