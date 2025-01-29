from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

app = Flask(__name__)
CORS(app)  # add this to allow requests from anywhere

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="bilstm_model.tflite")
interpreter.allocate_tensors()

# Load tokenizer and label encoder
with open('tokenizer.json', 'r') as f:
    tokenizer_data = f.read()
tokenizer = tokenizer_from_json(tokenizer_data)

with open('label_encoder.json', 'r') as f:
    label_encoder = json.load(f)

# Ensure keys are treated as strings for JSON lookup
label_encoder = {int(k): v for k, v in label_encoder.items()}

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        user_text = data.get('text', '')

        if not user_text.strip():
            return jsonify({'error': 'No input text provided'}), 400

        # Preprocess the input text
        processed_text = preprocess_text(user_text)
        sequence = tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=100, padding='post')

        # Perform inference
        input_data = np.array(padded_sequence, dtype=np.float32)
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

        # Get predicted class index and confidence score
        predicted_class_index = np.argmax(output_data)
        confidence_score = float(np.max(output_data))  # Convert to Python float for JSON

        # Get the corresponding label
        predicted_label = label_encoder.get(predicted_class_index, "Unknown")

        return jsonify({      
            'input_text': user_text,
            'predicted_status': predicted_label,
            'confidence_score': confidence_score
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
