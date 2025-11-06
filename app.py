from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import librosa
import pickle
import json
import os
from werkzeug.utils import secure_filename

# Charger le modèle et scaler
model = tf.keras.models.load_model('model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'aac'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_mfcc(audio_path, sr=8000, n_mfcc=40):
    y, _ = librosa.load(audio_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

def prepare_clinical_features(clinical_data):
    features = [
        clinical_data.get('age', 0.0),
        clinical_data.get('gender', 0.0),
        clinical_data.get('vhi', 0.0),
        clinical_data.get('rsi', 0.0),
        clinical_data.get('smoker', 0.0),
        clinical_data.get('coffee_cups', 0.0),
        clinical_data.get('water', 0.0),
        clinical_data.get('cigarettes', 0.0),
        clinical_data.get('alcohol_glasses', 0.0),
        clinical_data.get('carbonated', 0.0),
        clinical_data.get('carbonated_glasses', 0.0),
        clinical_data.get('tomatoes', 0.0),
        clinical_data.get('chocolate_grams', 0.0),
        clinical_data.get('cheese_grams', 0.0),
        clinical_data.get('citrus_number', 0.0)
    ]
    return scaler.transform(np.array(features).reshape(1, -1))[0]

@app.route('/health')
def health():
    return jsonify({"status": "ok", "message": "API Flask pour la prédiction de dysphonie"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'Aucun fichier audio fourni'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400

    if not allowed_file(audio_file.filename):
        return jsonify({'error': 'Format de fichier non supporté'}), 400

    if 'clinical_data' not in request.fields:
        return jsonify({'error': 'Données cliniques manquantes'}), 400

    filename = secure_filename(audio_file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    audio_file.save(filepath)

    try:
        mfcc_features = extract_mfcc(filepath)
        clinical_data_json = json.loads(request.fields['clinical_data'])
        clinical_features = prepare_clinical_features(clinical_data_json)

        audio_input = np.expand_dims(mfcc_features, axis=0)
        audio_input = np.expand_dims(audio_input, axis=-1)
        clinical_input = clinical_features.reshape(1, -1)

        predictions = model.predict([audio_input, clinical_input], verbose=0)
        probabilities = predictions[0]
        predicted_class = int(np.argmax(probabilities))

        class_names = [
            'healthy',
            'hyperkinetic dysphonia',
            'hypokinetic dysphonia',
            'reflux laryngitis'
        ]

        result = {
            'predictedClass': predicted_class,
            'predictedLabel': class_names[predicted_class],
            'confidence': float(probabilities[predicted_class]),
            'probabilities': {name: float(prob) for name, prob in zip(class_names, probabilities)}
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Erreur lors de la prédiction: {str(e)}'}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)