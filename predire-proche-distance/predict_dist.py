from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Charger le modèle KMeans
try:
    model = joblib.load(r'C:\Users\User\Desktop\algo_ml\predire-proche-distance\gradientboosting1.joblib')
    print("Modèle gradientboosting1 chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    model = None
# === Chargement des encoders ===
try:
    label_encoders = joblib.load(r'C:\Users\User\Desktop\algo_ml\predire-proche-distance\label_encoders.joblib')
    print("✅ Dictionnaire d'encodeurs chargé avec succès.")
except Exception as e:
    print(f"❌ Erreur lors du chargement des encodeurs : {e}")
    label_encoders = None

# === Page d'accueil (formulaire) ===
@app.route('/')
def home():
    return render_template('predict_dist.html')  # Ton HTML doit s'appeler index.html et être dans "templates/"

# === Route de prédiction ===
@app.route('/predict_dist', methods=['POST'])
def predict_dist():
    if request.method == 'POST':
        try:
            # Récupération des données du formulaire
            nom_entreprise = request.form['nom_entreprise']
            code_postal = request.form['code_postal']
            localite = request.form['localite']   # <<< correction ici
            delegation = request.form['delegation']

            # Préparation des données
            input_data = pd.DataFrame({
                'nom_entreprise': [nom_entreprise],
                'code_postal': [code_postal],
                'localite': [localite],
                'Delegation': [delegation]
            })

            # Encodage
            for col in ['nom_entreprise', 'code_postal', 'localite', 'Delegation']:
                if col in label_encoders:
                    input_data[col] = label_encoders[col].transform(input_data[col])

            # Prédiction
            distance_predite = model.predict(input_data)[0]
            prediction_text = f"La distance prédite est : {distance_predite:.2f} km."

            return render_template('predict_dist.html', prediction_text=prediction_text)

        except Exception as e:
            error_message = f"Erreur lors de la prédiction : {e}"
            return render_template('predict_dist.html', error_message=error_message)

# === Exécuter l'application ===
if __name__ == '__main__':
    app.run(debug=True, port=8003)