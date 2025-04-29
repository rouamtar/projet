from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Charger le modèle KMeans
try:
    model = joblib.load(r'C:\Users\User\Desktop\algo_ml\predire ca\random_forest_model3.joblib')
    print("Modèle randomforest chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    model = None
# Créer un encodeur pour les valeurs textuelles de la Délégation et Localisation
delegation_encoder = LabelEncoder()
localisation_encoder = LabelEncoder()

# Exemple d'encodage basé sur des valeurs déjà vues dans le dataset
delegations = ['Monastir', 'Sousse', 'Médenine', 'Le Kef', 'Tozeur', 'Ariana', 'Manouba', 'Bizerte', 'Siliana', 'Kairouan', 'Nabeul', 'Sfax', 'La Marsa', 'Carthage', 'Tunis', 'Ben Arous', 'Jendouba', 'Mahdia']
localisations = ['Jemmal', 'Kalaâ Kebira', 'Sayada', 'Zarzis', 'Dahmani', 'Degache', 'Ennasr', 'Douar Hicher', 'Sejnane', 'Gaâfour', 'Bouhajla', 'Ezzahra', 'Hammam Sousse', 'Rades', 'Carthage Byrsa', 'El Mourouj', 'Sidi Daoud', 'Tabarka', 'Hammamet','Kelibia']

delegation_encoder.fit(delegations)
localisation_encoder.fit(localisations)

@app.route('/')
def home():
    return render_template('predireE3.html')

@app.route('/predireE3', methods=['POST'])
def predict():
    try:
        # Récupérer les données du formulaire
        variable1 = float(request.form['variable1'])  # Quantités Totales Vendues
        variable2 = int(request.form['variable2'])    # Année
        variable6 = int(request.form['variable6'])    # Mois
        delegation = request.form['delegation']       # Délégation
        localisation = request.form['localisation']   # Localisation

        # Encodage des données textuelles
        delegation_encoded = delegation_encoder.transform([delegation])[0]
        localisation_encoded = localisation_encoder.transform([localisation])[0]

        # Créer un tableau avec les données de l'utilisateur
        data = np.array([[variable1, variable2, variable6, delegation_encoded, localisation_encoded]])

        # Faire la prédiction
        prediction = model.predict(data)

        # Afficher le résultat de la prédiction
        return render_template('predireE3.html', prediction_text=f"Le chiffre d'affaires prédit est : {prediction[0]} TND")
    except Exception as e:
        return render_template('predireE3.html', error_message=f"Une erreur s'est produite : {e}")

if __name__ == '__main__':
    app.run(debug=True, port=8002)  # Vous pouvez choisir un autre port