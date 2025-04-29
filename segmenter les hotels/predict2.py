from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Charger le modèle KMeans
try:
    kmeans = joblib.load(r'C:\Users\User\Desktop\algo_ml\segmenter les hotels\kmeans.joblib')
    print("Modèle KMeans chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    kmeans = None

# Liste des régions valides
regions = ['Djerba', 'Hammamet', 'Monastir', 'Sousse']
region_encoder = LabelEncoder()
region_encoder.fit(regions)

# Description des clusters
cluster_descriptions = {
    0: "Hôtels milieu de gamme avec des notes modérées à élevées.",
    1: "Hôtels de gamme supérieure, de qualité plus élevée.",
    2: "Hôtels haut de gamme ou de luxe, avec des prestations plus poussées.",
    3: "Hôtels économiques, de qualité plus basse avec des prix abordables."
}

@app.route('/', methods=['GET'])
def home():
    return render_template('predict2.html', prediction_text=None, error_message=None)

@app.route('/predict2', methods=['POST'])
def predict():
    try:
        region = request.form['region']
        rate = float(request.form['rate'])

        if region not in regions:
            return render_template('predict2.html', prediction_text=None, error_message=f"Erreur : La région '{region}' est invalide.", hotels_list=None)

        if kmeans is None:
            return render_template('predict2.html', prediction_text=None, error_message="Erreur : Modèle non chargé.", hotels_list=None)

        # Encoder la région
        region_encoded = region_encoder.transform([region])[0]

        # Correction du taux
        rate_corrected = rate / 2

        # Prédiction du cluster
        data = np.array([[region_encoded, rate_corrected]])
        cluster = kmeans.predict(data)[0]

        description = cluster_descriptions.get(cluster, "Cluster inconnu.")

        # Lire le fichier Excel
        hotels_df = pd.read_excel(r'C:\Users\User\Desktop\algo_ml\segmenter les hotels\hotels_avec_rate.xlsx')

        # Nettoyer la région
        hotels_df['region'] = hotels_df['region'].str.title()

        # Chercher les hôtels dans la région saisie ET dont la rate est proche de celle saisie
        tolerance = 0.5  # par exemple +-0.5 autour du rate
        hotels_matches = hotels_df[
            (hotels_df['region'].str.lower() == region.lower()) &
            (hotels_df['rate'] >= rate - tolerance) &
            (hotels_df['rate'] <= rate + tolerance)
        ]

        hotels_list = hotels_matches['name'].tolist()

        prediction_text = f" Les hotels dans cette région appartient au cluster  : {cluster} : {description}  Voici les hôtels trouvés dans {region} avec un rate proche de {rate} :"

        return render_template('predict2.html', prediction_text=prediction_text, error_message=None, hotels_list=hotels_list)

    except Exception as e:
        print(f"Erreur dans la prédiction: {e}")
        return render_template('predict2.html', prediction_text=None, error_message="Erreur lors de la prédiction. Détails: " + str(e), hotels_list=None)

if __name__ == '__main__':
    app.run(debug=True, port=3000)
