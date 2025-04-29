from flask import Flask, render_template, request
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Chemins vers les fichiers
chemin_model = r'C:\Users\User\Desktop\algo_ml\predire_prix-davion\modele_billet.joblib'
chemin_dataframe = r'C:\Users\User\Desktop\algo_ml\predire_prix-davion\Billet_source.csv'

# Chargement du modèle et du dataset
pipeline = joblib.load(chemin_model)
df = pd.read_csv(chemin_dataframe, sep=",")

# Mapping des compagnies (on garde uniquement la compagnie)
compagnie_mapping = {
    1: "KLM Royal Dutch Airlines",
    2: "Tunis Air",
    3: "Royal Air Maroc",
    4: "Air France",
    5: "Air Algérie",
    6: "Turkish Airlines",
    7: "ITA Airways"
}

# Préparation des dates dans le dataset
df['fk_Date_Depart'] = pd.to_datetime(df['fk_Date_Depart'], dayfirst=True, errors='coerce')
df['fk_Date_Arrivee'] = pd.to_datetime(df['fk_Date_Arrivee'], dayfirst=True, errors='coerce')
df['fk_Date_Depart_str'] = df['fk_Date_Depart'].dt.strftime("%d/%m/%Y")
df['fk_Date_Arrivee_str'] = df['fk_Date_Arrivee'].dt.strftime("%d/%m/%Y")

@app.route('/')
def home():
    return render_template('predire_avion.html')

@app.route('/predire_avion', methods=['GET', 'POST'])
def predire_avion():
    if request.method == 'POST':
        try:
            # Récupération des données du formulaire
            date_depart = datetime.strptime(request.form['date_depart'], "%Y-%m-%d").strftime("%d/%m/%Y")
            date_arrivee = datetime.strptime(request.form['date_arrivee'], "%Y-%m-%d").strftime("%d/%m/%Y")
            ville_depart = request.form['ville_depart'].strip().title()
            ville_arrivee = request.form['ville_arrivee'].strip().title()
            classe = request.form['classe']
            bagage = request.form['bagage']

            input_data = pd.DataFrame([{
                'fk_Date_Depart_str': date_depart,
                'fk_Date_Arrivee_str': date_arrivee,
                'fk_ville_depart': ville_depart,
                'fk_ville_Arrivee': ville_arrivee,
                'classe': classe,
                'bagage': bagage
            }])

            # Prédiction de prix
            predicted_price = round(pipeline.predict(input_data)[0], 2)

            # Filtrage selon classe et bagage uniquement
            match = df[
                (df['classe'] == classe) &
                (df['bagage'] == bagage)
            ]

            # Recherche compagnie
            compagnie_pred = "N/A"

            if not match.empty:
                if not match['fk_compagnie'].dropna().empty:
                    compagnie_id = match['fk_compagnie'].dropna().mode().iloc[0]
                    compagnie_pred = compagnie_mapping.get(compagnie_id, "N/A")

            return render_template('predire_avion.html',
                                   prix=predicted_price,
                                   compagnie=compagnie_pred)

        except Exception as e:
            return render_template('predire_avion.html', error_message=f"Erreur : {str(e)}")

    return render_template('predire_avion.html')

if __name__ == '__main__':
    app.run(debug=True, port=8007)
