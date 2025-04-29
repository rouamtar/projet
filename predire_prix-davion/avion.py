import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


import joblib
# Remplacer par le chemin vers ton fichier
fichier_csv = r'C:\Users\User\Desktop\algo_ml\predire_prix-davion\Billet_source.csv'

# Charger le fichier CSV dans un DataFrame
df = pd.read_csv(fichier_csv, sep=",")
df.head()
# ✅ Conversion des dates
df['fk_Date_Depart'] = pd.to_datetime(df['fk_Date_Depart'], dayfirst=True, errors='coerce')
df['fk_Date_Arrivee'] = pd.to_datetime(df['fk_Date_Arrivee'], dayfirst=True, errors='coerce')

# 📆 Garder dates format texte pour prédiction directe
df['fk_Date_Depart_str'] = df['fk_Date_Depart'].dt.strftime("%d/%m/%Y")
df['fk_Date_Arrivee_str'] = df['fk_Date_Arrivee'].dt.strftime("%d/%m/%Y")

# 🔎 Nettoyage
df = df.dropna(subset=[
    'fk_Date_Depart_str', 'fk_Date_Arrivee_str',
    'fk_ville_depart', 'fk_ville_Arrivee',
    'classe', 'bagage', 'prix_vol'
])

# ✅ Variables
X = df[[
    'fk_Date_Depart_str', 'fk_Date_Arrivee_str',
    'fk_ville_depart', 'fk_ville_Arrivee',
    'classe', 'bagage'
]]
y = df['prix_vol']

compagnie = df['fk_compagnie']
entreprise = df['fk_entreprise']

# 🔁 Encodage
categorical_cols = X.columns.tolist()
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# 🔨 Pipeline complet
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=150, random_state=42))
])

# 🎯 Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
compagnie_test = compagnie.loc[y_test.index]
entreprise_test = entreprise.loc[y_test.index]

# 🚀 Entraînement
pipeline.fit(X_train, y_train)

# 🔍 Évaluation
y_pred = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"✅ Modèle entraîné — RMSE : {rmse:.2f} DT")
print(f"📊 Coefficient de détermination (R²) : {r2:.2f}")

# 🔮 Prédiction personnalisée
def predict_billet(date_dep, date_arr, ville_dep, ville_arr, classe, bagage):
    input_data = pd.DataFrame([{
        'fk_Date_Depart_str': date_dep,
        'fk_Date_Arrivee_str': date_arr,
        'fk_ville_depart': ville_dep,
        'fk_ville_Arrivee': ville_arr,
        'classe': classe,
        'bagage': bagage
    }])
    predicted_price = pipeline.predict(input_data)[0]

    # Recherche de ligne similaire pour suggérer entreprise/compagnie
    match = df[
        (df['fk_ville_depart'] == ville_dep) &
        (df['fk_ville_Arrivee'] == ville_arr) &
        (df['classe'] == classe) &
        (df['bagage'] == bagage)
    ]
    entreprise_pred = match['fk_entreprise'].mode().iloc[0] if not match.empty else "N/A"
    compagnie_pred = match['fk_compagnie'].mode().iloc[0] if not match.empty else "N/A"

    return round(predicted_price, 2), entreprise_pred, compagnie_pred

# ✈️ Exemple
prix, ent, comp = predict_billet(
    date_dep="12/02/2024",
    date_arr="16/02/2024",
    ville_dep="Tunis",
    ville_arr="Rome",
    classe="Economique",
    bagage="Inclus"
)

print(f"\n🔮 Résultat de prédiction :")
print(f"💰 Prix estimé : {prix} DT")
print(f"🏢 Entreprise probable : {ent}")
print(f"🛫 Compagnie probable : {comp}")

# 💾 Sauvegarder le pipeline complet dans un fichier .joblib
modele_path = r'C:\Users\User\Desktop\algo_ml\predire_prix-davion\modele_billet.joblib'
joblib.dump(pipeline, modele_path)

print(f"✅ Modèle sauvegardé ici : {modele_path}")
