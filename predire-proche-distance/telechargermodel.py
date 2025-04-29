import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import joblib
# Remplacer par le chemin vers ton fichier
fichier_csv = r'C:\Users\User\Desktop\algo_ml\predire-proche-distance\df_magazin.csv'

# Charger le fichier CSV dans un DataFrame
df = pd.read_csv(fichier_csv, sep=",")
df.head()

# === Chargement des données ===

df = df.drop('nom_entreprise_clean', axis=1)

# === Cible : Distance ===
y = df['Distance']

# === Encodage des variables catégorielles avec LabelEncoder ===
categorical_cols = ['nom_entreprise', 'code_postal', 'localite', 'Delegation']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # on garde l'encodeur pour la prédiction future

# === Dataset final ===
X = df[categorical_cols]  # uniquement les variables encodées

# === Split Train/Test ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Modèles (KNN est inclus uniquement pour analyse, pas pour sauvegarde)
models = {
    'KNN': KNeighborsRegressor(n_neighbors=7),
    'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=7, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
}

# === Entraînement, évaluation et sélection du meilleur modèle (hors KNN)
best_model = None
best_r2 = -np.inf
best_name = ""
model_names = []
r2_scores = []

print("\n📊 Résultats des modèles :")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n🔹 {name} :")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    model_names.append(name)
    r2_scores.append(r2)

    if name != "KNN" and r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_name = name


print(f"\n✅ Modèle sauvegardé (hors KNN) : {best_name} avec R² = {best_r2:.4f}")

# === Affichage du graphique comparatif
y_pred_best = best_model.predict(X_test)
chemin_model = r'C:\Users\User\Desktop\algo_ml\predire-proche-distance\gradientboosting1.joblib'
joblib.dump(best_model, chemin_model)
print("✅ Modèle Gradient Boosting sauvegardé sous 'gradientboosting1.joblib'")
import joblib

# Sauvegarde du dictionnaire des encodeurs
chemin_encoders = r'C:\Users\User\Desktop\algo_ml\predire-proche-distance\label_encoders.joblib'
joblib.dump(label_encoders, chemin_encoders)

print(f"✅ Dictionnaire d'encodeurs sauvegardé sous : {chemin_encoders}")
