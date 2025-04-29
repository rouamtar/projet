import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Remplacer par le chemin vers ton fichier
fichier_csv = r'C:\Users\User\Desktop\algo_ml\predire ca\E1.csv'

# Charger le fichier CSV dans un DataFrame
df = pd.read_csv(fichier_csv, sep=";")
df.head()
df = df.drop(columns=['Prix_Unitaire_Totale'])

df['Prix_Total'] = df['Prix_Total'].astype(int)

# Transform year to relative value
min_year = df['Annee'].min()
df['Annee_relative'] = df['Annee'] - min_year
df = df.drop(columns=['Annee'])

# Target
y = df['Prix_Total']

# Remplacer l'encodage One-Hot par un encodage LabelEncoder pour les colonnes 'Delegation' et 'localite'
label_encoder_delegation = LabelEncoder()
label_encoder_localite = LabelEncoder()

df['Delegation'] = label_encoder_delegation.fit_transform(df['Delegation'])
df['localite'] = label_encoder_localite.fit_transform(df['localite'])

# Features
X = df.drop(columns=['Prix_Total'])

# Normaliser les caractéristiques numériques
scaler = MinMaxScaler()
X_scaled = X.copy()
X_scaled[['Quantite_Totale', 'Annee_relative']] = scaler.fit_transform(X_scaled[['Quantite_Totale', 'Annee_relative']])

# Séparation train-test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modèle Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Entraîner
model.fit(X_train, y_train)

# Prédiction
y_pred = model.predict(X_test)

# Évaluation
r2 = r2_score(y_test, y_pred)
print(f"Random Forest - R² Score: {r2:.4f}")

# Sauvegarder le modèle
import joblib
import os

# Définir l'emplacement
chemin_sauvegarde = r'C:\Users\User\Desktop\algo_ml\predire ca\random_forest_model.joblib'

# Sauvegarder le modèle
joblib.dump(model, chemin_sauvegarde)
print(f"\n✅ Modèle sauvegardé ici : {chemin_sauvegarde}")
