import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import joblib
# Remplacer par le chemin vers ton fichier
fichier_csv = r'C:\Users\User\Desktop\algo_ml\Recommand_best_product\source.csv'

# Charger le fichier CSV dans un DataFrame
df = pd.read_csv(fichier_csv, sep=",")
df.head()


# 3. Encodage des colonnes
le_marque = LabelEncoder()
le_taille = LabelEncoder()
le_nom = LabelEncoder()

df['Marque_enc'] = le_marque.fit_transform(df['Marque'])
df['Taille_enc'] = le_taille.fit_transform(df['Taille'])
df['Nom_enc'] = le_nom.fit_transform(df['Nom'])

# 4. Features et Target
X = df[['Marque_enc', 'Taille_enc', 'Prix', 'Categorie']]
y = df['Nom_enc']

# 5. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 6. Modèle Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy Random Forest : {accuracy:.2f}")

# 8. Fonction intelligente de recommandation
def top3_produits(marque=None, taille=None, categorie=None, prix_min=None, prix_max=None):
    if categorie is None:
        return ["❌ La catégorie est obligatoire."]

    # Filtrer uniquement la catégorie
    subset = df[df['Categorie'] == categorie].copy()
    if subset.empty:
        return ["❌ Aucun produit trouvé dans cette catégorie."]

    # Matching des critères
    subset['marque_match'] = (subset['Marque'] == marque) if marque else False
    subset['taille_match'] = (subset['Taille'] == taille) if taille else False
    subset['prix_match'] = (
        subset['Prix'].between(prix_min, prix_max)
        if prix_min is not None and prix_max is not None else False
    )

    # Score total
    subset['score'] = (
        subset['marque_match'].astype(int) +
        subset['taille_match'].astype(int) +
        subset['prix_match'].astype(int)
    )

    if subset['score'].max() == 0:
        return ["❌ Aucun produit proche trouvé. Essayez avec d'autres critères."]

    # Trier par score et prix
    subset = subset.sort_values(by=['score', 'Prix'], ascending=[False, True])
    subset = subset.drop_duplicates(subset=['Entreprise'])

    # Top 3
    top3 = subset.head(3)

    # Préparation des prédictions
    top3['Marque_enc'] = le_marque.transform(top3['Marque'])
    top3['Taille_enc'] = le_taille.transform(top3['Taille'])
    X_top3 = top3[['Marque_enc', 'Taille_enc', 'Prix', 'Categorie']]
    top3['Prediction'] = model.predict(X_top3)
    top3['Nom'] = le_nom.inverse_transform(top3['Prediction'])

    # Résultats
    resultats = [
        f"{row['Nom']} 🏢 {row['Entreprise']} 💰 {row['Prix']:.2f} DT"
        for _, row in top3.iterrows()
    ]

    return resultats

# 9. Exemple d’utilisation
resultats = top3_produits(
    marque="BOUCHERIE",
    taille=None,
    categorie=3,
    prix_min=5.0,
    prix_max=9.0
)

print("🔮 Top 3 suggestions :")
for res in resultats:
    print("🔹", res)
# 10. Sauvegarder le modèle
chemin_model = r'C:\Users\User\Desktop\algo_ml\Recommand_best_product\recommandation.joblib'
# Sauvegarde complète avec le modèle + encodeurs
objet = {
    'model': model,
    'le_marque': le_marque,
    'le_taille': le_taille,
    'le_nom': le_nom
}
joblib.dump(objet, chemin_model)
print("✅ Modèle et encodeurs sauvegardés dans 'recommandation.joblib'")



