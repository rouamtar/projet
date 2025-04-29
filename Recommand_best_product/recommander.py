from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Initialiser l'app Flask
app = Flask(__name__)

# Charger modèle + encoders
chemin_model = r'C:\Users\User\Desktop\algo_ml\Recommand_best_product\recommandation.joblib'
objet = joblib.load(chemin_model)

model = objet['model']
le_marque = objet['le_marque']
le_taille = objet['le_taille']
le_nom = objet['le_nom']

# Ajouter un LabelEncoder pour la catégorie
le_categorie = LabelEncoder()

# Charger DataFrame source
df = pd.read_csv(r'C:\Users\User\Desktop\algo_ml\Recommand_best_product\source.csv')

# Encoder la catégorie dans le DataFrame
df['Categorie_enc'] = le_categorie.fit_transform(df['Categorie'])

# Fonction de recommandation
def top3_produits(marque=None, taille=None, categorie=None, prix_min=None, prix_max=None):
    if not categorie:
        return ["❌ La catégorie est obligatoire."]

    # Encoder la catégorie choisie par l'utilisateur
    categorie_enc = le_categorie.transform([categorie])[0]

    # Filtrer uniquement par catégorie encodée
    subset = df[df['Categorie_enc'] == categorie_enc].copy()
    if subset.empty:
        return ["❌ Aucun produit trouvé dans cette catégorie."]


    # Matching des critères
    subset['marque_match'] = (subset['Marque'] == marque) if marque else False
    subset['taille_match'] = (subset['Taille'] == taille) if taille else False
    subset['prix_match'] = (
        subset['Prix'].between(float(prix_min), float(prix_max))
        if prix_min and prix_max else False
    )

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

    # Prendre le top 3
    top3 = subset.head(3)

        # Encodage des variables nécessaires pour la prédiction
    top3['Marque_enc'] = le_marque.transform(top3['Marque'])
    top3['Taille_enc'] = le_taille.transform(top3['Taille'])

    top3['Categorie'] = top3['Categorie_enc']  # Remettre le nom attendu par le modèle !

    X_top3 = top3[['Marque_enc', 'Taille_enc', 'Prix', 'Categorie']]

    # Prédiction avec le modèle
    top3['Prediction'] = model.predict(X_top3)
    top3['Nom'] = le_nom.inverse_transform(top3['Prediction'])

    # Formatage des résultats à retourner
    resultats = [
        f"{row['Nom']} 🏢 {row['Entreprise']} 💰 {row['Prix']:.2f} DT"
        for _, row in top3.iterrows()
    ]

    return resultats

# Page d'accueil
@app.route('/')
def home():
    return render_template('recommander.html')

# Route de prédiction
@app.route('/recommander', methods=['POST'])
def recommander():
    try:
        marque = request.form.get('marque')
        taille = request.form.get('taille')
        categorie = request.form.get('categorie')
        prix_min = request.form.get('prix_min')
        prix_max = request.form.get('prix_max')

        # Validation des entrées
        if not marque or not categorie:
            return render_template('recommander.html', error_message="❌ Marque et catégorie sont obligatoires.")

        # Validation et conversion des prix
        if prix_min and prix_max:
            try:
                prix_min = float(prix_min)
                prix_max = float(prix_max)
                if prix_min > prix_max:
                    return render_template('recommander.html', error_message="❌ Le prix minimum ne peut être supérieur au prix maximum.")
            except ValueError:
                return render_template('recommander.html', error_message="❌ Les prix doivent être des nombres valides.")
        else:
            prix_min = prix_max = None

        resultats = top3_produits(
            marque=marque.strip() if marque else None,
            taille=taille.strip() if taille else None,
            categorie=categorie.strip(),
            prix_min=prix_min,
            prix_max=prix_max
        )

        return render_template('recommander.html', resultats=resultats)

    except Exception as e:
        return render_template('recommander.html', error_message=f"Erreur : {str(e)}")

# Démarrer le serveur Flask
if __name__ == '__main__':
    app.run(debug=True, port=8005)
