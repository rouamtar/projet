import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

from sklearn.preprocessing import LabelEncoder



# Remplacer par le chemin vers ton fichier
fichier_excel = r'C:\Users\User\Desktop\algo_ml\hotels_avec_rate.xlsx'

# Charger le fichier Excel dans un DataFrame
df = pd.read_excel(fichier_excel)

# 1. Préparation des données
# Nettoyage des noms de région
corrections = {
    'hamamet': 'Hammamet',
    'hammet': 'Hammamet',
    'hammamet': 'Hammamet',
    'djerba': 'Djerba',
    'sousse': 'Sousse',
    'monastir': 'Monastir',
}

df['region'] = df['region'].str.lower().str.strip()
df['region'] = df['region'].replace(corrections)
df['region'] = df['region'].str.title()  # Pour avoir une belle mise en forme

# Encodage
region_encoder = LabelEncoder()
df['region_encoded'] = region_encoder.fit_transform(df['region'])

# === Affichage du codage des régions ===
print("\n=== Codage des régions ===")
for region, code in zip(region_encoder.classes_, region_encoder.transform(region_encoder.classes_)):
    print(f"{region} → {code}")

# 2. Données pour le clustering
data = df[['region_encoded', 'rate']].values

# 3. Détermination du nombre optimal de clusters (méthode du coude)
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

# 4. Clustering K-Means (K=4 par exemple)
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
labels_kmeans = kmeans.fit_predict(data)

# 5. Calcul des métriques
silhouette = silhouette_score(data, labels_kmeans)
db_score = davies_bouldin_score(data, labels_kmeans)

# 6. Interprétation des clusters
df['cluster'] = labels_kmeans
cluster_stats = df.groupby('cluster').agg({
    'rate': ['mean', 'min', 'max'],
    'region': lambda x: x.mode()[0]
}).reset_index()

print("\n=== Statistiques par cluster ===")
print(cluster_stats)

print(f"""
=== Résultats K-Means ===
- Nombre de clusters : {k}
- Score de Silhouette : {silhouette:.2f} (cible >0.5)
- Indice Davies-Bouldin : {db_score:.2f} (cible <1)

=== Interprétation recommandée ===
luster 0 : Hôtels milieu de gamme avec des notes modérées à élevées

Cluster 1 : Hôtels de gamme supérieure, de qualité plus élevée

Cluster 2 : Hôtels haut de gamme ou de luxe, avec des prestations plus poussées

Cluster 3 : Hôtels économiques, de qualité plus basse avec des prix abordables
""")
import joblib




chemin_model = r'C:\Users\User\Desktop\algo_ml\kmeans.joblib'

# Sauvegarde du modèle
joblib.dump(kmeans, chemin_model)
print("✅ Modèle KMeans sauvegardé sous 'kmeans_model.joblib'")