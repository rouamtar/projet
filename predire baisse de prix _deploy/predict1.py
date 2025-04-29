from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Charger le modèle KNN
try:
    knn_model = joblib.load(r'C:\Users\User\Desktop\algo_ml\predire baisse de prix _deploy\knn_model.joblib')
    print("Modèle chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    knn_model = None
# Dictionnaire pour convertir les codes en noms de produits
code_to_produit = {
    0: "Assouplissant",
    1: "Barre chocolatée",
    2: "Biscottes nature",
    3: "Biscuits fourrés",
    4: "Biscuits prouta",
    5: "Biscuits sans sucre",
    6: "Biscuits secs Fruit & fibre et en chocolat & noisette",
    7: "Boisson au jus existe en plusieurs saveurs",
    8: "Boisson gazeuse",
    9: "Boisson gazeuse",
    10: "Boisson gazeuse existe en plusieurs saveurs",
    11: "Boisson lactée",
    12: "Bonbons",
    13: "Boulettes de dinde",
    14: "Breadsticks Italiens",
    15: "Brosse à dents",
    16: "Brosses à dents",
    17: "Brownies chocolat",
    18: "Bâtonnets de pain au sésame",
    19: "Café soluble",
    20: "Cake",
    21: "Cake mini muffins",
    22: "Chamia",
    23: "Chamia aux amandes",
    24: "Chamia nature",
    25: "Chamia à tartiner existe en chocolat et noisette",
    26: "Champignons",
    27: "Champignons pieds",
    28: "Chicorée Café",
    29: "Chips",
    30: "Chocolat instantané",
    31: "Chocoline",
    32: "Cocktail fruits de mer",
    33: "Confiture",
    34: "Confiture d'abricot",
    35: "Cookies",
    36: "Cordon bleu pané",
    37: "Couches bébé",
    38: "Couches-bébé confort jumbo",
    39: "Crème dépilatoire",
    40: "Crevettes Décortiquées",
    41: "Crème de cuisson le squeeze",
    42: "Crème dessert",
    43: "Crème dessert au chocolat Liégeois",
    44: "Crème à tartiner aux noisettes",
    45: "Céréales",
    46: "Céréales",
    47: "Déodorant",
    48: "Déodorant SOUPLESSE",
    49: "Désodorisant existe en plusieurs senteurs",
    50: "Dentifrice",
    51: "Dentifrice Lilas EXTREME",
    52: "DÉLICE",
    53: "Eau de javel",
    54: "Eponges abrasives",
    55: "Escalope panée",
    56: "Essuie-tout",
    57: "Essuie-tout blanc",
    58: "Fromage EDAM",
    59: "Fromage en tranches",
    60: "Fromage fondu",
    61: "Fromage fondu râpé",
    62: "Fromage mozzarella pasta filata",
    63: "Fromage râpé",
    64: "Fromage râpé à l'emmental",
    65: "Fromage à tartiner Fraidoux",
    66: "GEL COIFFANT EFFET",
    67: "Gaufrettes",
    68: "Gel Machine",
    69: "Gel douche",
    70: "Gel douche",
    71: "Gel lavant mains",
    72: "Gel machine",
    73: "Glace",
    74: "Génoise",
    75: "Harissa",
    76: "Huile de maïs",
    77: "Huile végétale",
    78: "Jambon de dinde",
    79: "Jus orange",
    80: "Kabeb de dinde",
    81: "Ketchup",
    82: "Kit de coloration",
    83: "Lessive liquide",
    84: "Lessive poudre machine",
    85: "Lessive poudre machine",
    86: "Levure chimique",
    87: "Levure sèche instantanée",
    88: "Levure à gâteaux",
    89: "Lilas SUPREME",
    90: "Lingettes bébé",
    91: "Liquide vaisselle",
    92: "Liquide vaisselle existe en plusieurs formules",
    93: "Margarine",
    94: "Mayonnaise",
    95: "Miel",
    96: "Mozza pizza le boyau",
    97: "Nettoyant WC",
    98: "Nettoyant liquide de rinçage",
    99: "Nettoyant multi-usages",
    100: "Nettoyant vaisselle",
    101: "Papier hygiénique",
    102: "Penne rigate",
    103: "Préparation pour boisson instantanée existe en plusieurs saveurs",
    104: "Protège slip",
    105: "Préparation cake chocolat vanille",
    106: "Préparation chocolat chaud",
    107: "Préparation pour Muffin's",
    108: "Préparation pour cake sorgho",
    109: "Préparation pour fondant au chocolat",
    110: "Rasoirs jetables",
    111: "Ricotta",
    112: "SAVON DE MÉNAGE",
    113: "Salami dinde extra",
    114: "Salami famille",
    115: "Savon",
    116: "Savon de ménage",
    117: "Seiches nettoyées",
    118: "Serviettes féminines",
    119: "Serviettes hygiéniques",
    120: "Shampooing",
    121: "Soft Vaseline",
    122: "Spécialité laitière brassée",
    123: "Steak de dinde",
    124: "Sucre d'épilation",
    125: "Sucre vanilliné",
    126: "Tartelette",
    127: "Thon entier à l'huile d'olive",
    128: "Thon entier à l'huile végétale",
    129: "Thon entier à l'huile végétale",
    130: "Thé noir soluble",
    131: "Tomates pelées Cubées",
    132: "Vinaigre à ménage",
    133: "Yaourt à boire Maxi",
    134: "Yaourt aromatisé",
    135: "Yaourt à boire maxi",
    136: "Boisson en jus existe en plusieurs saveurs",
    137: "Jambon de poulet fumé",
    138: "Sauce pizza",
    139: "Œufs"
}
# Dictionnaire pour convertir les codes en noms de marques
code_to_marque = {
    0: "Afia",
    1: "Aqua",
    2: "Barilla",
    3: "Bimo",
    4: "Boga",
    5: "Brossard",
    6: "Calvé",
    7: "Céréal",
    8: "Chahia",
    9: "Chokini",
    10: "Cristaline",
    11: "Danette",
    12: "Danone",
    13: "Delice",
    14: "Délice",
    15: "Fanta",
    16: "Fraidoux",
    17: "Gringo",
    18: "Harissa Le Phare du Cap Bon",
    19: "Huilor",
    20: "Jebrane",
    21: "Juhayna",
    22: "Juver",
    23: "KitKat",
    24: "Lilas",
    25: "Marsa",
    26: "Maya",
    27: "Mina",
    28: "Mondelez",
    29: "Nestlé",
    30: "Oasis",
    31: "Panzani",
    32: "Président",
    33: "Prima",
    34: "Saida",
    35: "Sicam",
    36: "Sodibo",
    37: "Sopalin",
    38: "Spiga",
    39: "Stil",
    40: "Tanit",
    41: "Tropico",
    42: "Vitamilk",
    43: "Vitalait",
    44: "Zina"
}
encoding_marque = {
    'Marque_X': 0, 'Marque_Y': 1, 'Marque_Z': 2, # exemple, remplace par ton vrai dictionnaire
}


@app.route('/')
def home():
    return render_template('predict1.html')

@app.route('/predict1', methods=['POST'])
def predict():
    try:
        if knn_model is None:
            return render_template('predict1.html', error_message="Le modèle n'est pas disponible")
        
        # Récupérer les données du formulaire
        produit_code =  int(request.form['produit'])

        marque_code = int(request.form['variable2'])
        taille = int(request.form['variable3'])
        quantite = int(request.form['variable4'])
        annee = int(request.form['variable5'])
        mois = int(request.form['variable6'])
        jour = int(request.form['variable7'])

        # Validation des données
        if not (1 <= mois <= 12):
            return render_template('predict1.html', error_message="Mois doit être entre 1 et 12")
        if not (1 <= jour <= 31):
            return render_template('predict1.html', error_message="Jour doit être entre 1 et 31")

        # Préparation des données pour la prédiction
        data = [[produit_code, marque_code, taille, quantite, annee, mois, jour]]
        columns = ['Produit_encoded', 'Marque_encoded', 'Taille_encoded', 
                  'Quantite_Achetee', 'Annee', 'Mois', 'Jour']
        df = pd.DataFrame(data, columns=columns)

        # Faire la prédiction
        reduction = knn_model.predict(df)[0]

        # Récupérer le nom du produit
        nom_produit = code_to_produit.get(produit_code, "Produit inconnu")

        return render_template('predict1.html', 
                            prediction_text=f'Pour le produit "{nom_produit}", la réduction prédite est de : {reduction:.2f} %')

    except ValueError:
        return render_template('predict1.html', error_message="Veuillez entrer des valeurs numériques valides")
    except Exception as e:
        return render_template('predict1.html', error_message=f"Une erreur est survenue: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)