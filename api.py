from flask import Flask, request, jsonify
import joblib
import pandas as pd
from pathlib import Path
import git
app = Flask(__name__)

@app.route('/update_server', methods=['POST'])
def webhook():
    if request.method == 'POST':
        repo = git.Repo('path/to/git_repo')
        origin = repo.remotes.origin
    origin.pull()
    return 'Updated PythonAnywhere successfully', 200


# Obtenir le chemin complet du répertoire actuel
THIS_FOLDER = Path(__file__).parent.resolve()

# Charger le modèle entraîné
model_path = THIS_FOLDER / 'model_optimized.pkl'
model = joblib.load(model_path)

# Charger les données d'ID
id_data_path = THIS_FOLDER / 'id_data10.csv'
data = pd.read_csv(id_data_path)

# Charger X_train
xtrain_path = THIS_FOLDER / 'xtrain10.csv'
X_train = pd.read_csv(xtrain_path)

# Fusionner X_train avec les données d'ID
X_train = pd.merge(X_train, data[['SK_ID_CURR']], left_index=True, right_index=True)

shap_values_path = THIS_FOLDER / 'shap_valuesxz.pkl'

shap_values = joblib.load(shap_values_path)


# Récupérer les noms des fonctionnalités
features = X_train.columns.tolist()




@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer l'ID client à partir des données JSON de la requête
        id_client = request.json['id_client']
        print('ID client:', id_client)

        # Trouver les données du client correspondant à l'ID
        client_data = X_train[X_train['SK_ID_CURR'] == id_client]
        print('Données du client:', client_data)

        if client_data.empty:
            return jsonify({'error': 'ID client non trouvé'})

        # Supprimer la colonne 'SK_ID_CURR' avant de faire la prédiction
        client_data = client_data.drop('SK_ID_CURR', axis=1)

        # Prédire les probabilités avec le modèle
        probabilities = model.predict_proba(client_data)
        print('Probabilités:', probabilities)

        # Retourner les probabilités sous forme JSON
        return jsonify({'probabilities': probabilities.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/ids', methods=['GET'])
def get_ids():
    try:
        ids = data['SK_ID_CURR'].tolist()  # Obtenez la liste des ID clients à partir du dataframe des données
        return jsonify({'ids': ids})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/data', methods=['POST'])
def get_data():
    data = request.json
    selected_id = data['id_client']

    # Récupérer les valeurs SHAP pour le client sélectionné
    selected_shap_values = shap_values[selected_id]

    # Créer un dictionnaire contenant les valeurs SHAP pour chaque fonctionnalité
    shap_data = {feature: shap_value for feature, shap_value in zip(features, selected_shap_values)}

    return jsonify(shap_data)

@app.route('/all_data', methods=['POST'])
def get_all_data():
    try:
        # Récupérer les valeurs SHAP de tous les clients
        all_shap_values = shap_values.values()

        # Créer un dictionnaire contenant les fonctionnalités et la distribution totale
        distribution_data = {'Fonctionnalités': features, 'Distribution totale': all_shap_values}

        return jsonify(distribution_data)
    except Exception as e:
        return jsonify({'error': str(e)})
    


if __name__ == '__main__':
    app.run()