# INF8801A - Projet
# Prédiction de style de musique avec réseaux de neurones
Par Louis Harris, Ilyas Rahhali et Alexandre Richard

Article de référence : "Learning features from music audio with deep belief networks" (Philippe Hamel et Douglas Eck, 2010)

## Fichiers
- extract_data.py 	: Extraction des features et création d'un fichier de données DATA.data contenant les valeurs et les étiquettes des données de la base GTZAN
- DBN+MLP.ipynb 	: Notebook Jupyter avec l'implémentation du réseau DBN de l'article et implémentation d'un réseau pleinement connecté + tests
- lstm.py 			: Implémentation et test d'un réseau LSTM pour la prédiction des genres musicaux. Modifié à partir de https://github.com/ruohoruotsi/LSTM-Music-Genre-Classification

## Requis
- Python 3
- numpy
- matplotlib
- pytorch
- scikit-learn
- librosa
- pickle

## Execution
- Extraire les features des données à l'aide du code extract_data.py
- Test du modèle DBN et MLP ("Multi-layer perceptron") avec DBN+MLP.ipynb
- Test du modèle LSTM avec lstm.py
