import DBN
import numpy as np
import sklearn as skl
from sklearn.model_selection import train_test_split
import librosa
import os
import pickle

# Lecture des données
if(os.path.exists("..\\DATA.data")):
	filehandler = open("..\\DATA.data", 'rb')
	labels = pickle.load(filehandler)
	data = pickle.load(filehandler)
	filehandler.close()

else:
	dir_path = os.path.dirname(os.getcwd())
	print("Current directory is : " + dir_path)

	data = np.zeros((10,100,645,513), dtype=np.uint8) # 10 genres, 100 chansons par genre, 645 bouts par chanson, 513 valeurs par bout
	labels = np.zeros((10*100*645,1), dtype=np.float32)
	genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
	hop_length = 1024
	i = 0;
	for g in genres:
	    song_number = 0
	    for filename in os.listdir(dir_path + "\genres\\" + g):
	        songname = dir_path + "\genres\\" + g + "\\" + filename
	        y, sr = librosa.load(songname, mono=True, duration=30)
	        splited = [y[i:i + 1024] for i in range(0, len(y), 1024)]
	        splited = splited[0:len(splited) - 1]
	        for sub_song in range(0, len(splited)):
	            dft = abs(np.fft.fft(splited[sub_song]))
	            subset = dft[0:513]
	            data[genres.index(g), song_number, sub_song, :] = subset
	            labels[i,:] = genres.index(g)
	            i += 1
	        song_number += 1

	# Enregistrer ces données pour ne pas être obligé à tout recalculer lorsqu'on fait des tests
	filehandler = open("..\\DATA.data", 'wb')
	pickle.dump(labels, filehandler)
	pickle.dump(data, filehandler)
	filehandler.close()

data_DBN = data.reshape((-1,513))
# Division en ensembles de données et création des tenseurs
# Division pour avoir autant de données de chaque style
X_train, X_test, y_train, y_test = train_test_split(data_DBN, labels, test_size = 0.5, random_state = 0, shuffle = True)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.4, random_state = 0, shuffle = True)



# Création du DBN


# Entrainement du DBN


# Extraction des caractéristiques
# TODO : Échantillonage du training set pour obtenir 10 000


# TODO: Rouler le réseau DBN sur chaque segment de la base de données d'entrainement et extraire les activations
def run_and_extract_DBN(DBN, input, layers2extract=[3]):
	"""
	Roule le réseau DBN sur les données passées et extrait les activations des couches spécifiées

	Paramètres:
	--------
	DBN : Réseau DBN entrainé (de la classe DBN)
	input : Tenseur - données d'entrée à donner au réseau
	layers2extract : liste des numéros de couche du DBN où les activations doivent être extraites NON-IMPLÉMENTÉ
	"""

	# Rouler le réseau DBN
	_, activation = DBN.forward(input)
	# TODO : extraire les activations de la couche cachée spécifiée

	return activation



# Agréagation par segments de 5s
# TODO: faire l'agrégation par bouts de 5s de chaque chanson
# Chaque chanson = 30s @ 22050Hz -> 661 500 échantillons par segments de 1024 -> 646 morceaux
# 5s = environ 108 morceaux. Overlap de 2.5s -> par bons de 54 morceaux
labels = np.array([])
data = np.array([])
dbn = DBN()

nb_units_layer = 50
fc = 22050 # Fréquence d'échantillonage

# Calcul des activations pour chaque échantillon
activations = np.zeros((np.floor(len(data)/(5*fc)), nb_units_layer))
for i in range(0, len(data)):
	activations[i, :] = run_and_extract_DBN(dbn, data[i, :])

# Par bouts de 2.5s -> 54 morceaux
step = np.ceil(2.5 * fc)
step5 = np.ceil(5 * fc)

features = np.zeros((np.floor((len(data)-step)/step), nb_units_layer))
labs = np.zeros((np.floor((len(data)-step)/step), 1))
for i in range(features.shape[1]):
	features[i, :] = np.mean(activations[step*i:step*i+step5, :], axis=0) # moyenne par colonne (par feature)
	labs[i, :] = labels[step*i]

# Entrainement du classifieur SVM
svmclf = skl.svm.SVC(kernel='rbf')
print('Fitting the SVM classifier do the training data')
svmclf.fit(features, labs)


# Test et mesures des performances
# TODO : Rouler le classifieur sur l'ensemble de test
# TODO : validation croisée
test_data = np.array([])
test_labs = np.array([])

predictions = svmclf.predict(test_data)

acc = np.sum(predictions==test_labs) / len(test_labs)
print("Accuracy of SVM classification on test set : %2.2f %".format(acc*100))

# TODO: Mesurer la précision de la classification par le DBN seulement