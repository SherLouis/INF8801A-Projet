import numpy as np
import librosa
import os
import pickle
"""
Lecture des chansons de la base de données GTZAN, calcul des features et
enregistrement des données et des étiquettes dans un fichier de données
DATA.data
Les données GTZAN (répertoire "genres") doit se situer dans le même répertoire
que le présent code. 
Lien pour télécharger la base de données : http://opihi.cs.uvic.ca/sound/genres.tar.gz
"""

dir_path = os.path.dirname(os.getcwd())
print("Current directory is : " + dir_path)

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
""" Changer les valeurs ici pour modifier la façon dont les chansons sont séparés"""
hop_length = 1024 # Taille de chaque segment
n_by_song = (22050 * 30) // hop_length #Nombre de segments par chason de 30 secondes
n_fft = hop_length//2 +1
data = np.zeros((10, 100, n_by_song, n_fft), dtype=np.float32)  # 10 genres, 100 chansons par genre, 645 bouts par chanson, 513 valeurs par bout
labels = np.zeros((10 * 100 * n_by_song, 1), dtype=np.uint8)

i = 0
for g in genres: # pour chaqeu genre
    song_number = 0
    for filename in os.listdir(dir_path + "\genres\\" + g):
	    # pour chaque chanson de ce genre
        songname = dir_path + "\genres\\" + g + "\\" + filename
	    # Lecture de la chanson
        y, sr = librosa.load(songname, mono=True, duration=30)
	    # Division en segments
        splited = [y[i:min(len(y),i + hop_length)] for i in range(0, len(y), hop_length)]
        splited = splited[0:len(splited) - 1]
        for sub_song in range(0, len(splited)):
	        # pour chaque segment, calculer la DFT, la valeur absolue et ne conserver que la moitié
            dft = abs(np.fft.fft(splited[sub_song]))
            subset = dft[0:n_fft]
            data[genres.index(g), song_number, sub_song, :] = subset
            labels[i,:] = genres.index(g)
            i += 1
        song_number += 1

# Enregistrer ces données
filehandler = open("DATA.data", 'wb')
pickle.dump(labels, filehandler)
pickle.dump(data, filehandler)
filehandler.close()