#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Importation des librairies nécéssaires pour LSTM
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
import sklearn as skl
from sklearn.model_selection import train_test_split

#Importation des données préalablement formatées
filehandler = open("DATA_1024.data", 'rb')
print("Loading data")
#Les étiquettes des fragments de chanson
labels = pickle.load(filehandler)
#Les fragments de chanson
data = pickle.load(filehandler)
filehandler.close()

#Transformation de la matrice de donnée 
data = data.reshape((-1,513));
#Création de la mémoire (10 seconde) du LTSM
data = data.reshape((-1,215,513));
labels = labels[np.arange(0,len(labels),215)];

#Mélange de l'ordre des données
order = np.random.permutation(3000);
data = data[order,:,:];
labels = labels[order];

#Création de la matrice one-hot pour les étiquettes
n_values = np.max(labels) + 1
labels = np.squeeze(np.eye(n_values)[labels.reshape(-1)])

#Affichage des matrices de donné et d'étiquette
print("Data shape: " + str(data.shape))
print("Labels shape: " + str(labels.shape))

#Séparation des données (X) et étiquettes (y) en différents ensemble : ensemble d'entraînement (50%), ensemble de validation (20%) et ensemble de test (30%) 
print("Spliting the data")
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.5, random_state = 0, shuffle = False)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.4, random_state = 0, shuffle = False)

#Conversions en tenseurs pour la librairie torch
X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_train = y_train.squeeze()
y_test = torch.from_numpy(y_test)
y_test = y_test.squeeze()
X_val = torch.from_numpy(X_val)
y_val = torch.from_numpy(y_val)
y_val = y_val.squeeze()

#Affichage des dimentions des différents ensembles de donnée et d'étiquette
print("Training X shape: " + str(X_train.shape))
print("Training Y shape: " + str(y_train.shape))
print("Validation X shape: " + str(X_val.shape))
print("Validation Y shape: " + str(y_val.shape))
print("Test X shape: " + str(X_test.shape))
print("Test Y shape: " + str(y_test.shape))

# class LSTM extraite du dépôt github : https://github.com/ruohoruotsi/LSTM-Music-Genre-Classification
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=8, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # setup LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # setup output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        return (
            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
        )

    def forward(self, input):
        # lstm step => then ONLY take the sequence's final timetep to pass into the linear/dense layer
        # Note: lstm_out contains outputs for every step of the sequence we are looping over (for BPTT)
        # but we just need the output of the last step of the sequence, aka lstm_out[-1]
        lstm_out, hidden = self.lstm(input)
        logits = self.linear(lstm_out[-1])
        genre_scores = F.log_softmax(logits, dim=1)
        return genre_scores

    def get_accuracy(self, logits, target):
        """ compute accuracy for training round """
        corrects = (
            torch.max(logits, 1)[1].view(target.size()).data == target.data
        ).sum()
        accuracy = 100.0 * corrects / self.batch_size
        return accuracy.item()

#Nombre de donnée par group d'entraînement
batch_size = 43
#Nombre d'époque pour l'entraînement
num_epochs = 100

#Définition des dimensions réseau de neurone
print("Build LSTM RNN model ...")
model = LSTM(
    input_dim=513, hidden_dim=215, batch_size=batch_size, output_dim=10, num_layers=2
)
#Fonction de coût pour l'entraînement
loss_function = nn.NLLLoss()  

#Choix du "learning rate" de 0.0005
optimizer = optim.Adam(model.parameters(), lr=0.0005)

#Vérification si c'est possible de faire l'entraînement sur GPU à l'aide de cuda
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print("\nTraining on GPU")
else:
    print("\nNo GPU, training on CPU")

#Nombre de groupe d'entraînement total (35)
num_batches = int(X_train.shape[0] / batch_size)
#Nombre de groupe de validation total (13)
num_dev_batches = int(X_val.shape[0] / batch_size)
#Nombre de groupe de test total (20)
num_test_batches = int(X_test.shape[0] / batch_size)

#Initialisation des listes pour l'affichage des graphiques 
train_loss_list, train_accuracy_list, epoch_list = [], [], []
val_loss_list, val_accuracy_list, val_epoch_list = [], [], []

#Début de l'entrainement
print("Training ...")
for epoch in range(num_epochs):

    #Initialisation du coût et de la précision d'entraînement pour cette époque
    train_running_loss, train_acc = 0.0, 0.0

    #Initialisation de l'état caché
    model.hidden = model.init_hidden()
    #Boucle sur les différents groupes d'entraînement
    for i in range(num_batches):

        #Réinitialisation du gradient
        model.zero_grad()

        #Attribution des données et des étiquettes d'entraînement pour le groupe courant
        X_local_minibatch, y_local_minibatch = (
            X_train[i * batch_size : (i + 1) * batch_size,],
            y_train[i * batch_size : (i + 1) * batch_size,],
        )

        #Rotation de la matrice de donné pour la fonction de coût
        X_local_minibatch = X_local_minibatch.permute(1, 0, 2)
        #Transformation des étiquettes du format one-hot à des classes avec indice
        y_local_minibatch = torch.max(y_local_minibatch, 1)[1]

        #Ajout des données du nouveau groupe (forward pass)
        y_pred = model(X_local_minibatch)
        #Calcul du coût du group courant
        loss = loss_function(y_pred, y_local_minibatch)
        #Retour au coût précédent
        loss.backward()
        #Mise à jour des paramètres
        optimizer.step()

        #Calcul du coût d'entraînement total pour cet époque
        train_running_loss += loss.detach().item()
        #Calcul de la précision d'entraînement total pour cet époque
        train_acc += model.get_accuracy(y_pred, y_local_minibatch)

    #Affichage de l'époque, du coût et de la précision d'entrainement
    print(
        "Epoch:  %d | NLLoss: %.4f | Train Accuracy: %.2f"
        % (epoch, train_running_loss / num_batches, train_acc / num_batches)
    )
    
    #Ajout de l'époque, du coût et de la précision d'entrainement dans les listes pour les graphiques
    epoch_list.append(epoch)
    train_accuracy_list.append(train_acc / num_batches)
    train_loss_list.append(train_running_loss / num_batches)

    #Début de la validation
    print("Validation ...")
    #Calcul du coût et de la précision de validation un époque sur deux
    if epoch % 2 == 0:
        #Initialisation du coût et de la précision de validation pour cette époque
        val_running_loss, val_acc = 0.0, 0.0

        #Utilisation de torch.no_grad() et model.eval() pour calculer le coût et la précision de validation
        with torch.no_grad():
            model.eval()
            #Initialisation de l'état caché
            model.hidden = model.init_hidden()
            #Boucle sur les différents groupes de validation
            for i in range(num_dev_batches):
                #Attribution des données et des étiquettes de validation pour le groupe courant
                X_local_validation_minibatch, y_local_validation_minibatch = (
                    X_val[i * batch_size : (i + 1) * batch_size,],
                    y_val[i * batch_size : (i + 1) * batch_size,],
                )
                #Rotation de la matrice de donné pour la fonction de coût
                X_local_minibatch = X_local_validation_minibatch.permute(1, 0, 2)
                #Transformation des étiquettes du format one-hot à des classes avec indice
                y_local_minibatch = torch.max(y_local_validation_minibatch, 1)[1]

                #Ajout des données du nouveau groupe (forward pass)
                y_pred = model(X_local_minibatch)
                #Calcul du coût du group courant
                val_loss = loss_function(y_pred, y_local_minibatch)

                #Calcul du coût d'entraînement total pour cet époque
                val_running_loss += (val_loss.detach().item())
                #Calcul de la précision d'entraînement total pour cet époque
                val_acc += model.get_accuracy(y_pred, y_local_minibatch)
            
            #Réinitialisation du modèle d'entraînement entre les itérations de validation
            model.train()
            #Affichage de l'époque, du coût et de la précision d'entrainement et de validation
            print(
                "Epoch:  %d | NLLoss: %.4f | Train Accuracy: %.2f | Val Loss %.4f  | Val Accuracy: %.2f"
                % (
                    epoch,
                    train_running_loss / num_batches,
                    train_acc / num_batches,
                    val_running_loss / num_dev_batches,
                    val_acc / num_dev_batches,
                )
            )
            #Ajout de l'époque, du coût et de la précision de validation dans les listes pour les graphiques
            val_epoch_list.append(epoch)
            val_accuracy_list.append(val_acc / num_dev_batches)
            val_loss_list.append(val_running_loss / num_dev_batches)

#Affichage du graphique des coûts d'entraînement et de validation en fonction des époques
plt.figure()
plt.plot(val_epoch_list, val_loss_list)
plt.plot(epoch_list, train_loss_list)
plt.legend(['Validation', 'Training'])
plt.title("Évolution du coût selon l'époque")
plt.xlabel('Époque')
plt.ylabel('Coût')
plt.grid()

#Affichage du graphique des précision d'entraînement et de validation en fonction des époques
plt.figure()
plt.plot(val_epoch_list, val_accuracy_list)
plt.plot(epoch_list, train_accuracy_list)
plt.legend(['Validation', 'Training'])
plt.title("Évolution de la précision selon l'époque")
plt.xlabel('Époque')
plt.ylabel("Précision sur l'ensemble de validation")
plt.grid()

plt.show()


#Initialisation du coût et de la précision de test
test_running_loss, test_acc = 0.0, 0.0
#Utilisation de torch.no_grad() et model.eval() pour calculer le coût et la précision de test
with torch.no_grad():
    model.eval()
    #Initialisation de l'état caché
    model.hidden = model.init_hidden()
    #Boucle sur les différents groupes de test
    for i in range(num_test_batches):
        #Attribution des données et des étiquettes de validation pour le groupe courant
        X_local_test_minibatch, y_local_test_minibatch = (
            X_test[i * batch_size : (i + 1) * batch_size,],
            y_test[i * batch_size : (i + 1) * batch_size,],
        )
        #Rotation de la matrice de donné pour la fonction de coût
        X_local_minibatch = X_local_test_minibatch.permute(1, 0, 2)
        #Transformation des étiquettes du format one-hot à des classes avec indice
        y_local_minibatch = torch.max(y_local_test_minibatch, 1)[1]

        #Ajout des données du nouveau groupe (forward pass)
        y_pred = model(X_local_minibatch)
        #Calcul du coût du group courant
        test_loss = loss_function(y_pred, y_local_minibatch)

        #Calcul du coût de test
        test_running_loss += (val_loss.detach().item())
        #Calcul de la précision de test
        test_acc += model.get_accuracy(y_pred, y_local_minibatch)

    #Réinitialisation du modèle d'entraînement entre les itérations de test
    model.train()
    #Affichage du coût et de la précision de test
    print(
        "Test Loss %.4f  | test Accuracy: %.2f"
        % (
            test_running_loss / num_test_batches,
            test_acc / num_test_batches,
        )
    )