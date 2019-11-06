## Code de https://github.com/mehulrastogi/Deep-Belief-Network-pytorch

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from RBM import RBM


class DBN(nn.Module):
	def __init__(self,
	             visible_units=513,  # Taille de la couche d'entrée
	             hidden_units=[50, 50, 50],  # Couches cachées
	             k=2,
	             learning_rate=1e-5,
	             learning_rate_decay=False,
	             xavier_init=False,
	             increase_to_cd_k=False,
	             use_gpu=False
	             ):
		"""
		:param visible_units: Taille de la couche d'entrée (nombre de noeuds dans la première couche)
		:param hidden_units: Liste du nombre de noeuds dans chaque couche cachée
		:param k: Paramètre k pour l'entrainement des RBM
		:param learning_rate: taux d'apprentissage pour l'entrainement non-supervisé des RBM
		:param learning_rate_decay: True ou False (decay du learing rate ou pas)
		:param xavier_init: True ou False (initialisation xavier ou pas)
		:param increase_to_cd_k: False
		:param use_gpu: True ou False (utiliser GPU si CUDA disponible ou non)
		"""
		super(DBN, self).__init__()

		# Nombre de RBM = nombre de couches cachées
		self.n_layers = len(hidden_units)
		self.rbm_layers = []

		# Création des différents RBMs
		for i in range(self.n_layers):
			if i == 0: # Couche d'entrée
				input_size = visible_units
			else: # Couches cachées
				input_size = hidden_units[i - 1]
			rbm = RBM(visible_units=input_size,
			          hidden_units=hidden_units[i],
			          k=k,
			          learning_rate=learning_rate,
			          learning_rate_decay=learning_rate_decay,
			          xavier_init=xavier_init,
			          increase_to_cd_k=increase_to_cd_k,
			          use_gpu=use_gpu)

			self.rbm_layers.append(rbm)

		# Paramètres des RBMs
		self.W_rec = [nn.Parameter(self.rbm_layers[i].W.data.clone()) for i in range(self.n_layers)]
		self.W_gen = [nn.Parameter(self.rbm_layers[i].W.data) for i in range(self.n_layers)]
		self.bias_rec = [nn.Parameter(self.rbm_layers[i].h_bias.data.clone()) for i in range(self.n_layers)]
		self.bias_gen = [nn.Parameter(self.rbm_layers[i].v_bias.data) for i in range(self.n_layers)]

		# Enregistrement des paramètres dans le module pour pouvoir y accéder par noms
		for i in range(self.n_layers):
			self.register_parameter('W_rec%i' % i, self.W_rec[i])
			self.register_parameter('W_gen%i' % i, self.W_gen[i])
			self.register_parameter('bias_rec%i' % i, self.bias_rec[i])
			self.register_parameter('bias_gen%i' % i, self.bias_gen[i])

	def forward(self, input_data):
		"""
		Faire l'activation avant des couches RBM (forward pass)
		:param input_data: données d'entrée
		:return:
			p_v : probabilités
			v : valeurs (basée sur le sampling de la distribution dans les RBMs)
		"""
		v = input_data
		for i in range(len(self.rbm_layers)):
			v = v.view((v.shape[0], -1)).type(torch.FloatTensor)  # flatten
			p_v, v = self.rbm_layers[i].to_hidden(v)
		return p_v, v

	def reconstruct(self, input_data):
		"""
		Fait le forward pass et reconstruit l'entrée (backward pass)
		:param input_data: données d'entére
		:return:
			p_v : probabilités
			v : données reconstruites
		"""
		h = input_data
		p_h = 0
		for i in range(len(self.rbm_layers)):
			h = h.view((h.shape[0], -1)).type(torch.FloatTensor)  # flatten
			p_h, h = self.rbm_layers[i].to_hidden(h)

		v = h
		for i in range(len(self.rbm_layers) - 1, -1, -1):
			v = v.view((v.shape[0], -1)).type(torch.FloatTensor)
			p_v, v = self.rbm_layers[i].to_visible(v)
		return p_v, v

	def train_static(self, train_data, train_labels, num_epochs=50, batch_size=10):
		'''
        Greedy Layer By Layer training
        Keeping previous layers as static
        '''

		tmp = train_data

		for i in range(len(self.rbm_layers)):
			print("-" * 20)
			print("Training the {} st rbm layer".format(i + 1))

			tensor_x = tmp.type(torch.FloatTensor)  # transform to torch tensors
			tensor_y = train_labels.type(torch.FloatTensor)
			_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)  # create your datset
			_dataloader = torch.utils.data.DataLoader(_dataset, batch_size=batch_size,
			                                          drop_last=True)  # create your dataloader

			self.rbm_layers[i].train(_dataloader, num_epochs, batch_size)
			# print(train_data.shape)
			v = tmp.view((tmp.shape[0], -1)).type(torch.FloatTensor)  # flatten
			p_v, v = self.rbm_layers[i].forward(v)
			tmp = v
			# print(v.shape)
		return

	def train_ith(self, train_data, train_labels, num_epochs, batch_size, ith_layer):
		'''
        taking ith layer at once
        can be used for fine tuning
        '''
		if (ith_layer - 1 > len(self.rbm_layers) or ith_layer <= 0):
			print("Layer index out of range")
			return
		ith_layer = ith_layer - 1
		v = train_data.view((train_data.shape[0], -1)).type(torch.FloatTensor)

		for ith in range(ith_layer):
			p_v, v = self.rbm_layers[ith].forward(v)

		tmp = v
		tensor_x = tmp.type(torch.FloatTensor)  # transform to torch tensors
		tensor_y = train_labels.type(torch.FloatTensor)
		_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)  # create your datset
		_dataloader = torch.utils.data.DataLoader(_dataset, batch_size=batch_size, drop_last=True)
		self.rbm_layers[ith_layer].train(_dataloader, num_epochs, batch_size)
		return
