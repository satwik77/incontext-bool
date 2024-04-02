import os
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config,  ReformerModel, ReformerConfig, NystromformerModel, NystromformerConfig
# from transformers import LlamaModel

from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Lasso
import warnings
from sklearn import tree
import xgboost as xgb
from src.bool_tasks import cross_entropy
from src.baselines import NNModel, NNCosineModel, LeastSquaresModel, AveragingModel, LassoModel, DecisionTreeModel, XGBoostModel, NullClassifier, GDModel
from src.components.transformer import TransformerCLF
from src.components.retnet import RetnetModel
from src.components.hyena import HyenaModel
from src.components.dss import DSSModel
import pdb

from src.base_models import NeuralNetwork, ParallelNetworks

def build_model(conf):
	if conf.family == "san":
		model = TransformerModel(
			n_dims=conf.n_dims,
			n_positions=conf.n_positions,
			n_embd=conf.n_embd,
			n_layer=conf.n_layer,
			n_head=conf.n_head,
			freeze = conf.freeze,
		)
	elif conf.family == "mysan":
		model = TransformerCLF(
			n_dims=conf.n_dims,			
			d_model=conf.n_embd,
			n_layer=conf.n_layer,
			n_head=conf.n_head,
		)

	elif conf.family == "retnet":
		model = RetnetModel(
			n_dims=conf.n_dims,			
			d_model=conf.n_embd,
			n_layer=conf.n_layer,
			n_head=conf.n_head,
		)

	elif conf.family == "hyena":
		model = HyenaModel(
			n_dims=conf.n_dims,			
			d_model=conf.n_embd,
			n_layer=conf.n_layer,
			order=conf.order,
		)
	
	elif conf.family == "dss":
		model = DSSModel(
			n_dims=conf.n_dims,			
			d_model=conf.n_embd,
			n_layer=conf.n_layer,
		)
		
	elif conf.family == "gpt":
		model = GPT(
			n_dims=conf.n_dims,
			model_name = conf.model_name,
			freeze = conf.freeze,
		)
	elif conf.family == "llama":
		model = LLAMA(
			n_dims=conf.n_dims,
			llama_weights_path=conf.llama_weights_path,
			precision=conf.precision,
			model_name = conf.model_name,
			freeze = conf.freeze
		)
	elif conf.family == "lstm" or conf.family == "gru":
		model = LSTM(
			n_dims=conf.n_dims,
			n_embd=conf.n_embd,
			n_layer=conf.n_layer,
			rnn_type= conf.family,
			freeze = conf.freeze,
		)
	else:
		raise NotImplementedError

	return model


def get_relevant_baselines(task_name, n_dims):

	task_to_baselines = {
		"linear_regression": [
			(LeastSquaresModel, {}),
			(NNModel, {"n_neighbors": 3}),
			(AveragingModel, {}),
		],
		"linear_classification": [
			(NNModel, {"n_neighbors": 3}),
			(AveragingModel, {}),
		],
		"sparse_linear_regression": [
			(LeastSquaresModel, {}),
			(NNModel, {"n_neighbors": 3}),
			(AveragingModel, {}),
		]
		+ [(LassoModel, {"alpha": alpha}) for alpha in [1, 0.1, 0.01, 0.001, 0.0001]],
		"relu_2nn_regression": [
			(LeastSquaresModel, {}),
			(NNModel, {"n_neighbors": 3}),
			(AveragingModel, {}),
			(
				GDModel,
				{
					"model_class": NeuralNetwork,
					"model_class_args": {
						"in_size": n_dims,
						"hidden_size": 100,
						"out_size": 1,
					},
					"opt_alg": "adam",
					"batch_size": 100,
					"lr": 5e-3,
					"num_steps": 100,
				},
			),
		],
		"decision_tree": [
			(LeastSquaresModel, {}),
			(NNModel, {"n_neighbors": 3}),
			(DecisionTreeModel, {"max_depth": 4}),
			(DecisionTreeModel, {"max_depth": None}),
			(XGBoostModel, {}),
			(AveragingModel, {}),
		],
		"conjunction": [
			# (NNModel, {"n_neighbors": 3}),
			(NNModel, {"n_neighbors": 1}),
			(AveragingModel, {}),
			(NullClassifier, {}),
			
		] 
		,
		"teach_biconjunction": [
			(NNModel, {"n_neighbors": 3}),
			(AveragingModel, {}),
			(NullClassifier, {}),
		],
		"conjunction_long": [
			(NNModel, {"n_neighbors": 3}),
			(AveragingModel, {}),
			(NullClassifier, {}),
		],
		"mono_conjunction": [
			(NNModel, {"n_neighbors": 3}),
			(AveragingModel, {}),
			(NullClassifier, {}),
		],
		"teach_conjunction": [
			(NNModel, {"n_neighbors": 3}),
			(AveragingModel, {}),
			(NullClassifier, {}),
		]
		,
		"teach_conjunction_long": [
			(NNModel, {"n_neighbors": 3}),
			(AveragingModel, {}),
			(NullClassifier, {}),
		],
		"disjunction": [
			# (NNModel, {"n_neighbors": 3}),
			(NNModel, {"n_neighbors": 1}),
			(AveragingModel, {}),
			(NullClassifier, {}),
			
		],
		"nearest_neighbours": [
			(NNModel, {"n_neighbors": 1}),
			(NNCosineModel, {}),
			(AveragingModel, {}),
			(NullClassifier, {}),
			
		],
		'sparse_thres':[
			(NNModel, {"n_neighbors": 3}),
			(AveragingModel, {}),
			(NullClassifier, {}),
			
		],
		"parity": [
			(NNModel, {"n_neighbors": 3}),
			(AveragingModel, {}),
			(NullClassifier, {}),
		],
		"sparse_parity": [
			(NNModel, {"n_neighbors": 3}),
			(AveragingModel, {}),
			(NullClassifier, {}),
		] 
		,
		 "sparse_halfspace": [
			(NNModel, {"n_neighbors": 3}),
			(AveragingModel, {}),
			(NullClassifier, {}),
			
		]
		,
		"halfspace": [
			(NNModel, {"n_neighbors": 3}),
			(AveragingModel, {}),
			(NullClassifier, {}),
			
		],
		"int_halfspace": [
			(NNModel, {"n_neighbors": 3}),
			(AveragingModel, {}),
			(NullClassifier, {}),
		],
		"majority": [
			# (NNModel, {"n_neighbors": 3}),
			(NNModel, {"n_neighbors": 1}),
			(AveragingModel, {}),
			(NullClassifier, {}),
		],
		"full_majority": [
			(NNModel, {"n_neighbors": 3}),
			(AveragingModel, {}),
			(NullClassifier, {}),
		],
		"dnf": [
			(NNModel, {"n_neighbors": 3}),
			(AveragingModel, {}),
			(NullClassifier, {}),
		],
		"teach_dnf": [
			(NNModel, {"n_neighbors": 1}),
			(AveragingModel, {}),
			(NullClassifier, {}),
		],
		"cnf": [
			(NNModel, {"n_neighbors": 3}),
			(AveragingModel, {}),
			(NullClassifier, {}),
		],
		"dictator": [
			(NNModel, {"n_neighbors": 3}),
			(AveragingModel, {}),
			(NullClassifier, {}),
		],
	
	}
	if task_name not in task_to_baselines:
		base = [
			(NNModel, {"n_neighbors": 3}),
			(AveragingModel, {}),
			(NullClassifier, {}),
		]
	else:
		base = task_to_baselines[task_name]

	models = [model_cls(**kwargs) for model_cls, kwargs in base]
	return models


class TransformerModel(nn.Module):
	def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, freeze = 0):
		super(TransformerModel, self).__init__()
		configuration = GPT2Config(
			n_positions=2 * n_positions,
			n_embd=n_embd,
			n_layer=n_layer,
			n_head=n_head,
			resid_pdrop=0.0,
			embd_pdrop=0.0,
			attn_pdrop=0.0,
			use_cache=False,
		)
		self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

		self.n_positions = n_positions
		self.n_dims = n_dims
		self.n_embd = n_embd
		self._read_in = nn.Linear(n_dims, n_embd)
		self._backbone = GPT2Model(configuration)
		self._read_out = nn.Linear(n_embd, 1)

		if freeze >0:
			self._read_in = NeuralNetwork(n_dims, 256, self.n_embd)
			self._read_out = NeuralNetwork(self.n_embd, 256, 1)
			for param in self._backbone.parameters():
				if param.dim() > 1:
					torch.nn.init.xavier_uniform_(param)

			if freeze == 2:
				for param in self._backbone.parameters():
					param.requires_grad = False
			
				print('Froze all Transformer attention, layernorm and MLP parameters')

			elif freeze == 1:
				layers= len(self._backbone.h)
				for i in range(layers):
					block = self._backbone.h[i]
					for param in block.attn.parameters():
						param.requires_grad = False
					
					for param in block.mlp.parameters():
						param.requires_grad = False
				print('Froze all Transformer attention and MLP parameters')
				print('All LayerNorms are tunable')

			
			print('Tunable FFNs at the beginning and end')
			# print('Tunable Linear classifiers at the beginning and end')

		else:
			print('Transformer Normal Training: All parameters are tunable')
				

	@staticmethod
	def _combine(xs_b, ys_b):
		"""Interleaves the x's and the y's into a single sequence."""
		bsize, points, dim = xs_b.shape
		ys_b_wide = torch.cat(
			(
				ys_b.view(bsize, points, 1),
				torch.zeros(bsize, points, dim - 1, device=ys_b.device),
			),
			axis=2,
		)
		zs = torch.stack((xs_b, ys_b_wide), dim=2)
		zs = zs.view(bsize, 2 * points, dim)
		return zs

	def forward(self, xs, ys, inds=None):
		if inds is None:
			inds = torch.arange(ys.shape[1])
		else:
			inds = torch.tensor(inds)
			if max(inds) >= ys.shape[1] or min(inds) < 0:
				raise ValueError("inds contain indices where xs and ys are not defined")
		zs = self._combine(xs, ys)        
		embeds = self._read_in(zs)
		output = self._backbone(inputs_embeds=embeds).last_hidden_state
		prediction = self._read_out(output)
		return prediction[:, ::2, 0][:, inds]  # predict only on xs
	
	def get_attns(self, xs, ys, inds=None):
		if inds is None:
			inds = torch.arange(ys.shape[1])
		else:
			inds = torch.tensor(inds)
			if max(inds) >= ys.shape[1] or min(inds) < 0:
				raise ValueError("inds contain indices where xs and ys are not defined")
		zs = self._combine(xs, ys)        
		embeds = self._read_in(zs)
		attns = self._backbone(inputs_embeds=embeds, output_attentions=True).attentions
		return attns



class LSTM(nn.Module):
	def __init__(self, n_dims, n_embd=128, n_layer=2, rnn_type= 'lstm', freeze = 0):
		super(LSTM, self).__init__()
		self.name = f"embd={n_embd}_layer={n_layer}"

		self.drop = nn.Dropout(0.0)
		self.rnn_type = rnn_type
		self.n_dims = n_dims
		self.n_embd = n_embd
		self.n_layer = n_layer
		

		# self._read_in = NeuralNetwork(n_dims, 256, self.n_embd)
		if self.rnn_type.lower() == 'lstm':
			self._backbone = nn.LSTM(n_embd, n_embd, n_layer)
		elif self.rnn_type.lower() == 'gru':
			self._backbone = nn.GRU(n_embd, n_embd, n_layer)

		
		

		if freeze > 0:
			self._read_in = NeuralNetwork(n_dims, 256, self.n_embd)
			self._read_out = NeuralNetwork(self.n_embd, 256, 1)
			for param in self._backbone.parameters():
				if param.dim() > 1:
					torch.nn.init.xavier_uniform_(param)

			for param in self._backbone.parameters():
				param.requires_grad = False
			print('Froze all LSTM parameters')
		
		else:
			self._read_in = nn.Linear(n_dims, n_embd)
			self._read_out = nn.Linear(n_embd, 1)
			print('All {} parameters are tunable'.format(self.rnn_type))

	@staticmethod
	def _combine(xs_b, ys_b):
		"""Interleaves the x's and the y's into a single sequence."""
		bsize, points, dim = xs_b.shape
		ys_b_wide = torch.cat(
			(
				ys_b.view(bsize, points, 1),
				torch.zeros(bsize, points, dim - 1, device=ys_b.device),
			),
			axis=2,
		)
		zs = torch.stack((xs_b, ys_b_wide), dim=2)
		zs = zs.view(bsize, 2 * points, dim)
		return zs

	def forward(self, xs, ys, hidden=None, inds=None):
		if inds is None:
			inds = torch.arange(ys.shape[1])
		else:
			inds = torch.tensor(inds)
			if max(inds) >= ys.shape[1] or min(inds) < 0:
				raise ValueError("inds contain indices where xs and ys are not defined")
		
		if hidden is None:
			hidden = self.init_hidden(xs.size(0))
		
		zs = self._combine(xs, ys)
		lengths = torch.tensor([zs.size(1)] * zs.size(0))
		embeds = self._read_in(zs)
		embeds = embeds.transpose(0, 1)

		# emb_packed = nn.utils.rnn.pack_padded_sequence(embeds, lengths, enforce_sorted=False)
		# output_packed, hidden = self._backbone(emb_packed, hidden)
		output, hidden = self._backbone(embeds, hidden)
		# output, _ = nn.utils.rnn.pad_packed_sequence(output_packed)
		output = self.drop(output)
		output = output.transpose(0, 1)
		prediction = self._read_out(output)

		return prediction[:, ::2, 0][:, inds]  # predict only on xs
	
	
	def init_hidden(self, bsz):
		weight = next(self.parameters())
		if self.rnn_type.lower() == 'lstm':
			return (weight.new_zeros(self.n_layer, bsz, self.n_embd),
					weight.new_zeros(self.n_layer, bsz, self.n_embd))
		else:
			return weight.new_zeros(self.n_layer, bsz, self.n_embd)







class GPT(nn.Module):
	def __init__(self, n_dims, model_name = "gpt2-large", freeze= 0):
		super(GPT, self).__init__()
		
		self.name = model_name

		if 'gpt2' in model_name:
			# self._backbone = GPT2Model.from_pretrained(model_name)
			self._backbone = GPT2Model.from_pretrained(model_name, cache_dir = '/home/')
			self.n_embd = self._backbone.config.n_embd

		# Load GPT-J from HuggingFace on multiple GPUs for training
		elif 'bert' in model_name:
			self._backbone = AutoModel.from_pretrained('bert-base-cased', cache_dir = '/cache/')
			self.n_embd = self._backbone.config.hidden_size
			self._backbone.config.is_decoder = True
		
		elif 'opt' in model_name:
			self._backbone = OPTModel.from_pretrained('facebook/opt-1.3b', cache_dir = '/cache/')
			self.n_embd = self._backbone.config.hidden_size
			
		self.n_dims = n_dims
		
		self._read_in = nn.Linear(n_dims, self.n_embd)
		# self._read_in = NeuralNetwork(n_dims, 256, self.n_embd)
		# self._read_out = nn.Linear(self.n_embd, 1)
		self._read_out = NeuralNetwork(self.n_embd, 256, 1)

		if freeze >0:
			self._read_in = NeuralNetwork(n_dims, 256, self.n_embd)
			self._read_out = NeuralNetwork(self.n_embd, 256, 1)

			if freeze == 2:
				for param in self._backbone.parameters():
					param.requires_grad = False
			
				print('Froze all {} attention, layernorm and MLP parameters'.format(model_name))

			elif freeze == 1:
				if 'gpt' in model_name:
					layers= len(self._backbone.h)
					for i in range(layers):
						block = self._backbone.h[i]
						for param in block.attn.parameters():
							param.requires_grad = False
						
						for param in block.mlp.parameters():
							param.requires_grad = False
					print('Froze all {} attention and MLP parameters'.format(model_name))
					print('All LayerNorms are tunable')

			
			print('Tunable FFNs at the beginning and end')
			# print('Tunable linear and FFN at the beginning and end')

		else:
			print('{} Normal Training: All parameters are tunable'.format(model_name))


	

	@staticmethod
	def _combine(xs_b, ys_b):
		"""Interleaves the x's and the y's into a single sequence."""
		bsize, points, dim = xs_b.shape
		ys_b_wide = torch.cat(
			(
				ys_b.view(bsize, points, 1),
				torch.zeros(bsize, points, dim - 1, device=ys_b.device),
			),
			axis=2,
		)
		zs = torch.stack((xs_b, ys_b_wide), dim=2)
		zs = zs.view(bsize, 2 * points, dim)
		return zs

	def forward(self, xs, ys, inds=None):
		if inds is None:
			inds = torch.arange(ys.shape[1])
		else:
			inds = torch.tensor(inds)
			if max(inds) >= ys.shape[1] or min(inds) < 0:
				raise ValueError("inds contain indices where xs and ys are not defined")
		zs = self._combine(xs, ys)        
		embeds = self._read_in(zs)
		output = self._backbone(inputs_embeds=embeds).last_hidden_state
		prediction = self._read_out(output)
		return prediction[:, ::2, 0][:, inds]  # predict only on xs

	def get_attns(self, xs, ys, inds=None):
		if inds is None:
			inds = torch.arange(ys.shape[1])
		else:
			inds = torch.tensor(inds)
			if max(inds) >= ys.shape[1] or min(inds) < 0:
				raise ValueError("inds contain indices where xs and ys are not defined")
		zs = self._combine(xs, ys)        
		embeds = self._read_in(zs)
		attns = self._backbone(inputs_embeds=embeds, output_attentions=True).attentions
		return attns