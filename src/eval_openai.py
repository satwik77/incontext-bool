import json
import os
import sys
import argparse
import uuid
import pdb
import time
import random
import re

from munch import Munch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import yaml

import openai
from tenacity import retry
from tenacity.retry import retry_if_exception_type
from tenacity.wait import wait_random_exponential
import tiktoken

from text_generation import Client

import src.models as models
from src.samplers import get_data_sampler, sample_transformation
from src.bool_tasks_openai import get_task_sampler
from src.components.llm_models import select_model

def build_parser():
	# Data loading parameters
	parser = argparse.ArgumentParser(description='Evaluate OpenAI Models')

	parser.add_argument('-gpu', type=int, default=0, help='Specify the gpu to use')
	parser.add_argument('-run_name', type=str, default='default', help='run name for logs')
	parser.add_argument('-out_dir', type=str, default='models/', help='Output Directory')
	parser.add_argument('-plots_dir', type=str, default='plots/', help='Output Directory')
	parser.add_argument('-task', type=str, default='majority', help='Task')
	parser.add_argument('-data', type=str, default='boolean', choices= ['gaussian', 'boolean'],  help='Data Type')
	parser.add_argument('-increments', type=int, default=5, help='API call after every how many blocks of eg')
	parser.add_argument('-n_dims', type=int, default=24, help='latent dimension')
	parser.add_argument('-n_points', type=int, default=100, help='Number of points')
	parser.add_argument('-num_eval_examples', type=int, default=50, help='Number of examples')
	parser.add_argument('-api_key', type=str, default='<key>', help='Which OpenAI API Key to use')
	parser.add_argument('-prompt_type', type=str, default='basic', help='Which prompt to provide')
	parser.add_argument('-stop', type=str, default=['Input', '\n'], help='When to stop generation')
	parser.add_argument('-model_type', type=str, default='huggingface', choices=['completion', 'chat', 'huggingface', 'tgi'], help='Which type of model to use')
	parser.add_argument('-hf_model_type', type=str, default='causal', choices=['causal', 'llama', 'seq_to_seq'], help='Which type of hf model to use')
	parser.add_argument('-llama_weights_path', type=str, default='llama_weights_hf', help='Outer directory where hf converted LLaMA weights are kept')
	parser.add_argument('-model', type=str, default='gpt2', help='Which model to use')
	parser.add_argument('-port', type=int, default=8080, help='Port on which the model is hosted')
	parser.add_argument('-timeout', type=int, default=1000000, help='Timeout for the model')
	parser.add_argument('-max_tokens', type=int, default=2, help='Maximum number of tokens')
	parser.add_argument('-temperature', type=float, default=0.0, help='Sampling temperature')
	parser.add_argument('-top_p', type=float, default=1.0, help='top what percentage of tokens to be considered') # Alter this or temp, not both
	parser.add_argument('-n', type=int, default=1, help='number of completions to generate for each prompt')
	parser.add_argument('-presence_penalty', type=float, default=0.0, help='positive values increases model\'s likelihood to talk about new topics')
	parser.add_argument('-frequency_penalty', type=float, default=0.0, help='positive values decreases model\'s likelihood to repeat same line verbatim')
	parser.add_argument('-skip_baselines', dest='skip_baselines', action='store_true', help='Skip baselines')
	parser.add_argument('-no-skip_baselines', dest='skip_baselines', action='store_false', help='Not skipping baselines')
	parser.set_defaults(skip_baselines=False)
	parser.add_argument('-only_baselines', dest='only_baselines', action='store_true', help='Only baselines')
	parser.add_argument('-no-only_baselines', dest='only_baselines', action='store_false', help='Not only baselines')
	parser.set_defaults(only_baselines=False)

	return parser


@retry(
	retry=retry_if_exception_type(
			exception_types=(
				openai.error.RateLimitError,  # Rate limit exceeded (20 requests per minute)
				openai.error.APIConnectionError,  # Sometimes we get a connection error
				openai.error.ServiceUnavailableError,
				openai.error.APIError
				# OpenAIAPIError, # Sometimes we get APIError: Internal Error
			)
		),
		wait=wait_random_exponential(
			multiplier=2,
			max=120,
		),
	)
def _get_completion_response(engine, prompt, max_tokens, temperature, top_p, n, stop, presence_penalty, frequency_penalty, best_of, logprobs=1, echo=False):
	fin_prompt = "You are a learning algorithm. " + prompt
	return openai.Completion.create(engine=engine,
		prompt=fin_prompt,
		max_tokens=max_tokens,
		temperature=temperature,
		top_p=top_p,
		n=n,
		stop=stop,
		presence_penalty=presence_penalty,
		frequency_penalty=frequency_penalty,
		best_of=best_of,
		logprobs=logprobs,
		echo=echo
	)

@retry(
	retry=retry_if_exception_type(
			exception_types=(
				openai.error.RateLimitError,  # Rate limit exceeded (20 requests per minute)
				openai.error.APIConnectionError,  # Sometimes we get a connection error
				openai.error.ServiceUnavailableError,
				openai.error.APIError
				# OpenAIAPIError, # Sometimes we get APIError: Internal Error
			)
		),
		wait=wait_random_exponential(
			multiplier=1.5,
			max=5,
		),
	)
def _get_chat_response(engine, prompt, max_tokens, temperature, top_p, n, stop, presence_penalty, frequency_penalty):
	sys_prompt = "You are BoolGPT, a learning algorithm that learns boolean functions from given examples. Only respond with a 0 or 1."
	if "step" in prompt:
		sys_prompt = "You are BoolGPT, a learning algorithm that learns boolean functions from given examples. Answer by thinking step-by-step. Final token of the response must be a 0 or 1."
	return openai.ChatCompletion.create(
		model=engine,
		messages = [
			{"role": "system", "content": sys_prompt},
			{"role": "user", "content": prompt}
		],

		max_tokens=max_tokens,
		temperature=temperature,
		top_p=top_p,
		n=n,
		stop=stop,
		presence_penalty=presence_penalty,
		frequency_penalty=frequency_penalty
	)

def find_last_number(string):
	numbers = re.findall(r'\d+', string)
	if numbers:
		return int(numbers[-1])
	else:
		print('Untenable response Error: ', string.split("\n")[-1])
		return random.choice([0, 1])

def convert_ex(ls, y):
	inp = ""
	for ele in ls:
		if ele <= 0:
			inp = inp + "0 "
		else:
			inp = inp + "1 "
	inp = inp.strip()
	op = ""
	if y<=0:
		op="0"
	else:
		op="1"
	return inp, op

def get_prompt(prompt_type, xs, ys):
	num_ex = len(ys)-1
	if prompt_type == "basic":
		prompt = "You are given some examples of inputs and their corresponding labels. You need to learn the underlying boolean function represented by these input-label examples. Predict the label (either 0 or 1) for the final input.\n"
	elif prompt_type == "instr-mono_conjunction":
		prompt = "You are given some examples of inputs and their corresponding labels. You need to learn the underlying boolean function represented by these input-label examples. Predict the label (either 0 or 1) for the final input.\nThe underlying target function will be a conjunction of some subset of literals. Given a Boolean string as input, a conjunction of some k literals will output 1 if the input values at those k coordinates are 1 and will output 0 otherwise. For instance, for Boolean inputs of dimension 5, one of the conjunctions could be over the 2nd and 4th literal (or coordinate). Such a conjunction will output label 1 for the inputs such as  0 1 0 1 0 or 1 1 0 1 1 since the values at position 2 and 4 are both 1. For inputs such as 0 0 0 1 0 or 0 0 1 0 0, the label will be 0 since at least one of the values at coordinate 2 and 4 is 0.\n"
	elif prompt_type == "pac-mono_conjunction":
		with open("prompts/pac-mono_conjunction.txt", mode="r", encoding="utf-8") as f:
			algo = f.read()
		prompt = algo.strip() + "\n\nApply the above algorithm for learning the monotone conjunction represented by the following input-label examples. Predict the label (either 0 or 1) for the final input.\n"
	elif prompt_type == "pac-step-mono_conjunction":
		with open("prompts/pac-mono_conjunction.txt", mode="r", encoding="utf-8") as f:
			algo = f.read()
		prompt = algo.strip() + "\n\nApply the above algorithm for learning the monotone conjunction represented by the following input-label examples. Think step-by-step and show intermediate hypotheses. You must show how you calculate the outputs from the inputs using the hypothesis. Do not use the final input to modify the hypothesis. Finally, predict the label (either 0 or 1) for the final input as the last token.\n"
	elif prompt_type == "pac-step-mono_conjunction_python":
		with open("prompts/pac-mono_conjunction_python.txt", mode="r", encoding="utf-8") as f:
			algo = f.read()
		prompt = algo.strip() + "\n\nApply the above function for learning the monotone conjunction represented by the following input-label examples. Think step-by-step and show intermediate hypotheses. You must show how you calculate the outputs from the inputs using the hypothesis. Do not use the final input to modify the hypothesis. Finally, predict the label (either 0 or 1) for the final input as the last token.\n"
	for z in range(num_ex):
		inp, lab = convert_ex(xs[z], ys[z])
		prompt = prompt + "\nInput: " + inp + "\nLabel: " + lab
	inp, _ = convert_ex(xs[-1], ys[-1])
	prompt = prompt + "\nInput: " + inp + "\nLabel:"
	return prompt

class TGIModel():
	def __init__(self, port=8080, timeout=1000000):
		self.port = port
		self.timeout = timeout
		self.client = Client("http://127.0.0.1:" + str(port), timeout=timeout)
	def run(self, prompt, temperature=0.0, max_tokens=512, stop = []):
		fin_prompt = prompt
		if temperature > 0.0:
			response = self.client.generate(prompt=fin_prompt, max_new_tokens=max_tokens, temperature=temperature, stop_sequences=stop).generated_text
		else:
			response = self.client.generate(prompt=fin_prompt, do_sample=False, max_new_tokens=max_tokens, stop_sequences=stop).generated_text
		for s in stop:
			response = response.replace(s, '')
		return response

class LargeLanguageModel():
	def __init__(self, prompt_type, model_type, engine, hf_model_type, llama_weights_path, max_tokens, temperature, top_p, n, stop, presence_penalty, frequency_penalty, port, timeout):
		self.name = f"{engine}_prompt-{prompt_type}"

		self.prompt_type = prompt_type
		self.model_type = model_type
		self.engine = engine
		self.hf_model_type = hf_model_type
		self.llama_weights_path = llama_weights_path
		self.max_tokens = max_tokens
		self.temperature = temperature
		self.top_p = top_p
		self.n = n
		self.stop = stop
		self.presence_penalty = presence_penalty
		self.frequency_penalty = frequency_penalty
		self.encoding = None

		if self.model_type == "huggingface":
			self.model = select_model(model_type=self.hf_model_type, model_path=self.engine, llama_weights_path=self.llama_weights_path, max_tokens=self.max_tokens)
			self.model.load()
		elif self.model_type == "tgi":
			self.model = TGIModel(port = port, timeout = timeout)
		else:
			self.encoding = tiktoken.encoding_for_model(engine)

	def predict(self, xs, ys):
		prompt = get_prompt(self.prompt_type, xs, ys)
		fin_responses = []
		unten = 0
		if self.model_type == "completion":
			try:
				response = _get_completion_response(
					engine=self.engine,
					prompt=prompt,
					max_tokens=self.max_tokens,
					temperature=self.temperature,
					top_p=self.top_p,
					n=self.n,
					stop=self.stop,
					presence_penalty=self.presence_penalty,
					frequency_penalty=self.frequency_penalty,
					best_of=self.n+1,
					echo=False
				)
			except Exception as e:
				print('Request error: ', e)
				pdb.set_trace()
			response = response["choices"]
			for res in response:
				if "step" in self.prompt_type:
					pred_lab = find_last_number(res['text'].strip())
				try:
					pred_lab = int(res['text'].strip().split()[0])
					if pred_lab not in [0, 1]:
						raise Exception()
				except Exception as e:
					print('Untenable response Error: ', res['text'])
					pred_lab = random.choice([0, 1])
					unten = 1
				fin_responses.append(pred_lab)
		elif self.model_type == "chat":
			try:
				response = _get_chat_response(
					engine=self.engine,
					prompt=prompt, 
					max_tokens=self.max_tokens,
					temperature=self.temperature,
					top_p=self.top_p,
					n=self.n,
					stop=self.stop,
					presence_penalty=self.presence_penalty,
					frequency_penalty=self.frequency_penalty
				)
			except Exception as e:
				print('Request error: ', e)
				pdb.set_trace()
			response = response["choices"]
			for res in response:
				if "step" in self.prompt_type:
					pred_lab = find_last_number(res['message']['content'].strip())
				else:
					try:
						pred_lab = int(res['message']['content'].strip().split()[0])
						if pred_lab not in [0, 1]:
							raise Exception()
					except Exception as e:
						print('Untenable response Error: ', res['message']['content'])
						pred_lab = random.choice([0, 1])
						unten = 1
				fin_responses.append(pred_lab)
		elif self.model_type == "tgi":
			try:
				response = self.model.run(prompt=prompt, temperature=self.temperature, max_tokens=self.max_tokens, stop=self.stop)
			except Exception as e:
				print('Request error: ', e)
				pdb.set_trace()
			for res in response:
				if "step" in self.prompt_type:
					pred_lab = find_last_number(res.strip())
				else:
					try:
						pred_lab = int(res.strip().split()[0])
						if pred_lab not in [0, 1]:
							raise Exception()
					except Exception as e:
						print('Untenable response Error: ', res.strip().split()[0])
						pred_lab = random.choice([0, 1])
						unten = 1
				fin_responses.append(pred_lab)
		else:
			response = self.model.run(prompt=prompt, temperature=self.temperature, num_return_sequences=self.n)
			for res in response:
				if "step" in self.prompt_type:
					pred_lab = find_last_number(res.strip())
				try:
					pred_lab = int(res.strip().split()[0])
					if pred_lab not in [0, 1]:
						raise Exception()
				except Exception as e:
					print('Untenable response Error: ', res.strip().split()[0])
					pred_lab = random.choice([0, 1])
					unten = 1
				fin_responses.append(pred_lab)

		pred = max(fin_responses, key=fin_responses.count)

		if pred == 0:
			pred = -1

		return pred, prompt, unten

def get_model_from_run(run_path, step=-1, only_conf=False):
	config_path = os.path.join(run_path, "config.yaml")
	with open(config_path) as fp:  # we don't Quinfig it to avoid inherits
		conf = Munch.fromDict(yaml.safe_load(fp))
	if only_conf:
		return None, conf

	model = models.build_model(conf)

	if step == -1:
		state_path = os.path.join(run_path, "state.pt")
		state = torch.load(state_path)
		model.load_state_dict(state["model_state_dict"])
	else:
		model_path = os.path.join(run_path, f"model_{step}.pt")
		state_dict = torch.load(model_path)
		model.load_state_dict(state_dict)

	return model, conf


# Functions for evaluation


def eval_batch(model, task_sampler, xs, xs_p=None):
	task = task_sampler()
	if torch.cuda.is_available() and model.name.split("_")[0] in ["gpt2", "lstm"]:
		device = "cuda"
	else:
		device = "cpu"

	if xs_p is None:
		ys = task.evaluate(xs)
		pred = model(xs.to(device), ys.to(device)).detach()
		metrics = task.get_metric()(pred.cpu(), ys)
	else:
		b_size, n_points, _ = xs.shape
		metrics = torch.zeros(b_size, n_points)
		for i in range(n_points):
			xs_comb = torch.cat((xs[:, :i, :], xs_p[:, i:, :]), dim=1)
			ys = task.evaluate(xs_comb)

			pred = model(xs_comb.to(device), ys.to(device), inds=[i]).detach()
			metrics[:, i] = task.get_metric()(pred.cpu(), ys)[:, i]

	return metrics


def eval_batch_bool(model, task_sampler, n_points, bsize, device, increments, pac_learn=False):
	# pdb.set_trace()
	start_time = time.time()
	task = task_sampler()

	tot_tokens = 0

	prompt_struct = ""
	if isinstance(model, LargeLanguageModel):
		points_to_select = n_points+1
	else:
		points_to_select = n_points

	xs = task.sample_xs(points_to_select, bsize)
	ys = task.evaluate(xs)
	if bsize == 1 and ys.shape[0] != 1:
		ys = ys.unsqueeze(0)

	if isinstance(model, LargeLanguageModel):
		xs = xs[0].tolist() # remove batch size dimension
		ys = ys[0].tolist()
		all_preds = []
		all_golds = []
		if pac_learn:
			all_pacs = []
		untenables = []
		prompt_struct = ""
		for i in range(increments, len(ys), increments):
			prediction, t_prompt, unten = model.predict(xs[:i+1], ys[:i+1])
			if pac_learn:
				pac_learn_op = task.pac_learn(xs[:i+1], ys[:i+1])
				all_pacs.append(pac_learn_op)
			all_preds.append(prediction)
			all_golds.append(ys[i])
			if unten == 1:
				untenables.append(int(i/increments)-1)
			if prompt_struct == "" and i>len(ys)/10:
				prompt_struct = t_prompt
			if model.encoding is not None:
				tot_tokens += len(model.encoding.encode(t_prompt))
			print("Completed {} / {}...".format(i, len(ys)-1), end = '\r', flush = True)
		metrics = task.get_metric()(torch.Tensor(all_preds).unsqueeze(0), torch.Tensor(all_golds).unsqueeze(0))
		if pac_learn:
			pac_metrics = task.get_metric()(torch.Tensor(all_preds).unsqueeze(0), torch.Tensor(all_pacs).unsqueeze(0))
		for idx in untenables:
			metrics[0][idx] = 0
			if pac_learn:
				pac_metrics[0][idx] = 0

	else:
		pred = model(xs.to(device), ys.to(device)).detach()
		metrics = task.get_metric()(pred.cpu(), ys)
		if pac_learn:
			pac_metrics = metrics

	end_time = time.time()
	time_taken = end_time - start_time
	if pac_learn:
		return pac_metrics, time_taken, prompt_struct, tot_tokens
	return metrics, time_taken, prompt_struct, tot_tokens



# Functions for generating different kinds of train/test data


def gen_standard(data_sampler, n_points, b_size):
	xs = data_sampler.sample_xs(n_points, b_size)

	return xs, None


def gen_opposite_quadrants(data_sampler, n_points, b_size):
	xs = data_sampler.sample_xs(n_points, b_size)
	pattern = torch.randn([b_size, 1, xs.shape[2]]).sign()

	xs_train_pre = xs.abs() * pattern
	xs_test_post = -xs_train_pre

	return xs_train_pre, xs_test_post


def gen_random_quadrants(data_sampler, n_points, b_size):
	xs = data_sampler.sample_xs(n_points, b_size)
	pattern = torch.randn([b_size, 1, xs.shape[2]]).sign()

	xs_train_pre = xs.abs() * pattern
	xs_test_post = xs

	return xs_train_pre, xs_test_post


def gen_orthogonal_train_test(data_sampler, n_points, b_size):
	xs = data_sampler.sample_xs(n_points, b_size)
	n_dim = xs.shape[2]
	n_points = min(n_points, n_dim)
	# raise ValueError("number of points should be at most the dimension.")
	xs_train_pre = xs
	xs_test_post = torch.zeros(xs.shape)
	for i in range(n_points):
		xs_test_post_i = xs[:, i : i + 1, :]
		xs_train_pre_i = xs[:, :i, :]
		_, _, Vt = torch.linalg.svd(xs_train_pre_i, full_matrices=False)
		xs_train_pre_i_projection = Vt.transpose(1, 2) @ Vt
		xs_test_post_i_orthogonalized = (
			xs_test_post_i - xs_test_post_i @ xs_train_pre_i_projection
		)
		xs_test_post_i_normalized = (
			xs_test_post_i_orthogonalized
			* xs_test_post_i.norm(dim=2).unsqueeze(2)
			/ xs_test_post_i_orthogonalized.norm(dim=2).unsqueeze(2)
		)

		xs_test_post[:, i : i + 1, :] = xs_test_post_i_normalized

	return xs_train_pre, xs_test_post


def gen_overlapping_train_test(data_sampler, n_points, b_size):
	xs = data_sampler.sample_xs(n_points, b_size)
	xs_train_pre = xs
	xs_test_post = xs.clone()
	b_size = xs.shape[0]
	for i in range(1, n_points):
		xs_train_pre_i = xs[:, :i, :]
		perm = torch.stack([torch.randperm(i) for _ in range(b_size)]).unsqueeze(dim=1)
		ind_mat = (perm == 0) + 0.0
		xs_test_post[:, i : i + 1, :] = ind_mat @ xs_train_pre_i

	return xs_train_pre, xs_test_post


def aggregate_metrics(metrics, bootstrap_trials=1000):
	"""
	Takes as input a tensor of shape (num_eval, n_points) and returns a dict with
	per-point mean, stddev, and bootstrap limits
	"""
	results = {}
	results["mean"] = metrics.mean(dim=0)
	results["std"] = metrics.std(dim=0, unbiased=True)
	n = len(metrics)
	bootstrap_indices = torch.randint(n, size=(bootstrap_trials, n))
	bootstrap_means = metrics[bootstrap_indices].mean(dim=1).sort(dim=0)[0]
	results["bootstrap_low"] = bootstrap_means[int(0.05 * bootstrap_trials), :]
	results["bootstrap_high"] = bootstrap_means[int(0.95 * bootstrap_trials), :]

	return {k: v.tolist() for k, v in results.items()}


def eval_model(
	model,
	task_name,
	data_name,
	n_dims,
	n_points,
	prompting_strategy,
	num_eval_examples=1920,
	batch_size=64,
	data_sampler_kwargs={},
	task_sampler_kwargs={},
):
	"""
	Evaluate a model on a task with a variety of strategies.
	   Args:
	   - task: which base task we are evaluating on. E.g., "linear_regression"
	   - prompting_strategy: how to construct the prompt, e.g., "random_quadrants"
	   - num_eval_examples: total number of examples to evaluate on
	   - **sampler_kwargs: remaining arguments to pass directly to the sampler
	"""

	assert num_eval_examples % batch_size == 0
	
	task_sampler = get_task_sampler(
		task_name, n_dims, batch_size, **task_sampler_kwargs
	)

	if 'bool' not in data_name:
		data_sampler = get_data_sampler(data_name, n_dims, **data_sampler_kwargs)

	all_metrics = []

	generating_func = globals()[f"gen_{prompting_strategy}"]
	for i in range(num_eval_examples // batch_size):
		xs, xs_p = generating_func(data_sampler, n_points, batch_size)

		metrics = eval_batch(model, task_sampler, xs, xs_p)
		all_metrics.append(metrics)

	metrics = torch.cat(all_metrics, dim=0)

	return aggregate_metrics(metrics)



def eval_model_bool(
	model,
	task_name,
	data_name,
	n_dims,
	n_points,
	prompting_strategy,
	device,
	num_eval_examples=100,
	baseline_num_eval_examples=2000,
	data_sampler_kwargs={},
	task_sampler_kwargs={},
	increments=5,
	pac_learn=False
):
	"""
	Evaluate a model on a task with a variety of strategies.
	   Args:
	   - task: which base task we are evaluating on. E.g., "linear_regression"
	   - prompting_strategy: how to construct the prompt, e.g., "random_quadrants"
	   - num_eval_examples: total number of examples to evaluate on
	   - **sampler_kwargs: remaining arguments to pass directly to the sampler
	"""

	task_sampler = get_task_sampler(
		task_name, n_dims, 1, **task_sampler_kwargs
	)


	all_metrics = []
	times = []

	if isinstance(model, LargeLanguageModel):
		num_functions = num_eval_examples
	else:
		num_functions = baseline_num_eval_examples

	# generating_func = globals()[f"gen_{prompting_strategy}"]
	for i in range(num_functions):
		# xs, xs_p = generating_func(data_sampler, n_points, batch_size)

		metrics, time_taken, prompt_struct, tot_tokens = eval_batch_bool(model, task_sampler, n_points, 1, device, increments, pac_learn)
		all_metrics.append(metrics)
		times.append(time_taken)
		if isinstance(model, LargeLanguageModel) and i==0:
			print("------------------- Prompt structure -------------------------")
			print(prompt_struct)
			print("------------------- Total Number of tokens in all prompts per function -------------------------")
			print(tot_tokens)
		# print()
		print("Completed {} / {}...".format(i+1, num_functions), end = '\r', flush = True)
		print()

	print("Average time taken per example function: ", str(np.average(np.array(times))))
	print("Total time taken: ", str(np.sum(np.array(times))))

	metrics = torch.cat(all_metrics, dim=0)
	# pdb.set_trace()

	return aggregate_metrics(metrics)


def build_evals(conf):
	n_dims = conf.n_dims
	n_points = conf.n_points
	increments = conf.increments
	pac_learn = False
	if "pac" in conf.prompt_type:
		pac_learn = True

	task_name = conf.task
	data_name = conf.data

	base_kwargs = {
		"task_name": task_name,
		"increments": increments,
		"n_dims": n_dims,
		"n_points": n_points,
		"num_eval_examples": conf.num_eval_examples,
		"data_name": data_name,
		"prompting_strategy": "standard",
		"pac_learn": pac_learn
	}

	evaluation_kwargs = {}

	evaluation_kwargs["standard"] = {"prompting_strategy": "standard"}
	if task_name != "linear_regression":
		if task_name in ["relu_2nn_regression"]:
			evaluation_kwargs["linear_regression"] = {"task_name": "linear_regression"}
		for name, kwargs in evaluation_kwargs.items():
			# allow kwargs to override base_kwargs values
			evaluation_kwargs[name] = base_kwargs.copy()
			evaluation_kwargs[name].update(kwargs)

		return evaluation_kwargs

	for strategy in [
		"random_quadrants",
		"orthogonal_train_test",
		"overlapping_train_test",
	]:
		evaluation_kwargs[strategy] = {"prompting_strategy": strategy}

	for method in ["half_subspace", "skewed"]:
		if "subspace" in method:
			eigenvals = torch.zeros(n_dims)
			eigenvals[: n_dims // 2] = 1
		else:
			eigenvals = 1 / (torch.arange(n_dims) + 1)

		scale = sample_transformation(eigenvals, normalize=True)
		evaluation_kwargs[f"{method}"] = {
			"data_sampler_kwargs": {"scale": scale},
		}

	for dim in ["x", "y"]:
		for scale in [0.333, 0.5, 2, 3]:
			if dim == "x":
				eigenvals = scale * torch.ones(n_dims)
				t = sample_transformation(eigenvals)
				scaling_args = {"data_sampler_kwargs": {"scale": t}}
			else:
				eigenvals = scale * torch.ones(n_dims)
				scaling_args = {"task_sampler_kwargs": {"scale": scale}}

			evaluation_kwargs[f"scale-{dim}={scale}"] = scaling_args

	evaluation_kwargs[f"noisyLR"] = {
		"task_sampler_kwargs": {"renormalize_ys": True, "noise_std": 1},
		"task_name": "noisy_linear_regression",
	}

	for name, kwargs in evaluation_kwargs.items():
		# allow kwargs to override base_kwargs values
		evaluation_kwargs[name] = base_kwargs.copy()
		evaluation_kwargs[name].update(kwargs)

	return evaluation_kwargs


def compute_evals(all_models, evaluation_kwargs, device, save_path=None, plots_path=None):
	try:
		with open(save_path) as fp:
			all_metrics = json.load(fp)
	except Exception:
		all_metrics = {}
	
	data_type = evaluation_kwargs["standard"]["data_name"]

	for eval_name, kwargs in tqdm(evaluation_kwargs.items()):
		metrics = {}
		if eval_name in all_metrics:
			metrics = all_metrics[eval_name]
		for model in all_models:
			if model.name in metrics:
				continue
			
			if data_type == "boolean":
				
				metrics[model.name] = eval_model_bool(model, device=device, **kwargs)
			else:
				metrics[model.name] = eval_model(model, **kwargs)

		all_metrics[eval_name] = metrics

	if save_path is not None:
		with open(save_path, "w") as fp:
			json.dump(all_metrics, fp, indent=2)

	if plots_path is not None:
		with open(plots_path, "w") as fp:
			json.dump(all_metrics, fp, indent=2)

	return all_metrics


def get_run_metrics(
	args, device, skip_baselines=False
):
	model = LargeLanguageModel(
		prompt_type=args.prompt_type, 
		model_type=args.model_type,
		engine=args.model,
		hf_model_type=args.hf_model_type,
		llama_weights_path=args.llama_weights_path,
		max_tokens=args.max_tokens,
		temperature=args.temperature,
		top_p=args.top_p,
		n=args.n,
		stop=args.stop,
		presence_penalty=args.presence_penalty,
		frequency_penalty=args.frequency_penalty,
		port=args.port,
		timeout=args.timeout
	)
	if args.only_baselines:
		all_models = []
		plots_path = os.path.join(args.plots_dir, "baselines.json")
	else:
		all_models = [model]
		plots_path = os.path.join(args.plots_dir, args.model+"_prompt-"+args.prompt_type+".json")
	if not skip_baselines:
		all_models += models.get_relevant_baselines(args.task, args.n_dims)
	evaluation_kwargs = build_evals(args)

	save_path = os.path.join(args.out_dir, "metrics.json")

	all_metrics = compute_evals(all_models, evaluation_kwargs, device=device, save_path=save_path, plots_path=plots_path)
	return all_metrics



def conf_to_model_name(conf):
	if conf.family == "gpt2":
		return {
			(3, 2): "Transformer-xs",
			(6, 4): "Transformer-small",
			(12, 8): "Transformer",
		}[(conf.n_layer, conf.n_head)]
	else:
		return conf.name


def baseline_names(name):
	if "OLS" in name:
		return "Least Squares"
	if name == "averaging":
		return "Averaging"
	if "NN" in name:
		k = name.split("_")[1].split("=")[1]
		return f"{k}-Nearest Neighbors"
	if "lasso" in name:
		alpha = name.split("_")[1].split("=")[1]
		return f"Lasso (alpha={alpha})"
	if "gd" in name:
		return "2-layer NN, GD"
	if "decision_tree" in name:
		return "Greedy Tree Learning"
	if "xgboost" in name:
		return "XGBoost"
	return name


def read_run_dir(run_dir):
	all_runs = {}
	for task in os.listdir(run_dir):
		task_dir = os.path.join(run_dir, task)
		# for run_id in os.listdir(task_dir):
		for run_id in ["f8e3a309-aac1-4dc7-aebf-e34ca8ea1085"]:
			run_path = os.path.join(task_dir, run_id)
			_, conf = get_model_from_run(run_path, only_conf=True)
			params = {}
			params["run_id"] = run_id
			params["task"] = task
			params["model"] = conf_to_model_name(conf)
			params["kwargs"] = "_".join(
				f"{k}={v}" for k, v in conf.task_kwargs.items()
			)
			num_tasks = (
				conf.num_tasks if "num_tasks" in conf else None
			)
			params["num_tasks"] = num_tasks if num_tasks is not None else -1
			num_examples = (
				conf.num_training_examples
				if "num_training_examples" in conf
				else None
			)
			params["num_examples"] = num_examples if num_examples is not None else -1
			params["n_dims"] = conf.n_dims
			params["n_layer"] = conf.n_layer
			params["n_head"] = conf.n_head
			params["run_name"] = conf.name

			for k, v in params.items():
				if k not in all_runs:
					all_runs[k] = []
				all_runs[k].append(v)

	df = pd.DataFrame(all_runs).sort_values("run_name")
	assert len(df) == len(df.run_name.unique())
	return df

if __name__ == "__main__":
	parser = build_parser()
	args = parser.parse_args()

	openai.api_key = "Key" # Provide openai key

	run_id = str(uuid.uuid4())
	print(f"Run ID: {run_id}")

	# if args.model_type == "huggingface" and args.hf_model_type == "llama":
	# 	args.model = "llama"

	if args.run_name == "default":
		args.run_name = args.model + "-" + args.prompt_type +  "-" + str(args.temperature) + "-" + str(args.n)

	args.out_dir = os.path.join(args.out_dir, args.task)
	args.out_dir = os.path.join(args.out_dir, args.run_name)
	args.out_dir = os.path.join(args.out_dir, run_id)

	args.plots_dir = os.path.join(args.plots_dir, args.task+"-dim"+str(args.n_dims)+"-pts"+str(args.n_points)+"-fn"+str(args.num_eval_examples))

	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)

	if not os.path.exists(args.plots_dir):
		os.makedirs(args.plots_dir)
	
	device = torch.device("cuda:{}".format(args.gpu))

	metrics = get_run_metrics(args, device=device, skip_baselines=args.skip_baselines)