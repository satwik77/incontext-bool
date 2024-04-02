import os
import argparse

import torch
from typing import Optional

from pydantic import BaseModel
from transformers import (
	PreTrainedModel,
	PreTrainedTokenizer,
	AutoModelForSeq2SeqLM,
	AutoTokenizer,
	AutoModelForCausalLM,
	LlamaForCausalLM,
	LlamaTokenizer,
	AutoModel
)

class EvalModel(BaseModel, arbitrary_types_allowed=True):
	max_input_length: int = 512
	max_tokens: int = 512

	def run(self, prompt: str) -> str:
		raise NotImplementedError

	def check_valid_length(self, text: str) -> bool:
		raise NotImplementedError


class SeqToSeqModel(EvalModel):
	model_path: str
	llama_weights_path: str
	model: Optional[PreTrainedModel]
	tokenizer: Optional[PreTrainedTokenizer]
	device: torch.device

	def load(self):
		if self.model is None:
			self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
			self.model.eval()
			self.model.to(self.device)
		if self.tokenizer is None:
			self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

	def run(self, prompt: str) -> str:
		self.load()
		inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
		outputs = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
		return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

	def check_valid_length(self, text: str) -> bool:
		self.load()
		inputs = self.tokenizer(text)
		return len(inputs.input_ids) <= self.max_input_length


class CausalModel(SeqToSeqModel):
	def load(self):
		if self.model is None:
			self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
			self.model.eval()
			self.model.to(self.device)
		if self.tokenizer is None:
			self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

	def run(self, prompt: str, temperature=0.0, num_return_sequences=1) -> str:
		self.load()
		inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
		if temperature == 0.0:
			outputs = self.model.generate(
				**inputs,
				do_sample=False,
				num_return_sequences=1,
				max_new_tokens=self.max_tokens,
				pad_token_id=self.tokenizer.eos_token_id,  # Avoid pad token warning
			)
		else:
			outputs = self.model.generate(
				**inputs,
				do_sample=True,
				temperature=temperature,
				num_return_sequences=num_return_sequences,
				max_new_tokens=self.max_tokens,
				pad_token_id=self.tokenizer.eos_token_id,  # Avoid pad token warning
			)
		# batch_size, length = inputs.input_ids.shape
		# print(length)
		# print(outputs)
		fin_ops = []
		for z in range(outputs.shape[0]):
			fin_ops.append(self.tokenizer.decode(outputs[z], skip_special_tokens=True).split(prompt)[1])
		return fin_ops


class LlamaModel(SeqToSeqModel):
	use_template: bool = False
	"""
	Not officially supported by AutoModelForCausalLM, so we need the specific class
	Optionally, we can use the prompt template from: https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
	"""

	def load(self):
		final_path = os.path.join(self.llama_weights_path, self.model_path)
		if self.tokenizer is None:
			self.tokenizer = LlamaTokenizer.from_pretrained(final_path, torch_dtype=torch.float16) #torch_dtype=torch.float16
		if self.model is None:
			self.model = LlamaForCausalLM.from_pretrained(final_path, torch_dtype=torch.float16)
			self.model.eval()
			self.model.to(self.device)

	def run(self, prompt: str, temperature=0.0, num_return_sequences=1) -> str:
		if self.use_template:
			template = (
				"Generate more creative instructions and corresponding preferred and rejected responses. "
			)
			text = template.format_map(dict(instruction=prompt))
		else:
			text = prompt

		self.load()
		inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
		if temperature == 0.0:
			outputs = self.model.generate(
				**inputs,
				do_sample=False,
				num_return_sequences=1,
				max_new_tokens=self.max_tokens
			)
		else:
			outputs = self.model.generate(
				**inputs,
				do_sample=True,
				temperature=temperature,
				num_return_sequences=num_return_sequences,
				max_new_tokens=self.max_tokens
			)
		batch_size, length = inputs.input_ids.shape
		return self.tokenizer.decode(outputs[0, length:], skip_special_tokens=True)


def select_model(max_input_length=512, max_tokens=512, model_type="causal", model_path="facebook/opt-1.3b", llama_weights_path='/home/', model=None, tokenizer=None, device=torch.device("cuda:0"), use_template=False) -> EvalModel:
	model_map = dict(
		seq_to_seq=SeqToSeqModel,
		causal=CausalModel,
		llama=LlamaModel,
	)
	model_class = model_map.get(model_type)
	if model_class is None:
		raise ValueError(f"{model_type}. Choose from {list(model_map.keys())}")
	return model_class(max_input_length=max_input_length, max_tokens=max_tokens, model_path=model_path, llama_weights_path=llama_weights_path, model=model, tokenizer=tokenizer, device=device, use_template=use_template)

def test_model(
	prompt: str = "Write an email to a professor asking for deadline extension on the assignment.",
	model_type: str = "llama",
	model_path: str = "facebook/opt-iml-1.3b",
):
	model = select_model(model_type=model_type, model_path=model_path)
	print(model.run(prompt))

if __name__ == "__main__":
	test_model()