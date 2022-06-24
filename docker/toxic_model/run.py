from flask import Flask, request, jsonify
from markupsafe import escape
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import os
import torch

classes = ['Toxic','Severely Toxic','Obscene','Identity Attack','Insult','Threat','Sexually Explicit']

class Predictor:
	def __init__(self, model_dir='.', device='cpu'):
		self.tokenizer = RobertaTokenizer.from_pretrained(model_dir, use_fast=False, local_files_only=True)
		self.model = RobertaForSequenceClassification.from_pretrained(
			model_dir, num_labels=7, problem_type="multi_label_classification", local_files_only=True).to(device)
		self.device = device
		
	def to(self, device):
		self.device = device
		self.model = self.model.to(device)

	def predict_text(self, s : str, logits : bool = False):
		with torch.no_grad():
			x = self.model(**self.tokenize_text(s))
		x = x.logits[0]
		if not logits:
			x = torch.sigmoid(x)
		return x.cpu()

	def tokenize_text(self, s : str):
		tok = self.tokenizer([s])
		return {k: torch.tensor(v, device=self.device) for k, v in tok.items()}

	def __call__(self, s : str, **kwargs):
		return self.predict_text(s, **kwargs)
	
if __name__ == "__main__":
	app = Flask(__name__)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	p = Predictor('model', device=device)
	
	@app.route("/")
	def base():
		return "<p>Running</>"
		
	@app.route("/predict", methods=['GET', 'POST'])
	def predict(name):
		s = content['text']
		return jsonify({k: v.item() for k, v in zip(classes, p(s))})