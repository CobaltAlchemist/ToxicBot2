import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tdata
import torch.utils.data as tdata
import wandb
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_metric

class ToxicDatasetBert(tdata.Dataset):
	def __init__(self, tokenizer, root='.'):
		self.dframe = pd.read_csv(os.path.join(root, 'train.csv'))
		self.toxcols = ['toxic','severe_toxicity','obscene','identity_attack','insult','threat','sexual_explicit']
		self.hatecols = ['asian','atheist','bisexual','black','buddhist','christian','female','heterosexual','hindu','homosexual_gay_or_lesbian','intellectual_or_learning_disability','jewish','latino','male','muslim','other_disability','other_gender','other_race_or_ethnicity','other_religion','other_sexual_orientation','physical_disability','psychiatric_or_mental_illness','transgender','white']
		self.emojis = ['funny','wow','sad','likes','disagree']
		self.tokenizer = tokenizer

	def __len__(self,):
		return len(self.dframe)

	def __getitem__(self, idx):
		row = self.dframe.iloc[idx]
		x = self.tokenizer(row.comment_text, truncation=True, max_length=256)
		x['input_ids'] = torch.tensor(x['input_ids'], dtype=torch.long)
		x['attention_mask'] = torch.tensor(x['attention_mask'], dtype=torch.long)
		toxicity = torch.tensor(row[self.toxcols], dtype=torch.float)
		#hate = torch.tensor(row[self.hatecols].fillna(0), dtype=torch.float)
		#emojis = torch.tensor(row[self.emojis], dtype=torch.int32)
		#hatemask = torch.tensor(~row[self.hatecols].isna())
		return {**x, 'labels':toxicity}#, hate, hatemask, emojis)

	def make_tokenizer(self, vocab_size=20_000):
		tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
		trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=vocab_size)
		tokenizer.pre_tokenizer = Whitespace()
		tokenizer.train_from_iterator(self.dframe.comment_text, trainer)
		return tokenizer
		
class Metrics:
	def __init__(self):
		self.acc = load_metric("accuracy", config_name='multilabel', average='macro')
		self.f1 = load_metric("f1", config_name='multilabel', average='macro')
		self.prec = load_metric("precision", config_name='multilabel', average='macro')
		self.recall = load_metric("recall", config_name='multilabel', average='macro')

	def __call__(self, eval_pred):
		logits, labels = eval_pred
		predictions = (labels > 0.).astype(np.int)
		labels = (labels > 0.5).astype(np.int)
		ret = {
			'accuracy': self.acc.compute(predictions=predictions, references=labels),
			'f1': self.f1.compute(predictions=predictions, references=labels, average='macro'),
			'precision': self.prec.compute(predictions=predictions, references=labels, average='macro'),
			'recall': self.recall.compute(predictions=predictions, references=labels, average='macro'),
		}
		return ret

if __name__ == "__main__":
	tokenizer = RobertaTokenizer.from_pretrained('roberta-base', use_fast=False)
	model = RobertaForSequenceClassification.from_pretrained(
		"roberta-base",
		num_labels=7,
		problem_type="multi_label_classification"
	)
	
	dset = ToxicDatasetBert(tokenizer)
	data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=256, padding='max_length', pad_to_multiple_of=8)
	vset, tset = tdata.random_split(dset, [len(dset) // 10, len(dset) - len(dset) // 10])
	
	metric = Metrics()
	
	training_args = TrainingArguments(
		output_dir="./models/Roberta Fine Tuned New",
		learning_rate=5e-5,
		per_device_train_batch_size=64,
		per_device_eval_batch_size=1024,
		num_train_epochs=3,
		evaluation_strategy='epoch',
		dataloader_num_workers=4,
		weight_decay=0.01,
		tf32=True,
	)
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=tset,
		eval_dataset=vset,
		tokenizer=tokenizer,
		data_collator=data_collator,
		compute_metrics=metric,
	)
	
	trainer.train("models\Roberta Fine Tuned New\checkpoint-47500")
