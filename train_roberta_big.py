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
		
def all_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = (logits > 0.).astype(np.int64)
    references = (labels > 0.5).astype(np.int64)
    acc = load_metric("accuracy", config_name='multilabel', average='macro')
    f1 = load_metric("f1", config_name='multilabel', average='macro')
    prec = load_metric("precision", config_name='multilabel', average='macro')
    recall = load_metric("recall", config_name='multilabel', average='macro')
    ret = {
        'accuracy': acc.compute(predictions=predictions, references=references),
        'f1': f1.compute(predictions=predictions, references=references, average='macro'),
        'precision': prec.compute(predictions=predictions, references=references, average='macro'),
        'recall': recall.compute(predictions=predictions, references=references, average='macro'),
    }
    return ret

if __name__ == "__main__":
	tokenizer = RobertaTokenizer.from_pretrained('roberta-large', use_fast=False)
	model = RobertaForSequenceClassification.from_pretrained(
		"roberta-large",
		num_labels=7,
		problem_type="multi_label_classification"
	)
	
	dset = ToxicDatasetBert(tokenizer)
	data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=256, padding='max_length', pad_to_multiple_of=8)
	vset, tset = tdata.random_split(dset, [len(dset) // 10, len(dset) - len(dset) // 10])
	
	training_args = TrainingArguments(
		output_dir="./models/RobertaLarge2e",
		learning_rate=5e-5,
		per_device_train_batch_size=16,
		gradient_accumulation_steps=4,
		per_device_eval_batch_size=256,
		num_train_epochs=2,
		save_strategy='epoch',
		evaluation_strategy='epoch',
		dataloader_num_workers=4,
		weight_decay=0.01,
		bf16=True,
		tf32=True,
		optim='adamw_torch',
		
	)
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=tset,
		eval_dataset=vset,
		tokenizer=tokenizer,
		data_collator=data_collator,
		compute_metrics=all_metrics,
	)
	
	trainer.train()
