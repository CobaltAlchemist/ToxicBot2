import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tdata
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import torchsummary
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

class ToxicDataset(tdata.Dataset):
	def __init__(self, root='.'):
		self.dframe = pd.read_csv(os.path.join(root, 'train.csv'))
		self.toxcols = ['toxic','severe_toxicity','obscene','identity_attack','insult','threat','sexual_explicit']
		self.hatecols = ['asian','atheist','bisexual','black','buddhist','christian','female','heterosexual','hindu','homosexual_gay_or_lesbian','intellectual_or_learning_disability','jewish','latino','male','muslim','other_disability','other_gender','other_race_or_ethnicity','other_religion','other_sexual_orientation','physical_disability','psychiatric_or_mental_illness','transgender','white']
		self.emojis = ['funny','wow','sad','likes','disagree']
		
	def __len__(self,):
		return len(self.dframe)
	
	def __getitem__(self, idx):
		row = self.dframe.iloc[idx]
		x = row.comment_text
		toxicity = torch.tensor(row[self.toxcols], dtype=torch.float)
		hate = torch.tensor(row[self.hatecols].fillna(0), dtype=torch.float)
		emojis = torch.tensor(row[self.emojis], dtype=torch.int32)
		hatemask = torch.tensor(~row[self.hatecols].isna())
		return {'input': x, 'toxicity': toxicity, 'hate': hate, 'hatemask': hatemask, 'reactions': emojis}
	
	def make_tokenizer(self, vocab_size=20_000):
		tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
		trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=vocab_size)
		tokenizer.pre_tokenizer = Whitespace()
		tokenizer.train_from_iterator(self.dframe.comment_text, trainer)
		return tokenizer
		
def cyclic_encoding(x, resolution = 64):
	arr = torch.zeros((*x.shape, resolution), dtype=torch.double)
	arr[...,:] = x[...,None]
	arr /= 10000 ** (torch.arange(0,resolution, dtype=torch.double) / resolution)
	arr[...,::2] = torch.sin(arr[...,::2])
	arr[...,1::2] = torch.cos(arr[...,1::2])
	return arr

def positional_encoding(length = 256, dims = 128):
	return cyclic_encoding(torch.arange(length), resolution=dims)

class XformPosEnc(pl.LightningModule):
	def __init__(self, emb_dims = 512, hidden:int=128, heads=8, layers:int=2, max_seqlen = 0, vocab_size=20_000, dropout=0.3, tokenizer=None, lr=1e-3, classes:int=[6], **kwargs):
		super().__init__()
		self.embeddings = nn.Embedding(vocab_size, emb_dims)
		base = nn.TransformerEncoderLayer(
			d_model=hidden,
			nhead=heads,
			dropout=dropout,
			batch_first=True,
		)
		self.body = nn.Sequential(
			nn.Conv1d(emb_dims, hidden, 1),
			nn.ReLU(),
		)
		self.transformer =  nn.TransformerEncoder(base,layers)
		self.linear = nn.ModuleList([nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(hidden*2, hidden),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden, c)
		) for c in classes])
		self.tokenizer = tokenizer
		self.lr = lr
		self.max_seq = max_seqlen
		
		self.f1 = nn.ModuleDict({k: torchmetrics.F1(num_classes=7, average='macro') for k in ['tr', 'val', 'test']})

	def forward(self, x):
		x = self.embeddings(x)
		x = x + positional_encoding(*x.shape[1:]).to(x)
		x = torch.transpose(x, 1, 2)
		x = self.body(x)
		x = torch.transpose(x, 1, 2)
		x = self.transformer(x)
		x = torch.cat((x.mean(1), x.max(1)[0]), dim=1)
		return [l(x) for l in self.linear]
	
	def shared_step(self, batch, idx, label='tr'):
		s = batch['input']
		toxicity = batch['toxicity']
		hate = batch['hate']
		hatemask = batch['hatemask']
		reactions = batch['reactions']
		ids = [torch.tensor(x.ids[-self.max_seq:]).to(toxicity.device) for x in self.tokenizer.encode_batch(s)]
		x = nn.utils.rnn.pad_sequence(ids, batch_first=True)
		t, h, r = self(x)
		loss = F.binary_cross_entropy_with_logits(t, toxicity)
		loss = loss + F.binary_cross_entropy_with_logits(h[hatemask], hate[hatemask])
		#loss = loss + F.mse_loss(r, reactions)
		self.f1[label](torch.sigmoid(t), (toxicity > 0.5).int())
		return loss
	
	def training_step(self, batch, batch_idx):
		self.log(f'f1', self.f1['tr'], prog_bar=True)
		loss = self.shared_step(batch, batch_idx, 'tr')
		return loss
	
	def validation_step(self, batch, batch_idx):
		return self.shared_step(batch, batch_idx, 'val')
	
	def validation_epoch_end(self, outs):
		self.log('val_f1', self.f1['val'].compute(), prog_bar=True)

	def configure_optimizers(self):
		return torch.optim.AdamW(self.parameters(), lr=self.lr)
		
if __name__ == "__main__":
	dset = ToxicDataset()
	BATCH_SIZE = 32
	EMB_DIMS = 2048
	HIDDEN = 1024
	HEADS = 8
	LAYERS = 6
	VOCAB_SIZE=10_000
	DROPOUT = 0.1
	CLASSES = []
	MAX_SEQ = 128
	LR = 1e-5
	for items in dset:
		CLASSES.extend([len(items['toxicity']), len(items['hate']), len(items['reactions'])])
		break
	tok = dset.make_tokenizer(VOCAB_SIZE)
	tset, vset = tdata.random_split(dset, [len(dset) // 10, len(dset) - len(dset) // 10])
	tload = tdata.DataLoader(dset, batch_size=BATCH_SIZE, num_workers=4)
	vload = tdata.DataLoader(dset, batch_size=BATCH_SIZE, num_workers=4)
	net = XformPosEnc(emb_dims=EMB_DIMS, hidden=HIDDEN, heads=HEADS, layers=LAYERS, vocab_size=VOCAB_SIZE, max_seqlen=MAX_SEQ, dropout=DROPOUT, lr=LR, classes=CLASSES, tokenizer=tok)
	torchsummary.summary(net, (128,), dtypes=(torch.long,), device='cpu')
	trainer = pl.Trainer(max_epochs=10, gpus=1)
	trainer.fit(net, tload, vload)
