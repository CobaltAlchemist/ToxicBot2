import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from asyncio import Lock

class Predictor:
    def __init__(self, model_dir='.', device='cpu'):
        self.tokenizer = RobertaTokenizer.from_pretrained(model_dir, use_fast=False, local_files_only=True)
        self.model = RobertaForSequenceClassification.from_pretrained(
            model_dir, num_labels=7, problem_type="multi_label_classification", local_files_only=True)
		self.lock = Lock()

    async def predict_text(self, s : str, logits : bool = False):
		async with self.lock:
            with torch.no_grad():
                x = self.model(**{k: torch.tensor(v) for k, v in self.tokenizer([s]).items()})
            x = x.logits[0]
            if not logits:
                x = torch.sigmoid(x)
            return x.numpy()

    def tokenize_text(self, s : str):
        return self.tokenizer.tokenize(s)

    async def __call__(self, s : str, **kwargs):
        return await self.predict_text(s, **kwargs)

if __name__ == "__main__":
    p = Predictor('model', device='cpu')
    print(p("Hello world"))
    print(p.predict_text("Hi there"))
