from dataclasses import dataclass
from functools import partial

import random
import jax
import jax.numpy as jnp
from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel
from transformers import CLIPProcessor, FlaxCLIPModel
from flax.jax_utils import replicate, unreplicate
from functools import partial
from dalle_mini import DalleBartProcessor
from flax.training.common_utils import shard_prng_key, shard
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from markupsafe import escape
from glob import glob
from importlib import import_module
import os


DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"
DALLE_COMMIT_ID = None
VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"
CLIP_REPO = "openai/clip-vit-base-patch32"
CLIP_COMMIT_ID = None

@dataclass
class DalleRequest:
	text: str
	n: int = 8
	k: float = None
	p: float = None
	temp: float = None
	cond_scale: float = 10.0
	return_single: bool = False

class DallePModel:
	def __init__(self, model_dir = None):
		if model_dir is None:
			model_dir = os.path.dirname(os.path.realpath(__file__))
		self.model, self.params = DalleBart.from_pretrained(
			DALLE_MODEL,
			cache_dir=model_dir,
			revision=DALLE_COMMIT_ID,
			dtype=jnp.float16,
			_do_init=False
		)
		self.vqgan, self.vqgan_params = VQModel.from_pretrained(
			VQGAN_REPO,
			cache_dir=model_dir,
			revision=VQGAN_COMMIT_ID,
			_do_init=False
		)
		self.processor = DalleBartProcessor.from_pretrained(
			DALLE_MODEL,
			cache_dir=model_dir,
			revision=DALLE_COMMIT_ID)
		self.clip, self.clip_params = FlaxCLIPModel.from_pretrained(
			CLIP_REPO,
			cache_dir=model_dir,
			revision=CLIP_COMMIT_ID,
			dtype=jnp.float16, 
			_do_init=False
		)
		self.clip_processor = CLIPProcessor.from_pretrained(CLIP_REPO, revision=CLIP_COMMIT_ID, cache_dir=model_dir)
		self.params = replicate(self.params)

	@staticmethod
	@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(0, 4, 5, 6, 7))
	def p_generate(
		model, tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
	):
		return model.generate(
			**tokenized_prompt,
			prng_key=key,
			params=params,
			top_k=top_k,
			top_p=top_p,
			temperature=temperature,
			condition_scale=condition_scale,
		)

	@staticmethod
	def p_decode(vqgan, indices, params):
		return vqgan.decode_code(indices, params=params)
	
	@staticmethod
	def p_clip(clip, inputs, params):
		logits = clip(params=params, **inputs).logits_per_image
		return logits

	def tokenize(self, prompts):
		tokenized_prompts = self.processor(prompts)
		tokenized_prompt = replicate(tokenized_prompts)
		return tokenized_prompt

	def __call__(self, request):
		seed = random.randint(0, 2**32 - 1)
		key = jax.random.PRNGKey(seed)
		tokens = self.tokenize([request.text]*request.n)
		key = jax.random.PRNGKey(seed)
		key, subkey = jax.random.split(key)
		encoded_images = self.p_generate(
			self.model,
			tokens,
			shard_prng_key(subkey),
			self.params,
			request.k,
			request.p,
			request.temp,
			request.cond_scale
		)
		encoded_images = encoded_images.sequences[..., 1:]
		encoded_images = unreplicate(encoded_images)
		decoded_images = self.p_decode(self.vqgan, encoded_images, self.vqgan_params)
		decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
		
		images = [np.asarray(x * 255, dtype=np.uint8) for x in decoded_images]
		
		if not request.return_single:
			return images
		clip_inputs = self.clip_processor(
			text=[request.text] * jax.device_count(),
			images=images,
			return_tensors="np",
			padding="max_length",
			max_length=77,
			truncation=True,
		).data
		logits = self.p_clip(self.clip, shard(clip_inputs), self.clip_params)
		p = request.n
		logits = np.asarray([logits[:, i::p, i] for i in range(p)]).squeeze()
		best = logits.argmax()
		return [images[best]]
		
def image_to_str(img):
	buf = BytesIO()
	img = Image.fromarray(img)
	img.save(buf, format='PNG')
	img_str = b64encode(buf.getvalue())
	return img_str.decode('utf-8')

def environ_default(name, val):
	if name in os.environ:
		return
	os.environ[name] = val

def create_app():
	app = Flask(__name__)
	environ_default('MODEL_DIR', '/model')
	dalle = DallePModel(os.environ['MODEL_DIR'])
	
	@app.route("/")
	def base():
		return "<p>Running</>"
		
	@app.route("/predict", methods=['GET', 'POST'])
	def predict():
		r = DalleRequest(request.json)
		imgs = dalle(r)
		return jsonify({'images': [image_to_str(x) for x in imgs]})
		
	return app
	
if __name__ == "__main__":
	app = create_app()