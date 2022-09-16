from flask import Flask, request, jsonify
from markupsafe import escape
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import os
import torch
from io import BytesIO
from base64 import b64encode


def image_to_str(img):
    buf = BytesIO()
    img.save(buf, format='PNG')
    img_str = b64encode(buf.getvalue())
    return img_str.decode('utf-8')


def create_app():
    app = Flask(__name__)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    access_token = os.environ["ACCESS_TOKEN"]
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        use_auth_token=access_token,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        revision="fp16" if torch.cuda.is_available() else None,
    ).to("cuda")

    @app.route("/")
    def base():
        return "<p>Running</>"

    @app.route("/predict", methods=['GET', 'POST'])
    def predict():
        s = request.json['text']
        with autocast('cuda', dtype=torch.half):
            images = pipe(s, guidance_scale=7.)
        return jsonify({
            'nsfw': images['nsfw_content_detected'],
            'image': image_to_str(images['sample'][0])
        })


if __name__ == "__main__":
    app = create_app()
