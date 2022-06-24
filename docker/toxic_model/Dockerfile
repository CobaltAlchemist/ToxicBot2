FROM docker pull huggingface/transformers-pytorch-gpu

RUN python3 -m pip install --no-cache-dir --upgrade \
	Flask

COPY run.py

CMD [ "python3" "run.py" ]