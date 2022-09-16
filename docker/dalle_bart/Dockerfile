FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

RUN apt-get update && apt-get install -y \
  git \
  python3 \
  python3-pip \
  && rm -rf /var/lib/apt/lists/*
  
RUN pip3 install --no-cache-dir --upgrade https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.8+cuda11.cudnn805-cp38-none-manylinux2014_x86_64.whl
RUN pip3 install --no-cache-dir --upgrade https://storage.googleapis.com/jax-releases/jax/jax-0.3.8.tar.gz

RUN pip install --upgrade --no-cache-dir -q \
  git+https://github.com/borisdayma/dalle-mini.git \
  git+https://github.com/patil-suraj/vqgan-jax.git

RUN pip install --no-cache-dir flask
RUN wandb login --anonymously

COPY inference.py .

ENV FLASK_APP=service
ENV FLASK_RUN_EXTRA_FILES=/usr/bin/flask/plugins
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_ALLOCATOR=platform

CMD [ "python3", "inference.py" ]