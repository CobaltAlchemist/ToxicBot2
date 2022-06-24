FROM docker pull huggingface/transformers-pytorch-gpu

RUN python3 -m pip install --no-cache-dir --upgrade Flask
	
ENV FLASK_APP=run
EXPOSE 5000
COPY run.py

CMD [ "python3" "-m" "flask" "run" "--host=0.0.0.0" ]