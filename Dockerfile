FROM python:3.8.13-slim-bullseye

RUN python -m pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
	pillow \
	numpy \
	discord.py==1.7.3 \
	pandas \
	requests

RUN mkdir toxiclib
COPY toxiclib/__init__.py toxiclib/.
COPY toxiclib/constants.py toxiclib/.
COPY toxiclib/ddb.py toxiclib/.
COPY toxiclib/toxicbot.py toxiclib/.
COPY run.py .

CMD [ "python", "run.py" ]