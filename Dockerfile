FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# FROM python:3.9

# WORKDIR /code

# COPY ./requirements.txt /code/requirements.txt

# RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# for open cv
RUN apt-get update && apt-get install libgl1  -y

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/ChatAnything

COPY --chown=user . $HOME/ChatAnything

RUN pip install -r requirements.txt

CMD python app.py
