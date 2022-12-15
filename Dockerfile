FROM python:3.9

COPY requirements.txt /
COPY ncf_torch.py /
COPY training.py /
COPY data data/

WORKDIR /

RUN pip install -r requirements.txt