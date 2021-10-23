
# https://towardsdatascience.com/build-and-run-a-docker-container-for-your-machine-learning-model-60209c2d7a7f

FROM jupyter/scipy-notebook

RUN pip install joblib

# We can for example bind a host directory in the container using WORKDIR in the Dockerfile:
# ./mydata
# WORKDIR /home/k8s/minikube/dockers-images/5G/AI-for-IoT

# A volume is a persistent data stored in /var/lib/docker/volumes/...
# VOLUME /Users/Xavi/Desktop/code/data

COPY train.csv ./train.csv
COPY test.csv ./test.csv
COPY train-rul.py ./train-rul.py
COPY inference-rul.py ./inference-rul.py

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


RUN python3 train-rul.py