FROM jupyter/scipy-notebook

WORKDIR /var/lib/python

COPY . . 

RUN pip install --no-cache-dir -r requirements.txt

# COPY train.csv ./train.csv
# COPY test.csv ./test.csv
# COPY train.py ./train.py
# COPY inference.py ./inference.py

# RUN python3 train.py # this will run in the run stage in Jenkins

