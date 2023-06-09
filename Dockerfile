FROM ubuntu

RUN apt-get update && apt-get install -y --no-install-recommends --no-install-suggests wget vim git sudo make gcc g++ python3 python3-pip
RUN pip3 install tensorflow
RUN pip3 install numpy

WORKDIR /home