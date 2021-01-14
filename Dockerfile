# https://github.com/aws/deep-learning-containers/blob/master/available_images.md
# https://github.com/aws/deep-learning-containers
# aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.3.1-gpu-py37-cu110-ubuntu18.04

# Update system
RUN apt-get update
# RUN apt-get upgrade -y

# Upgrade pip
RUN pip install --upgrade pip

# dependencies
COPY requirements.docker.txt /tmp
# COPY requirements.build.txt /tmp
COPY requirements.test.txt /tmp

# RUN pip install -r /tmp/requirements.build.txt
RUN pip install -r /tmp/requirements.test.txt
RUN pip install -r /tmp/requirements.docker.txt

# MNIST data set [load]
# http://yann.lecun.com/exdb/mnist/
# mnist
RUN mkdir /tmp/projects
RUN mkdir /tmp/projects/mnist
RUN mkdir /tmp/projects/mnist/data
RUN wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
RUN wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
RUN wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
RUN wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
RUN mv ./*.gz /tmp/projects/mnist/data

COPY projects/mnist/dicebox.config /tmp/projects/mnist
COPY projects/mnist/mnist_decoder.py /tmp/projects/mnist

RUN cd /tmp && python ./projects/mnist/mnist_decoder.py
RUN mv /tmp/data/test /tmp/projects/mnist/mnist_test_28x28
RUN mv /tmp/data/train /tmp/projects/mnist/mnist_train_28x28
RUN rm -r /tmp/data

# Move source over
COPY /src /tmp/src

# TODO: Entry point
# ENTRYPOINT ["top", "-b"]
# ENTRYPOINT ["top", "-b"]
