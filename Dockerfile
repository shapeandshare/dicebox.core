# TODO: swap to aws ml image ...
# from tensorflow/tensorflow:latest-devel-gpu

# Move source over
COPY /src /tmp/src


# Data ..
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get remove python3 -y
RUN apt-get install python3.7 -y
RUN apt-get install python3-venv -y

# python3.7
RUN python3.7

# TODO: Entry point