# This Dockerfile starts with the rayproject/ray image and installs application specific dependencies
# and copies application source files
FROM rayproject/ray
RUN conda --version
RUN apt-get update && apt-get -y install gcc && apt-get -y install g++
RUN mkdir /app
COPY winequality-red.csv /app
COPY source /app/source
COPY requirements.txt /app
RUN apt-get install nano
RUN pip install -r /app/requirements.txt
WORKDIR /app/