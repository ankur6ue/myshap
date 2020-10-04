# This Dockerfile starts with the rayproject/ray image and installs application specific dependencies
# and copies application source files
FROM rayproject/ray as build-env
RUN conda --version
RUN apt-get update && apt-get -y install gcc && apt-get -y install g++
RUN mkdir /app
COPY requirements.txt /app
RUN pip install -r /app/requirements.txt
RUN apt-get install nano
# Now start with build-env and copy the application source and data
FROM build-env as shap-image
COPY winequality-red.csv /app
COPY source /app/source
COPY env.list /app
WORKDIR /app/