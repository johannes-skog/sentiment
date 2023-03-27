FROM pytorch/pytorch as base

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get -y install \
    curl \

# install az-cli
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash

RUN pip3 install pandas 

FROM base

COPY requirements.txt /tmp

RUN pip install -r /tmp/requirements.txt

RUN apt-get -y install screen

# ENTRYPOINT /.venv/bin/pip3 install -r /tmp/requirements.txt
# ENTRYPOINT ["/.venv/bin/pip3", "install",  "-r", "/tmp/requirements.txt"]
