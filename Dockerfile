FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime

ENV PYTHONUNBUFFERED TRUE
HEALTHCHECK NONE

# PREREQUISITE
RUN apt-get update && apt-get install -y software-properties-common rsync
RUN add-apt-repository -y ppa:git-core/ppa && apt-get update && apt-get install -y apt-utils git libglib2.0-dev openjdk-11-jdk && apt-get update

# TORCHSERVE
RUN git clone https://github.com/pytorch/serve.git
RUN /bin/bash -c "cd ./serve && python ./ts_scripts/install_dependencies.py --cuda=cu102"
RUN pip install torchserve torch-model-archiver torch-workflow-archiver

COPY ./deployment/dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh

RUN mkdir -p /home/model-server/ && mkdir -p /home/model-server/tmp
COPY ./deployment/config.properties /home/model-server/config.properties

WORKDIR /home/model-server
ENV TEMP=/home/model-server/tmp
ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]
