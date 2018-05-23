FROM debian:stretch-slim

LABEL maintainer="packaging@redmintnetwork.fr"
EXPOSE 8077

COPY repo /repo

RUN echo "deb file:///repo stretch main" > /etc/apt/sources.list.d/loudml-1.x.list && \
    apt-get update

RUN apt-get install -y python3-pip

RUN apt-get install -y --allow-unauthenticated \
    loudml loudml-api loudml-base loudml-influx loudml-elastic loudml-api

COPY loudml.yml /etc/loudml/config.yml
ENTRYPOINT /usr/bin/loudmld
