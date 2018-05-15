FROM debian:stretch-slim

LABEL maintainer="packaging@redmintnetwork.fr"
EXPOSE 8077

RUN echo "deb http://loudml.s3-website-eu-west-1.amazonaws.com/repo/deb/ stretch main" > /etc/apt/sources.list.d/loudml-1.x.list && \
    apt-get update && \
    apt-get install -y python3-pip && \
    pip3 install 'dateutils>=0.6.6' && \
    pip3 install 'elasticsearch>=5.4.0' && \
    pip3 install 'influxdb>=5.0.0' && \
    pip3 install 'hyperopt==0.1' && \
    pip3 install 'tensorflow<=1.3.1,>=1.3.0'

RUN apt-get install -y --allow-unauthenticated loudml

COPY loudml.yml /etc/loudml/config.yml
ENTRYPOINT /usr/bin/loudmld
