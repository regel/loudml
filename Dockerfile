ARG extras_require=cpu
ARG base_image=python:3.6-slim 
ARG gpu=false

FROM $base_image AS builder
ARG extras_require

RUN apt-get update -qq && \
  apt-get install -y --no-install-recommends \
  build-essential \
  python3-venv \
  pkg-config \
  git-core \
  openssl

# copy files
COPY . /build/

# change working directory
WORKDIR /build

# install dependencies
RUN python -m venv /opt/venv && \
  . /opt/venv/bin/activate && \
  pip install --no-cache-dir -U 'pip<20' && \
  pip install --no-cache-dir -r requirements.txt && \
  pip install --no-cache-dir .[$extras_require] && \
  rm -rf dist *.egg-info

# start a new build stage
FROM $base_image AS runner
ARG gpu

# copy everything from /opt
COPY --from=builder /opt/venv /opt/venv

RUN mkdir /var/lib/loudml && \
	chgrp -R 0 /var/lib/loudml && \
	chmod -R g=u /var/lib/loudml && \
	mkdir /etc/loudml && \
{ if [ "x$gpu" = "xtrue" ] ; then /bin/echo -e '\
---\n\
buckets: []\n\
server:\n\
  listen: 0.0.0.0:8077\n\
inference:\n\
  num_cpus: 1\n\
  num_gpus: 0\n\
training:\n\
  num_cpus: 1\n\
  num_gpus: 1\n\
  batch_size: 256\n\
' \
>> /etc/loudml/config.yml ; \
else /bin/echo -e '\
---\n\
buckets: []\n\
server:\n\
  listen: 0.0.0.0:8077\n\
' \
>> /etc/loudml/config.yml ; \
fi ; }

# make sure we use the virtualenv
ENV PATH="/opt/venv/bin:$PATH"

# update permissions & change user to not run as root
WORKDIR /app
RUN chgrp -R 0 /app && chmod -R g=u /app
USER 1001

# create a volume for temporary data
VOLUME /tmp

# change shell
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# the entry point
EXPOSE 8077
LABEL maintainer="packaging@loudml.io"
ENTRYPOINT ["loudmld"]
