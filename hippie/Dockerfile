FROM debian:stretch-slim

RUN apt-get update \
	&& apt-get install -y curl wget \
	&& apt-get purge -y

RUN useradd -ms /bin/bash nvm

# Copy and set permission for nvm directory
COPY . /home/nvm/.nvm/
RUN chown nvm:nvm -R "/home/nvm/.nvm"

# Set sudoer for "nvm"
RUN echo 'nvm ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers

ENV DOCKERIZE_VERSION v0.6.1
RUN wget https://github.com/jwilder/dockerize/releases/download/$DOCKERIZE_VERSION/dockerize-linux-amd64-$DOCKERIZE_VERSION.tar.gz \
    && tar -C /usr/local/bin -xzvf dockerize-linux-amd64-$DOCKERIZE_VERSION.tar.gz \
    && rm dockerize-linux-amd64-$DOCKERIZE_VERSION.tar.gz

# Switch to user "nvm" from now
USER nvm

# nvm
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.0/install.sh | bash
RUN echo 'export NVM_DIR="$HOME/.nvm"'                                       >> "$HOME/.bashrc"
RUN echo '[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"  # This loads nvm' >> "$HOME/.bashrc"
RUN echo '[ -s "$NVM_DIR/bash_completion" ] && . "$NVM_DIR/bash_completion" # This loads nvm bash_completion' >> "$HOME/.bashrc"

# Set WORKDIR to nvm directory
WORKDIR /home/nvm/.nvm

# nodejs and tools
RUN bash -c 'source $HOME/.nvm/nvm.sh   && \
    nvm install --lts                   && \
    npm install --prefix "$HOME/.nvm/"  && \
    npm ci'

#ENTRYPOINT ["/bin/bash"]
