FROM python:3.12-slim@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704

RUN apt-get update && apt-get install -y --no-install-recommends \
      ruby-full build-essential zlib1g-dev \
      tmux \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Non-root user matching the host
ARG UID=1000
ARG GID=1000
ARG UNAME=dev

RUN groupadd -g 110 render || true \
 && groupadd -g ${GID} ${UNAME} || true \
 && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${UNAME} \
 && groupadd -g 44 video 2>/dev/null || true \
 && usermod -aG render,video ${UNAME}

ENV HOME=/home/${UNAME}
ENV GEM_HOME=${HOME}/gems
ENV PATH=${HOME}/.local/bin:${GEM_HOME}/bin:${PATH}

ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PIP_NO_CACHE_DIR=1
ENV MPLBACKEND=Agg

COPY dockers/profile.d/10-dev-path.sh /etc/profile.d/10-dev-path.sh

RUN chown ${UID}:${GID} /workspace
USER ${UNAME}

RUN gem install jekyll bundler

COPY --chown=${UID}:${GID} requirements.txt requirements.txt
RUN pip install -r requirements.txt

CMD ["bash"]
