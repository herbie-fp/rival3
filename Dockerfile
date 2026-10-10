# Linux amd64 artifact for Rival's timing evaluation. The digest is the
# linux/amd64 Ubuntu 24.04 manifest, resolved on 2026-10-10.
FROM ubuntu@sha256:f610ab94648195aa356059f5b41d6085c9d4d903c072430cdd1af7bdb646106b

ARG RACKET_VERSION=8.18
ARG RUST_VERSION=1.85.0
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/racket/bin:/root/.cargo/bin:${PATH} \
    MPLBACKEND=Agg

# These are the Ubuntu 24.04 (noble) package revisions used for the arithmetic
# stack. Racket is installed below because noble ships Racket 8.10, not 8.18.
RUN apt-get update \
 && apt-get install --yes --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
      libgmp-dev=2:6.3.0+dfsg-2ubuntu6.1 \
      libmpfi-dev=1.5.3+ds-6build1 \
      libmpfr-dev=4.2.1-1build1.1 \
      python3-matplotlib=3.6.3-1ubuntu5 \
      python3-numpy=1:1.26.4+ds-6ubuntu1 \
      python3-pandas=2.1.4+dfsg-7 \
      sollya=8.0+ds-2build3 \
      xz-utils \
 && rm -rf /var/lib/apt/lists/*

RUN curl --fail --location --silent --show-error \
      --output /tmp/racket-installer.sh \
      "https://mirror.racket-lang.org/installers/${RACKET_VERSION}/racket-${RACKET_VERSION}-x86_64-linux-buster-cs.sh" \
 && echo "72dfa7602edfe5bd2761af246709ebe93e0fd7f3a0fe98e39bd00418ab7abc70  /tmp/racket-installer.sh" | sha256sum --check \
 && sh /tmp/racket-installer.sh --in-place --dest /opt/racket \
 && rm /tmp/racket-installer.sh

RUN curl --fail --location --silent --show-error --output /tmp/rustup-init.sh https://sh.rustup.rs \
 && sh /tmp/rustup-init.sh -y --profile minimal --default-toolchain "${RUST_VERSION}" \
 && rm /tmp/rustup-init.sh

WORKDIR /workspace
COPY . /workspace

# Compile the Racket FFI once at build time. The data remains compressed in the
# image and is expanded only into /tmp while an evaluation is running.
RUN make install

CMD ["bash"]
